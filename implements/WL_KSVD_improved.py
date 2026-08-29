import numpy as np
import random
import networkx as nx
from typing import List
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from karateclub.utils.treefeatures import WeisfeilerLehmanHashing
from ksvd import ApproximateKSVD
from collections import Counter
from utils.elbow import find_elbow_cut, find_energy_cut

class WL_KSVD():
    r""" An implementation of WL_KSVD

    Args:
        wl_iterations (int): Number of Weisfeiler-Lehman iterations. Default is 2.
        attributed (bool): Presence of graph attributes. Default is False.
        dimensions (int): Dimensionality of embedding. Default is 128.
        workers (int): Number of cores. Default is 4.
        down_sampling (float): Down sampling frequency. Default is 0.0001.
        epochs (int): Number of epochs. Default is 10.
        learning_rate (float): HogWild! learning rate. Default is 0.025.
        min_count (int): Minimal count of graph feature occurrences. Default is 5.
        seed (int): Random seed for the model. Default is 42.
        erase_base_features (bool): Erasing the base features. Default is False.

        n_vocab: Number of preliminary vocabulary size.  Default is 1000
        n_atoms: Number of dictionary elements (atoms). Default is 128
        n_non_zero_coefs: Number of nonzero coefficients to target. Default is 10
        max_iter: Maximum number of iterations. Default is 10
        tol: Tolerance for error. Default is 1e-6

    """

    def __init__(
        self,
        wl_iterations: int = 2,
        attributed: bool = True,
        dimensions: int = 1024,
        workers: int = 4,
        down_sampling: float = 0.0001,
        epochs: int = 10,
        learning_rate: float = 0.025,
        min_count: int = 5,
        min_features: int = 50,
        seed: int = 42,
        erase_base_features: bool = True,
        n_vocab: int = 1700,
        n_atoms: int = 128,
        n_non_zero_coefs: int = 10,
        max_iter: int = 10,
        tol: float = 1e-6,
        selection: str = "energy",
        energy: float = 0.99,
        y_vocab_train: list = []

    ):
        self.wl_iterations = wl_iterations
        self.attributed = attributed
        self.dimensions = dimensions
        self.workers = workers
        self.down_sampling = down_sampling
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.min_features = min_features
        self.seed = seed
        self.erase_base_features = erase_base_features
        self.n_vocab = n_vocab
        self.n_atoms = n_atoms
        self.n_non_zero_coefs = n_non_zero_coefs
        self.max_iter = max_iter
        self.tol = tol
        self.selection = selection
        self.energy = energy
        self.y_vocab_train = y_vocab_train


    def createWLhash(self, graph_list):

        documents = []
        # TODO: parallel implementation
        for graph in graph_list:
            g = self._check_graph(graph)

            document = WeisfeilerLehmanHashing(
                g, self.wl_iterations, self.attributed, self.erase_base_features)

            documents.append(document)

        documents = [
            TaggedDocument(words=doc.get_graph_features(), tags=[str(i)])
            for i, doc in enumerate(documents)
        ]

        return documents

    def create_vocab(self, corpus, labels):
        unique_classes = sorted(set(labels))
        n_classes = len(unique_classes)

        # Per-class document frequency and class sizes
        class_df = {c: Counter() for c in unique_classes}
        class_counts = Counter(labels)

        for doc, label in zip(corpus, labels):
            unique_words = set(doc.words)
            for word in unique_words:
                class_df[label][word] += 1

        # ── Expose to interpreter ─────────────────────────────────────────────
        # Stored before scoring so the interpreter always has the full statistics,
        # even for tokens that are later trimmed from the final vocabulary.
        self.class_df = class_df
        self.class_counts = class_counts
        # ─────────────────────────────────────────────────────────────────────

        all_words = set()
        for df in class_df.values():
            all_words.update(df.keys())

        scored_vocab = []

        for word in all_words:
            # Normalized document frequency per class
            p = {
                c: class_df[c][word] / class_counts[c]
                for c in unique_classes
            }

            # Mean pairwise Hellinger distance
            hellinger_sum = 0.0
            n_pairs = 0
            for i in range(n_classes):
                for j in range(i + 1, n_classes):
                    ci, cj = unique_classes[i], unique_classes[j]
                    hellinger_sum += abs(np.sqrt(p[ci]) - np.sqrt(p[cj]))
                    n_pairs += 1

            discriminative_score = hellinger_sum / n_pairs if n_pairs > 0 else 0.0

            total_presence = sum(p.values()) / n_classes

            score = total_presence * discriminative_score
            scored_vocab.append((word, score))

        scored_vocab.sort(key=lambda x: x[1], reverse=True)

        scores = np.array([x[1] for x in scored_vocab])

        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score
            scored_vocab = [(word, score / max_score) for word, score in scored_vocab]


        #-------------------------------------
        #------------ Mean - Std -------------
        #-------------------------------------
        # threshold = scores.mean() - scores.std()
        # trimmed_vocab = [item for item in scored_vocab if item[1] >= threshold]

        #-------------------------------------
        #-------- Adaptive Feature Cut -------
        #-------------------------------------
        # scored_vocab is sorted descending, so `scores` is a decreasing curve.
        # Both cuts are data-driven (no fixed percentile). The energy cut keeps
        # the top features covering `self.energy` of the summed score, which
        # reaches into the weak tail the elbow discards -- trading a little
        # runtime for the AUC that tail carries. See utils/elbow.py.
        print(f"Total Features {len(scores)}")
        if self.selection == "elbow":
            n_keep, threshold = find_elbow_cut(scores, sorted_desc=True)
            print(f"elbow cut at index {n_keep} (threshold {threshold:.6g})")
        elif self.selection == "energy":
            n_keep, threshold = find_energy_cut(
                scores, energy=self.energy, sorted_desc=True,
                min_keep=self.min_features,
            )
            print(f"energy cut ({self.energy:.4g}) at index {n_keep} "
                  f"(threshold {threshold:.6g})")
        elif self.selection == "none":
            # No cut: score and rank as usual, then keep everything. Exists for
            # the comparison arms in pca/, which need the full scored vocabulary
            # so that selection is the ONLY thing differing between arms.
            #
            # Not expressible as energy=1.0: features scoring exactly 0 (equal
            # presence in both classes) make the cumulative-score curve plateau
            # before the last rank, so an energy target of 1.0 still cuts them.
            # MUTAG has such features; nci_full does not. Hence an explicit path.
            n_keep, threshold = len(scores), float(scores[-1])
            print(f"no cut: keeping all {n_keep} features "
                  f"(lowest score {threshold:.6g})")
        else:
            raise ValueError(
                f"unknown selection method {self.selection!r}; "
                "expected 'energy', 'elbow' or 'none'"
            )

        # # Keep the full (pre-trim) score curve so the elbow can be plotted later,
        # # once the artifact bundle directory exists (see utils/export.py).
        self.selection_scores_ = scores
        trimmed_vocab = scored_vocab[:n_keep]

        print(f"Selected {len(trimmed_vocab)} features via adaptive selection")
        if len(trimmed_vocab) < self.min_features:
            trimmed_vocab = scored_vocab[:self.n_vocab]

        self.n_vocab = len(trimmed_vocab)
        return trimmed_vocab

    def calc_coefficients(self, corpus, vocab):

        sparse_vector = np.zeros([len(corpus), self.n_vocab])

        i = 0
        for corpus in corpus:
            words = corpus.words

            words_count = Counter(corpus.words)
            j = 0
            for atom, _ in vocab:
                sparse_vector[i][j] = words_count[atom]
                j = j + 1

            i = i + 1

        return sparse_vector

    def fit(self, graphs: List[nx.classes.graph.Graph]):
        """
        Fitting a WL_KSVD model.
        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        """
        self._set_seed()

        documents = self.createWLhash(graphs)


        self._vocab = self.create_vocab(documents, self.y_vocab_train)

        x = self.calc_coefficients(documents, self._vocab)

        aksvd = ApproximateKSVD(n_components=self.dimensions, max_iter=self.max_iter, tol=self.tol,
                 transform_n_nonzero_coefs=self.n_non_zero_coefs)
        self._dictionary = aksvd.fit(x).components_

        self._embedding = aksvd.transform(x)

        self.aksvd = aksvd

        return self

    def get_embedding(self) -> np.array:
        r"""Getting the embedding of graphs.
        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        return np.array(self._embedding)

    def infer(self, graphs) -> np.array:
        """Infer the graph embeddings.

        Arg types:
            * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
        Return types:
            * **embedding** *(Numpy array)* - The embedding of graphs.
        """
        self._set_seed()

        documents = self.createWLhash(graphs)

        x = self.calc_coefficients(documents, self._vocab)

        embedding = self.aksvd.transform(x)

        return embedding


    #-------------------------------------------------------------------------

    def _set_seed(self):
        """Creating the initial random seed."""
        random.seed(self.seed)
        np.random.seed(self.seed)

    @staticmethod
    def _ensure_integrity(graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
        """Ensure walk traversal conditions."""
        edge_list = [(index, index) for index in range(graph.number_of_nodes())]
        graph.add_edges_from(edge_list)

        return graph

    @staticmethod
    def _check_indexing(graph: nx.classes.graph.Graph):
        """Checking the consecutive numeric indexing."""
        numeric_indices = [index for index in range(graph.number_of_nodes())]
        node_indices = sorted([node for node in graph.nodes()])

        assert numeric_indices == node_indices, "The node indexing is wrong."

    def _check_graph(self, graph: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
        """Check the Karate Club assumptions about the graph."""
        self._check_indexing(graph)
        graph = self._ensure_integrity(graph)

        return graph

    def _check_graphs(self, graphs: List[nx.classes.graph.Graph]):
        """Check the Karate Club assumptions for a list of graphs."""
        graphs = [self._check_graph(graph) for graph in graphs]

        return graphs