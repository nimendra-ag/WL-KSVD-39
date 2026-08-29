import os
import gzip
import networkx as nx
from rdkit import Chem
from rdkit import RDLogger


RDLogger.DisableLog("rdApp.*")


DATASETS_DIR = "datasets"


class GraphDataLoader:
    def __init__(self):
        self.nci_full_graphs, self.nci_full_labels = self.load_nci_full()
        self._initialized = True

    def load_nci_full(self, id=1):
        """
        id - (1, 33, 41, 47, 81, 83, 109, 123, 145)
        """
        print('Loading NCI dataset')
        DATASET_DIR = "datasets/NCI_full"  # change this
        graphs = []
        y = []

        filename = f"{id}total-connect.sdf"
        filepath = os.path.join(DATASET_DIR, filename)

        supplier = Chem.SDMolSupplier(filepath, removeHs=False)
        for mol in supplier:
            if mol is None:
                continue

            G = nx.Graph()

            # Add atoms as nodes
            for atom in mol.GetAtoms():
                G.add_node(
                    atom.GetIdx(),
                    feature=atom.GetSymbol()   # WL uses node labels
                )

            # Add bonds as edges
            for bond in mol.GetBonds():
                G.add_edge(
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    bond_type=str(bond.GetBondType()),
                    bond_order=bond.GetBondTypeAsDouble(),
                    aromatic=bond.GetIsAromatic(),
                    in_ring=bond.IsInRing(),
                    conjugated=bond.GetIsConjugated(),
                    stereo=str(bond.GetStereo())
                )

            # Get graph label
            # In NCI, class label is stored as a molecule property
            label = int(float(mol.GetProp("value")))
            graphs.append(G)
            y.append(label)

        print(f"Loaded {len(graphs)} graphs")
        return graphs, y