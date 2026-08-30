import os
import time
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from implements.WL_KSVD_improved import WL_KSVD
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from utils.graph_data import GraphDataLoader
from utils.elbow import plot_cdf_curve

N_RUNS = 5
# Dictionary size: number of KSVD atoms, i.e. the embedding dimensionality.
n_dimensions = 128
DATASET_ID = 1
DATASET_NAME = f"NCI_full / {DATASET_ID}total-connect.sdf"
# Slug used to name this dataset's output folders (one per dataset, not per run).
DATASET_SLUG = f"nci_full_{DATASET_ID}"

# One timestamp for the whole execution, shared by the results file and the
# per-seed analytics folder so the two can be matched up afterwards.
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
# results/analytics/<dataset>/<execution timestamp>/seed_<seed>_cdf.png
ANALYTICS_DIR = os.path.join(RESULTS_DIR, "analytics", DATASET_SLUG, RUN_TIMESTAMP)

graphDataLoader = GraphDataLoader(dataset_id=DATASET_ID)
graphs, y = graphDataLoader.nci_full_graphs, graphDataLoader.nci_full_labels

# metrics collected across runs, per model
results = {
    "Logistic Regression": {"auc": [], "f1": []},
    "Gradient Boosting": {"auc": [], "f1": []},
    "SVM": {"auc": [], "f1": []},
    "Random Forest": {"auc": [], "f1": []},
}

# per-run bookkeeping: the seed and the vocabulary size the adaptive cut kept.
# Feature counts are shared by every model in a run, so they live here rather
# than being duplicated inside `results`.
run_seeds = []
run_features_selected = []
run_features_total = []

start_time = time.perf_counter()

for run in range(N_RUNS):
    seed = 42 + run
    print(f"\n===== Run {run + 1}/{N_RUNS} (seed={seed}) =====")

    # Over-fit protection
    G_train, G_test, y_train, y_test = train_test_split(graphs, y, test_size=0.2, random_state=seed)

    # divide the train set further into vocab training and ML training sets
    G_vocab_train, G_ML_train, y_vocab_train, y_ML_train = train_test_split(
        G_train, y_train, test_size=0.75, random_state=seed
    )

    # Fit the WL+KSVD model on the vocab training set
    print("Fitting the embedding model")
    model = WL_KSVD(seed=seed, dimensions=n_dimensions, y_vocab_train=y_vocab_train)
    model.fit(G_vocab_train)
    X_vocab_train = model.get_embedding()

    # Vocabulary size after the adaptive (energy) cut, out of the full WL feature set
    n_features_selected = model.n_vocab
    n_features_total = len(model.selection_scores_)
    run_seeds.append(seed)
    run_features_selected.append(n_features_selected)
    run_features_total.append(n_features_total)
    print(f"Features kept: {n_features_selected} / {n_features_total}")

    # Curve B: how much of the summed discriminative score the kept features
    # hold. One figure per seed, since each seed trains on a different split.
    cdf_path = plot_cdf_curve(
        model.selection_scores_,
        os.path.join(ANALYTICS_DIR, f"seed_{seed}_cdf.png"),
        title=f"{DATASET_SLUG} / seed {seed}",
        selection=model.selection,
        energy=model.energy,
    )
    print(f"Feature-selection CDF saved to {cdf_path}")

    # Infer the embedding of the ML training set
    X_ML_train = model.infer(G_ML_train)
    X_ML_test = model.infer(G_test)

    # Scaling the embedding such that sparsity is preserved
    scaler = MaxAbsScaler()
    X_ML_train_scaled = scaler.fit_transform(X_ML_train)
    X_ML_test_scaled = scaler.transform(X_ML_test)

    # Applying ML models
    print("Applying ML models")

    print("Predicting with Logistic Regression")
    clf = LogisticRegression(random_state=seed).fit(X_ML_train_scaled, y_ML_train)
    y_proba = clf.predict_proba(X_ML_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, clf.predict(X_ML_test_scaled))
    results["Logistic Regression"]["auc"].append(auc)
    results["Logistic Regression"]["f1"].append(f1)
    print(f"AUC: {auc:.4f}, F1: {f1:.4f}")

    print("Predicting with Gradient Boosting")
    clf = GradientBoostingClassifier(random_state=seed).fit(X_ML_train_scaled, y_ML_train)
    y_proba = clf.predict_proba(X_ML_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, clf.predict(X_ML_test_scaled))
    results["Gradient Boosting"]["auc"].append(auc)
    results["Gradient Boosting"]["f1"].append(f1)
    print(f"AUC: {auc:.4f}, F1: {f1:.4f}")

    print("Predicting with SVM")
    clf = CalibratedClassifierCV(LinearSVC(random_state=seed)).fit(X_ML_train_scaled, y_ML_train)
    y_proba = clf.predict_proba(X_ML_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, clf.predict(X_ML_test_scaled))
    results["SVM"]["auc"].append(auc)
    results["SVM"]["f1"].append(f1)
    print(f"AUC: {auc:.4f}, F1: {f1:.4f}")

    print("Predicting with Random Forest")
    clf = RandomForestClassifier(random_state=seed).fit(X_ML_train_scaled, y_ML_train)
    y_proba = clf.predict_proba(X_ML_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, clf.predict(X_ML_test_scaled))
    results["Random Forest"]["auc"].append(auc)
    results["Random Forest"]["f1"].append(f1)
    print(f"AUC: {auc:.4f}, F1: {f1:.4f}")

end_time = time.perf_counter()
duration = end_time - start_time

# Summarize mean/std across runs
summary_lines = [
    f"WL+KSVD results over {N_RUNS} runs",
    f"Dataset: {DATASET_NAME} ({len(graphs)} graphs)",
    f"Dictionary size (KSVD atoms): {n_dimensions}",
    f"Generated: {datetime.now().isoformat(timespec='seconds')}",
    f"Total execution time: {duration:.2f} seconds",
    "",
]

header = f"{'run':>4} {'seed':>6} {'features':>10} {'AUC':>8} {'F1':>8}"

print("\n===== Summary (mean +/- std over {} runs) =====".format(N_RUNS))
print(f"Dataset: {DATASET_NAME} | Dictionary size: {n_dimensions}")
for model_name, metrics in results.items():
    auc_mean, auc_std = np.mean(metrics["auc"]), np.std(metrics["auc"])
    f1_mean, f1_std = np.mean(metrics["f1"]), np.std(metrics["f1"])
    line = f"{model_name}: AUC = {auc_mean:.4f} +/- {auc_std:.4f}, F1 = {f1_mean:.4f} +/- {f1_std:.4f}"
    print(line)
    summary_lines.append(line)

    # Per-run detail: seed and kept-feature count next to that run's scores
    print("  " + header)
    summary_lines.append("  " + header)
    for i in range(len(metrics["auc"])):
        features = f"{run_features_selected[i]}/{run_features_total[i]}"
        row = (f"{i + 1:>4} {run_seeds[i]:>6} {features:>10} "
               f"{metrics['auc'][i]:>8.4f} {metrics['f1'][i]:>8.4f}")
        print("  " + row)
        summary_lines.append("  " + row)
    summary_lines.append("")

# Save results to file
os.makedirs(RESULTS_DIR, exist_ok=True)
results_path = os.path.join(RESULTS_DIR, f"wl_aksvd_improved_results_nci_{graphDataLoader.dataset_id}_{n_dimensions}_{RUN_TIMESTAMP}.txt")

with open(results_path, "w") as f:
    f.write("\n".join(summary_lines) + "\n")

print(f"\nResults saved to {results_path}")
