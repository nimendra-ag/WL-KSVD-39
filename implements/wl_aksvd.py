import os
import time
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from implements.WL_KSVD import WL_KSVD
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from utils.graph_data import GraphDataLoader

N_RUNS = 5
n_dimensions = 128

DATASET_ID = 1
DATASET_NAME = f"NCI_full / {DATASET_ID}total-connect.sdf"

graphDataLoader = GraphDataLoader(dataset_id=DATASET_ID)
graphs, y = graphDataLoader.nci_full_graphs, graphDataLoader.nci_full_labels

# metrics collected across runs, per model
results = {
    "Logistic Regression": {"auc": [], "f1": []},
    "Gradient Boosting": {"auc": [], "f1": []},
    "SVM": {"auc": [], "f1": []},
    "Random Forest": {"auc": [], "f1": []},
}

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
    model = WL_KSVD(seed=seed, dimensions=n_dimensions)
    model.fit(G_vocab_train)
    X_vocab_train = model.get_embedding()

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
    f"Generated: {datetime.now().isoformat(timespec='seconds')}",
    f"Total execution time: {duration:.2f} seconds",
    "",
]

print("\n===== Summary (mean +/- std over {} runs) =====".format(N_RUNS))
for model_name, metrics in results.items():
    auc_mean, auc_std = np.mean(metrics["auc"]), np.std(metrics["auc"])
    f1_mean, f1_std = np.mean(metrics["f1"]), np.std(metrics["f1"])
    line = f"{model_name}: AUC = {auc_mean:.4f} +/- {auc_std:.4f}, F1 = {f1_mean:.4f} +/- {f1_std:.4f}"
    print(line)
    summary_lines.append(line)
    summary_lines.append(f"  AUC per run: {['%.4f' % v for v in metrics['auc']]}")
    summary_lines.append(f"  F1 per run:  {['%.4f' % v for v in metrics['f1']]}")

# Save results to file
results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(results_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_path = os.path.join(results_dir, f"wl_aksvd_results_nci_{graphDataLoader.dataset_id}_{n_dimensions}_{timestamp}.txt")

with open(results_path, "w") as f:
    f.write("\n".join(summary_lines) + "\n")

print(f"\nResults saved to {results_path}")
