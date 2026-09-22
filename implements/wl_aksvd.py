import os
import time
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from implements.WL_KSVD import WL_KSVD
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
    recall_score, precision_score, matthews_corrcoef,
    balanced_accuracy_score, confusion_matrix,
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from utils.graph_data import GraphDataLoader

N_RUNS = 5
# Dictionary size: number of KSVD atoms, i.e. the embedding dimensionality.
n_dimensions = 128

DATASET_ID = 1
DATASET_NAME = f"NCI_full / {DATASET_ID}total-connect.sdf"

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

graphDataLoader = GraphDataLoader(dataset_id=DATASET_ID)
graphs, y = graphDataLoader.nci_full_graphs, graphDataLoader.nci_full_labels


# ── Helper: evaluate one classifier and return all metrics ──────────────
def evaluate_classifier(clf, X_test, y_test):
    """Return a dict of metrics for a fitted classifier."""
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[-1, 1]).ravel()

    return {
        # ── global metrics ──
        "roc_auc":      roc_auc_score(y_test, y_proba),
        "pr_auc":       average_precision_score(y_test, y_proba),
        "f1":           f1_score(y_test, y_pred),
        "mcc":          matthews_corrcoef(y_test, y_pred),
        "balanced_acc": balanced_accuracy_score(y_test, y_pred),
        # ── minority (active, label=1) ──
        "min_recall":    tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "min_precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "min_tp": tp,
        "min_fn": fn,
        # ── majority (inactive, label=-1) ──
        "maj_recall":    tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "maj_precision": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        "maj_tn": tn,
        "maj_fp": fp,
    }


# ── Metric keys tracked per model ──────────────────────────────────────
METRIC_KEYS = [
    "roc_auc", "pr_auc", "f1", "mcc", "balanced_acc",
    "min_recall", "min_precision",
    "maj_recall", "maj_precision",
]

MODEL_NAMES = [
    "Logistic Regression",
    "Gradient Boosting",
    "SVM",
    "Random Forest",
]

results = {name: {k: [] for k in METRIC_KEYS} for name in MODEL_NAMES}

# per-run bookkeeping
run_seeds = []
run_features_selected = []
run_features_total = []

start_time = time.perf_counter()

for run in range(N_RUNS):
    seed = 42 + run
    print(f"\n===== Run {run + 1}/{N_RUNS} (seed={seed}) =====")

    # Over-fit protection
    G_train, G_test, y_train, y_test = train_test_split(
        graphs, y, test_size=0.2, random_state=seed
    )

    # divide the train set further into vocab training and ML training sets
    G_vocab_train, G_ML_train, y_vocab_train, y_ML_train = train_test_split(
        G_train, y_train, test_size=0.75, random_state=seed
    )

    # Fit the WL+KSVD model on the vocab training set
    print("Fitting the embedding model")
    model = WL_KSVD(seed=seed, dimensions=n_dimensions)
    model.fit(G_vocab_train)
    X_vocab_train = model.get_embedding()

    # Vocabulary size after the top-n_vocab trim, out of the full WL feature set
    n_features_selected = model.n_vocab
    n_features_total = model.n_features_total_
    run_seeds.append(seed)
    run_features_selected.append(n_features_selected)
    run_features_total.append(n_features_total)
    print(f"Features kept: {n_features_selected} / {n_features_total}")

    # Infer the embedding of the ML training set
    X_ML_train = model.infer(G_ML_train)
    X_ML_test = model.infer(G_test)

    # Scaling the embedding such that sparsity is preserved
    scaler = MaxAbsScaler()
    X_ML_train_scaled = scaler.fit_transform(X_ML_train)
    X_ML_test_scaled = scaler.transform(X_ML_test)

    # ── Applying ML models ──────────────────────────────────────────────
    print("Applying ML models")

    classifiers = {
        "Logistic Regression": LogisticRegression(random_state=seed),
        "Gradient Boosting":   GradientBoostingClassifier(random_state=seed),
        "SVM":                 CalibratedClassifierCV(LinearSVC(random_state=seed)),
        "Random Forest":       RandomForestClassifier(random_state=seed),
    }

    for model_name, clf in classifiers.items():
        print(f"Predicting with {model_name}")
        clf.fit(X_ML_train_scaled, y_ML_train)
        m = evaluate_classifier(clf, X_ML_test_scaled, y_test)

        for k in METRIC_KEYS:
            results[model_name][k].append(m[k])

        print(f"  ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
              f"F1={m['f1']:.4f}  MCC={m['mcc']:.4f}  "
              f"BalAcc={m['balanced_acc']:.4f}")
        print(f"  Minority  → recall={m['min_recall']:.4f}  "
              f"precision={m['min_precision']:.4f}  "
              f"(TP={m['min_tp']}, FN={m['min_fn']})")
        print(f"  Majority  → recall={m['maj_recall']:.4f}  "
              f"precision={m['maj_precision']:.4f}  "
              f"(TN={m['maj_tn']}, FP={m['maj_fp']})")

end_time = time.perf_counter()
duration = end_time - start_time


# ── Summary ─────────────────────────────────────────────────────────────
summary_lines = [
    f"WL+KSVD (baseline / frequency) results over {N_RUNS} runs",
    f"Dataset: {DATASET_NAME} ({len(graphs)} graphs)",
    f"Dictionary size (KSVD atoms): {n_dimensions}",
    f"Generated: {datetime.now().isoformat(timespec='seconds')}",
    f"Total execution time: {duration:.2f} seconds",
    "",
]

print(f"\n{'='*80}")
print(f"Summary (mean ± std over {N_RUNS} runs)")
print(f"Dataset: {DATASET_NAME} | Dictionary size: {n_dimensions}")
print(f"{'='*80}")

for model_name in MODEL_NAMES:
    metrics = results[model_name]

    # ── Per-metric mean ± std ──
    line_global = (
        f"{model_name}:\n"
        f"  ROC-AUC      = {np.mean(metrics['roc_auc']):.4f} ± {np.std(metrics['roc_auc']):.4f}\n"
        f"  PR-AUC       = {np.mean(metrics['pr_auc']):.4f} ± {np.std(metrics['pr_auc']):.4f}\n"
        f"  F1           = {np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}\n"
        f"  MCC          = {np.mean(metrics['mcc']):.4f} ± {np.std(metrics['mcc']):.4f}\n"
        f"  Balanced Acc = {np.mean(metrics['balanced_acc']):.4f} ± {np.std(metrics['balanced_acc']):.4f}\n"
        f"  ── Minority (active) ──\n"
        f"    Recall     = {np.mean(metrics['min_recall']):.4f} ± {np.std(metrics['min_recall']):.4f}\n"
        f"    Precision  = {np.mean(metrics['min_precision']):.4f} ± {np.std(metrics['min_precision']):.4f}\n"
        f"  ── Majority (inactive) ──\n"
        f"    Recall     = {np.mean(metrics['maj_recall']):.4f} ± {np.std(metrics['maj_recall']):.4f}\n"
        f"    Precision  = {np.mean(metrics['maj_precision']):.4f} ± {np.std(metrics['maj_precision']):.4f}"
    )
    print(line_global)
    summary_lines.append(line_global)

    # ── Per-run detail ──
    header = (f"  {'run':>4} {'seed':>6} {'features':>12} "
              f"{'ROC-AUC':>8} {'PR-AUC':>8} {'F1':>6} {'MCC':>6} "
              f"{'MinRec':>7} {'MinPre':>7} {'MajRec':>7} {'MajPre':>7}")
    print(header)
    summary_lines.append(header)

    for i in range(N_RUNS):
        features = f"{run_features_selected[i]}/{run_features_total[i]}"
        row = (
            f"  {i+1:>4} {run_seeds[i]:>6} {features:>12} "
            f"{metrics['roc_auc'][i]:>8.4f} "
            f"{metrics['pr_auc'][i]:>8.4f} "
            f"{metrics['f1'][i]:>6.4f} "
            f"{metrics['mcc'][i]:>6.4f} "
            f"{metrics['min_recall'][i]:>7.4f} "
            f"{metrics['min_precision'][i]:>7.4f} "
            f"{metrics['maj_recall'][i]:>7.4f} "
            f"{metrics['maj_precision'][i]:>7.4f}"
        )
        print(row)
        summary_lines.append(row)
    summary_lines.append("")

# Save results to file
results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(
    results_dir,
    f"wl_aksvd_results_nci_{graphDataLoader.dataset_id}_{n_dimensions}_{RUN_TIMESTAMP}.txt",
)

with open(results_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines) + "\n")

print(f"\nResults saved to {results_path}")