"""
Evaluation and model comparison.

Produces (every figure saved at 300 dpi):

  Quantitative   - metrics table (CSV), grouped bar chart, per-model confusion
                   matrices, ROC-overlay, PR-overlay.
  Computational  - #params, weights size, training time, inference latency /
                   throughput  -> comparison bars + accuracy-vs-params scatter.
  Qualitative    - training-history curves and a grid of sample test
                   predictions for every model.

Run:  python evaluate.py
"""
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score,
    matthews_corrcoef, balanced_accuracy_score)

import config as C
C.enable_gpu_memory_growth()
import data_loader as D
from models import build_model

plt.rcParams["savefig.dpi"] = C.PLOT_DPI
plt.rcParams["figure.dpi"] = 110
sns.set_style("whitegrid")
_PALETTE = {"EfficientNetB4": "#1f77b4", "MobileNetV2": "#ff7f0e",
            "ResNet50": "#2ca02c", "DenseNet121": "#d62728"}


# --------------------------------------------------------------------------- #
def load_trained_model(name):
    model, _, last_conv = build_model(name)
    model.load_weights(os.path.join(C.MODELS_DIR, f"{name}.weights.h5"))
    return model, last_conv


def _specificity(cm):
    tn, fp = cm[0, 0], cm[0, 1]
    return tn / (tn + fp) if (tn + fp) else 0.0


def _measure_latency(model, X, n=100):
    """Return (ms/image single-sample latency, images/sec batched throughput)."""
    x1 = X[:1]
    model.predict(x1, verbose=0)                       # warm-up
    t0 = time.time()
    for i in range(n):
        model.predict(X[i:i + 1], verbose=0)
    latency_ms = (time.time() - t0) / n * 1000.0

    t0 = time.time()
    model.predict(X, batch_size=C.BATCH_SIZE, verbose=0)
    throughput = len(X) / (time.time() - t0)
    return latency_ms, throughput


# --------------------------------------------------------------------------- #
def evaluate_all():
    cache = D.build_or_load_cache()
    X_test, y_test = cache["X_test"], cache["y_test"].astype(int)

    rows, roc_data, pr_data, cms, preds_store = [], {}, {}, {}, {}

    for name in C.MODEL_NAMES:
        wpath = os.path.join(C.MODELS_DIR, f"{name}.weights.h5")
        if not os.path.exists(wpath):
            print(f"[evaluate] skipping {name} (no trained weights)")
            continue
        print(f"[evaluate] {name} ...")
        model, _ = load_trained_model(name)

        proba = model.predict(X_test, batch_size=C.BATCH_SIZE, verbose=0).ravel()
        pred = (proba >= 0.5).astype(int)
        preds_store[name] = (proba, pred)

        cm = confusion_matrix(y_test, pred)
        cms[name] = cm
        latency_ms, throughput = _measure_latency(model, X_test)

        # computational info saved during training (if present)
        comp_path = os.path.join(C.RESULTS_DIR, f"{name}_compute.json")
        comp = json.load(open(comp_path)) if os.path.exists(comp_path) else {}

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Balanced Acc": balanced_accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall (Sens)": recall_score(y_test, pred, zero_division=0),
            "Specificity": _specificity(cm),
            "F1": f1_score(y_test, pred, zero_division=0),
            "AUC": roc_auc_score(y_test, proba),
            "AP (PR-AUC)": average_precision_score(y_test, proba),
            "MCC": matthews_corrcoef(y_test, pred),
            "Params (M)": comp.get("total_params", model.count_params()) / 1e6,
            "Size (MB)": comp.get("weights_size_mb", np.nan),
            "Train time (min)": comp.get("train_time_sec", np.nan) / 60.0
                if comp.get("train_time_sec") else np.nan,
            "Latency (ms/img)": latency_ms,
            "Throughput (img/s)": throughput,
        })

        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_data[name] = (fpr, tpr, roc_auc_score(y_test, proba))
        prec, rec, _ = precision_recall_curve(y_test, proba)
        pr_data[name] = (rec, prec, average_precision_score(y_test, proba))

        tf.keras.backend.clear_session()

    if not rows:
        print("No trained models found - run train.py first.")
        return

    df = pd.DataFrame(rows).set_index("Model").round(4)
    df.to_csv(os.path.join(C.RESULTS_DIR, "comparison_metrics.csv"))
    print("\n===== Comparison metrics =====")
    print(df.to_string())

    # ---- figures ------------------------------------------------------------
    _plot_metric_bars(df)
    _plot_confusion_matrices(cms, y_test)
    _plot_roc(roc_data)
    _plot_pr(pr_data)
    _plot_computational(df)
    _plot_history_curves()
    _plot_sample_predictions(cache, preds_store)
    print(f"\nAll evaluation artefacts saved under: {C.PLOTS_DIR}")
    return df


# --------------------------------------------------------------------------- #
def _save(fig, fname):
    path = os.path.join(C.PLOTS_DIR, fname)
    fig.savefig(path, dpi=C.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved", os.path.basename(path))


def _plot_metric_bars(df):
    metrics = ["Accuracy", "Balanced Acc", "Precision", "Recall (Sens)",
               "Specificity", "F1", "AUC", "MCC"]
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(metrics))
    w = 0.8 / len(df)
    for i, (name, r) in enumerate(df.iterrows()):
        ax.bar(x + i * w, [r[m] for m in metrics], w, label=name,
               color=_PALETTE.get(name))
    ax.set_xticks(x + w * (len(df) - 1) / 2)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_title("Quantitative comparison on the BUS-BRA test set")
    ax.legend(ncol=len(df), loc="lower center", bbox_to_anchor=(0.5, -0.28))
    _save(fig, "quantitative_metric_bars.png")


def _plot_confusion_matrices(cms, y_test):
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES)
        acc = np.trace(cm) / cm.sum()
        ax.set_title(f"{name}\nacc={acc:.3f}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.suptitle("Confusion matrices (test set)", y=1.03, fontsize=13)
    _save(fig, "confusion_matrices.png")


def _plot_roc(roc_data):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, lw=2, color=_PALETTE.get(name),
                label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves"); ax.legend(loc="lower right")
    _save(fig, "roc_curves.png")


def _plot_pr(pr_data):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, (rec, prec, ap) in pr_data.items():
        ax.plot(rec, prec, lw=2, color=_PALETTE.get(name),
                label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves"); ax.legend(loc="lower left")
    _save(fig, "precision_recall_curves.png")


def _plot_computational(df):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    names = list(df.index)
    colors = [_PALETTE.get(n) for n in names]

    axes[0, 0].bar(names, df["Params (M)"], color=colors)
    axes[0, 0].set_title("Model size - parameters (M)"); axes[0, 0].set_ylabel("Millions")

    axes[0, 1].bar(names, df["Latency (ms/img)"], color=colors)
    axes[0, 1].set_title("Single-image inference latency"); axes[0, 1].set_ylabel("ms / image")

    axes[1, 0].bar(names, df["Train time (min)"], color=colors)
    axes[1, 0].set_title("Training wall-clock time"); axes[1, 0].set_ylabel("minutes")

    ax = axes[1, 1]
    ax.scatter(df["Params (M)"], df["Accuracy"], s=140, c=colors)
    for n in names:
        ax.annotate(n, (df.loc[n, "Params (M)"], df.loc[n, "Accuracy"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Params (M)"); ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs. model size")
    for a in axes.ravel()[:3]:
        a.tick_params(axis="x", rotation=20)
    fig.suptitle("Computational comparison", y=1.01, fontsize=14)
    _save(fig, "computational_comparison.png")


def _plot_history_curves():
    files = {n: os.path.join(C.RESULTS_DIR, f"{n}_history.json")
             for n in C.MODEL_NAMES}
    files = {n: p for n, p in files.items() if os.path.exists(p)}
    if not files:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, p in files.items():
        h = json.load(open(p))
        c = _PALETTE.get(name)
        if "val_loss" in h:
            axes[0].plot(h["val_loss"], color=c, label=name)
        if "val_auc" in h:
            axes[1].plot(h["val_auc"], color=c, label=name)
    axes[0].set_title("Validation loss"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[1].set_title("Validation AUC"); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("AUC")
    axes[0].legend(); axes[1].legend()
    _save(fig, "training_history.png")


def _plot_sample_predictions(cache, preds_store, n_samples=8):
    X_test, y_test = cache["X_test"], cache["y_test"].astype(int)
    rng = np.random.RandomState(C.SEED)
    idx = rng.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)
    models = list(preds_store.keys())

    fig, axes = plt.subplots(len(models), len(idx),
                             figsize=(2.0 * len(idx), 2.2 * len(models)))
    axes = np.atleast_2d(axes)
    for r, name in enumerate(models):
        proba, pred = preds_store[name]
        for c, i in enumerate(idx):
            ax = axes[r, c]
            ax.imshow(X_test[i].squeeze(), cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            ok = pred[i] == y_test[i]
            ax.set_title(f"P:{C.CLASS_NAMES[pred[i]]}({proba[i]:.2f})",
                         color="green" if ok else "red", fontsize=8)
            if c == 0:
                ax.set_ylabel(name, fontsize=9)
            if r == 0:
                ax.text(0.5, 1.35, f"True:{C.CLASS_NAMES[y_test[i]]}",
                        transform=ax.transAxes, ha="center", fontsize=8,
                        fontweight="bold")
    fig.suptitle("Qualitative results - sample test predictions", y=1.01)
    _save(fig, "qualitative_sample_predictions.png")


if __name__ == "__main__":
    evaluate_all()
