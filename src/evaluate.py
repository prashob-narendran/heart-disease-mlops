"""Model evaluation: metrics and plots.

All plot functions save a PNG and return its path, so the training script
can log the same file to MLflow and keep a copy in reports/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import FEATURE_DESCRIPTIONS, FEATURES, RANDOM_SEED
from src.eda import CLASS_COLORS, INK, INK_2, SURFACE, set_style

set_style()

# Scoring used by GridSearchCV; also the metric names logged to MLflow.
# zero_division=0 keeps heavily regularised grid points (which may predict
# no positives in a fold) from flooding the log with warnings.
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}

MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]  # fixed order
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "svm_rbf": "SVM (RBF)",
}


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, path: Path) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "blues", ["#f0f6fd", "#2a78d6"]))
    labels = [["True negative", "False positive"],
              ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            dark = cm[i, j] > cm.max() * 0.6
            ax.text(j, i, f"{cm[i, j]}\n{labels[i][j]}", ha="center",
                    va="center", fontsize=9,
                    color="white" if dark else INK)
    ax.set_xticks([0, 1], ["No disease", "Disease"])
    ax.set_yticks([0, 1], ["No disease", "Disease"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, loc="left")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curves(curves: dict, path: Path, title="ROC curve - test set"):
    """curves: {model_name: (y_true, y_proba)}"""
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    for i, (name, (y_true, y_proba)) in enumerate(curves.items()):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, linewidth=2, color=MODEL_COLORS[i % 4],
                label=f"{MODEL_LABELS.get(name, name)}  (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=INK_2, linewidth=1,
            label="Chance")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def permutation_importance_frame(model, X, y, n_repeats=30) -> pd.DataFrame:
    """Drop in ROC-AUC when each ORIGINAL feature is shuffled.

    Works for every model type and is computed on the 13 clinical features
    (before one-hot encoding), so it is directly interpretable.
    """
    res = permutation_importance(model, X, y, scoring="roc_auc",
                                 n_repeats=n_repeats,
                                 random_state=RANDOM_SEED, n_jobs=-1)
    return (pd.DataFrame({"feature": FEATURES,
                          "importance_mean": res.importances_mean,
                          "importance_std": res.importances_std})
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True))


def plot_importance(imp: pd.DataFrame, title: str, path: Path) -> Path:
    d = imp.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.barh([FEATURE_DESCRIPTIONS[f] for f in d["feature"]],
            d["importance_mean"], xerr=d["importance_std"],
            color=CLASS_COLORS[0], height=0.6,
            error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_xlabel("Mean drop in test ROC-AUC when feature is shuffled")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_comparison(table: pd.DataFrame, path: Path) -> Path:
    """Grouped bars of CV mean metrics with std error bars, per model."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    models = list(table.index)
    x = np.arange(len(metrics))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, m in enumerate(models):
        means = [table.loc[m, f"cv_{k}_mean"] for k in metrics]
        stds = [table.loc[m, f"cv_{k}_std"] for k in metrics]
        ax.bar(x + (i - (len(models) - 1) / 2) * w, means, width=w * 0.92,
               yerr=stds, color=MODEL_COLORS[i % 4],
               error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2},
               label=MODEL_LABELS.get(m, m), edgecolor=SURFACE)
    ax.set_xticks(x, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("5-fold CV score (mean ± std)")
    ax.set_title("Model comparison - cross-validation on training set",
                 loc="left")
    ax.legend(ncol=len(models), loc="upper left", bbox_to_anchor=(0, -0.1))
    ax.grid(axis="x", visible=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
