"""Exploratory data analysis helpers.

Every plotting function returns the matplotlib Figure and saves a PNG to
reports/figures/ so the same images can go straight into the report.
Used by notebooks/01_eda.ipynb and by `python -m src.eda`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

# Scripts, tests, CI and the API only SAVE figures, so use the file-only
# "Agg" backend. Without this, Windows defaults to Tk (an on-screen window
# toolkit) and fails if Tcl/Tk is not fully installed. Inside Jupyter
# (ipykernel loaded) the notebook's inline backend is left untouched.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    CATEGORY_LABELS,
    FEATURE_DESCRIPTIONS,
    FIGURES_DIR,
    NUMERIC_FEATURES,
    TARGET,
)

# ---------------------------------------------------------------- style
# Two-class palette (validated for colour-vision deficiency): blue = no
# disease, orange = disease. Text never uses the series colours.
CLASS_COLORS = {0: "#2a78d6", 1: "#eb6834"}
CLASS_NAMES = CATEGORY_LABELS[TARGET]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#c23b3a"]
)


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "font.size": 9,
        "figure.dpi": 110,
        "savefig.dpi": 150,
    })


set_style()


def _save(fig: plt.Figure, name: str, out_dir: Path | None) -> plt.Figure:
    out_dir = Path(out_dir or FIGURES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    return fig


def _class_legend(fig: plt.Figure, **kw) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[k])
               for k in (0, 1)]
    fig.legend(handles, [CLASS_NAMES[0], CLASS_NAMES[1]],
               loc="upper right", ncol=2, **kw)


# ---------------------------------------------------------------- tables
def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for numeric features, split by class."""
    rows = []
    for col in NUMERIC_FEATURES:
        s = df[col].astype(float)
        rows.append({
            "feature": col,
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "median": s.median(),
            "max": s.max(),
            "mean_no_disease": s[df[TARGET] == 0].mean(),
            "mean_disease": s[df[TARGET] == 1].mean(),
        })
    return pd.DataFrame(rows).set_index("feature").round(2)


def outlier_table(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """Count values beyond Tukey fences (Q1 - k*IQR, Q3 + k*IQR)."""
    rows = []
    for col in NUMERIC_FEATURES:
        s = df[col].astype(float).dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        mask = (s < lo) | (s > hi)
        rows.append({
            "feature": col, "lower_fence": lo, "upper_fence": hi,
            "n_outliers": int(mask.sum()),
            "outlier_values": sorted(s[mask].round(1).tolist()),
        })
    return pd.DataFrame(rows).set_index("feature")


def disease_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    """Disease prevalence within each level of each categorical feature."""
    rows = []
    for col in CATEGORICAL_FEATURES:
        grp = df.groupby(col, dropna=True)[TARGET].agg(["count", "mean"])
        for code, r in grp.iterrows():
            rows.append({
                "feature": col,
                "level": CATEGORY_LABELS[col].get(int(code), code),
                "n": int(r["count"]),
                "disease_rate_%": round(r["mean"] * 100, 1),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- plots
def plot_class_balance(df: pd.DataFrame, out_dir=None) -> plt.Figure:
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.4))
    bars = ax.bar([CLASS_NAMES[i] for i in counts.index], counts.values,
                  color=[CLASS_COLORS[i] for i in counts.index], width=0.55)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 3,
                f"{v}  ({v / counts.sum():.0%})", ha="center", va="bottom",
                color=INK, fontsize=9)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_ylabel("Patients")
    ax.set_title("Target class balance")
    ax.grid(axis="x", visible=False)
    return _save(fig, "01_class_balance", out_dir)


def plot_numeric_distributions(df: pd.DataFrame, out_dir=None) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6))
    axes = axes.ravel()
    for ax, col in zip(axes, NUMERIC_FEATURES):
        for cls in (0, 1):
            data = df.loc[df[TARGET] == cls, col].astype(float).dropna()
            sns.histplot(data, ax=ax, color=CLASS_COLORS[cls], stat="density",
                         bins=18, alpha=0.35, edgecolor=SURFACE, linewidth=1)
            sns.kdeplot(data, ax=ax, color=CLASS_COLORS[cls], linewidth=2,
                        cut=0)
        ax.set_title(FEATURE_DESCRIPTIONS[col])
        ax.set_xlabel("")
        ax.set_ylabel("Density")
    axes[-1].axis("off")
    fig.suptitle("Numeric feature distributions by diagnosis",
                 x=0.01, ha="left", fontsize=13, fontweight="bold",
                 color=INK)
    _class_legend(fig, bbox_to_anchor=(0.99, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, "02_numeric_distributions", out_dir)


def plot_numeric_boxplots(df: pd.DataFrame, out_dir=None) -> plt.Figure:
    fig, axes = plt.subplots(1, len(NUMERIC_FEATURES), figsize=(13, 3.8))
    for ax, col in zip(axes, NUMERIC_FEATURES):
        data = [df.loc[df[TARGET] == c, col].astype(float).dropna()
                for c in (0, 1)]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True,
                        medianprops={"color": INK, "linewidth": 1.5},
                        whiskerprops={"color": INK_2},
                        capprops={"color": INK_2},
                        flierprops={"marker": "o", "markersize": 4,
                                    "markerfacecolor": "none",
                                    "markeredgecolor": INK_2})
        for patch, c in zip(bp["boxes"], (0, 1)):
            patch.set_facecolor(CLASS_COLORS[c])
            patch.set_alpha(0.75)
            patch.set_edgecolor(SURFACE)
        ax.set_xticks([1, 2], ["No", "Yes"])
        ax.set_xlabel("Disease")
        ax.set_title(FEATURE_DESCRIPTIONS[col], fontsize=9.5)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Outlier check - numeric features by diagnosis",
                 x=0.01, ha="left", fontsize=13, fontweight="bold",
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, "03_numeric_boxplots", out_dir)


def plot_categorical_counts(df: pd.DataFrame, out_dir=None) -> plt.Figure:
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.4))
    for ax, col in zip(axes.ravel(), CATEGORICAL_FEATURES):
        ct = pd.crosstab(df[col], df[TARGET])
        ct = ct.reindex(columns=[0, 1], fill_value=0)
        labels = [CATEGORY_LABELS[col].get(int(i), str(i)) for i in ct.index]
        x = np.arange(len(ct))
        w = 0.38
        for j, cls in enumerate((0, 1)):
            ax.bar(x + (j - 0.5) * (w + 0.02), ct[cls].values, width=w,
                   color=CLASS_COLORS[cls])
        rate = ct[1] / ct.sum(axis=1)
        top = ct.values.max()
        for xi, r in zip(x, rate):
            ax.text(xi, top * 1.06, f"{r:.0%}", ha="center", fontsize=8,
                    color=INK_2)
        ax.set_ylim(0, top * 1.2)
        ax.set_xticks(x, labels, rotation=20 if len(labels) > 2 else 0,
                      ha="right" if len(labels) > 2 else "center")
        ax.set_title(FEATURE_DESCRIPTIONS[col])
        ax.grid(axis="x", visible=False)
        missing = int(df[col].isna().sum())
        if missing:
            ax.set_xlabel(f"{missing} missing excluded", color=INK_2,
                          fontsize=8)
    fig.suptitle("Categorical features by diagnosis "
                 "(% above bars = disease rate in that group)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold",
                 color=INK)
    _class_legend(fig, bbox_to_anchor=(0.99, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "04_categorical_counts", out_dir)


def plot_correlation_heatmap(df: pd.DataFrame, out_dir=None) -> plt.Figure:
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    corr = df[cols].astype(float).corr(method="pearson")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(9.5, 8))
    sns.heatmap(corr, mask=mask, cmap=DIVERGING, vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", annot_kws={"size": 7.5},
                linewidths=1.5, linecolor=SURFACE, square=True,
                cbar_kws={"shrink": 0.7, "label": "Pearson r"}, ax=ax)
    ax.grid(False)
    ax.set_title("Correlation matrix (categorical codes treated as ordinal)",
                 loc="left")
    return _save(fig, "05_correlation_heatmap", out_dir)


def plot_target_correlation(df: pd.DataFrame, out_dir=None) -> plt.Figure:
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    corr = (df[cols + [TARGET]].astype(float).corr()[TARGET]
            .drop(TARGET).sort_values())
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [CLASS_COLORS[1] if v > 0 else CLASS_COLORS[0] for v in corr]
    ax.barh([FEATURE_DESCRIPTIONS[c] for c in corr.index], corr.values,
            color=colors, height=0.6)
    for i, v in enumerate(corr.values):
        ax.text(v + (0.015 if v >= 0 else -0.015), i, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8,
                color=INK)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    lim = np.abs(corr.values).max() * 1.25
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Pearson r with disease  (orange = higher value -> "
                  "more disease)")
    ax.set_title("Feature correlation with the target", loc="left")
    ax.grid(axis="y", visible=False)
    return _save(fig, "06_target_correlation", out_dir)


def plot_age_vs_thalach(df: pd.DataFrame, out_dir=None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for cls in (0, 1):
        sub = df[df[TARGET] == cls]
        ax.scatter(sub["age"].astype(float), sub["thalach"], s=30,
                   color=CLASS_COLORS[cls], alpha=0.8, edgecolor=SURFACE,
                   linewidth=1, label=CLASS_NAMES[cls])
    ax.set_xlabel(FEATURE_DESCRIPTIONS["age"])
    ax.set_ylabel(FEATURE_DESCRIPTIONS["thalach"])
    ax.set_title("Age vs maximum heart rate", loc="left")
    ax.legend(loc="lower left")
    return _save(fig, "07_age_vs_thalach", out_dir)


def plot_missing_values(df: pd.DataFrame, out_dir=None) -> plt.Figure:
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values()
    fig, ax = plt.subplots(figsize=(5, 2.4))
    ax.barh(miss.index, miss.values, color="#2a78d6", height=0.5)
    for i, v in enumerate(miss.values):
        ax.text(v + 0.1, i, f"{v} ({v / len(df):.1%})", va="center",
                fontsize=8.5, color=INK)
    ax.set_xlim(0, miss.max() * 1.6 if len(miss) else 1)
    ax.set_xlabel("Missing rows")
    ax.set_title("Missing values after cleaning", loc="left")
    ax.grid(axis="y", visible=False)
    return _save(fig, "00_missing_values", out_dir)


ALL_PLOTS = [
    plot_missing_values,
    plot_class_balance,
    plot_numeric_distributions,
    plot_numeric_boxplots,
    plot_categorical_counts,
    plot_correlation_heatmap,
    plot_target_correlation,
    plot_age_vs_thalach,
]


def run_all(df: pd.DataFrame, out_dir=None) -> list[Path]:
    """Generate every EDA figure; returns the saved file paths."""
    for fn in ALL_PLOTS:
        plt.close(fn(df, out_dir))
    out_dir = Path(out_dir or FIGURES_DIR)
    return sorted(out_dir.glob("*.png"))


if __name__ == "__main__":
    from src.data_processing import load_clean

    paths = run_all(load_clean())
    print("\n".join(str(p) for p in paths))
