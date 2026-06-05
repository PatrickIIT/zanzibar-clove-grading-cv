"""
visualisation.py
----------------
Publication-quality visualisation utilities for the thesis figures.

Functions
~~~~~~~~~
- ``benchmark_lollipop``    — ranked lollipop chart of all 22 architecture F1 scores
- ``training_curves``       — loss + accuracy / F1 curves for a training run
- ``confusion_matrix_fig``  — annotated 4×4 or 5×5 confusion matrix heatmap
- ``feature_ablation_bar``  — bar chart for handcrafted feature ablation results
- ``quantisation_scatter``  — accuracy vs model size scatter for TFLite variants
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

# Default style
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":      150,
})

GRADE_NAMES  = ["Grade I", "Grade II", "Grade III", "Grade IV"]
FAMILY_COLORS = {
    "Classical ML":  "#4e79a7",
    "ResNet":        "#f28e2b",
    "DenseNet":      "#e15759",
    "EfficientNet":  "#76b7b2",
    "Multi-Scale":   "#59a14f",
    "ViT":           "#edc948",
    "Decomposed":    "#b07aa1",
    "Negative":      "#bab0ac",
}


# --------------------------------------------------------------------------- #
# Benchmark lollipop chart
# --------------------------------------------------------------------------- #
def benchmark_lollipop(
    results: Dict[str, Tuple[float, str]],
    title: str = "22-Architecture Benchmark — Macro F1 on Held-Out Test Set",
    output_path: Optional[str | Path] = None,
    figsize: Tuple[int, int] = (12, 9),
) -> plt.Figure:
    """Ranked lollipop chart of architecture macro-F1 scores.

    Parameters
    ----------
    results : dict
        ``{model_name: (f1_percent, family_name)}`` e.g.
        ``{"ResNet-50": (99.71, "ResNet"), "VGG-16": (14.94, "Negative")}``.
    title : str
    output_path : str | Path, optional
        If provided, saves the figure to this path.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Sort descending by F1
    sorted_items = sorted(results.items(), key=lambda kv: kv[1][0], reverse=True)
    names    = [k for k, _ in sorted_items]
    scores   = [v[0] for _, v in sorted_items]
    families = [v[1] for _, v in sorted_items]
    colors   = [FAMILY_COLORS.get(f, "#888888") for f in families]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos   = np.arange(len(names))

    # Lollipop stems
    ax.hlines(y_pos, 0, scores, colors=colors, linewidth=1.8, alpha=0.7)
    # Dots
    ax.scatter(scores, y_pos, c=colors, s=60, zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Macro F1 (%)", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlim(0, 103)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))

    # Annotate each bar
    for score, y in zip(scores, y_pos):
        ax.text(score + 0.4, y, f"{score:.2f}%", va="center", fontsize=7.5)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=col, markersize=9, label=fam)
        for fam, col in FAMILY_COLORS.items()
        if fam in families
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=8.5, framealpha=0.8)

    ax.invert_yaxis()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"[visualisation] Saved benchmark lollipop → {output_path}")
    return fig


# --------------------------------------------------------------------------- #
# Training curves
# --------------------------------------------------------------------------- #
def training_curves(
    history: Dict[str, list],
    model_name: str = "Model",
    output_path: Optional[str | Path] = None,
    figsize: Tuple[int, int] = (12, 4),
) -> plt.Figure:
    """Plot training / validation loss and accuracy (or F1) curves.

    Parameters
    ----------
    history : dict
        Keys: ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc``
        (and optionally ``val_f1``).
    model_name : str
    output_path : str | Path, optional
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    ncols  = 3 if "val_f1" in history else 2
    fig, axes = plt.subplots(1, ncols, figsize=figsize)

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train loss")
    ax.plot(epochs, history["val_loss"],   label="Val loss", linestyle="--")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train acc")
    ax.plot(epochs, [a * 100 for a in history["val_acc"]],   label="Val acc",  linestyle="--")
    ax.set_title("Accuracy (%)")
    ax.set_xlabel("Epoch")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
    ax.legend(fontsize=8)

    # Macro F1 (optional)
    if "val_f1" in history and ncols == 3:
        ax = axes[2]
        ax.plot(epochs, [f * 100 for f in history["val_f1"]], color="#e15759", label="Val macro F1")
        ax.set_title("Val Macro F1 (%)")
        ax.set_xlabel("Epoch")
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
        ax.legend(fontsize=8)

    fig.suptitle(f"{model_name} — Training Curves", fontsize=13)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# Confusion matrix
# --------------------------------------------------------------------------- #
def confusion_matrix_fig(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    normalise: bool = True,
    output_path: Optional[str | Path] = None,
    figsize: Tuple[int, int] = (6, 5),
) -> plt.Figure:
    """Annotated confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Raw confusion matrix of shape ``(n_classes, n_classes)``.
    class_names : list[str], optional
        Defaults to GRADE_NAMES.
    normalise : bool
        If True, normalise rows to sum to 1 (recall-normalised).
    output_path : str | Path, optional
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    names = class_names or GRADE_NAMES
    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot  = cm.astype(float) / np.maximum(row_sums, 1)
        fmt      = ".2f"
        vmin, vmax = 0.0, 1.0
    else:
        cm_plot = cm
        fmt     = "d"
        vmin, vmax = 0, cm.max()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# Feature ablation bar chart
# --------------------------------------------------------------------------- #
def feature_ablation_bar(
    ablation_results: Dict[str, float],
    title: str = "Handcrafted Feature Ablation — Macro F1 (SVM-RBF)",
    output_path: Optional[str | Path] = None,
    figsize: Tuple[int, int] = (8, 4),
) -> plt.Figure:
    """Horizontal bar chart for feature ablation results.

    Parameters
    ----------
    ablation_results : dict
        ``{descriptor_combination: macro_f1_percent}`` e.g.
        ``{"Hist only": 88.1, "Hist + Stat": 92.4, "All (93-dim)": 96.23}``.
    """
    names  = list(ablation_results.keys())
    scores = list(ablation_results.values())
    y_pos  = np.arange(len(names))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(y_pos, scores, color="#4e79a7", edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Macro F1 (%)")
    ax.set_title(title)
    ax.set_xlim(0, 103)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))

    for bar, score in zip(bars, scores):
        ax.text(
            score + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}%", va="center", fontsize=9,
        )

    ax.invert_yaxis()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# Quantisation scatter
# --------------------------------------------------------------------------- #
def quantisation_scatter(
    variants: Dict[str, Tuple[float, float]],
    title: str = "TFLite Quantisation — Accuracy vs Model Size",
    output_path: Optional[str | Path] = None,
    figsize: Tuple[int, int] = (7, 5),
) -> plt.Figure:
    """Scatter plot of accuracy vs model size for TFLite quantisation variants.

    Parameters
    ----------
    variants : dict
        ``{label: (size_mb, accuracy_percent)}`` e.g.
        ``{"INT8 (4.1 MB)": (4.1, 80.15), "FP16 (7.1 MB)": (7.1, 81.62),
           "Dynamic (3.9 MB)": (3.9, 80.88), "Full precision": (18.0, 84.56)}``.
    """
    fig, ax = plt.subplots(figsize=figsize)
    palette = ["#4e79a7", "#f28e2b", "#e15759", "#59a14f"]

    for (label, (size, acc)), color in zip(variants.items(), palette):
        ax.scatter(size, acc, s=120, color=color, zorder=5, label=label)
        ax.annotate(
            label, (size, acc),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8.5,
        )

    ax.set_xlabel("Model size (MB)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig
