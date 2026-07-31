"""Reusable plotting and input-validation helpers for classifier comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9")
"""Okabe-Ito colour-blind friendly palette."""


@dataclass(frozen=True)
class PredictionSet:
    """Aligned binary labels and predicted probabilities for several models."""

    labels: np.ndarray
    probabilities: Mapping[str, np.ndarray]


def configure_style() -> None:
    """Apply a consistent, legible Matplotlib style for manuscript figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def display_name(model_name: str) -> str:
    """Convert a machine-readable model name into a concise figure label."""
    return model_name.replace("_", " ").title()


def save_figure(figure: Figure, output_directory: Path, stem: str) -> tuple[Path, Path]:
    """Save a figure in publication-ready PNG and vector PDF formats."""
    output_directory.mkdir(parents=True, exist_ok=True)
    png_path = output_directory / f"{stem}.png"
    pdf_path = output_directory / f"{stem}.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def plot_roc(predictions: PredictionSet) -> Figure:
    """Plot ROC curves and AUROC values for every classifier."""
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    for colour, (name, values) in zip(PALETTE, predictions.probabilities.items()):
        fpr, tpr, _ = roc_curve(predictions.labels, values)
        auc = roc_auc_score(predictions.labels, values)
        axis.plot(fpr, tpr, color=colour, linewidth=2, label=f"{display_name(name)} (AUROC = {auc:.3f})")
    axis.plot([0, 1], [0, 1], "--", color="0.35", linewidth=1, label="Chance")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate", title="Receiver operating characteristic")
    axis.legend(loc="lower right", frameon=False)
    figure.tight_layout()
    return figure


def plot_precision_recall(predictions: PredictionSet) -> Figure:
    """Plot precision-recall curves and average precision values."""
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    prevalence = float(np.mean(predictions.labels))
    for colour, (name, values) in zip(PALETTE, predictions.probabilities.items()):
        precision, recall, _ = precision_recall_curve(predictions.labels, values)
        score = average_precision_score(predictions.labels, values)
        axis.plot(recall, precision, color=colour, linewidth=2, label=f"{display_name(name)} (AP = {score:.3f})")
    axis.axhline(prevalence, linestyle="--", color="0.35", linewidth=1, label=f"Prevalence = {prevalence:.3f}")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    axis.legend(loc="lower left", frameon=False)
    figure.tight_layout()
    return figure


def _calibration_points(labels: np.ndarray, values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Return observed fractions and mean probabilities for non-empty bins."""
    observed, mean_probability = calibration_curve(labels, values, n_bins=bins, strategy="uniform")
    return observed, mean_probability


def plot_calibration(predictions: PredictionSet, bins: int) -> Figure:
    """Plot calibration curves for all classifiers on one comparable axis."""
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    axis.plot([0, 1], [0, 1], "--", color="0.35", linewidth=1, label="Perfect calibration")
    for colour, (name, values) in zip(PALETTE, predictions.probabilities.items()):
        observed, mean_probability = _calibration_points(predictions.labels, values, bins)
        axis.plot(mean_probability, observed, marker="o", color=colour, linewidth=2, label=display_name(name))
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed positive fraction", title="Calibration curves")
    axis.legend(loc="upper left", frameon=False)
    figure.tight_layout()
    return figure


def plot_confusion_matrices(predictions: PredictionSet, threshold: float) -> Figure:
    """Plot one thresholded confusion matrix per classifier."""
    names = list(predictions.probabilities)
    figure, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 4.0), squeeze=False)
    for axis, name in zip(axes[0], names):
        matrix = confusion_matrix(predictions.labels, predictions.probabilities[name] >= threshold, labels=(0, 1))
        ConfusionMatrixDisplay(matrix, display_labels=("Negative", "Positive")).plot(
            ax=axis, cmap="Blues", colorbar=False, values_format="d"
        )
        axis.set_title(display_name(name))
    figure.suptitle(f"Confusion matrices (threshold = {threshold:g})", y=1.03)
    figure.tight_layout()
    return figure


def plot_probability_distributions(predictions: PredictionSet) -> Figure:
    """Plot predicted-probability histograms by true class for every classifier."""
    names = list(predictions.probabilities)
    figure, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 3.8), sharey=True, squeeze=False)
    bins = np.linspace(0, 1, 21)
    negative = predictions.labels == 0
    positive = predictions.labels == 1
    for axis, name in zip(axes[0], names):
        values = predictions.probabilities[name]
        axis.hist(values[negative], bins=bins, density=True, alpha=0.7, color="#0072B2", label="Negative")
        axis.hist(values[positive], bins=bins, density=True, alpha=0.6, color="#D55E00", label="Positive")
        axis.set(title=display_name(name), xlim=(0, 1), xlabel="Predicted probability")
    axes[0, 0].set_ylabel("Density")
    axes[0, -1].legend(frameon=False)
    figure.suptitle("Predicted probability distributions by true class", y=1.03)
    figure.tight_layout()
    return figure


def plot_reliability_diagram(predictions: PredictionSet, bins: int) -> Figure:
    """Plot per-model reliability diagrams with predicted-probability frequencies."""
    names = list(predictions.probabilities)
    figure, axes = plt.subplots(2, len(names), figsize=(4.1 * len(names), 6.1), gridspec_kw={"height_ratios": (3, 1)}, sharex="col", squeeze=False)
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    for column, (colour, name) in enumerate(zip(PALETTE, names)):
        values = predictions.probabilities[name]
        observed, mean_probability = _calibration_points(predictions.labels, values, bins)
        calibration_axis, histogram_axis = axes[:, column]
        calibration_axis.plot([0, 1], [0, 1], "--", color="0.35", linewidth=1)
        calibration_axis.plot(mean_probability, observed, marker="o", color=colour, linewidth=2)
        calibration_axis.set(title=display_name(name), xlim=(0, 1), ylim=(0, 1), ylabel="Observed fraction" if column == 0 else "")
        counts, _ = np.histogram(values, bins=bin_edges)
        histogram_axis.bar(bin_centres, counts, width=1 / bins * 0.9, color=colour, alpha=0.8)
        histogram_axis.set(xlim=(0, 1), xlabel="Predicted probability", ylabel="Count" if column == 0 else "")
    figure.suptitle("Reliability diagrams", y=1.02)
    figure.tight_layout()
    return figure


def validate_prediction_set(frame: pd.DataFrame, label_column: str, probability_columns: Sequence[str]) -> PredictionSet:
    """Validate and convert detected aligned-prediction columns into arrays."""
    labels = pd.to_numeric(frame[label_column], errors="raise").to_numpy(dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("The label column must contain both binary classes 0 and 1.")
    probabilities: dict[str, np.ndarray] = {}
    for column in probability_columns:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if np.isnan(values).any() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"Probability column {column!r} must contain values in [0, 1].")
        probabilities[column.removeprefix("probability_")] = values
    return PredictionSet(labels=labels, probabilities=probabilities)
