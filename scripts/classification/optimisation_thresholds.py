"""Validation-only threshold search and operating-point metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score


def threshold_analysis(
    labels: np.ndarray,
    probabilities: np.ndarray,
    method: str,
    target_sensitivity: float = 0.8,
) -> tuple[float, pd.DataFrame]:
    """Select a threshold using validation labels only and retain all candidates."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Threshold selection requires both validation classes.")
    candidates = np.unique(np.r_[0.0, probabilities, 1.0])
    rows = []
    for threshold in candidates:
        prediction = probabilities >= threshold
        tp = int(((prediction == 1) & (labels == 1)).sum())
        tn = int(((prediction == 0) & (labels == 0)).sum())
        fp = int(((prediction == 1) & (labels == 0)).sum())
        fn = int(((prediction == 0) & (labels == 1)).sum())
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        rows.append(
            {
                "threshold": threshold,
                "f1": f1_score(labels, prediction, zero_division=0),
                "youden_j": sensitivity + specificity - 1,
                "balanced_accuracy": balanced_accuracy_score(labels, prediction),
                "sensitivity": sensitivity,
                "specificity": specificity,
            }
        )
    table = pd.DataFrame(rows)
    if method == "fixed_0.5":
        selected = 0.5
    elif method == "max_f1":
        selected = float(table.loc[table.f1.idxmax(), "threshold"])
    elif method == "youden":
        selected = float(table.loc[table.youden_j.idxmax(), "threshold"])
    elif method == "balanced_accuracy":
        selected = float(table.loc[table.balanced_accuracy.idxmax(), "threshold"])
    elif method == "target_sensitivity":
        eligible = table.loc[table.sensitivity >= target_sensitivity]
        selected = float(
            (eligible if not eligible.empty else table)
            .sort_values(["specificity", "threshold"], ascending=[False, False])
            .iloc[0]
            .threshold
        )
    else:
        raise ValueError(f"Unsupported threshold method: {method}")
    table["selected"] = np.isclose(table.threshold, selected)
    return selected, table
