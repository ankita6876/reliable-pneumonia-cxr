"""Reusable, model-agnostic utilities for MC-dropout uncertainty analysis."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn


def active_dropout_modules(model: nn.Module) -> list[tuple[str, float]]:
    """Return named stochastic dropout modules with a non-zero probability."""
    return [
        (name, float(module.p))
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.dropout._DropoutNd) and module.p > 0.0
    ]


def dropout_modules(model: nn.Module) -> list[tuple[str, float]]:
    """Return every explicit dropout module, including inactive zero-rate ones."""
    return [
        (name, float(module.p))
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.dropout._DropoutNd)
    ]


def enable_mc_dropout(model: nn.Module) -> list[tuple[str, float]]:
    """Set a model to evaluation mode, then enable only non-zero dropout modules.

    Keeping the parent model in evaluation mode preserves BatchNorm running
    statistics and makes MC inference safe for a checkpointed classifier.
    """
    model.eval()
    modules = active_dropout_modules(model)
    for name, _ in modules:
        dict(model.named_modules())[name].train()
    return modules


def binary_uncertainty(probability_samples: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate binary predictive uncertainty from ``[passes, cases]`` samples."""
    samples = np.asarray(probability_samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 1:
        raise ValueError("probability_samples must have shape [mc_passes, cases].")
    clipped = np.clip(samples, 1e-12, 1.0 - 1e-12)
    mean = clipped.mean(axis=0)
    entropy = _binary_entropy(mean)
    expected_entropy = _binary_entropy(clipped).mean(axis=0)
    votes = (clipped >= 0.5).sum(axis=0)
    variation_ratio = 1.0 - np.maximum(votes, clipped.shape[0] - votes) / clipped.shape[0]
    return {
        "mean_probability": mean,
        "predictive_variance": clipped.var(axis=0),
        "predictive_standard_deviation": clipped.std(axis=0),
        "predictive_entropy": entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": entropy - expected_entropy,
        "variation_ratio": variation_ratio,
    }


def uncertainty_summary(frame: pd.DataFrame, group_column: str | None = None) -> pd.DataFrame:
    """Summarise standard uncertainty measures overall or by a categorical column."""
    measures = ["predictive_entropy", "mutual_information", "mc_standard_deviation"]
    missing = set(measures) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing uncertainty column(s): {', '.join(sorted(missing))}.")
    groups: Iterable[tuple[object, pd.DataFrame]] = [("overall", frame)] if group_column is None else frame.groupby(group_column, dropna=False)
    rows = []
    for group, subset in groups:
        row: dict[str, object] = {"group": group, "n": len(subset)}
        for measure in measures:
            row[f"{measure}_mean"] = float(subset[measure].mean())
            row[f"{measure}_median"] = float(subset[measure].median())
            row[f"{measure}_standard_deviation"] = float(subset[measure].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def selective_prediction(frame: pd.DataFrame, uncertainty_column: str) -> tuple[pd.DataFrame, float]:
    """Return risk/accuracy by retained coverage and area under the risk-coverage curve."""
    required = {uncertainty_column, "correctness"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing selective-prediction column(s): {', '.join(sorted(missing))}.")
    ordered = frame.sort_values(uncertainty_column, kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    if n == 0:
        raise ValueError("Selective prediction requires at least one case.")
    rows = []
    for requested in range(100, 9, -10):
        retained = max(1, int(np.ceil(n * requested / 100)))
        accuracy = float(ordered.loc[:retained - 1, "correctness"].mean())
        rows.append({"requested_coverage_percent": requested, "coverage": retained / n, "n_retained": retained,
                     "accuracy": accuracy, "risk": 1.0 - accuracy})
    result = pd.DataFrame(rows).sort_values("coverage")
    aurc = float(
        np.trapezoid(
            result["risk"].to_numpy(dtype=float),
            result["coverage"].to_numpy(dtype=float),
        )
    )
    return result.reset_index(drop=True), aurc


def _binary_entropy(probability: np.ndarray) -> np.ndarray:
    """Return Bernoulli entropy in natural-log units."""
    probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return -(probability * np.log(probability) + (1.0 - probability) * np.log(1.0 - probability))
