"""Evaluate a frozen two-branch logit fusion.

This script does not train models or select fusion weights.
It combines previously generated probabilities using a
source-selected, frozen weighted-logit rule.

Example
-------
python scripts/evaluate_complementary_fusion.py \
    --predictions predictions.csv \
    --label-column binary_target \
    --primary-column resnet50_original_probability \
    --complementary-column complementary_probability \
    --primary-weight 0.65 \
    --threshold 0.5414520502090454 \
    --output metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


EPS = 1e-7


def probability_to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(
        np.asarray(p, dtype=np.float64),
        EPS,
        1.0 - EPS,
    )
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)

    # Numerically stable sigmoid.
    out = np.empty_like(z)

    positive = z >= 0

    out[positive] = (
        1.0
        / (1.0 + np.exp(-z[positive]))
    )

    ez = np.exp(z[~positive])

    out[~positive] = (
        ez / (1.0 + ez)
    )

    return out


def fuse_probabilities(
    primary: np.ndarray,
    complementary: np.ndarray,
    primary_weight: float,
) -> np.ndarray:

    if not 0.0 <= primary_weight <= 1.0:
        raise ValueError(
            "primary_weight must be in [0, 1]"
        )

    complementary_weight = (
        1.0 - primary_weight
    )

    fused_logit = (
        primary_weight
        * probability_to_logit(primary)
        +
        complementary_weight
        * probability_to_logit(complementary)
    )

    return sigmoid(fused_logit)


def evaluate(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
) -> dict[str, float]:

    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=np.float64)

    pred = (p >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    precision = (
        tp / (tp + fp)
        if tp + fp
        else np.nan
    )

    f1 = (
        2.0 * precision * sensitivity
        / (precision + sensitivity)
        if precision + sensitivity
        else np.nan
    )

    return {
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "NLL": log_loss(y, p, labels=[0, 1]),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "BalancedAccuracy":
            (sensitivity + specificity) / 2.0,
        "F1": f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        required=True,
    )

    parser.add_argument(
        "--label-column",
        default="binary_target",
    )

    parser.add_argument(
        "--primary-column",
        default="resnet50_original_probability",
    )

    parser.add_argument(
        "--complementary-column",
        default="complementary_probability",
    )

    parser.add_argument(
        "--primary-weight",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5414520502090454,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.predictions)

    required = [
        args.label_column,
        args.primary_column,
        args.complementary_column,
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    y = df[
        args.label_column
    ].to_numpy(dtype=int)

    primary = df[
        args.primary_column
    ].to_numpy(dtype=np.float64)

    complementary = df[
        args.complementary_column
    ].to_numpy(dtype=np.float64)

    fusion = fuse_probabilities(
        primary,
        complementary,
        args.primary_weight,
    )

    rows = []

    for name, probs in [
        ("primary", primary),
        ("complementary", complementary),
        ("frozen_fusion", fusion),
    ]:

        metrics = evaluate(
            y,
            probs,
            args.threshold,
        )

        metrics["model"] = name
        rows.append(metrics)

    result = pd.DataFrame(rows).set_index(
        "model"
    )

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(output)

    print(
        result.to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
