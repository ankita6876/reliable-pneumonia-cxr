"""Paired patient-cluster bootstrap for frozen fusion.

Compares frozen fusion predictions against a primary model
without model selection or target-domain tuning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


def bootstrap(
    df: pd.DataFrame,
    label_column: str,
    patient_column: str,
    primary_column: str,
    fusion_column: str,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)

    groups = [
        (
            g[label_column].to_numpy(dtype=np.int8),
            g[primary_column].to_numpy(dtype=np.float64),
            g[fusion_column].to_numpy(dtype=np.float64),
        )
        for _, g in df.groupby(
            patient_column,
            sort=False,
        )
    ]

    n = len(groups)

    auc_delta = []
    ap_delta = []

    for _ in range(replicates):

        sampled = rng.integers(
            0,
            n,
            size=n,
        )

        y = np.concatenate(
            [groups[i][0] for i in sampled]
        )

        primary = np.concatenate(
            [groups[i][1] for i in sampled]
        )

        fusion = np.concatenate(
            [groups[i][2] for i in sampled]
        )

        if np.unique(y).size < 2:
            continue

        auc_delta.append(
            roc_auc_score(y, fusion)
            - roc_auc_score(y, primary)
        )

        ap_delta.append(
            average_precision_score(y, fusion)
            - average_precision_score(y, primary)
        )

    return (
        np.asarray(auc_delta),
        np.asarray(ap_delta),
    )


def sign_pvalue(values: np.ndarray) -> float:

    n = len(values)

    lower = (
        np.sum(values <= 0) + 1
    ) / (n + 1)

    upper = (
        np.sum(values >= 0) + 1
    ) / (n + 1)

    return min(
        1.0,
        2.0 * min(lower, upper),
    )


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
        "--patient-column",
        default="patient_id",
    )

    parser.add_argument(
        "--primary-column",
        default="resnet50_original_probability",
    )

    parser.add_argument(
        "--fusion-column",
        default="fusion_probability",
    )

    parser.add_argument(
        "--replicates",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    df = pd.read_csv(
        args.predictions
    )

    y = df[
        args.label_column
    ].to_numpy(dtype=int)

    primary = df[
        args.primary_column
    ].to_numpy(dtype=np.float64)

    fusion = df[
        args.fusion_column
    ].to_numpy(dtype=np.float64)

    observed_auc = (
        roc_auc_score(y, fusion)
        - roc_auc_score(y, primary)
    )

    observed_ap = (
        average_precision_score(y, fusion)
        - average_precision_score(y, primary)
    )

    d_auc, d_ap = bootstrap(
        df=df,
        label_column=args.label_column,
        patient_column=args.patient_column,
        primary_column=args.primary_column,
        fusion_column=args.fusion_column,
        replicates=args.replicates,
        seed=args.seed,
    )

    auc_ci = np.percentile(
        d_auc,
        [2.5, 97.5],
    )

    ap_ci = np.percentile(
        d_ap,
        [2.5, 97.5],
    )

    result = pd.DataFrame([
        {
            "rows": len(df),
            "patients":
                df[args.patient_column].nunique(),
            "bootstrap_replicates": len(d_auc),

            "delta_auroc": observed_auc,
            "delta_auroc_ci_low": auc_ci[0],
            "delta_auroc_ci_high": auc_ci[1],
            "delta_auroc_p":
                sign_pvalue(d_auc),

            "delta_auprc": observed_ap,
            "delta_auprc_ci_low": ap_ci[0],
            "delta_auprc_ci_high": ap_ci[1],
            "delta_auprc_p":
                sign_pvalue(d_ap),
        }
    ])

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        output,
        index=False
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
