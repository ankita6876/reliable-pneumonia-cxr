"""Explicit CheXpert uncertainty-label strategies for experiment preparation."""

from typing import Literal

import pandas as pd


LabelStrategy = Literal["u_zero", "u_one", "ignore", "soft_uncertain"]
SUPPORTED_LABEL_STRATEGIES = frozenset(
    {"u_zero", "u_one", "ignore", "soft_uncertain"}
)
VALID_INPUT_LABELS = frozenset({1, 0, -1})


def apply_label_strategy(
    records: pd.DataFrame,
    strategy: LabelStrategy | str,
    uncertain_soft_target: float = 0.5,
    uncertain_sample_weight: float = 0.5,
) -> pd.DataFrame:
    """Return a copied DataFrame with one predefined uncertainty strategy applied.

    ``u_zero`` maps uncertain labels to negative, ``u_one`` maps them to
    positive, ``ignore`` removes uncertain-label rows, and ``soft_uncertain``
    assigns configurable target and loss-weight values to uncertain rows. The
    original label is retained in ``raw_pneumonia_label`` while the training
    target and loss weight are written to separate columns.

    Raises:
        ValueError: If the strategy or input labels are unsupported.
        KeyError: If ``pneumonia_label`` is absent.
    """
    if strategy not in SUPPORTED_LABEL_STRATEGIES:
        expected = ", ".join(sorted(SUPPORTED_LABEL_STRATEGIES))
        raise ValueError(f"Unsupported label strategy {strategy!r}. Expected one of: {expected}.")
    if "pneumonia_label" not in records:
        raise KeyError("Input records are missing required column: pneumonia_label")

    if not 0.0 <= uncertain_soft_target <= 1.0:
        raise ValueError("uncertain_soft_target must be between 0 and 1.")
    if uncertain_sample_weight <= 0.0:
        raise ValueError("uncertain_sample_weight must be positive.")

    source_column = (
        "raw_pneumonia_label" if "raw_pneumonia_label" in records else "pneumonia_label"
    )
    labels = pd.to_numeric(records[source_column], errors="coerce")
    invalid_labels = labels.isna() | ~labels.isin(VALID_INPUT_LABELS)
    if invalid_labels.any():
        invalid_values = records.loc[invalid_labels, "pneumonia_label"].unique().tolist()
        raise ValueError(
            "Invalid Pneumonia labels; expected only 1, 0, or -1. "
            f"Found: {invalid_values}"
        )

    if strategy == "ignore":
        retained_mask = labels != -1
        result = records.loc[retained_mask].copy()
        targets = labels.loc[retained_mask].astype("float32")
        weights = pd.Series(1.0, index=result.index, dtype="float32")
    else:
        result = records.copy()
        if strategy == "u_zero":
            targets = labels.mask(labels == -1, 0).astype("float32")
        elif strategy == "u_one":
            targets = labels.mask(labels == -1, 1).astype("float32")
        else:
            targets = labels.mask(labels == -1, uncertain_soft_target).astype("float32")
        weights = pd.Series(1.0, index=result.index, dtype="float32")
        if strategy == "soft_uncertain":
            weights.loc[labels == -1] = uncertain_sample_weight

    result["raw_pneumonia_label"] = labels.loc[result.index].astype("int64")
    result["training_target"] = targets
    result["sample_loss_weight"] = weights
    result["pneumonia_label"] = targets
    return result
