"""Explicit CheXpert uncertainty-label strategies for experiment preparation."""

from typing import Literal

import pandas as pd


LabelStrategy = Literal["u_zero", "u_one", "ignore"]
SUPPORTED_LABEL_STRATEGIES = frozenset({"u_zero", "u_one", "ignore"})
VALID_INPUT_LABELS = frozenset({1, 0, -1})


def apply_label_strategy(
    records: pd.DataFrame, strategy: LabelStrategy | str
) -> pd.DataFrame:
    """Return a copied DataFrame with one predefined uncertainty strategy applied.

    ``u_zero`` maps uncertain labels to negative, ``u_one`` maps them to
    positive, and ``ignore`` removes uncertain-label rows. All retained labels
    are returned as binary integer values while all other columns are preserved.

    Raises:
        ValueError: If the strategy or input labels are unsupported.
        KeyError: If ``pneumonia_label`` is absent.
    """
    if strategy not in SUPPORTED_LABEL_STRATEGIES:
        expected = ", ".join(sorted(SUPPORTED_LABEL_STRATEGIES))
        raise ValueError(f"Unsupported label strategy {strategy!r}. Expected one of: {expected}.")
    if "pneumonia_label" not in records:
        raise KeyError("Input records are missing required column: pneumonia_label")

    labels = pd.to_numeric(records["pneumonia_label"], errors="coerce")
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
        result["pneumonia_label"] = labels.loc[retained_mask].astype("int64")
        return result

    result = records.copy()
    mapped_labels = labels.mask(labels == -1, 0 if strategy == "u_zero" else 1)
    result["pneumonia_label"] = mapped_labels.astype("int64")
    return result
