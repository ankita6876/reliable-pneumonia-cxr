"""Tests for predefined CheXpert uncertain-label strategies."""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.data.label_strategy import apply_label_strategy  # noqa: E402


def _records() -> pd.DataFrame:
    """Create ordered records with all supported input labels."""
    return pd.DataFrame(
        {
            "patient_id": ["patient3", "patient1", "patient2"],
            "pneumonia_label": [1, -1, 0],
            "image_path": ["three.jpg", "one.jpg", "two.jpg"],
        },
        index=[7, 3, 9],
    )


def test_u_zero_maps_uncertain_labels_to_zero() -> None:
    """The u_zero experiment treats -1 labels as negative."""
    result = apply_label_strategy(_records(), "u_zero")

    assert result["pneumonia_label"].tolist() == [1, 0, 0]
    assert set(result["pneumonia_label"]) == {0, 1}


def test_u_one_maps_uncertain_labels_to_one() -> None:
    """The u_one experiment treats -1 labels as positive."""
    result = apply_label_strategy(_records(), "u_one")

    assert result["pneumonia_label"].tolist() == [1, 1, 0]
    assert set(result["pneumonia_label"]) == {0, 1}


def test_ignore_removes_uncertain_rows_and_preserves_retained_order() -> None:
    """The ignore experiment drops only -1 rows without reordering other rows."""
    result = apply_label_strategy(_records(), "ignore")

    assert result.index.tolist() == [7, 9]
    assert result["patient_id"].tolist() == ["patient3", "patient2"]
    assert result["pneumonia_label"].tolist() == [1, 0]


def test_input_dataframe_is_not_modified() -> None:
    """Every strategy operates on a new DataFrame rather than mutating the input."""
    records = _records()
    original = records.copy(deep=True)

    result = apply_label_strategy(records, "u_one")

    pd.testing.assert_frame_equal(records, original)
    assert result is not records


def test_invalid_labels_raise_clear_error() -> None:
    """Only 1, 0, and -1 are valid inputs to uncertainty strategies."""
    records = _records()
    records.loc[3, "pneumonia_label"] = 2

    with pytest.raises(ValueError, match="Invalid Pneumonia labels"):
        apply_label_strategy(records, "u_zero")


def test_unsupported_strategy_raises_clear_error() -> None:
    """The strategy must be one of the three predefined experiments."""
    with pytest.raises(ValueError, match="Unsupported label strategy"):
        apply_label_strategy(_records(), "drop_uncertain")


def test_missing_label_column_raises_clear_error() -> None:
    """A manifest without the source label column cannot be transformed."""
    with pytest.raises(KeyError, match="pneumonia_label"):
        apply_label_strategy(pd.DataFrame({"patient_id": ["patient1"]}), "u_zero")
