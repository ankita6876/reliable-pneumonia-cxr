"""Tests for CheXpert CSV metadata validation."""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.data.chexpert_metadata import (  # noqa: E402
    MetadataValidationError,
    extract_patient_id,
    validate_chexpert_metadata,
)


REQUIRED_ROW = {
    "Sex": "Female",
    "Age": 50,
    "AP/PA": "AP",
}


def _write_csv(root: Path, split_name: str, rows: list[dict[str, object]]) -> None:
    """Write a small synthetic metadata CSV."""
    pd.DataFrame(rows).to_csv(root / f"{split_name}.csv", index=False)


def _row(path: str, view: str, pneumonia: object) -> dict[str, object]:
    """Build a complete synthetic CheXpert metadata row."""
    return {"Path": path, "Frontal/Lateral": view, "Pneumonia": pneumonia, **REQUIRED_ROW}


def test_validates_metadata_and_reports_requested_counts(tmp_path: Path) -> None:
    """Valid train and valid CSVs produce independent summary counts."""
    root = tmp_path / "chexpert"
    train_image = root / "train" / "patient00001" / "study1" / "view1_frontal.jpg"
    valid_image = root / "valid" / "patient00002" / "study1" / "view1_lateral.jpg"
    train_image.parent.mkdir(parents=True)
    valid_image.parent.mkdir(parents=True)
    train_image.touch()
    valid_image.touch()
    _write_csv(
        root,
        "train",
        [
            _row("CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg", "Frontal", 1),
            _row("train/patient00001/study1/view1_frontal.jpg", "Frontal", -1),
        ],
    )
    _write_csv(
        root,
        "valid",
        [_row("valid/patient00002/study1/view1_lateral.jpg", "Lateral", 0)],
    )

    validation = validate_chexpert_metadata(root)

    assert validation.train.row_count == 2
    assert validation.train.unique_patient_count == 1
    assert validation.train.frontal_image_count == 2
    assert validation.train.lateral_image_count == 0
    assert validation.train.pneumonia_label_counts == {"1": 1, "0": 0, "-1": 1, "missing": 0}
    assert validation.train.missing_image_path_count == 0
    assert validation.valid.lateral_image_count == 1
    assert not validation.overlapping_patient_ids


def test_reports_missing_paths_duplicates_and_patient_overlap(tmp_path: Path) -> None:
    """Missing images, duplicate rows, and overlapping patients are all reported."""
    root = tmp_path / "chexpert"
    image = root / "train" / "patient00001" / "study1" / "view1_frontal.jpg"
    image.parent.mkdir(parents=True)
    image.touch()
    duplicate_row = _row("train/patient00001/study1/view1_frontal.jpg", "Frontal", None)
    _write_csv(root, "train", [duplicate_row, duplicate_row])
    _write_csv(
        root,
        "valid",
        [_row("valid/patient00001/study1/missing.jpg", "Lateral", 0)],
    )

    validation = validate_chexpert_metadata(root)

    assert validation.train.duplicate_metadata_row_count == 1
    assert validation.train.pneumonia_label_counts["missing"] == 2
    assert validation.valid.missing_image_path_count == 1
    assert validation.overlapping_patient_ids == frozenset({"patient00001"})


def test_missing_required_columns_raise_clear_error(tmp_path: Path) -> None:
    """Incomplete metadata CSVs identify their missing required columns."""
    root = tmp_path / "chexpert"
    root.mkdir()
    pd.DataFrame({"Path": ["train/patient00001/study1/view.jpg"]}).to_csv(
        root / "train.csv", index=False
    )
    _write_csv(root, "valid", [_row("valid/patient00002/study1/view.jpg", "Frontal", 1)])

    with pytest.raises(MetadataValidationError, match="missing required column"):
        validate_chexpert_metadata(root)


def test_missing_csv_raises_clear_error(tmp_path: Path) -> None:
    """A missing split CSV is reported before validation."""
    root = tmp_path / "chexpert"
    root.mkdir()

    with pytest.raises(FileNotFoundError, match="train.csv was not found"):
        validate_chexpert_metadata(root)


def test_extract_patient_id_from_chexpert_path() -> None:
    """Patient directory names are extracted from standard CheXpert paths."""
    image_path = Path("train/patient00001/study1/view1_frontal.jpg")

    assert extract_patient_id(image_path) == "patient00001"
