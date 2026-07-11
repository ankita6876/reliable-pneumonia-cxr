"""Tests for deterministic patient-level CheXpert cohort splitting."""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.data.splitting import (  # noqa: E402
    SplitValidationError,
    split_cohort_records,
)


def _cohort_records(patients_per_label: int = 20) -> pd.DataFrame:
    """Build synthetic records with two images per patient and per label."""
    records: list[dict[str, object]] = []
    patient_number = 1
    for label in (1, 0, -1):
        for _ in range(patients_per_label):
            patient_id = f"patient{patient_number:05d}"
            for image_number in (1, 2):
                records.append(
                    {
                        "patient_id": patient_id,
                        "study_id": "study1",
                        "image_path": f"train/{patient_id}/study1/view{image_number}.jpg",
                        "original_csv_path": f"train/{patient_id}/study1/view{image_number}.jpg",
                        "pneumonia_label": label,
                        "sex": "Female",
                        "age": 50,
                        "ap_pa_view": "AP" if image_number == 1 else "PA",
                    }
                )
            patient_number += 1
    return pd.DataFrame(records)


def test_split_is_deterministic_for_same_seed() -> None:
    """The same seed produces exactly the same patient assignments."""
    records = _cohort_records()

    first = split_cohort_records(records, seed=42)
    second = split_cohort_records(records, seed=42)

    assert first.records["split"].tolist() == second.records["split"].tolist()


def test_split_changes_with_different_seed() -> None:
    """A different seed changes at least one assignment in a sufficiently large cohort."""
    records = _cohort_records()

    first = split_cohort_records(records, seed=42)
    second = split_cohort_records(records, seed=7)

    assert first.records["split"].tolist() != second.records["split"].tolist()


def test_split_has_no_overlap_and_preserves_every_row() -> None:
    """All input rows are retained and all split-pair overlaps are zero."""
    records = _cohort_records()

    splits = split_cohort_records(records)

    assert len(splits.records) == len(records)
    assert splits.rows_before == splits.rows_after
    assert splits.missing_row_count == 0
    assert splits.overlapping_patient_counts == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
    assert not splits.has_failures


def test_all_images_for_a_patient_stay_together() -> None:
    """A patient's multiple image rows receive one and only one split label."""
    splits = split_cohort_records(_cohort_records())

    assignments_per_patient = splits.records.groupby("patient_id")["split"].nunique()

    assert assignments_per_patient.eq(1).all()


def test_label_distribution_is_approximately_preserved() -> None:
    """Each label stratum follows the 70/15/15 allocation within patient rounding."""
    splits = split_cohort_records(_cohort_records(patients_per_label=20))

    for split_name, expected_patients in (("train", 14), ("validation", 3), ("test", 3)):
        split_records = splits.records.loc[splits.records["split"] == split_name]
        counts = split_records.groupby("pneumonia_label")["patient_id"].nunique()
        assert counts.to_dict() == {1: expected_patients, 0: expected_patients, -1: expected_patients}


@pytest.mark.parametrize("proportions", [(0.7, 0.2, 0.2), (0.7, 0.3), (0.7, 0.3, 0.0)])
def test_invalid_split_proportions_raise_clear_error(
    proportions: tuple[float, ...],
) -> None:
    """Invalid split proportions cannot create a cohort partition."""
    with pytest.raises(SplitValidationError, match="proportions"):
        split_cohort_records(_cohort_records(), proportions=proportions)  # type: ignore[arg-type]
