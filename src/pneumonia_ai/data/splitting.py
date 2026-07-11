"""Deterministic patient-level splitting for CheXpert cohort records."""

from dataclasses import dataclass
from pathlib import Path
import random

import pandas as pd


SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_PROPORTIONS = (0.70, 0.15, 0.15)
VALID_LABELS = frozenset({1, 0, -1})
REQUIRED_COHORT_COLUMNS = ("patient_id", "image_path", "pneumonia_label", "ap_pa_view")


class SplitValidationError(ValueError):
    """Raised when cohort records cannot be safely split."""


@dataclass(frozen=True)
class SplitReport:
    """Counts and label distribution for one patient-level split."""

    split: str
    row_count: int
    unique_patient_count: int
    positive_count: int
    negative_count: int
    uncertain_count: int
    positive_percentage: float
    negative_percentage: float
    uncertain_percentage: float
    ap_count: int
    pa_count: int


@dataclass(frozen=True)
class CheXpertSplits:
    """Split cohort records, summaries, and integrity checks."""

    records: pd.DataFrame
    reports: dict[str, SplitReport]
    overlapping_patient_counts: dict[str, int]
    rows_before: int
    rows_after: int
    missing_row_count: int
    duplicate_assignment_count: int

    @property
    def has_failures(self) -> bool:
        """Return whether a split-integrity requirement failed."""
        return (
            any(self.overlapping_patient_counts.values())
            or self.missing_row_count != 0
            or self.duplicate_assignment_count != 0
            or self.rows_before != self.rows_after
        )


def split_chexpert_cohort(
    cohort_path: Path | str,
    seed: int = 42,
    proportions: tuple[float, float, float] = DEFAULT_PROPORTIONS,
) -> CheXpertSplits:
    """Read and split a cohort CSV using deterministic patient-level assignment."""
    records = pd.read_csv(cohort_path)
    return split_cohort_records(records, seed=seed, proportions=proportions)


def split_cohort_records(
    records: pd.DataFrame,
    seed: int = 42,
    proportions: tuple[float, float, float] = DEFAULT_PROPORTIONS,
) -> CheXpertSplits:
    """Split cohort records while retaining every image for a patient together."""
    _validate_records(records)
    _validate_proportions(proportions)
    patient_ids = records["patient_id"].astype("string")
    labels = pd.to_numeric(records["pneumonia_label"], errors="coerce").astype("int64")
    patient_labels = _patient_strata(patient_ids, labels)
    assignments = _assign_patients(patient_labels, seed, proportions)

    split_records = records.copy()
    split_records["split"] = patient_ids.map(assignments)
    reports = {
        split_name: _split_report(split_records, split_name, labels)
        for split_name in SPLIT_NAMES
    }
    overlaps = _overlapping_patient_counts(split_records)
    duplicate_assignments = int(
        (split_records.groupby("patient_id")["split"].nunique() > 1).sum()
    )
    rows_before = len(records)
    rows_after = len(split_records)
    return CheXpertSplits(
        records=split_records,
        reports=reports,
        overlapping_patient_counts=overlaps,
        rows_before=rows_before,
        rows_after=rows_after,
        missing_row_count=rows_before - rows_after,
        duplicate_assignment_count=duplicate_assignments,
    )


def write_chexpert_splits(splits: CheXpertSplits, output_path: Path | str) -> None:
    """Write split records to CSV without changing the cohort path columns."""
    splits.records.to_csv(Path(output_path), index=False)


def _validate_records(records: pd.DataFrame) -> None:
    """Validate essential cohort fields and original pneumonia labels."""
    missing_columns = [column for column in REQUIRED_COHORT_COLUMNS if column not in records]
    if missing_columns:
        raise SplitValidationError(
            f"Cohort CSV is missing required column(s): {', '.join(missing_columns)}"
        )
    patient_ids = records["patient_id"].astype("string")
    if patient_ids.isna().any() or (patient_ids.str.strip() == "").any():
        raise SplitValidationError("Cohort CSV contains missing patient IDs.")

    labels = pd.to_numeric(records["pneumonia_label"], errors="coerce")
    invalid_labels = labels.isna() | ~labels.isin(VALID_LABELS)
    if invalid_labels.any():
        invalid_values = records.loc[invalid_labels, "pneumonia_label"].unique().tolist()
        raise SplitValidationError(
            "Cohort CSV contains invalid Pneumonia labels; expected only 1, 0, or -1. "
            f"Found: {invalid_values}"
        )


def _validate_proportions(proportions: tuple[float, float, float]) -> None:
    """Validate the three split proportions."""
    if len(proportions) != len(SPLIT_NAMES) or any(value <= 0 for value in proportions):
        raise SplitValidationError("Split proportions must contain three positive values.")
    if abs(sum(proportions) - 1.0) > 1e-9:
        raise SplitValidationError("Split proportions must sum to 1.0.")


def _patient_strata(patient_ids: pd.Series, labels: pd.Series) -> dict[str, int]:
    """Assign each patient a deterministic dominant-label stratum."""
    patient_labels: dict[str, int] = {}
    grouped_labels = pd.DataFrame({"patient_id": patient_ids, "label": labels}).groupby(
        "patient_id", sort=True
    )
    for patient_id, group in grouped_labels:
        label_counts = group["label"].value_counts()
        largest_count = label_counts.max()
        patient_labels[str(patient_id)] = max(
            int(label)
            for label, count in label_counts.items()
            if count == largest_count
        )
    return patient_labels


def _assign_patients(
    patient_labels: dict[str, int],
    seed: int,
    proportions: tuple[float, float, float],
) -> dict[str, str]:
    """Allocate shuffled patients within label strata to the requested proportions."""
    patients_by_label = {label: [] for label in sorted(VALID_LABELS)}
    for patient_id, label in patient_labels.items():
        patients_by_label[label].append(patient_id)

    random_generator = random.Random(seed)
    assignments: dict[str, str] = {}
    for label in sorted(VALID_LABELS):
        patient_ids = sorted(patients_by_label[label])
        random_generator.shuffle(patient_ids)
        counts = _allocate_counts(len(patient_ids), proportions)
        position = 0
        for split_name, count in zip(SPLIT_NAMES, counts, strict=True):
            for patient_id in patient_ids[position : position + count]:
                assignments[patient_id] = split_name
            position += count
    return assignments


def _allocate_counts(
    total: int, proportions: tuple[float, float, float]
) -> tuple[int, int, int]:
    """Allocate a finite patient count by largest-remainder rounding."""
    raw_counts = [total * proportion for proportion in proportions]
    counts = [int(value) for value in raw_counts]
    remainder = total - sum(counts)
    ranked_indices = sorted(
        range(len(SPLIT_NAMES)), key=lambda index: (raw_counts[index] - counts[index], -index), reverse=True
    )
    for index in ranked_indices[:remainder]:
        counts[index] += 1
    return tuple(counts)  # type: ignore[return-value]


def _split_report(
    records: pd.DataFrame, split_name: str, labels: pd.Series
) -> SplitReport:
    """Summarize rows, patients, labels, and projection views for one split."""
    split_mask = records["split"] == split_name
    split_records = records.loc[split_mask]
    split_labels = labels.loc[split_mask]
    row_count = len(split_records)
    ap_pa_views = split_records["ap_pa_view"].astype("string").str.strip().str.upper()
    positive_count = int((split_labels == 1).sum())
    negative_count = int((split_labels == 0).sum())
    uncertain_count = int((split_labels == -1).sum())
    return SplitReport(
        split=split_name,
        row_count=row_count,
        unique_patient_count=split_records["patient_id"].nunique(),
        positive_count=positive_count,
        negative_count=negative_count,
        uncertain_count=uncertain_count,
        positive_percentage=_percentage(positive_count, row_count),
        negative_percentage=_percentage(negative_count, row_count),
        uncertain_percentage=_percentage(uncertain_count, row_count),
        ap_count=int((ap_pa_views == "AP").sum()),
        pa_count=int((ap_pa_views == "PA").sum()),
    )


def _percentage(count: int, total: int) -> float:
    """Return a percentage, including an empty-split-safe value."""
    return 0.0 if total == 0 else 100 * count / total


def _overlapping_patient_counts(records: pd.DataFrame) -> dict[str, int]:
    """Return overlap counts for all split pairs."""
    patient_sets = {
        split_name: set(records.loc[records["split"] == split_name, "patient_id"])
        for split_name in SPLIT_NAMES
    }
    return {
        "train_validation": len(patient_sets["train"] & patient_sets["validation"]),
        "train_test": len(patient_sets["train"] & patient_sets["test"]),
        "validation_test": len(patient_sets["validation"] & patient_sets["test"]),
    }
