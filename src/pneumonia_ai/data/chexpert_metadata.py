"""Validation of CheXpert CSV metadata without loading image contents."""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re

import pandas as pd


DATASET_DIRECTORY = "CheXpert-v1.0-small"
REQUIRED_COLUMNS = (
    "Path",
    "Sex",
    "Age",
    "Frontal/Lateral",
    "AP/PA",
    "Pneumonia",
)
PATIENT_ID_PATTERN = re.compile(r"^patient\d+$", re.IGNORECASE)


class MetadataValidationError(ValueError):
    """Raised when a metadata CSV cannot satisfy the validation requirements."""


@dataclass(frozen=True)
class SplitMetadataReport:
    """Validation summary for one CheXpert metadata CSV."""

    csv_path: Path
    row_count: int
    unique_patient_count: int
    frontal_image_count: int
    lateral_image_count: int
    pneumonia_label_counts: dict[str, int]
    missing_image_path_count: int
    duplicate_metadata_row_count: int
    patient_ids: frozenset[str]


@dataclass(frozen=True)
class CheXpertMetadataValidation:
    """Combined metadata validation result for CheXpert train and valid CSVs."""

    root_path: Path
    train: SplitMetadataReport
    valid: SplitMetadataReport
    overlapping_patient_ids: frozenset[str]

    @property
    def has_failures(self) -> bool:
        """Return whether missing images or cross-split patients were found."""
        return (
            self.train.missing_image_path_count > 0
            or self.valid.missing_image_path_count > 0
            or bool(self.overlapping_patient_ids)
        )


def validate_chexpert_metadata(root: Path | str) -> CheXpertMetadataValidation:
    """Validate train and valid CheXpert metadata against the supplied root.

    Only CSV contents and filesystem path metadata are read; referenced image files
    are never opened.

    Raises:
        FileNotFoundError: If a required CSV is absent.
        MetadataValidationError: If a required CSV column is absent.
    """
    root_path = Path(root).expanduser().resolve()
    train = _validate_split(root_path, "train")
    valid = _validate_split(root_path, "valid")
    return CheXpertMetadataValidation(
        root_path=root_path,
        train=train,
        valid=valid,
        overlapping_patient_ids=train.patient_ids & valid.patient_ids,
    )


def _validate_split(root_path: Path, split_name: str) -> SplitMetadataReport:
    """Load and validate one split CSV."""
    csv_path = _find_csv(root_path, split_name)
    dataframe = pd.read_csv(csv_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe]
    if missing_columns:
        raise MetadataValidationError(
            f"{csv_path} is missing required column(s): {', '.join(missing_columns)}"
        )

    resolved_image_paths = [
        _resolve_image_path(root_path, csv_value) for csv_value in dataframe["Path"]
    ]
    patient_ids = frozenset(
        patient_id
        for image_path in resolved_image_paths
        if (patient_id := extract_patient_id(image_path)) is not None
    )
    view_types = dataframe["Frontal/Lateral"].astype("string").str.strip().str.casefold()
    pneumonia_labels = pd.to_numeric(dataframe["Pneumonia"], errors="coerce")
    return SplitMetadataReport(
        csv_path=csv_path,
        row_count=len(dataframe),
        unique_patient_count=len(patient_ids),
        frontal_image_count=int((view_types == "frontal").sum()),
        lateral_image_count=int((view_types == "lateral").sum()),
        pneumonia_label_counts={
            "1": int((pneumonia_labels == 1).sum()),
            "0": int((pneumonia_labels == 0).sum()),
            "-1": int((pneumonia_labels == -1).sum()),
            "missing": int(pneumonia_labels.isna().sum()),
        },
        missing_image_path_count=sum(not image_path.is_file() for image_path in resolved_image_paths),
        duplicate_metadata_row_count=int(dataframe.duplicated().sum()),
        patient_ids=patient_ids,
    )


def _find_csv(root_path: Path, split_name: str) -> Path:
    """Find a required CSV in either supported extracted-root layout."""
    direct_path = root_path / f"{split_name}.csv"
    nested_path = root_path / DATASET_DIRECTORY / f"{split_name}.csv"
    if direct_path.is_file():
        return direct_path
    if nested_path.is_file():
        return nested_path
    raise FileNotFoundError(
        f"Required {split_name}.csv was not found at {direct_path} or {nested_path}."
    )


def _resolve_image_path(root_path: Path, csv_path_value: object) -> Path:
    """Resolve a CSV image path for direct and nested extracted-root layouts."""
    path_parts = PurePosixPath(str(csv_path_value).replace("\\", "/")).parts
    direct_path = root_path.joinpath(*path_parts)
    if not path_parts or path_parts[0] != DATASET_DIRECTORY:
        return direct_path

    stripped_path = root_path.joinpath(*path_parts[1:])
    if direct_path.is_file() or not stripped_path.is_file():
        return direct_path
    return stripped_path


def extract_patient_id(image_path: Path) -> str | None:
    """Extract a CheXpert patient directory name from an image path."""
    for path_part in image_path.parts:
        if PATIENT_ID_PATTERN.fullmatch(path_part):
            return path_part
    return None
