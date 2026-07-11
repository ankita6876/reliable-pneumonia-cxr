"""Build the eligible CheXpert pneumonia cohort from train metadata."""

from dataclasses import dataclass
from pathlib import Path, PurePath
import re

import pandas as pd

from pneumonia_ai.data.chexpert_metadata import (
    MetadataValidationError,
    REQUIRED_COLUMNS,
    _find_csv,
    _resolve_image_path,
    extract_patient_id,
)


STUDY_ID_PATTERN = re.compile(r"^study\d+$", re.IGNORECASE)
ELIGIBLE_PNEUMONIA_LABELS = (1, 0, -1)
COHORT_COLUMNS = (
    "patient_id",
    "study_id",
    "image_path",
    "original_csv_path",
    "pneumonia_label",
    "sex",
    "age",
    "ap_pa_view",
)


@dataclass(frozen=True)
class CheXpertCohort:
    """Eligible CheXpert pneumonia records and their construction audit."""

    root_path: Path
    train_csv_path: Path
    records: pd.DataFrame
    total_eligible_rows: int
    unique_patients: int
    positive_count: int
    negative_count: int
    uncertain_count: int
    ap_count: int
    pa_count: int
    missing_ap_pa_count: int
    excluded_lateral_rows: int
    excluded_missing_label_rows: int
    missing_image_path_count: int
    duplicate_row_count: int
    duplicate_image_path_count: int
    conflicting_label_image_path_count: int

    @property
    def has_failures(self) -> bool:
        """Return whether missing images or conflicting labels were found."""
        return (
            self.missing_image_path_count > 0
            or self.conflicting_label_image_path_count > 0
        )


def build_chexpert_cohort(root: Path | str) -> CheXpertCohort:
    """Construct an unsplit, frontal CheXpert pneumonia cohort from ``train.csv``.

    Image files are only checked for path existence; their contents are never read.

    Raises:
        FileNotFoundError: If ``train.csv`` cannot be found.
        MetadataValidationError: If train metadata lacks required columns.
    """
    root_path = Path(root).expanduser().resolve()
    train_csv_path = _find_csv(root_path, "train")
    dataframe = pd.read_csv(train_csv_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe]
    if missing_columns:
        raise MetadataValidationError(
            f"{train_csv_path} is missing required column(s): {', '.join(missing_columns)}"
        )

    view_types = dataframe["Frontal/Lateral"].astype("string").str.strip().str.casefold()
    pneumonia_labels = pd.to_numeric(dataframe["Pneumonia"], errors="coerce")
    original_paths = dataframe["Path"].astype("string")
    image_paths = original_paths.map(lambda value: _resolve_image_path(root_path, value))
    is_artifact = image_paths.map(lambda path: path.name.startswith("._"))
    eligible_mask = (
        (view_types == "frontal")
        & pneumonia_labels.isin(ELIGIBLE_PNEUMONIA_LABELS)
        & ~is_artifact
    )
    eligible_dataframe = dataframe.loc[eligible_mask].copy()
    eligible_paths = image_paths.loc[eligible_mask]
    eligible_labels = pneumonia_labels.loc[eligible_mask].astype("int64")
    eligible_views = dataframe.loc[eligible_mask, "AP/PA"].astype("string").str.strip()
    records = _build_records(
        root_path,
        eligible_dataframe,
        eligible_paths,
        eligible_labels,
        eligible_views,
    )
    duplicate_paths = records["image_path"].duplicated()
    conflicting_paths = _conflicting_image_path_count(records)
    ap_pa_normalized = eligible_views.str.upper()
    return CheXpertCohort(
        root_path=root_path,
        train_csv_path=train_csv_path,
        records=records,
        total_eligible_rows=len(records),
        unique_patients=records["patient_id"].nunique(),
        positive_count=int((eligible_labels == 1).sum()),
        negative_count=int((eligible_labels == 0).sum()),
        uncertain_count=int((eligible_labels == -1).sum()),
        ap_count=int((ap_pa_normalized == "AP").sum()),
        pa_count=int((ap_pa_normalized == "PA").sum()),
        missing_ap_pa_count=int((eligible_views.isna() | (eligible_views == "")).sum()),
        excluded_lateral_rows=int((view_types == "lateral").sum()),
        excluded_missing_label_rows=int(pneumonia_labels.isna().sum()),
        missing_image_path_count=sum(not path.is_file() for path in eligible_paths),
        duplicate_row_count=int(eligible_dataframe.duplicated().sum()),
        duplicate_image_path_count=int(duplicate_paths.sum()),
        conflicting_label_image_path_count=conflicting_paths,
    )


def write_chexpert_cohort(cohort: CheXpertCohort, output_path: Path | str) -> None:
    """Write cohort records to CSV with only CheXpert-root-relative image paths."""
    cohort.records.to_csv(Path(output_path), index=False)


def extract_study_id(image_path: Path) -> str | None:
    """Extract a CheXpert study directory name from an image path."""
    for path_part in image_path.parts:
        if STUDY_ID_PATTERN.fullmatch(path_part):
            return path_part
    return None


def _build_records(
    root_path: Path,
    dataframe: pd.DataFrame,
    image_paths: pd.Series,
    pneumonia_labels: pd.Series,
    ap_pa_views: pd.Series,
) -> pd.DataFrame:
    """Create serializable, root-relative cohort records."""
    records = pd.DataFrame(
        {
            "patient_id": image_paths.map(extract_patient_id),
            "study_id": image_paths.map(extract_study_id),
            "image_path": image_paths.map(
                lambda path: _relative_manifest_path(path, root_path)
            ),
            "original_csv_path": dataframe["Path"].astype("string"),
            "pneumonia_label": pneumonia_labels,
            "sex": dataframe["Sex"],
            "age": dataframe["Age"],
            "ap_pa_view": ap_pa_views,
        }
    )
    return records.loc[:, COHORT_COLUMNS]


def _relative_manifest_path(path: PurePath, root_path: PurePath) -> str:
    """Serialize a root-relative dataset path with portable POSIX separators."""
    return path.relative_to(root_path).as_posix()


def _conflicting_image_path_count(records: pd.DataFrame) -> int:
    """Count image paths assigned more than one retained pneumonia label."""
    labels_per_path = records.groupby("image_path")["pneumonia_label"].nunique()
    return int((labels_per_path > 1).sum())
