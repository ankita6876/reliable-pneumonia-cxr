"""Tests for construction of an unsplit CheXpert pneumonia cohort."""

import sys
from pathlib import Path, PureWindowsPath

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.data.chexpert_cohort import (  # noqa: E402
    _relative_manifest_path,
    build_chexpert_cohort,
    extract_study_id,
    write_chexpert_cohort,
)
from pneumonia_ai.data.chexpert_metadata import MetadataValidationError  # noqa: E402


def _row(path: str, view: str, label: object, ap_pa: object = "AP") -> dict[str, object]:
    """Build one complete synthetic CheXpert train row."""
    return {
        "Path": path,
        "Sex": "Female",
        "Age": 50,
        "Frontal/Lateral": view,
        "AP/PA": ap_pa,
        "Pneumonia": label,
    }


def _write_train(root: Path, rows: list[dict[str, object]]) -> None:
    """Write synthetic train metadata."""
    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)


def test_builds_frontal_cohort_and_retains_uncertain_labels(tmp_path: Path) -> None:
    """Only frontal 1, 0, and -1 rows become eligible without label remapping."""
    root = tmp_path / "chexpert"
    frontal = root / "train" / "patient00001" / "study1" / "view1_frontal.jpg"
    uncertain = root / "train" / "patient00002" / "study2" / "view1_frontal.jpg"
    frontal.parent.mkdir(parents=True)
    uncertain.parent.mkdir(parents=True)
    frontal.touch()
    uncertain.touch()
    _write_train(
        root,
        [
            _row("CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg", "Frontal", 1),
            _row("train/patient00002/study2/view1_frontal.jpg", "Frontal", -1, "PA"),
            _row("train/patient00003/study1/view1_lateral.jpg", "Lateral", 0),
            _row("train/patient00004/study1/view1.jpg", "Frontal", None),
            _row("train/patient00005/study1/._view1.jpg", "Frontal", 0),
        ],
    )

    cohort = build_chexpert_cohort(root)

    assert cohort.total_eligible_rows == 2
    assert cohort.unique_patients == 2
    assert cohort.positive_count == 1
    assert cohort.negative_count == 0
    assert cohort.uncertain_count == 1
    assert cohort.ap_count == 1
    assert cohort.pa_count == 1
    assert cohort.excluded_lateral_rows == 1
    assert cohort.excluded_missing_label_rows == 1
    assert cohort.records["pneumonia_label"].tolist() == [1, -1]
    assert cohort.records["image_path"].tolist() == [
        "train/patient00001/study1/view1_frontal.jpg",
        "train/patient00002/study2/view1_frontal.jpg",
    ]


def test_reports_missing_paths_duplicates_and_conflicting_labels(tmp_path: Path) -> None:
    """Audit fields identify missing files, duplicate rows, paths, and conflicts."""
    root = tmp_path / "chexpert"
    image = root / "train" / "patient00001" / "study1" / "view1.jpg"
    image.parent.mkdir(parents=True)
    image.touch()
    row = _row("train/patient00001/study1/view1.jpg", "Frontal", 1)
    _write_train(
        root,
        [
            row,
            row,
            _row("train/patient00001/study1/view1.jpg", "Frontal", 0),
            _row("train/patient00002/study1/missing.jpg", "Frontal", 0, None),
        ],
    )

    cohort = build_chexpert_cohort(root)

    assert cohort.missing_image_path_count == 1
    assert cohort.duplicate_row_count == 1
    assert cohort.duplicate_image_path_count == 2
    assert cohort.conflicting_label_image_path_count == 1
    assert cohort.missing_ap_pa_count == 1
    assert cohort.has_failures


def test_output_records_use_relative_image_paths(tmp_path: Path) -> None:
    """CSV output contains image paths relative to the supplied root."""
    root = tmp_path / "chexpert"
    image = root / "train" / "patient00001" / "study1" / "view1.jpg"
    image.parent.mkdir(parents=True)
    image.touch()
    _write_train(root, [_row("train/patient00001/study1/view1.jpg", "Frontal", 0)])

    output_path = tmp_path / "cohort.csv"
    write_chexpert_cohort(build_chexpert_cohort(root), output_path)
    saved = pd.read_csv(output_path)

    assert saved.loc[0, "image_path"] == "train/patient00001/study1/view1.jpg"
    assert not Path(saved.loc[0, "image_path"]).is_absolute()


def test_manifest_path_serialization_uses_forward_slashes_for_windows_paths() -> None:
    """Windows-compatible paths serialize with portable POSIX separators."""
    root = PureWindowsPath("C:/datasets/chexpert")
    image_path = root / "train" / "patient00001" / "study1" / "view1.jpg"

    serialized = _relative_manifest_path(image_path, root)

    assert serialized == "train/patient00001/study1/view1.jpg"
    assert "\\" not in serialized


def test_missing_required_columns_raise_clear_error(tmp_path: Path) -> None:
    """The cohort builder reports incomplete train metadata."""
    root = tmp_path / "chexpert"
    root.mkdir()
    pd.DataFrame({"Path": ["train/patient00001/study1/view1.jpg"]}).to_csv(
        root / "train.csv", index=False
    )

    with pytest.raises(MetadataValidationError, match="missing required column"):
        build_chexpert_cohort(root)


def test_extract_study_id_from_chexpert_path() -> None:
    """Study directory names are extracted from standard CheXpert paths."""
    assert extract_study_id(Path("train/patient00001/study12/view1.jpg")) == "study12"
