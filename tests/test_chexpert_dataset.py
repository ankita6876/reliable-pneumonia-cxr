"""Tests for the CheXpert PyTorch dataset."""

import sys
from pathlib import Path

import pandas as pd
from PIL import Image
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.data.chexpert_dataset import (  # noqa: E402
    CheXpertPneumoniaDataset,
    DatasetValidationError,
)


def _manifest_row(patient_id: str, split: str, label: object) -> dict[str, object]:
    """Create a portable synthetic manifest row."""
    return {
        "patient_id": patient_id,
        "study_id": "study1",
        "image_path": f"train/{patient_id}/study1/view1.jpg",
        "pneumonia_label": label,
        "ap_pa_view": "AP",
        "split": split,
    }


def _write_image(root: Path, relative_path: str) -> None:
    """Create a grayscale dummy image at a POSIX-style relative path."""
    image_path = root.joinpath(*relative_path.split("/"))
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (12, 10), color=128).save(image_path)


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a synthetic split manifest."""
    pd.DataFrame(rows).to_csv(path, index=False)


def test_filters_requested_split_and_preserves_all_labels(tmp_path: Path) -> None:
    """Only the requested split is exposed, with labels 1, 0, and -1 unchanged."""
    root = tmp_path / "chexpert"
    rows = [
        _manifest_row("patient00001", "train", 1),
        _manifest_row("patient00002", "train", 0),
        _manifest_row("patient00003", "train", -1),
        _manifest_row("patient00004", "validation", 1),
    ]
    for row in rows:
        _write_image(root, str(row["image_path"]))
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, rows)

    dataset = CheXpertPneumoniaDataset(root, manifest_path, "train")

    assert len(dataset) == 3
    assert [dataset[index]["label"].item() for index in range(len(dataset))] == [1.0, 0.0, -1.0]


def test_converts_grayscale_to_rgb_and_applies_transform(tmp_path: Path) -> None:
    """Pillow images are RGB before the optional caller-provided transform runs."""
    root = tmp_path / "chexpert"
    row = _manifest_row("patient00001", "train", 1)
    _write_image(root, str(row["image_path"]))
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [row])
    received_modes: list[str] = []

    def transform(image: Image.Image) -> torch.Tensor:
        received_modes.append(image.mode)
        return torch.ones((3, 4, 5))

    transformed_dataset = CheXpertPneumoniaDataset(root, manifest_path, "train", transform)
    plain_dataset = CheXpertPneumoniaDataset(root, manifest_path, "train")

    assert plain_dataset[0]["image"].mode == "RGB"
    assert torch.equal(transformed_dataset[0]["image"], torch.ones((3, 4, 5)))
    assert received_modes == ["RGB"]


def test_returned_image_path_is_relative_posix_path(tmp_path: Path) -> None:
    """Returned image metadata is relative and portable across operating systems."""
    root = tmp_path / "chexpert"
    row = _manifest_row("patient00001", "train", 0)
    _write_image(root, "train/patient00001/study1/view1.jpg")
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [row])

    sample = CheXpertPneumoniaDataset(root, manifest_path, "train")[0]

    returned_path = sample["image_path"]

    assert returned_path == "train/patient00001/study1/view1.jpg"
    assert not Path(returned_path).is_absolute()
    assert "/" in returned_path
    assert "\\" not in returned_path


def test_raises_for_missing_image(tmp_path: Path) -> None:
    """A manifest image that is absent from the root raises a clear error."""
    root = tmp_path / "chexpert"
    root.mkdir()
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [_manifest_row("patient00001", "train", 1)])

    with pytest.raises(FileNotFoundError, match="Image file"):
        CheXpertPneumoniaDataset(root, manifest_path, "train")


def test_raises_for_invalid_split_invalid_label_and_empty_split(tmp_path: Path) -> None:
    """Invalid split names, labels, and empty requested splits are rejected clearly."""
    root = tmp_path / "chexpert"
    valid_row = _manifest_row("patient00001", "train", 1)
    _write_image(root, str(valid_row["image_path"]))
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [valid_row])

    with pytest.raises(DatasetValidationError, match="Invalid split"):
        CheXpertPneumoniaDataset(root, manifest_path, "development")
    with pytest.raises(DatasetValidationError, match="contains no rows"):
        CheXpertPneumoniaDataset(root, manifest_path, "validation")

    invalid_manifest = tmp_path / "invalid_manifest.csv"
    invalid_row = _manifest_row("patient00001", "train", 2)
    _write_manifest(invalid_manifest, [invalid_row])
    with pytest.raises(DatasetValidationError, match="invalid Pneumonia labels"):
        CheXpertPneumoniaDataset(root, invalid_manifest, "train")


def test_raises_for_missing_manifest_root_and_columns(tmp_path: Path) -> None:
    """Root, manifest, and schema errors are raised before image loading."""
    root = tmp_path / "chexpert"
    root.mkdir()
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [_manifest_row("patient00001", "train", 1)])

    with pytest.raises(FileNotFoundError, match="dataset root"):
        CheXpertPneumoniaDataset(tmp_path / "missing", manifest_path, "train")
    with pytest.raises(FileNotFoundError, match="Split manifest"):
        CheXpertPneumoniaDataset(root, tmp_path / "missing.csv", "train")

    incomplete_manifest = tmp_path / "incomplete.csv"
    pd.DataFrame({"split": ["train"]}).to_csv(incomplete_manifest, index=False)
    with pytest.raises(DatasetValidationError, match="required column"):
        CheXpertPneumoniaDataset(root, incomplete_manifest, "train")
