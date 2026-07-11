"""Tests for CheXpert filesystem inventory."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.data.inventory import inventory_chexpert  # noqa: E402


def test_valid_inventory_counts_files_and_directories(tmp_path: Path) -> None:
    """The inventory reports supported images and general filesystem counts."""
    root = tmp_path / "chexpert"
    nested = root / "train" / "patient001"
    nested.mkdir(parents=True)
    (root / "valid.JPG").touch()
    (nested / "view.jpeg").touch()
    (nested / "mask.png").touch()
    (nested / "notes.txt").touch()

    inventory = inventory_chexpert(root)

    assert inventory.root_path == root.resolve()
    assert inventory.total_files == 4
    assert inventory.image_files_by_extension == {".jpg": 1, ".jpeg": 1, ".png": 1}
    assert inventory.total_image_files == 3
    assert inventory.directory_count == 2
    assert root / "valid.JPG" in inventory.representative_paths
    assert nested / "view.jpeg" in inventory.representative_paths


def test_missing_root_raises_clear_error(tmp_path: Path) -> None:
    """A nonexistent root is rejected before scanning."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        inventory_chexpert(tmp_path / "missing")


def test_file_root_raises_clear_error(tmp_path: Path) -> None:
    """A file cannot be used as the dataset root."""
    root_file = tmp_path / "not-a-directory.txt"
    root_file.touch()

    with pytest.raises(NotADirectoryError, match="not a directory"):
        inventory_chexpert(root_file)


def test_no_images_found_raises_clear_error(tmp_path: Path) -> None:
    """A directory without supported image files is rejected."""
    root = tmp_path / "chexpert"
    root.mkdir()
    (root / "labels.csv").touch()

    with pytest.raises(ValueError, match="No image files found"):
        inventory_chexpert(root)


def test_csv_discovery_records_full_paths(tmp_path: Path) -> None:
    """CSV files are recorded at every nested depth using full paths."""
    root = tmp_path / "chexpert"
    nested = root / "metadata" / "train"
    nested.mkdir(parents=True)
    top_level_csv = root / "valid.csv"
    nested_csv = nested / "labels.CSV"
    top_level_csv.touch()
    nested_csv.touch()
    (nested / "view.png").touch()

    inventory = inventory_chexpert(root)

    assert inventory.csv_files == (nested_csv, top_level_csv)


def test_metadata_artifact_image_is_not_counted(tmp_path: Path) -> None:
    """AppleDouble metadata artifacts are not treated as medical images."""
    root = tmp_path / "chexpert"
    root.mkdir()
    (root / "view.jpg").touch()
    (root / "._view.jpg").touch()

    inventory = inventory_chexpert(root)

    assert inventory.total_files == 2
    assert inventory.image_files_by_extension == {".jpg": 1, ".jpeg": 0, ".png": 0}
