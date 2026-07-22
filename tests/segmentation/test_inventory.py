from pathlib import Path

from pneumonia_ai.segmentation.inventory import inventory_dataset


def test_inventory_missing_directory(tmp_path: Path) -> None:
    result = inventory_dataset(tmp_path / "missing")

    assert result.exists is False
    assert result.total_files == 0
    assert result.image_files == 0
    assert result.other_files == 0
    assert result.extensions == {}


def test_inventory_counts_supported_images(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    nested = dataset_root / "nested"
    nested.mkdir(parents=True)

    (dataset_root / "image_1.png").write_bytes(b"png")
    (nested / "image_2.JPG").write_bytes(b"jpg")
    (nested / "mask.tiff").write_bytes(b"tiff")
    (dataset_root / "metadata.csv").write_text("id\n1\n", encoding="utf-8")

    result = inventory_dataset(dataset_root)

    assert result.exists is True
    assert result.total_files == 4
    assert result.image_files == 3
    assert result.other_files == 1
    assert result.extensions == {
        ".csv": 1,
        ".jpg": 1,
        ".png": 1,
        ".tiff": 1,
    }


def test_inventory_handles_files_without_extensions(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    (dataset_root / "README").write_text("dataset", encoding="utf-8")

    result = inventory_dataset(dataset_root)

    assert result.total_files == 1
    assert result.image_files == 0
    assert result.other_files == 1
    assert result.extensions == {"<no_extension>": 1}
