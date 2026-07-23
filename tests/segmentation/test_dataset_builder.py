from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from pneumonia_ai.segmentation.dataset_builder import (
    MontgomeryDatasetError,
    build_montgomery_dataset,
    validate_filename_matching,
)
from pneumonia_ai.segmentation.mask_utils import merge_lung_masks
from pneumonia_ai.segmentation.metadata import (
    MontgomeryMetadataRecord,
    build_metadata_dataframe,
)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def _create_dataset(root: Path) -> None:
    image = np.zeros((4, 5), dtype=np.uint8)
    _write_image(root / "CXR_png" / "MCUCXR_0002_0.png", image)
    _write_image(root / "CXR_png" / "MCUCXR_0001_0.png", image)
    _write_image(root / "ManualMask" / "leftMask" / "MCUCXR_0002_0.png", image)
    _write_image(root / "ManualMask" / "leftMask" / "MCUCXR_0001_0.png", image)
    _write_image(root / "ManualMask" / "rightMask" / "MCUCXR_0002_0.png", image)
    _write_image(root / "ManualMask" / "rightMask" / "MCUCXR_0001_0.png", image)


def test_filename_matching_rejects_missing_and_unexpected_masks(tmp_path: Path) -> None:
    images = [tmp_path / "image_a.png"]
    left_masks = [tmp_path / "image_b.png"]

    with pytest.raises(MontgomeryDatasetError, match="missing left mask"):
        validate_filename_matching(images, left_masks, images)


def test_merge_lung_masks_is_binary_union() -> None:
    left = np.array([[0, 3], [0, 0]], dtype=np.uint8)
    right = np.array([[9, 0], [0, 1]], dtype=np.uint8)

    merged = merge_lung_masks(left, right)

    np.testing.assert_array_equal(
        merged, np.array([[255, 255], [0, 255]], dtype=np.uint8)
    )


def test_build_dataset_writes_merged_masks_and_metadata(tmp_path: Path) -> None:
    _create_dataset(tmp_path)
    left_path = tmp_path / "ManualMask" / "leftMask" / "MCUCXR_0001_0.png"
    right_path = tmp_path / "ManualMask" / "rightMask" / "MCUCXR_0001_0.png"
    _write_image(left_path, np.array([[1, 0, 0, 0, 0]] * 4, dtype=np.uint8))
    _write_image(right_path, np.array([[0, 0, 1, 0, 0]] * 4, dtype=np.uint8))

    metadata = build_montgomery_dataset(tmp_path)

    assert metadata["image_id"].tolist() == ["MCUCXR_0001_0", "MCUCXR_0002_0"]
    assert metadata[["width", "height"]].values.tolist() == [[5, 4], [5, 4]]
    assert (tmp_path / "processed" / "metadata.csv").is_file()
    persisted = pd.read_csv(tmp_path / "processed" / "metadata.csv")
    assert persisted.equals(metadata)
    merged = cv2.imread(
        str(tmp_path / "merged_masks" / "MCUCXR_0001_0.png"), cv2.IMREAD_GRAYSCALE
    )
    np.testing.assert_array_equal(
        merged, np.array([[255, 0, 255, 0, 0]] * 4, dtype=np.uint8)
    )


def test_build_dataset_detects_missing_mask(tmp_path: Path) -> None:
    _create_dataset(tmp_path)
    (tmp_path / "ManualMask" / "rightMask" / "MCUCXR_0002_0.png").unlink()

    with pytest.raises(MontgomeryDatasetError, match="missing right mask"):
        build_montgomery_dataset(tmp_path)


def test_build_metadata_dataframe_orders_records(tmp_path: Path) -> None:
    record = MontgomeryMetadataRecord.from_paths(
        image_path=tmp_path / "b.png",
        left_mask_path=tmp_path / "left.png",
        right_mask_path=tmp_path / "right.png",
        merged_mask_path=tmp_path / "merged.png",
        width=10,
        height=20,
    )

    dataframe = build_metadata_dataframe([record])

    assert dataframe.columns.tolist() == [
        "image_id",
        "image_path",
        "left_mask_path",
        "right_mask_path",
        "merged_mask_path",
        "width",
        "height",
    ]
    assert dataframe.loc[0, "image_id"] == "b"
