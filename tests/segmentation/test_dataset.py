from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from pneumonia_ai.segmentation.dataset import (
    MontgomerySegmentationDataset,
    SegmentationMetadataError,
)
from pneumonia_ai.segmentation.transforms import TrainingTransform


def _write_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), array)


def _write_metadata(tmp_path: Path, image: np.ndarray, mask: np.ndarray) -> Path:
    image_path = tmp_path / "images" / "sample.png"
    mask_path = tmp_path / "masks" / "sample.png"
    _write_image(image_path, image)
    _write_image(mask_path, mask)
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {
                "image_id": "sample",
                "image_path": image_path,
                "merged_mask_path": mask_path,
            }
        ]
    ).to_csv(metadata_path, index=False)
    return metadata_path


def test_dataset_returns_normalized_single_channel_tensors(tmp_path: Path) -> None:
    metadata_path = _write_metadata(
        tmp_path,
        np.array([[0, 255], [128, 64]], dtype=np.uint8),
        np.array([[0, 3], [255, 0]], dtype=np.uint8),
    )

    image, mask, metadata = MontgomerySegmentationDataset(metadata_path, image_size=8)[0]

    assert image.shape == (1, 8, 8)
    assert mask.shape == (1, 8, 8)
    assert image.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert 0.0 <= image.min() <= image.max() <= 1.0
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert metadata == {
        "image_id": "sample",
        "image_path": str(tmp_path / "images" / "sample.png"),
        "merged_mask_path": str(tmp_path / "masks" / "sample.png"),
    }


def test_training_transform_keeps_image_mask_geometry_synchronized(tmp_path: Path) -> None:
    pattern = np.array(
        [[0, 255, 0, 0], [0, 255, 255, 0], [0, 0, 255, 0], [0, 0, 0, 0]],
        dtype=np.uint8,
    )
    metadata_path = _write_metadata(tmp_path, pattern, pattern)
    transform = TrainingTransform(
        horizontal_flip_probability=1.0,
        max_rotation_degrees=0.0,
        max_translation_fraction=0.0,
        min_scale=1.0,
        max_scale=1.0,
        min_brightness=1.0,
        max_brightness=1.0,
        min_contrast=1.0,
        max_contrast=1.0,
    )

    image, mask, _ = MontgomerySegmentationDataset(
        metadata_path, image_size=4, transform=transform
    )[0]

    assert torch.equal(image, mask)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_dataset_raises_for_missing_sample_file(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {
                "image_id": "missing",
                "image_path": tmp_path / "missing-image.png",
                "merged_mask_path": tmp_path / "missing-mask.png",
            }
        ]
    ).to_csv(metadata_path, index=False)

    dataset = MontgomerySegmentationDataset(metadata_path)

    with pytest.raises(FileNotFoundError, match="Chest x-ray file does not exist"):
        dataset[0]


@pytest.mark.parametrize(
    "dataframe",
    [
        pd.DataFrame([{"image_id": "sample"}]),
        pd.DataFrame(columns=["image_id", "image_path", "merged_mask_path"]),
        pd.DataFrame(
            [
                {"image_id": "duplicate", "image_path": "a", "merged_mask_path": "a"},
                {"image_id": "duplicate", "image_path": "b", "merged_mask_path": "b"},
            ]
        ),
    ],
)
def test_dataset_rejects_invalid_metadata(tmp_path: Path, dataframe: pd.DataFrame) -> None:
    metadata_path = tmp_path / "invalid.csv"
    dataframe.to_csv(metadata_path, index=False)

    with pytest.raises(SegmentationMetadataError):
        MontgomerySegmentationDataset(metadata_path)
