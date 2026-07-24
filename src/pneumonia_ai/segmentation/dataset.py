"""PyTorch dataset for paired Montgomery chest x-rays and lung masks."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from pneumonia_ai.segmentation.transforms import EvaluationTransform, resize_image_and_mask


REQUIRED_METADATA_COLUMNS = ("image_id", "image_path", "merged_mask_path")
PairTransform = Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]


class SegmentationMetadataError(ValueError):
    """Raised when a segmentation manifest cannot describe valid samples."""


def load_segmentation_metadata(metadata_path: Path | str) -> pd.DataFrame:
    """Load and validate the columns required by the segmentation pipeline."""

    path = Path(metadata_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Metadata CSV does not exist: {path}")

    metadata = pd.read_csv(path)
    missing_columns = [
        column for column in REQUIRED_METADATA_COLUMNS if column not in metadata.columns
    ]
    if missing_columns:
        raise SegmentationMetadataError(
            f"Metadata CSV is missing required columns: {', '.join(missing_columns)}"
        )
    if metadata.empty:
        raise SegmentationMetadataError("Metadata CSV contains no samples.")
    if metadata.loc[:, list(REQUIRED_METADATA_COLUMNS)].isnull().any().any():
        raise SegmentationMetadataError("Metadata CSV contains missing required values.")
    if metadata["image_id"].astype(str).duplicated().any():
        raise SegmentationMetadataError("Metadata CSV contains duplicate image_id values.")
    return metadata.reset_index(drop=True)


class MontgomerySegmentationDataset(Dataset[tuple[Tensor, Tensor, dict[str, str]]]):
    """Load normalized, resized Montgomery x-rays and binary lung masks."""

    def __init__(
        self,
        metadata_path: Path | str,
        image_size: int = 512,
        transform: PairTransform | None = None,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be a positive integer.")
        self.metadata = load_segmentation_metadata(metadata_path)
        self.image_size = image_size
        self.transform = transform or EvaluationTransform()

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, dict[str, str]]:
        record = self.metadata.iloc[index]
        image_path = Path(str(record["image_path"])).expanduser()
        mask_path = Path(str(record["merged_mask_path"])).expanduser()
        image = _read_grayscale(image_path, "chest x-ray")
        mask = (_read_grayscale(mask_path, "merged mask") > 0.0).astype(np.float32)
        image_tensor = torch.from_numpy(image).unsqueeze(0).to(dtype=torch.float32)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(dtype=torch.float32)
        image_tensor, mask_tensor = resize_image_and_mask(
            image_tensor, mask_tensor, self.image_size
        )
        image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)
        metadata: dict[str, str] = {
            "image_id": str(record["image_id"]),
            "image_path": str(image_path),
            "merged_mask_path": str(mask_path),
        }
        return (
            image_tensor.clamp(0.0, 1.0).to(dtype=torch.float32),
            (mask_tensor > 0.5).to(dtype=torch.float32),
            metadata,
        )


def _read_grayscale(path: Path, description: str) -> np.ndarray:
    """Read one grayscale image and scale its pixels into ``[0, 1]``."""

    if not path.is_file():
        raise FileNotFoundError(f"{description.capitalize()} file does not exist: {path}")
    array = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if array is None:
        raise OSError(f"Unable to decode {description}: {path}")
    return array.astype(np.float32) / 255.0
