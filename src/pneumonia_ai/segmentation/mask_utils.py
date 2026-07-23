"""Utilities for validating and merging lung segmentation masks."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class MaskValidationError(ValueError):
    """Raised when a segmentation mask cannot be used safely."""


def read_mask(mask_path: Path | str) -> np.ndarray:
    """Read a mask as a single-channel image.

    Raises:
        MaskValidationError: If OpenCV cannot decode the image.
    """

    path = Path(mask_path)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise MaskValidationError(f"Unable to read mask image: {path}")
    return mask


def merge_lung_masks(left_mask: np.ndarray, right_mask: np.ndarray) -> np.ndarray:
    """Combine left and right lung masks into a binary ``uint8`` mask.

    The returned image contains only 0 and 255 so it can be written directly as
    a conventional binary PNG.
    """

    if left_mask.ndim != 2 or right_mask.ndim != 2:
        raise MaskValidationError("Lung masks must be single-channel images.")
    if left_mask.shape != right_mask.shape:
        raise MaskValidationError(
            "Left and right masks have different resolutions: "
            f"{left_mask.shape} and {right_mask.shape}."
        )

    return np.where((left_mask > 0) | (right_mask > 0), 255, 0).astype(np.uint8)


def save_mask(mask: np.ndarray, output_path: Path | str) -> None:
    """Write a binary mask, raising when OpenCV cannot save it."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask):
        raise OSError(f"Unable to write merged mask: {path}")
