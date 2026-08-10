"""Deterministic segmentation-guided classifier image preparation."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
import torch

if TYPE_CHECKING:
    from pneumonia_ai.segmentation.inference import FrozenLungSegmenter


class InputMode(StrEnum):
    ORIGINAL = "original"
    HARD_MASKED = "hard_masked"
    SOFT_MASKED = "soft_masked"
    LUNG_CROP = "lung_crop"


def prepare_classifier_image(
    image: Image.Image,
    mode: InputMode | str = InputMode.ORIGINAL,
    *,
    segmenter: FrozenLungSegmenter | None = None,
    threshold: float = 0.5,
    soft_mask_outside_factor: float = 0.20,
    crop_padding: int = 0,
    output_size: int | tuple[int, int] | None = None,
    probability_mask: torch.Tensor | None = None,
) -> Image.Image:
    """Create an ``L`` image for classifier transforms without using labels.

    Empty crop masks fall back to the complete image. ``output_size`` is applied
    to every mode when supplied (integer means square dimensions).
    """
    try:
        resolved_mode = InputMode(mode)
    except ValueError as error:
        raise ValueError("mode must be one of: original, hard_masked, soft_masked, lung_crop.") from error
    size = _validate_output_size(output_size)
    if crop_padding < 0:
        raise ValueError("crop_padding must be non-negative.")
    if isinstance(soft_mask_outside_factor, bool) or not _is_unit_interval(soft_mask_outside_factor):
        raise ValueError("soft_mask_outside_factor must be between 0 and 1 inclusive.")
    grayscale = image.convert("L")
    if resolved_mode is InputMode.ORIGINAL:
        return _resize(grayscale, size)
    if segmenter is None and probability_mask is None:
        raise ValueError(f"input mode {resolved_mode.value!r} requires a FrozenLungSegmenter.")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1.")
    probability = probability_mask if probability_mask is not None else segmenter.predict_proba(grayscale)
    if probability.shape != (grayscale.height, grayscale.width):
        raise ValueError("Probability mask dimensions must match the input image.")
    mask = probability.detach().cpu().numpy() >= threshold
    array = np.asarray(grayscale).copy()
    if resolved_mode is InputMode.HARD_MASKED:
        array[~mask] = 0
        return _resize(Image.fromarray(array, mode="L"), size)
    if resolved_mode is InputMode.SOFT_MASKED:
        # Work in float before the explicit uint8 conversion so the attenuation
        # is deterministic and never relies on NumPy's in-place cast semantics.
        softened = array.astype(np.float32)
        softened[~mask] *= float(soft_mask_outside_factor)
        return _resize(Image.fromarray(softened.astype(np.uint8), mode="L"), size)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return _resize(grayscale, size)
    left, right = max(0, xs.min() - crop_padding), min(grayscale.width, xs.max() + crop_padding + 1)
    top, bottom = max(0, ys.min() - crop_padding), min(grayscale.height, ys.max() + crop_padding + 1)
    return _resize(grayscale.crop((left, top, right, bottom)), size)


def _validate_output_size(value: int | tuple[int, int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    size = (value, value) if isinstance(value, int) else value
    if len(size) != 2 or any(not isinstance(item, int) or item <= 0 for item in size):
        raise ValueError("output_size must be a positive integer or (width, height) tuple.")
    return size


def _resize(image: Image.Image, size: tuple[int, int] | None) -> Image.Image:
    return image.copy() if size is None else image.resize(size, Image.Resampling.BILINEAR)


def _is_unit_interval(value: object) -> bool:
    try:
        return 0.0 <= float(value) <= 1.0
    except (TypeError, ValueError, OverflowError):
        return False
