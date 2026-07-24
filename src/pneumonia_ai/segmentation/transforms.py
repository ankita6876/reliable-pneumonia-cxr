"""Synchronized transforms for lung-segmentation image and mask pairs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional


def resize_image_and_mask(
    image: Tensor, mask: Tensor, image_size: int
) -> tuple[Tensor, Tensor]:
    """Resize an image/mask pair with interpolation appropriate to each."""

    size = [image_size, image_size]
    return (
        transform_functional.resize(
            image, size, interpolation=InterpolationMode.BILINEAR, antialias=True
        ),
        transform_functional.resize(
            mask, size, interpolation=InterpolationMode.NEAREST
        ),
    )


@dataclass(frozen=True)
class EvaluationTransform:
    """Deterministic no-op transform for validation and test samples."""

    def __call__(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        return image, mask


@dataclass(frozen=True)
class TrainingTransform:
    """Apply stochastic, synchronized geometric augmentation to a sample pair."""

    horizontal_flip_probability: float = 0.5
    max_rotation_degrees: float = 10.0
    max_translation_fraction: float = 0.05
    min_scale: float = 0.95
    max_scale: float = 1.05
    min_brightness: float = 0.9
    max_brightness: float = 1.1
    min_contrast: float = 0.9
    max_contrast: float = 1.1

    def __call__(self, image: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Transform both tensors while applying photometric changes only to image."""

        if torch.rand(()) < self.horizontal_flip_probability:
            image = transform_functional.hflip(image)
            mask = transform_functional.hflip(mask)

        angle = _sample_uniform(-self.max_rotation_degrees, self.max_rotation_degrees)
        max_x_translation = image.shape[-1] * self.max_translation_fraction
        max_y_translation = image.shape[-2] * self.max_translation_fraction
        translate = [
            round(_sample_uniform(-max_x_translation, max_x_translation)),
            round(_sample_uniform(-max_y_translation, max_y_translation)),
        ]
        scale = _sample_uniform(self.min_scale, self.max_scale)
        image = transform_functional.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        mask = transform_functional.affine(
            mask,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0.0,
        )
        image = transform_functional.adjust_brightness(
            image, _sample_uniform(self.min_brightness, self.max_brightness)
        )
        image = transform_functional.adjust_contrast(
            image, _sample_uniform(self.min_contrast, self.max_contrast)
        )
        return image.clamp_(0.0, 1.0), (mask > 0.5).to(dtype=torch.float32)


def _sample_uniform(minimum: float, maximum: float) -> float:
    """Sample one value from a closed float range using PyTorch's RNG."""

    return minimum + (maximum - minimum) * torch.rand(()).item()
