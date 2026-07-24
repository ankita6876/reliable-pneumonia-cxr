"""Loss functions for binary lung segmentation logits."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional


def _validate_binary_shapes(logits: Tensor, targets: Tensor) -> None:
    """Validate batched single-channel binary segmentation tensors."""

    if logits.ndim != 4 or targets.ndim != 4:
        raise ValueError("logits and targets must have shape [B, 1, H, W].")
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits and targets must have equal shapes, got {logits.shape} and {targets.shape}."
        )
    if logits.shape[1] != 1:
        raise ValueError("Binary segmentation requires a single-channel tensor.")


def dice_loss_from_logits(logits: Tensor, targets: Tensor, smooth: float = 1.0) -> Tensor:
    """Return mean binary Dice loss computed stably from raw logits."""

    _validate_binary_shapes(logits, targets)
    if smooth <= 0:
        raise ValueError("smooth must be positive.")
    probabilities = torch.sigmoid(logits)
    target_values = targets.to(dtype=probabilities.dtype)
    dimensions = (1, 2, 3)
    intersection = (probabilities * target_values).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + target_values.sum(dim=dimensions)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


class BinaryDiceLoss(nn.Module):
    """Binary Dice loss module accepting raw logits and binary masks."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        if smooth <= 0:
            raise ValueError("smooth must be positive.")
        self.smooth = smooth

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute Dice loss."""

        return dice_loss_from_logits(logits, targets, smooth=self.smooth)


class BCEWithLogitsDiceLoss(nn.Module):
    """Weighted sum of BCE-with-logits and binary Dice losses."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
            raise ValueError("At least one non-negative loss weight must be positive.")
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the configured BCE and Dice weighted sum."""

        _validate_binary_shapes(logits, targets)
        target_values = targets.to(dtype=logits.dtype)
        bce = functional.binary_cross_entropy_with_logits(logits, target_values)
        dice = dice_loss_from_logits(logits, target_values, smooth=self.smooth)
        return self.bce_weight * bce + self.dice_weight * dice
