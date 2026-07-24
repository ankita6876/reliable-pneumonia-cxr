"""Gradient-free per-sample metrics for binary segmentation logits."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from pneumonia_ai.segmentation.losses import _validate_binary_shapes


@dataclass(frozen=True)
class SegmentationMetrics:
    """Mean per-sample binary segmentation metrics."""

    dice: float
    iou: float
    sensitivity: float
    specificity: float
    precision: float

    def to_dict(self) -> dict[str, float]:
        """Return metrics suitable for logging or serialization."""

        return asdict(self)


@torch.no_grad()
def per_sample_metrics(
    logits: Tensor, targets: Tensor, threshold: float = 0.5
) -> dict[str, Tensor]:
    """Return one metric value per sample after thresholding sigmoid logits.

    Empty target and prediction masks receive Dice, IoU, sensitivity,
    specificity, and precision of one.  A non-empty prediction against an
    empty target receives zero Dice, IoU, and precision.
    """

    _validate_binary_shapes(logits, targets)
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between zero and one.")
    predictions = torch.sigmoid(logits) >= threshold
    truth = targets.to(dtype=torch.bool)
    dimensions = (1, 2, 3)
    true_positive = (predictions & truth).sum(dim=dimensions, dtype=torch.float32)
    false_positive = (predictions & ~truth).sum(dim=dimensions, dtype=torch.float32)
    false_negative = (~predictions & truth).sum(dim=dimensions, dtype=torch.float32)
    true_negative = (~predictions & ~truth).sum(dim=dimensions, dtype=torch.float32)

    dice = _safe_ratio(2.0 * true_positive, 2.0 * true_positive + false_positive + false_negative)
    iou = _safe_ratio(true_positive, true_positive + false_positive + false_negative)
    sensitivity = _safe_ratio(true_positive, true_positive + false_negative)
    specificity = _safe_ratio(true_negative, true_negative + false_positive)
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    return {
        "dice": dice,
        "iou": iou,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
    }


@torch.no_grad()
def segmentation_metrics(
    logits: Tensor, targets: Tensor, threshold: float = 0.5
) -> SegmentationMetrics:
    """Aggregate per-sample segmentation metrics without retaining gradients."""

    values = per_sample_metrics(logits, targets, threshold=threshold)
    return SegmentationMetrics(**{name: value.mean().item() for name, value in values.items()})


def _safe_ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
    """Divide with an explicit perfect score for empty denominators."""

    return torch.where(denominator > 0, numerator / denominator, torch.ones_like(denominator))
