"""Binary losses used by the focused optimisation experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Stable binary focal loss with optional positive-class weighting."""

    def __init__(self, gamma: float = 2.0, pos_weight: float | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=_weight(self.pos_weight, logits),
        )
        probability = torch.sigmoid(logits)
        p_t = probability * targets + (1 - probability) * (1 - targets)
        return ((1 - p_t).pow(self.gamma) * bce).mean()


class AsymmetricFocalLoss(nn.Module):
    """Binary asymmetric focal loss that down-weights easy negatives more strongly."""

    def __init__(
        self, gamma_positive: float = 1.0, gamma_negative: float = 4.0
    ) -> None:
        super().__init__()
        self.gamma_positive = gamma_positive
        self.gamma_negative = gamma_negative

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        positive = (
            targets
            * (1 - probability).pow(self.gamma_positive)
            * torch.log(probability)
        )
        negative = (
            (1 - targets)
            * probability.pow(self.gamma_negative)
            * torch.log(1 - probability)
        )
        return -(positive + negative).mean()


def build_loss(name: str, pos_weight: float | None = None) -> nn.Module:
    """Create one supported loss without implicit class-weighting changes."""
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "weighted_bce":
        return nn.BCEWithLogitsLoss(pos_weight=_weight(pos_weight, None))
    if name == "focal":
        return FocalLoss(pos_weight=pos_weight)
    if name == "asymmetric_focal":
        return AsymmetricFocalLoss()
    raise ValueError(f"Unsupported loss: {name}")


def _weight(value: float | None, reference: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return torch.tensor(value, device=None if reference is None else reference.device)
