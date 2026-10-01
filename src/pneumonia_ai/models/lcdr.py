"""Lung-Context Disentanglement Regularization (LCDR).

LCDR preserves the ordinary full-image ResNet50 classifier pathway.

During training only, a frozen lung probability map partitions an
intermediate layer2 feature tensor into lung and complementary-context
representations. A context adversary receives the context representation
through gradient reversal.

At inference, the auxiliary pathway is not required: prediction is the
ordinary ResNet50 full-image forward pass.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class _GradientReversalFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return -ctx.alpha * grad_output, None


def gradient_reverse(x: Tensor, alpha: float) -> Tensor:
    return _GradientReversalFunction.apply(x, float(alpha))


class LCDRResNet50(nn.Module):
    """P7-compatible ResNet50 plus training-only context adversary."""

    def __init__(
        self,
        pretrained: bool = True,
        grl_alpha: float = 0.03,
    ) -> None:
        super().__init__()

        weights = (
            ResNet50_Weights.IMAGENET1K_V2
            if pretrained
            else None
        )

        self.backbone = resnet50(weights=weights)

        # Preserve P7 binary-classification head semantics.
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

        self.grl_alpha = float(grl_alpha)

        # layer2 output dimension for ResNet50 = 512.
        # Recorded LCDR adversary: linear 512 -> 1.
        self.context_adversary = nn.Linear(512, 1)

    @staticmethod
    def _masked_pool(features: Tensor, mask: Tensor) -> Tensor:
        numerator = (features * mask).sum(dim=(2, 3))
        denominator = mask.sum(dim=(2, 3)).clamp_min(1e-6)
        return numerator / denominator

    def _forward_to_layer2(self, image: Tensor) -> Tensor:

        x = self.backbone.conv1(image)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)

        return x

    def _forward_from_layer2(self, x: Tensor) -> Tensor:

        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)

        return x.squeeze(1)

    def forward(
        self,
        image: Tensor,
        lung_map: Tensor | None = None,
        *,
        return_aux: bool = False,
    ):

        # ----------------------------------------------------
        # Ordinary inference path.
        # Exactly a normal full-image ResNet50 prediction.
        # ----------------------------------------------------
        if lung_map is None and not return_aux:
            return self.backbone(image).squeeze(1)

        # ----------------------------------------------------
        # Training/diagnostic path.
        # ----------------------------------------------------
        if lung_map is None:
            raise ValueError(
                "lung_map is required when return_aux=True."
            )

        features = self._forward_to_layer2(image)

        if lung_map.ndim == 3:
            lung_map = lung_map.unsqueeze(1)

        if lung_map.ndim != 4 or lung_map.shape[1] != 1:
            raise ValueError(
                "lung_map must have shape Bx1xHxW or BxHxW."
            )

        lung_map = lung_map.to(
            device=features.device,
            dtype=features.dtype,
        )

        lung_map = F.interpolate(
            lung_map,
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)

        context_map = 1.0 - lung_map

        lung_features = self._masked_pool(
            features,
            lung_map,
        )

        context_features = self._masked_pool(
            features,
            context_map,
        )

        # Main classifier is unchanged.
        classification_logit = self._forward_from_layer2(
            features
        )

        # Context adversary:
        # forward values unchanged;
        # backbone receives -alpha * adversary gradient;
        # adversary head receives ordinary +gradient.
        reversed_context = gradient_reverse(
            context_features,
            self.grl_alpha,
        )

        context_logit = self.context_adversary(
            reversed_context
        ).squeeze(1)

        if not return_aux:
            return classification_logit

        return {
            "logit": classification_logit,
            "context_logit": context_logit,
            "lung_features": lung_features,
            "context_features": context_features,
            "lung_map_feature": lung_map,
        }
