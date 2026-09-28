"""Lung-Guided Context-Preserving Network (LGCP-Net).

The model keeps the complete chest radiograph as input and uses a frozen
lung-segmentation probability map as an anatomical prior at feature level.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class LGCPNet(nn.Module):
    """ResNet50 classifier with lung-guided and complementary-context features."""

    def __init__(
        self,
        pretrained: bool = True,
        projection_dim: int = 256,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        # Preserve the convolutional ResNet feature extractor but remove
        # global pooling and the original ImageNet classifier.
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        feature_dim = backbone.fc.in_features  # 2048 for ResNet50

        self.lung_projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.context_projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # A learned gate controls how much complementary contextual
        # information is allowed into the final representation.
        self.context_gate = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
            nn.Sigmoid(),
        )

        self.classifier = nn.Linear(projection_dim, 1)

    @staticmethod
    def _masked_pool(features: Tensor, mask: Tensor) -> Tensor:
        """Probability-weighted spatial pooling."""

        numerator = (features * mask).sum(dim=(2, 3))
        denominator = mask.sum(dim=(2, 3)).clamp_min(1e-6)
        return numerator / denominator

    def forward(
        self,
        image: Tensor,
        lung_map: Tensor,
        *,
        return_aux: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        """Predict pneumonia from full CXR plus a lung probability map."""

        features = self.backbone(image)

        if lung_map.ndim == 3:
            lung_map = lung_map.unsqueeze(1)

        if lung_map.ndim != 4 or lung_map.shape[1] != 1:
            raise ValueError("lung_map must have shape Bx1xHxW or BxHxW.")

        lung_map = lung_map.to(device=features.device, dtype=features.dtype)
        lung_map = F.interpolate(
            lung_map,
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)

        context_map = 1.0 - lung_map

        lung_features = self._masked_pool(features, lung_map)
        context_features = self._masked_pool(features, context_map)

        lung_embedding = self.lung_projection(lung_features)
        context_embedding = self.context_projection(context_features)

        gate = self.context_gate(
            torch.cat([lung_embedding, context_embedding], dim=1)
        )

        # Lung evidence forms the primary representation.
        # Context can complement it, but only through the learned gate.
        fused = lung_embedding + gate * context_embedding

        logit = self.classifier(fused).squeeze(1)

        if not return_aux:
            return logit

        return {
            "logit": logit,
            "lung_embedding": lung_embedding,
            "context_embedding": context_embedding,
            "context_gate": gate,
            "lung_map_feature": lung_map,
        }
