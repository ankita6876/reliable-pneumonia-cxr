"""LGCP-v2: Lung-Guided Residual Context-Preserving Network.

The full-image representation is preserved as the primary pathway.
Frozen lung probability maps provide anatomical guidance through
residual lung and contextual feature pathways rather than replacing
the global representation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class LGCPv2Net(nn.Module):
    """ResNet50 with global, lung-guided, and controlled context pathways."""

    def __init__(
        self,
        pretrained: bool = True,
        projection_dim: int = 256,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        weights = (
            ResNet50_Weights.IMAGENET1K_V2
            if pretrained
            else None
        )

        backbone = resnet50(weights=weights)

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

        feature_dim = backbone.fc.in_features  # 2048

        # Primary pathway: preserve the complete CXR representation.
        self.global_projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Anatomically guided residual pathway.
        self.lung_projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Complementary context remains available but controlled.
        self.context_projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Learn how strongly anatomy should modify the global stream.
        self.lung_gate = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
            nn.Sigmoid(),
        )

        # Context gate sees global + lung + context information.
        self.context_gate = nn.Sequential(
            nn.Linear(projection_dim * 3, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
            nn.Sigmoid(),
        )

        self.fusion_norm = nn.LayerNorm(projection_dim)

        self.classifier = nn.Linear(
            projection_dim,
            1,
        )

    @staticmethod
    def _masked_pool(
        features: Tensor,
        mask: Tensor,
    ) -> Tensor:
        numerator = (
            features * mask
        ).sum(dim=(2, 3))

        denominator = (
            mask.sum(dim=(2, 3))
            .clamp_min(1e-6)
        )

        return numerator / denominator

    def forward(
        self,
        image: Tensor,
        lung_map: Tensor,
        *,
        return_aux: bool = False,
    ) -> Tensor | dict[str, Tensor]:

        features = self.backbone(image)

        if lung_map.ndim == 3:
            lung_map = lung_map.unsqueeze(1)

        if (
            lung_map.ndim != 4
            or lung_map.shape[1] != 1
        ):
            raise ValueError(
                "lung_map must have shape "
                "Bx1xHxW or BxHxW."
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

        # --------------------------------------------------
        # 1. Preserve complete full-image evidence.
        # --------------------------------------------------
        global_features = F.adaptive_avg_pool2d(
            features,
            output_size=1,
        ).flatten(1)

        # --------------------------------------------------
        # 2. Extract anatomical and contextual evidence.
        # --------------------------------------------------
        lung_features = self._masked_pool(
            features,
            lung_map,
        )

        context_features = self._masked_pool(
            features,
            context_map,
        )

        global_embedding = self.global_projection(
            global_features
        )

        lung_embedding = self.lung_projection(
            lung_features
        )

        context_embedding = self.context_projection(
            context_features
        )

        # --------------------------------------------------
        # 3. Residual anatomical guidance.
        # --------------------------------------------------
        lung_gate = self.lung_gate(
            torch.cat(
                [
                    global_embedding,
                    lung_embedding,
                ],
                dim=1,
            )
        )

        lung_residual = (
            lung_gate * lung_embedding
        )

        # --------------------------------------------------
        # 4. Controlled complementary context.
        # --------------------------------------------------
        context_gate = self.context_gate(
            torch.cat(
                [
                    global_embedding,
                    lung_embedding,
                    context_embedding,
                ],
                dim=1,
            )
        )

        context_residual = (
            context_gate * context_embedding
        )

        # Full-image pathway is never removed.
        fused = (
            global_embedding
            + lung_residual
            + context_residual
        )

        fused = self.fusion_norm(fused)

        logit = self.classifier(
            fused
        ).squeeze(1)

        if not return_aux:
            return logit

        return {
            "logit": logit,
            "global_embedding": global_embedding,
            "lung_embedding": lung_embedding,
            "context_embedding": context_embedding,
            "lung_gate": lung_gate,
            "context_gate": context_gate,
            "lung_residual": lung_residual,
            "context_residual": context_residual,
            "lung_map_feature": lung_map,
        }
