"""Factory for the predefined binary chest-X-ray classifier backbones."""

import timm
from torch import nn


SUPPORTED_MODEL_NAMES = (
    "densenet121",
    "convnext_tiny",
    "tf_efficientnetv2_s",
)


def create_model(name: str, pretrained: bool = True) -> nn.Module:
    """Create a predefined backbone with exactly one binary output logit.

    Args:
        name: One of :data:`SUPPORTED_MODEL_NAMES`.
        pretrained: Whether to initialize from timm pretrained weights.

    Raises:
        ValueError: If ``name`` is not a predefined study backbone.
    """
    if name not in SUPPORTED_MODEL_NAMES:
        supported = ", ".join(SUPPORTED_MODEL_NAMES)
        raise ValueError(f"Unsupported model name {name!r}. Supported models: {supported}.")
    return timm.create_model(name, pretrained=pretrained, num_classes=1)
