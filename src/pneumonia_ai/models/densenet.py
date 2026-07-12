"""Backward-compatible DenseNet121 model construction."""

from torch import nn

from pneumonia_ai.models.factory import create_model


def create_densenet121(pretrained: bool = True) -> nn.Module:
    """Create a DenseNet121 with one binary-classification output logit."""
    return create_model("densenet121", pretrained=pretrained)
