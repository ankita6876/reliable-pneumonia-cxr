"""DenseNet model construction for binary pneumonia classification."""

import timm
from torch import nn


def create_densenet121(pretrained: bool = True) -> nn.Module:
    """Create a DenseNet121 with one binary-classification output logit."""
    return timm.create_model("densenet121", pretrained=pretrained, num_classes=1)
