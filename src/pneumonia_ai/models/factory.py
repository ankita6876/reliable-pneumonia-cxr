"""Factory for the predefined binary chest-X-ray classifier backbones."""

import timm
from torch import nn


SUPPORTED_MODEL_NAMES = (
    "densenet121",
    "efficientnet_b0",
    "convnext_tiny",
    "tf_efficientnetv2_s",
    "xrv_densenet121_all",
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
    if name == "xrv_densenet121_all":
        return _create_xrv_densenet121_all(pretrained)
    return timm.create_model(name, pretrained=pretrained, num_classes=1)


def _create_xrv_densenet121_all(pretrained: bool) -> nn.Module:
    """Create an XRV DenseNet121 feature extractor with a new binary head."""
    try:
        import torchxrayvision as xrv
    except ImportError as error:
        raise ImportError(
            "torchxrayvision is required for xrv_densenet121_all. "
            "Install project dependencies before selecting this model."
        ) from error

    weights = "densenet121-res224-all" if pretrained else None
    model = xrv.models.DenseNet(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model.apply_sigmoid = False
    model.op_threshs = None
    return model
