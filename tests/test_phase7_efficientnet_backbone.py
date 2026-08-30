import torch

from pneumonia_ai.models.factory import (
    SUPPORTED_MODEL_NAMES,
    create_model,
)


def test_efficientnet_b0_is_supported():
    assert "efficientnet_b0" in SUPPORTED_MODEL_NAMES


def test_efficientnet_b0_returns_one_logit_per_image():
    model = create_model(
        "efficientnet_b0",
        pretrained=False,
    )
    model.eval()

    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 1)


def test_efficientnet_b0_classifier_parameter_contract():
    model = create_model(
        "efficientnet_b0",
        pretrained=False,
    )

    classifier_names = [
        name
        for name, _ in model.named_parameters()
        if name.startswith("classifier")
    ]

    assert classifier_names == [
        "classifier.weight",
        "classifier.bias",
    ]
