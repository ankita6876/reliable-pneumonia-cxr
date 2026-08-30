import torch
import pytest

from pneumonia_ai.models.factory import (
    SUPPORTED_MODEL_NAMES,
    create_model,
)
from scripts.classification.optimisation_config import OptimisationConfig
from scripts.classification.run_optimisation_experiment import (
    _is_classifier_parameter,
    configure_fine_tuning,
    differential_parameter_groups,
)


def test_resnet50_is_supported():
    assert "resnet50" in SUPPORTED_MODEL_NAMES


def test_resnet50_returns_one_logit():
    model = create_model("resnet50", pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 1)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("classifier.weight", True),
        ("classifier.bias", True),
        ("fc.weight", True),
        ("fc.bias", True),
        ("conv1.weight", False),
        ("layer1.0.conv1.weight", False),
    ],
)
def test_classifier_parameter_detection(name, expected):
    assert _is_classifier_parameter(name) is expected


@pytest.mark.parametrize(
    ("backbone", "expected_prefix"),
    [
        ("densenet121", "classifier."),
        ("efficientnet_b0", "classifier."),
        ("resnet50", "fc."),
    ],
)
def test_freezing_keeps_only_head_trainable(backbone, expected_prefix):
    model = create_model(backbone, pretrained=False)

    configure_fine_tuning(
        model,
        freeze_backbone=True,
    )

    trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    assert trainable
    assert all(
        name.startswith(expected_prefix)
        for name in trainable
    )


@pytest.mark.parametrize(
    "backbone",
    [
        "densenet121",
        "efficientnet_b0",
        "resnet50",
    ],
)
def test_differential_groups_separate_body_and_head(backbone):
    model = create_model(backbone, pretrained=False)

    config = OptimisationConfig(
        experiment="phase7_test",
        backbone=backbone,
        backbone_learning_rate=1e-5,
        head_learning_rate=1e-4,
    )

    configure_fine_tuning(
        model,
        freeze_backbone=False,
    )

    groups = differential_parameter_groups(
        model,
        config,
    )

    assert len(groups) == 2

    assert groups[0]["lr"] == pytest.approx(1e-5)
    assert groups[1]["lr"] == pytest.approx(1e-4)

    head_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if _is_classifier_parameter(name)
    }

    grouped_head_ids = {
        id(parameter)
        for parameter in groups[1]["params"]
    }

    assert grouped_head_ids == head_ids
