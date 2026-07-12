"""Tests for predefined binary classifier construction."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.models.factory import (  # noqa: E402
    SUPPORTED_MODEL_NAMES,
    create_model,
)


@pytest.mark.parametrize("model_name", SUPPORTED_MODEL_NAMES)
def test_predefined_models_return_one_logit_per_image(model_name: str) -> None:
    """Every predefined model emits one binary-classification logit per image."""
    model = create_model(model_name, pretrained=False)
    model.eval()
    channels = 1 if model_name == "xrv_densenet121_all" else 3

    with torch.inference_mode():
        output = model(torch.randn(2, channels, 224, 224))

    assert output.shape == (2, 1)


def test_factory_rejects_unsupported_model_name() -> None:
    """Only protocol-defined backbones can be constructed."""
    with pytest.raises(ValueError, match="Unsupported model name 'resnet50'"):
        create_model("resnet50", pretrained=False)


def test_xrv_factory_does_not_request_weights_when_pretrained_is_disabled(monkeypatch) -> None:
    """The XRV factory passes ``None`` rather than a downloadable weight identifier."""
    requested_weights: list[str | None] = []

    class FakeDenseNet(torch.nn.Module):
        def __init__(self, weights=None) -> None:
            super().__init__()
            requested_weights.append(weights)
            self.classifier = torch.nn.Linear(4, 18)
            self.apply_sigmoid = True
            self.op_threshs = torch.ones(18)

    fake_xrv = SimpleNamespace(models=SimpleNamespace(DenseNet=FakeDenseNet))
    monkeypatch.setitem(sys.modules, "torchxrayvision", fake_xrv)

    model = create_model("xrv_densenet121_all", pretrained=False)

    assert requested_weights == [None]
    assert model.classifier.out_features == 1
    assert model.op_threshs is None
