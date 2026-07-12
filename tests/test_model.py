"""Tests for predefined binary classifier construction."""

import sys
from pathlib import Path

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

    with torch.inference_mode():
        output = model(torch.randn(2, 3, 224, 224))

    assert output.shape == (2, 1)


def test_factory_rejects_unsupported_model_name() -> None:
    """Only protocol-defined backbones can be constructed."""
    with pytest.raises(ValueError, match="Unsupported model name 'resnet50'"):
        create_model("resnet50", pretrained=False)
