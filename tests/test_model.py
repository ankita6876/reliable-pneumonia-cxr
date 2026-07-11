"""Tests for DenseNet121 binary output construction."""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.models.densenet import create_densenet121  # noqa: E402


def test_densenet121_returns_one_logit_per_image() -> None:
    """DenseNet121 emits one binary-classification logit for each image."""
    model = create_densenet121(pretrained=False)

    output = model(torch.randn(2, 3, 224, 224))

    assert output.shape == (2, 1)
