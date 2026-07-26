from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest
import torch

from pneumonia_ai.segmentation.inference import FrozenLungSegmenter
from pneumonia_ai.segmentation.model import UNet


def _checkpoint(path: Path) -> Path:
    model = UNet(base_channels=2, depth=1)
    torch.save({"model_config": {"in_channels": 1, "out_channels": 1, "base_channels": 2, "depth": 1}, "model_state": model.state_dict(), "training_config": {"image_size": 8}}, path)
    return path


def test_frozen_segmenter_reconstructs_freezes_and_predicts(tmp_path: Path) -> None:
    segmenter = FrozenLungSegmenter(_checkpoint(tmp_path / "tiny.pt"))
    image = Image.new("L", (11, 7), color=120)

    first = segmenter.predict_proba(image)
    second = segmenter.predict_proba(image)

    assert first.shape == (7, 11)
    assert torch.all((first >= 0) & (first <= 1))
    assert torch.equal(first, second)
    assert not segmenter.model.training
    assert all(not parameter.requires_grad for parameter in segmenter.model.parameters())
    assert segmenter.predict(image, threshold=0.5).dtype is torch.bool


def test_frozen_segmenter_rejects_malformed_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({}, path)
    with pytest.raises(ValueError, match="model_config"):
        FrozenLungSegmenter(path)
