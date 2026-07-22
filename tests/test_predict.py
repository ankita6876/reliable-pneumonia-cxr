"""Tests for portable prediction checkpoint configuration resolution."""

from pathlib import Path
import sys

import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.predict import (  # noqa: E402
    _prepare_prediction_manifest,
    _resolve_checkpoint_configuration,
)


def _configuration() -> dict[str, object]:
    return {
        "model": {"name": "densenet121", "preprocessing": "imagenet"},
        "training": {"image_size": 224, "batch_size": 16},
        "label_strategy": {"name": "ignore"},
    }


def test_resolves_embedded_checkpoint_configuration(tmp_path: Path) -> None:
    """New checkpoints use their embedded configuration without filesystem lookup."""
    configuration = _configuration()
    state = {"model_state_dict": {}, "configuration": configuration}

    assert _resolve_checkpoint_configuration(state, tmp_path / "checkpoint.pt") is configuration


def test_resolves_legacy_checkpoint_from_run_configuration(tmp_path: Path) -> None:
    """Original baseline checkpoints reuse the immutable config.yaml beside them."""
    configuration = _configuration()
    checkpoint = tmp_path / "best_validation_auroc.pt"
    (tmp_path / "config.yaml").write_text(
        "model:\n  name: densenet121\n  preprocessing: imagenet\n"
        "training:\n  image_size: 224\n  batch_size: 16\n"
        "label_strategy:\n  name: ignore\n",
        encoding="utf-8",
    )
    state = {"epoch": 3, "model_state_dict": {}, "auroc": 0.8}

    assert _resolve_checkpoint_configuration(state, checkpoint) == configuration


def test_legacy_checkpoint_without_canonical_configuration_fails(tmp_path: Path) -> None:
    """Legacy weight-only checkpoints must not silently guess a configuration."""
    state = {"epoch": 3, "model_state_dict": {}, "auroc": 0.8}

    with pytest.raises(ValueError, match="canonical run configuration is missing"):
        _resolve_checkpoint_configuration(state, tmp_path / "best_validation_auroc.pt")


def test_prediction_manifest_uses_packaged_test_images_only() -> None:
    """Packaged test images replace source paths without admitting validation rows."""
    manifest = pd.DataFrame(
        {
            "split": ["test", "validation"],
            "image_path": ["train/source-test.jpg", "train/source-validation.jpg"],
            "packaged_image_path": ["images/train/source-test.jpg", "images/train/source-validation.jpg"],
        }
    )

    selected = _prepare_prediction_manifest(manifest, "test")

    assert selected["split"].tolist() == ["test"]
    assert selected["image_path"].tolist() == ["images/train/source-test.jpg"]
