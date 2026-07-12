"""Tests for privacy-safe experiment metadata."""

import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.training.experiment import (  # noqa: E402
    create_run_directory,
    write_resolved_config,
    write_run_manifest,
)


def test_run_manifest_records_metadata_without_dataset_paths(tmp_path: Path) -> None:
    """Manifest records reproducibility data while excluding absolute input paths."""
    run_directory, run_id, timestamp = create_run_directory(tmp_path, "densenet121")
    dataset_root = tmp_path / "private-dataset"
    split_manifest = dataset_root / "splits" / "patients.csv"
    metadata = write_run_manifest(
        run_directory,
        run_id=run_id,
        timestamp=timestamp,
        model_name="densenet121",
        pretrained=False,
        uncertain_label_strategy="ignore",
        seed=42,
        image_size=224,
        batch_size=2,
        learning_rate=0.0001,
        epoch_count=1,
        device=torch.device("cpu"),
        split_manifest_path=split_manifest,
        train_sample_count=2,
        validation_sample_count=1,
        dry_run=True,
        amp_enabled=False,
        pos_weight=1.5,
    )
    saved_metadata = json.loads((run_directory / "run_manifest.json").read_text())

    assert run_directory.name == f"densenet121_{timestamp}_{run_id}"
    assert saved_metadata == metadata
    assert saved_metadata["split_manifest_filename"] == "patients.csv"
    assert saved_metadata["dry_run"] is True
    assert saved_metadata["amp_enabled"] is False
    assert saved_metadata["pos_weight"] == 1.5
    assert str(dataset_root) not in (run_directory / "run_manifest.json").read_text()
    assert "dataset_root" not in saved_metadata


def test_resolved_dry_run_config_records_effective_settings(tmp_path: Path) -> None:
    """Dry-run snapshots show that downloads and full training are disabled."""
    config = {
        "model": {"name": "densenet121", "pretrained": True},
        "training": {"epochs": 10, "batch_size": 16},
    }
    write_resolved_config(tmp_path, config, dry_run=True)
    saved_config = yaml.safe_load((tmp_path / "config.yaml").read_text())

    assert saved_config["dry_run"] is True
    assert saved_config["model"]["pretrained"] is False
    assert saved_config["training"] == {"epochs": 1, "batch_size": 2}
    assert config["model"]["pretrained"] is True


def test_normal_run_manifest_marks_dry_run_false(tmp_path: Path) -> None:
    """Normal-run metadata explicitly distinguishes it from a dry run."""
    run_directory, run_id, timestamp = create_run_directory(tmp_path, "convnext_tiny")
    metadata = write_run_manifest(
        run_directory,
        run_id=run_id,
        timestamp=timestamp,
        model_name="convnext_tiny",
        pretrained=True,
        uncertain_label_strategy="ignore",
        seed=42,
        image_size=224,
        batch_size=16,
        learning_rate=0.0001,
        epoch_count=10,
        device=torch.device("cpu"),
        split_manifest_path=tmp_path / "manifest.csv",
        train_sample_count=10,
        validation_sample_count=5,
        dry_run=False,
        amp_enabled=False,
        pos_weight=None,
    )

    assert metadata["dry_run"] is False


def test_resolved_config_redacts_absolute_paths(tmp_path: Path) -> None:
    """Configuration snapshots do not persist user-specific absolute paths."""
    config = {
        "output_dir": str(tmp_path / "outputs"),
        "model": {"name": "densenet121", "pretrained": False},
        "training": {"epochs": 1, "batch_size": 2},
    }
    write_resolved_config(tmp_path, config, dry_run=False)

    saved_config = (tmp_path / "config.yaml").read_text()
    assert str(tmp_path) not in saved_config
    assert "<redacted-absolute-path>" in saved_config
