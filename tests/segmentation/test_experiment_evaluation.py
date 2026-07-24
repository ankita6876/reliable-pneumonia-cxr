from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

import pneumonia_ai.segmentation.experiment as experiment
from pneumonia_ai.segmentation.evaluation import evaluate_checkpoint
from pneumonia_ai.segmentation.experiment import ExperimentConfig, run_experiment
from pneumonia_ai.segmentation.model import UNet
from pneumonia_ai.segmentation.training import EpochSummary


def _write_split(tmp_path: Path, name: str, sample_count: int = 2) -> Path:
    records = []
    for index in range(sample_count):
        image = np.zeros((16, 16), dtype=np.uint8)
        image[3:10, 4:11] = 255
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[4:9, 5:10] = 255
        image_path = tmp_path / "images" / f"{name}-{index}.png"
        mask_path = tmp_path / "masks" / f"{name}-{index}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(image_path), image)
        assert cv2.imwrite(str(mask_path), mask)
        records.append(
            {"image_id": f"{name}-{index}", "image_path": image_path, "merged_mask_path": mask_path}
        )
    split_path = tmp_path / f"{name}.csv"
    pd.DataFrame(records).to_csv(split_path, index=False)
    return split_path


def _config(tmp_path: Path, **overrides: object) -> ExperimentConfig:
    values: dict[str, object] = {
        "train_csv": _write_split(tmp_path, "train"),
        "validation_csv": _write_split(tmp_path, "validation"),
        "output_directory": tmp_path / "run",
        "image_size": 16,
        "batch_size": 1,
        "epochs": 3,
        "base_channels": 2,
        "depth": 1,
        "patience": 1,
    }
    values.update(overrides)
    return ExperimentConfig(**values)  # type: ignore[arg-type]


def _summary(dice: float) -> EpochSummary:
    return EpochSummary(0.5, dice, dice, dice, dice, dice, 2)


def test_experiment_selects_best_checkpoint_stops_early_and_writes_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation_scores = iter([0.8, 0.7])

    def train_summary_after_optimizer_step(*args: object, **kwargs: object) -> EpochSummary:
        optimizer = args[2]
        assert isinstance(optimizer, torch.optim.Optimizer)
        optimizer.step()
        return _summary(0.4)

    monkeypatch.setattr(experiment, "train_one_epoch", train_summary_after_optimizer_step)
    monkeypatch.setattr(
        experiment, "validate_one_epoch", lambda *args, **kwargs: _summary(next(validation_scores))
    )
    config = _config(tmp_path)

    history = run_experiment(config)

    assert len(history) == 2
    assert (config.output_directory / "best_checkpoint.pt").is_file()
    assert (config.output_directory / "latest_checkpoint.pt").is_file()
    checkpoint = torch.load(config.output_directory / "best_checkpoint.pt", weights_only=False)
    assert checkpoint["best_validation_dice"] == 0.8
    assert len(pd.read_csv(config.output_directory / "history.csv")) == 2
    assert json.loads((config.output_directory / "resolved_config.json").read_text()) ["image_size"] == 16
    manifest = json.loads((config.output_directory / "run_manifest.json").read_text())
    assert str(tmp_path) not in json.dumps(manifest)
    assert manifest["train_samples"] == manifest["validation_samples"] == 2


def test_atomic_checkpoint_failure_preserves_existing_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"previous")
    monkeypatch.setattr(torch, "save", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("write failed")))

    with pytest.raises(OSError, match="write failed"):
        experiment._atomic_torch_save({"value": 1}, checkpoint_path)

    assert checkpoint_path.read_bytes() == b"previous"


def test_resume_rejects_incompatible_model_configuration(tmp_path: Path) -> None:
    checkpoint = {
        "model_state": {},
        "optimizer_state": {},
        "epoch": 1,
        "best_validation_dice": 0.5,
        "model_config": {"in_channels": 1, "out_channels": 1, "base_channels": 2, "depth": 1},
        "seed": 42,
    }

    with pytest.raises(ValueError, match="incompatible"):
        experiment._validate_checkpoint(
            checkpoint,
            {"in_channels": 1, "out_channels": 1, "base_channels": 4, "depth": 1},
            42,
        )


def test_tiny_cpu_experiment_and_evaluation_outputs(tmp_path: Path) -> None:
    config = _config(tmp_path, epochs=1, patience=1)

    run_experiment(config)
    metrics = evaluate_checkpoint(
        config.output_directory / "best_checkpoint.pt",
        config.validation_csv,
        tmp_path / "evaluation",
        batch_size=1,
        qualitative_count=2,
    )

    assert metrics["samples"] == 2
    per_sample = pd.read_csv(tmp_path / "evaluation" / "per_sample_metrics.csv")
    assert per_sample["image_id"].tolist() == ["validation-0", "validation-1"]
    assert (tmp_path / "evaluation" / "aggregate_metrics.json").is_file()
    masks = sorted((tmp_path / "evaluation" / "prediction_masks").glob("*.png"))
    assert len(masks) == 2
    for path in masks:
        values = set(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).ravel().tolist())
        assert values <= {0, 255}
    assert len(list((tmp_path / "evaluation" / "qualitative").glob("*.png"))) == 2


def test_training_does_not_accept_or_access_a_test_split(tmp_path: Path) -> None:
    config = _config(tmp_path, epochs=1)

    assert "test" not in ExperimentConfig.__dataclass_fields__
    assert not (tmp_path / "test.csv").exists()
    run_experiment(config)


def test_evaluation_rejects_incompatible_checkpoint_state(tmp_path: Path) -> None:
    split = _write_split(tmp_path, "evaluation")
    checkpoint_path = tmp_path / "checkpoint.pt"
    model = UNet(base_channels=2, depth=1)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": {},
            "epoch": 1,
            "best_validation_dice": 0.0,
            "model_config": {"in_channels": 1, "out_channels": 1, "base_channels": 2, "depth": 1},
            "training_config": {"image_size": 16},
            "seed": 42,
        },
        checkpoint_path,
    )

    result = evaluate_checkpoint(checkpoint_path, split, tmp_path / "outputs", qualitative_count=0)

    assert result["samples"] == 2
