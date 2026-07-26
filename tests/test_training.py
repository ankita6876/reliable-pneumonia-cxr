"""Tests for baseline training engine reproducibility and development-only use."""

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.training import engine  # noqa: E402
from pneumonia_ai.training.engine import (  # noqa: E402
    UNCERTAIN_LABEL_STRATEGY,
    EpochMetrics,
    amp_is_enabled,
    calculate_pos_weight,
    count_raw_training_labels,
    create_development_loaders,
    run_training,
    train_one_epoch,
    validate_one_epoch,
)
from pneumonia_ai.training.seed import seed_everything  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
from scripts import train_baseline  # noqa: E402


class _ToyDataset(Dataset[dict[str, torch.Tensor]]):
    """Small in-memory dataset for one-step engine tests."""

    def __init__(self) -> None:
        self.images = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        self.labels = torch.tensor([0.0, 1.0])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": self.images[index], "label": self.labels[index]}


def test_seed_everything_is_deterministic() -> None:
    """The shared seed function controls Python, NumPy, and PyTorch randomness."""
    seed_everything(42)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_everything(42)
    second = (random.random(), np.random.rand(), torch.rand(1).item())

    assert first == second


def test_one_training_and_validation_step() -> None:
    """One batch supports forward, BCE loss, backward, optimizer, and metrics."""
    dataset = _ToyDataset()
    loader = DataLoader(dataset, batch_size=2)
    model = nn.Sequential(nn.Linear(2, 1))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model[0].weight.detach().clone()

    training_metrics = train_one_epoch(
        model, loader, criterion, optimizer, torch.device("cpu"), max_batches=1
    )
    validation_metrics = validate_one_epoch(
        model, loader, criterion, torch.device("cpu"), max_batches=1
    )

    assert training_metrics.loss > 0
    assert not torch.equal(before, model[0].weight)
    assert validation_metrics.loss > 0
    assert validation_metrics.auroc is not None
    assert validation_metrics.auprc is not None


def test_development_dataset_builder_never_requests_test_split(
    tmp_path: Path, monkeypatch
) -> None:
    """The pipeline constructs only train and validation datasets after ignore filtering."""
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "split": ["train", "validation", "test"],
            "pneumonia_label": [1, -1, 0],
            "image_path": ["one.jpg", "two.jpg", "three.jpg"],
        }
    ).to_csv(manifest_path, index=False)
    requested_splits: list[str] = []

    class FakeDataset:
        def __init__(self, root, manifest, split, transform) -> None:
            requested_splits.append(split)

    monkeypatch.setattr(engine, "CheXpertPneumoniaDataset", FakeDataset)

    engine.build_train_validation_datasets(
        tmp_path,
        manifest_path,
        tmp_path / "outputs",
        lambda image: image,
        lambda image: image,
    )

    assert requested_splits == ["train", "validation"]
    assert UNCERTAIN_LABEL_STRATEGY == "ignore"


def test_dry_run_disables_pretrained_weights(monkeypatch) -> None:
    """Dry runs never request pretrained weights, avoiding downloads."""
    requested: dict[str, object] = {}

    def fake_create_model(name: str, pretrained: bool) -> SimpleNamespace:
        requested.update(name=name, pretrained=pretrained)
        return SimpleNamespace()

    monkeypatch.setattr(train_baseline, "create_model", fake_create_model)

    train_baseline._create_model_from_config(
        {"name": "densenet121", "pretrained": True}, dry_run=True
    )

    assert requested == {"name": "densenet121", "pretrained": False}


def test_xrv_dry_run_disables_pretrained_weights(monkeypatch) -> None:
    """The domain-pretrained model also avoids weight downloads during dry runs."""
    requested: dict[str, object] = {}

    def fake_create_model(name: str, pretrained: bool) -> SimpleNamespace:
        requested.update(name=name, pretrained=pretrained)
        return SimpleNamespace()

    monkeypatch.setattr(train_baseline, "create_model", fake_create_model)

    train_baseline._create_model_from_config(
        {"name": "xrv_densenet121_all", "pretrained": True}, dry_run=True
    )

    assert requested == {"name": "xrv_densenet121_all", "pretrained": False}


def test_torchxrayvision_transform_is_single_channel_and_uses_xrv_range() -> None:
    """XRV preprocessing preserves grayscale and avoids ImageNet RGB normalization."""
    _, validation_transform = train_baseline._transforms(224, "torchxrayvision")

    transformed = validation_transform(Image.new("RGB", (300, 250), color=0))

    assert transformed.shape == (1, 224, 224)
    assert torch.all(transformed == -1024.0)


def test_transform_factory_selects_requested_preprocessing() -> None:
    """ImageNet and TorchXRayVision paths produce their model-specific channel counts."""
    _, imagenet_validation = train_baseline._transforms(224, "imagenet")
    _, xrv_validation = train_baseline._transforms(224, "torchxrayvision")
    image = Image.new("RGB", (224, 224), color=128)

    assert imagenet_validation(image).shape == (3, 224, 224)
    assert xrv_validation(image).shape == (1, 224, 224)


def test_amp_is_disabled_on_cpu() -> None:
    """AMP requests are safely disabled when the selected device is CPU."""
    assert amp_is_enabled(True, torch.device("cpu")) is False


def test_calculate_pos_weight_and_ablation_disable() -> None:
    """Positive-class weighting uses the post-strategy train label balance."""
    assert calculate_pos_weight([0, 0, 0, 1]) == 3.0
    assert calculate_pos_weight([0, 1], enabled=False) is None
    assert calculate_pos_weight([0.0, 0.5, 1.0]) == 1.0


def test_raw_training_counts_preserve_uncertain_rows_before_strategy(tmp_path: Path) -> None:
    """Run metadata distinguishes source uncertainty from the strategy's retained rows."""
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {"split": ["train", "train", "train", "validation"], "pneumonia_label": [0, 1, -1, -1]}
    ).to_csv(manifest_path, index=False)

    assert count_raw_training_labels(manifest_path) == (2, 1)


def test_weighted_unreduced_loss_applies_sample_weights_before_averaging() -> None:
    """The training reduction uses weighted per-sample BCE values, not batch mean BCE."""
    loss = engine._weighted_mean_loss(
        torch.tensor([2.0, 4.0]), torch.tensor([1.0, 0.5])
    )

    assert loss.item() == pytest.approx(4.0 / 1.5)


class _UncertainValidationDataset(Dataset[dict[str, torch.Tensor]]):
    """Definite labels plus a deliberately high-loss uncertain validation row."""

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image": torch.tensor([0.0 if index < 2 else 100.0]),
            "target": torch.tensor([0.0, 1.0, 0.5][index]),
            "label": torch.tensor([0.0, 1.0, 0.5][index]),
            "raw_label": torch.tensor([0.0, 1.0, -1.0][index]),
            "sample_weight": torch.tensor(1.0),
        }


def test_validation_filters_uncertain_labels_for_comparable_selection() -> None:
    """An uncertain validation row cannot affect validation loss, AUROC, or AUPRC."""
    loader = DataLoader(_UncertainValidationDataset(), batch_size=3)
    metrics = validate_one_epoch(
        nn.Identity(),
        loader,
        nn.BCEWithLogitsLoss(reduction="none"),
        torch.device("cpu"),
    )

    assert metrics.loss == pytest.approx(float(torch.log(torch.tensor(2.0))))
    assert metrics.auroc == pytest.approx(0.5)
    assert metrics.auprc == pytest.approx(0.5)


def test_strategy_builds_share_definite_validation_records(tmp_path: Path, monkeypatch) -> None:
    """All strategies train differently but select models on the same definite rows."""
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "split": ["train", "train", "train", "validation", "validation", "validation", "test"],
            "pneumonia_label": [0, 1, -1, 0, 1, -1, 1],
            "image_path": [f"image-{index}.jpg" for index in range(7)],
        }
    ).to_csv(manifest_path, index=False)
    captured: dict[str, list[pd.DataFrame]] = {}

    class FakeDataset:
        def __init__(self, root, manifest, split, transform) -> None:
            captured.setdefault(split, []).append(pd.read_csv(manifest))

    monkeypatch.setattr(engine, "CheXpertPneumoniaDataset", FakeDataset)
    validation_raw_labels: list[list[int]] = []
    training_lengths: list[int] = []
    for strategy in ("ignore", "u_zero", "u_one", "soft_uncertain"):
        engine.build_train_validation_datasets(
            tmp_path,
            manifest_path,
            tmp_path / "outputs",
            lambda image: image,
            lambda image: image,
            label_strategy=strategy,
        )
        training_records, validation_records = captured["train"].pop(), captured["validation"].pop()
        training_lengths.append(len(training_records.loc[training_records["split"] == "train"]))
        validation_raw_labels.append(
            validation_records.loc[validation_records["split"] == "validation", "raw_pneumonia_label"].tolist()
        )

    assert validation_raw_labels == [[0, 1]] * 4
    assert training_lengths == [2, 3, 3, 3]


def test_cpu_loaders_disable_pinning_and_persistent_workers() -> None:
    """CPU development loaders avoid CUDA-specific worker memory settings."""
    dataset = _ToyDataset()
    train_loader, validation_loader = create_development_loaders(
        dataset, dataset, batch_size=2, num_workers=0, seed=42
    )

    assert train_loader.pin_memory is False
    assert validation_loader.persistent_workers is False


def _training_arguments(tmp_path: Path) -> dict[str, object]:
    """Return compact CPU-only settings for trainer control-flow tests."""
    dataset = _ToyDataset()
    loader = DataLoader(dataset, batch_size=2)
    return {
        "model": nn.Linear(2, 1),
        "train_loader": loader,
        "validation_loader": loader,
        "output_dir": tmp_path,
        "epochs": 4,
        "learning_rate": 0.1,
        "weight_decay": 0.0,
        "early_stopping_patience": 2,
        "early_stopping_min_delta": 0.01,
        "scheduler_factor": 0.5,
        "scheduler_patience": 0,
        "scheduler_min_lr": 0.01,
        "pos_weight": None,
        "amp_requested": True,
        "configuration": {"model": "toy", "training": "test"},
        "device": torch.device("cpu"),
    }


def test_early_stopping_scheduler_and_best_checkpoint(monkeypatch, tmp_path: Path) -> None:
    """Plateau loss lowers LR and non-improving AUROC stops after configured patience."""
    validation_metrics = iter(
        [
            EpochMetrics(loss=1.0, auroc=0.8, auprc=0.7),
            EpochMetrics(loss=1.0, auroc=0.8, auprc=0.7),
            EpochMetrics(loss=1.0, auroc=0.8, auprc=0.7),
        ]
    )
    monkeypatch.setattr(engine, "train_one_epoch", lambda *args: EpochMetrics(loss=1.0))
    monkeypatch.setattr(engine, "validate_one_epoch", lambda *args: next(validation_metrics))

    history = run_training(**_training_arguments(tmp_path))

    checkpoint_paths = list(tmp_path.glob("*.pt"))
    best_checkpoint = torch.load(
        tmp_path / "best_validation_auroc.pt", map_location="cpu", weights_only=False
    )
    last_checkpoint = torch.load(
        tmp_path / "last_checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert len(history) == 3
    assert history.loc[1, "learning_rate"] == 0.05
    assert {path.name for path in checkpoint_paths} == {
        "best_validation_auroc.pt",
        "last_checkpoint.pt",
    }
    assert set(best_checkpoint) >= {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
        "best_validation_auroc",
        "configuration",
    }
    assert last_checkpoint["epoch"] == 3


def test_resume_restores_best_checkpoint_and_rejects_incompatible_config(
    monkeypatch, tmp_path: Path
) -> None:
    """Resume continues in place from checkpoint state and rejects configuration drift."""
    first_validation_metrics = iter(
        [EpochMetrics(loss=1.0, auroc=0.8, auprc=0.7)]
    )
    monkeypatch.setattr(engine, "train_one_epoch", lambda *args: EpochMetrics(loss=1.0))
    monkeypatch.setattr(
        engine, "validate_one_epoch", lambda *args: next(first_validation_metrics)
    )
    arguments = _training_arguments(tmp_path)
    arguments["epochs"] = 1
    run_training(**arguments)
    checkpoint_path = tmp_path / "best_validation_auroc.pt"

    resumed_validation_metrics = iter([EpochMetrics(loss=0.9, auroc=0.9, auprc=0.8)])
    monkeypatch.setattr(
        engine, "validate_one_epoch", lambda *args: next(resumed_validation_metrics)
    )
    arguments["epochs"] = 2
    history = run_training(**arguments, resume_checkpoint=checkpoint_path)

    assert history["epoch"].tolist() == [1, 2]
    incompatible = dict(arguments)
    incompatible["configuration"] = {"model": "other"}
    with pytest.raises(ValueError, match="incompatible"):
        run_training(**incompatible, resume_checkpoint=checkpoint_path)


def test_atomic_checkpoint_is_loadable_and_preserves_previous_checkpoint_on_failure(
    monkeypatch, tmp_path: Path
) -> None:
    """Atomic save leaves no temporary file and never destroys a valid prior checkpoint."""
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode="min")
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    checkpoint_path = tmp_path / "best_validation_auroc.pt"
    configuration = {"model": "toy"}

    engine._save_checkpoint_atomic(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=1,
        best_validation_auroc=0.8,
        configuration=configuration,
    )

    temporary_path = tmp_path / "best_validation_auroc.tmp.pt"
    saved_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint_path.is_file()
    assert not temporary_path.exists()
    assert saved_checkpoint["epoch"] == 1

    def failing_save(*args, **kwargs) -> None:
        raise RuntimeError("simulated checkpoint write failure")

    monkeypatch.setattr(engine.torch, "save", failing_save)
    with pytest.raises(RuntimeError, match="simulated checkpoint write failure"):
        engine._save_checkpoint_atomic(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=2,
            best_validation_auroc=0.9,
            configuration=configuration,
        )

    preserved_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert not temporary_path.exists()
    assert preserved_checkpoint["epoch"] == 1
