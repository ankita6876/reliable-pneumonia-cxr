"""Tests for baseline training engine reproducibility and development-only use."""

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.training import engine  # noqa: E402
from pneumonia_ai.training.engine import (  # noqa: E402
    UNCERTAIN_LABEL_STRATEGY,
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
