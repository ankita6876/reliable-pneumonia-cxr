from pathlib import Path
import json
import sys

import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image
from torchvision import transforms

from scripts.classification.optimisation_config import (OptimisationConfig, configuration_differences, load_config, validate_controlled_a_series)
from scripts.classification.optimisation_losses import AsymmetricFocalLoss, FocalLoss
from scripts.classification.optimisation_thresholds import threshold_analysis
from scripts.classification.run_optimisation_experiment import (
    _prepare_run_directory,
    build_transforms,
    configure_fine_tuning,
    differential_parameter_groups,
    run_experiment,
)
from scripts.train_baseline import _transforms as build_classifier_transforms
from scripts.classification.build_optimisation_leaderboard import build_leaderboard
from pneumonia_ai.segmentation.cache import MaskCache


def test_pretrained_configuration_propagates(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("experiment: run\npretrained: true\n")
    assert load_config(path).pretrained is True


def test_a_series_has_only_the_intended_resolved_differences() -> None:
    root = Path(__file__).resolve().parents[1] / "configs" / "classification_optimisation"
    configs = [load_config(root / name) for name in (
        "A0_reproduce_current.yaml", "A1_pretrained.yaml", "A2_pretrained_longer.yaml", "A3_pretrained_light_aug.yaml", "A4_pretrained_progressive.yaml", "A5_pretrained_progressive_focal.yaml",
    )]
    resolved = [config.to_dict() for config in configs]
    assert configuration_differences(configs[0], configs[1]) == {"pretrained"}
    assert configuration_differences(configs[1], configs[2]) == {"epochs", "early_stopping_patience"}
    assert configuration_differences(configs[2], configs[3]) == {"horizontal_flip"}
    assert configuration_differences(configs[3], configs[4]) == {"learning_rate", "backbone_learning_rate", "head_learning_rate", "weight_decay"}
    assert configuration_differences(configs[4], configs[5]) == {"loss"}
    validate_controlled_a_series(root)


def test_a4_original_control_differs_only_by_input_mode() -> None:
    root = Path(__file__).resolve().parents[1] / "configs" / "classification_optimisation"
    hard, original = load_config(root / "A4_pretrained_progressive.yaml"), load_config(root / "A4_original_control.yaml")
    assert configuration_differences(hard, original) == {"input_mode"}
    assert {
        key for key in hard.to_dict() if hard.to_dict()[key] != original.to_dict()[key]
    } == {"experiment", "input_mode"}


def test_a4_soft_masked_preserves_a4_controls_and_records_frozen_factor() -> None:
    root = Path(__file__).resolve().parents[1] / "configs" / "classification_optimisation"
    hard = load_config(root / "A4_pretrained_progressive.yaml")
    soft = load_config(root / "A4_soft_masked.yaml")
    assert soft.input_mode == "soft_masked" and soft.mask_threshold == .5 and soft.soft_mask_outside_factor == .20
    assert configuration_differences(hard, soft) == {"input_mode"}


def test_a4_original_control_transform_is_identical_to_a4() -> None:
    root = Path(__file__).resolve().parents[1] / "configs" / "classification_optimisation"
    hard, original = load_config(root / "A4_pretrained_progressive.yaml"), load_config(root / "A4_original_control.yaml")
    image = Image.new("RGB", (240, 200), color=128)
    hard_train, hard_validation = build_transforms(hard); original_train, original_validation = build_transforms(original)
    assert [type(x) for x in hard_train.transforms] == [type(x) for x in original_train.transforms]
    torch.testing.assert_close(hard_validation(image), original_validation(image), rtol=0, atol=0)


def test_augmentation_configuration_controls_horizontal_flip() -> None:
    historical, validation = build_transforms(OptimisationConfig(experiment="a"))
    no_flip, _ = build_transforms(OptimisationConfig(experiment="b", horizontal_flip=False))
    assert any(isinstance(item, transforms.RandomHorizontalFlip) for item in historical.transforms)
    assert not any(isinstance(item, transforms.RandomHorizontalFlip) for item in no_flip.transforms)
    assert any(isinstance(item, transforms.RandomRotation) for item in historical.transforms)
    assert any(isinstance(item, transforms.RandomRotation) for item in no_flip.transforms)
    assert all(not isinstance(item, transforms.RandomHorizontalFlip) for item in validation.transforms)
    image = Image.new("RGB", (240, 200), color=128)
    torch.testing.assert_close(validation(image), validation(image), rtol=0, atol=0)


def test_augmentation_none_has_no_random_training_transform() -> None:
    train, validation = build_transforms(
        OptimisationConfig(experiment="none", augmentation="none", horizontal_flip=False, rotation_degrees=0)
    )
    image = Image.new("RGB", (240, 200), color=(5, 110, 205))
    assert not any(isinstance(item, (transforms.RandomHorizontalFlip, transforms.RandomRotation)) for item in train.transforms)
    torch.testing.assert_close(train(image), validation(image), rtol=0, atol=0)


def test_mask_cache_rejects_incompatible_metadata(tmp_path: Path) -> None:
    cache = MaskCache(tmp_path / "cache")
    metadata = {"schema_version": 1, "segmentation_checkpoint_sha256": "first"}
    cache.validate_or_initialise_metadata(metadata)
    cache.validate_or_initialise_metadata(metadata)
    with pytest.raises(ValueError, match="incompatible"):
        cache.validate_or_initialise_metadata({**metadata, "segmentation_checkpoint_sha256": "other"})


def test_unsupported_augmentation_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("experiment: invalid\naugmentation: light\n", encoding="utf-8")
    with pytest.raises(ValueError, match="none or historical"):
        load_config(path)


@pytest.mark.parametrize("factor", ("-0.01", "1.01"))
def test_optimisation_soft_factor_validation_is_enforced(tmp_path: Path, factor: str) -> None:
    path = tmp_path / "invalid-soft-mask.yaml"
    path.write_text(f"experiment: invalid\ninput_mode: soft_masked\nsoft_mask_outside_factor: {factor}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="soft_mask_outside_factor"):
        load_config(path)


@pytest.mark.parametrize(
    ("backbone", "preprocessing", "expected_channels"),
    [
        ("densenet121", "imagenet", 3),
        ("xrv_densenet121_all", "torchxrayvision", 1),
    ],
)
def test_optimisation_validation_transform_matches_classifier_pipeline(
    backbone: str, preprocessing: str, expected_channels: int
) -> None:
    """Abort on any shape, dtype, channel, normalization, or value divergence."""
    config = OptimisationConfig(
        experiment="pipeline-equivalence",
        backbone=backbone,
        preprocessing=preprocessing,
    )
    _, original_validation = build_classifier_transforms(224, preprocessing)
    _, optimisation_validation = build_transforms(config)
    image = Image.new("RGB", (311, 197), color=(32, 128, 224))

    original = original_validation(image)
    optimisation = optimisation_validation(image)

    assert optimisation.shape == original.shape == (expected_channels, 224, 224)
    assert optimisation.dtype == original.dtype == torch.float32
    assert optimisation.shape[0] == expected_channels
    torch.testing.assert_close(optimisation, original, rtol=0, atol=0)


def test_focal_losses_prefer_correct_logits() -> None:
    target = torch.tensor([1.0, 0.0])
    assert FocalLoss()(torch.tensor([3.0, -3.0]), target) < FocalLoss()(
        torch.tensor([-3.0, 3.0]), target
    )
    assert AsymmetricFocalLoss()(
        torch.tensor([3.0, -3.0]), target
    ) < AsymmetricFocalLoss()(torch.tensor([-3.0, 3.0]), target)


def test_threshold_selection_uses_candidates_and_target_sensitivity() -> None:
    threshold, table = threshold_analysis(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.4, 0.6, 0.9]),
        "target_sensitivity",
        0.8,
    )
    assert table.selected.sum() == 1
    assert table.loc[table.selected, "sensitivity"].item() >= 0.8
    assert 0 <= threshold <= 1


def test_progressive_freezing_and_differential_groups() -> None:
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2))
    model.classifier = nn.Linear(2, 1)  # type: ignore[attr-defined]
    configure_fine_tuning(model, True)
    assert model[0].training is False and not model[1].weight.requires_grad
    groups = differential_parameter_groups(model, OptimisationConfig(experiment="x"))
    assert len(groups) == 1 and groups[0]["lr"] == 1e-4
    configure_fine_tuning(model, False)
    assert model[1].weight.requires_grad


def test_development_runner_never_accepts_a_test_only_manifest(tmp_path: Path) -> None:
    config = OptimisationConfig(experiment="blocked")
    manifest = tmp_path / "splits.csv"
    manifest.write_text(
        "split,patient_id,study_id,image_path,pneumonia_label\n"
        "test,p,s,i.png,1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="train and validation"):
        run_experiment(config, manifest, tmp_path, output_root=tmp_path / "out")


def test_completed_run_overwrite_protection(tmp_path: Path) -> None:
    output = tmp_path / "out" / "done"
    output.mkdir(parents=True)
    (output / "run_summary.json").write_text('{"status": "completed"}')
    with pytest.raises(FileExistsError):
        _prepare_run_directory(output, resume=False, restart=False, force=False)


def test_missing_manifest_does_not_create_output_directory(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    with pytest.raises(FileNotFoundError, match="Split manifest"):
        run_experiment(
            OptimisationConfig(experiment="missing_manifest"),
            tmp_path / "missing.csv",
            tmp_path,
            output_root=output_root,
        )
    assert not (output_root / "missing_manifest").exists()


def test_missing_image_root_does_not_create_output_directory(tmp_path: Path) -> None:
    manifest = tmp_path / "splits.csv"
    manifest.write_text(
        "split,patient_id,study_id,image_path,pneumonia_label\n"
        "test,p,s,input.png,1\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "out"
    with pytest.raises(NotADirectoryError, match="Image root"):
        run_experiment(
            OptimisationConfig(experiment="missing_image_root"),
            manifest,
            tmp_path / "missing-images",
            output_root=output_root,
        )
    assert not (output_root / "missing_image_root").exists()


def test_preflight_accepts_complete_existing_mask_cache_without_segmentation_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.classification.run_optimisation_experiment as runner

    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (12, 12), color=128).save(image_root / "train.png")
    Image.new("RGB", (12, 12), color=64).save(image_root / "validation.png")
    manifest = tmp_path / "splits.csv"
    manifest.write_text(
        "split,patient_id,study_id,image_path,pneumonia_label\n"
        "train,p1,s1,train.png,1\n"
        "validation,p2,s2,validation.png,0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "create_model", lambda *args, **kwargs: nn.Identity())
    cache = MaskCache(tmp_path / "mask_cache")
    for name in ("train.png", "validation.png"):
        source = image_root / name
        cache.set(name, torch.ones((12, 12)), source_path=source)

    runner._preflight(
        OptimisationConfig(experiment="prepared_inputs"),
        manifest,
        image_root,
        None,
        cache.directory,
        "cpu",
    )
    train_set, _ = runner._development_datasets(
        OptimisationConfig(experiment="prepared_inputs"),
        manifest,
        image_root,
        tmp_path,
        None,
        cache,
    )
    assert train_set[0]["image"].shape == (3, 224, 224)


def test_preflight_rejects_incomplete_mask_cache(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (12, 12)).save(image_root / "train.png")
    Image.new("RGB", (12, 12)).save(image_root / "validation.png")
    manifest = tmp_path / "splits.csv"
    manifest.write_text(
        "split,patient_id,study_id,image_path,pneumonia_label\n"
        "train,p1,s1,train.png,1\n"
        "validation,p2,s2,validation.png,0\n",
        encoding="utf-8",
    )
    cache = MaskCache(tmp_path / "mask_cache")
    cache.set("train", torch.ones((12, 12)), source_path=image_root / "train.png")

    with pytest.raises(ValueError, match="Mask cache is incomplete"):
        run_experiment(
            OptimisationConfig(experiment="incomplete_cache"),
            manifest,
            image_root,
            mask_cache_path=cache.directory,
            output_root=tmp_path / "out",
        )
    assert not (tmp_path / "out" / "incomplete_cache").exists()


def test_preflight_rejects_missing_mask_cache_and_segmentation_checkpoint(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "splits.csv"
    manifest.write_text(
        "split,patient_id,study_id,image_path,pneumonia_label\n"
        "train,p1,s1,input.png,1\n"
        "validation,p2,s2,input.png,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="refusing to fall back to unmasked"):
        run_experiment(
            OptimisationConfig(experiment="missing_masking_input"),
            manifest,
            tmp_path,
            output_root=tmp_path / "out",
        )
    assert not (tmp_path / "out" / "missing_masking_input").exists()


def test_original_mode_needs_no_segmentation_or_mask_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.classification.run_optimisation_experiment as runner
    image_root = tmp_path / "images"; image_root.mkdir()
    Image.new("RGB", (12, 12)).save(image_root / "train.png"); Image.new("RGB", (12, 12)).save(image_root / "validation.png")
    manifest = tmp_path / "splits.csv"
    manifest.write_text("split,patient_id,study_id,image_path,pneumonia_label\ntrain,p1,s1,train.png,1\nvalidation,p2,s2,validation.png,0\n")
    monkeypatch.setattr(runner, "create_model", lambda *args, **kwargs: nn.Identity())
    train_set, validation_set = runner._development_datasets(
        OptimisationConfig(experiment="original", input_mode="original"),
        manifest, image_root, tmp_path, None, None,
    )
    assert train_set.input_mode.value == "original"
    assert validation_set.input_mode.value == "original"
    assert train_set.lung_segmenter is None and validation_set.lung_segmenter is None
    assert train_set.mask_cache is None and validation_set.mask_cache is None
    monkeypatch.setattr(
        runner,
        "_masking_dependencies",
        lambda *args, **kwargs: pytest.fail("original mode must not resolve masking dependencies"),
    )
    runner._preflight(
        OptimisationConfig(experiment="original", input_mode="original"),
        manifest,
        image_root,
        tmp_path / "unused-segmenter.pt",
        tmp_path / "unused-cache",
        "cpu",
    )
    assert not (tmp_path / "shared_mask_cache").exists()


@pytest.mark.parametrize(("device", "cuda_available"), (("cpu", False), ("cuda", True)))
def test_original_run_reaches_training_setup_without_masking_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    cuda_available: bool,
) -> None:
    """Original-image runs must not touch cache or segmentation state on either device."""
    import scripts.classification.run_optimisation_experiment as runner

    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (12, 12)).save(image_root / "train.png")
    Image.new("RGB", (12, 12)).save(image_root / "validation.png")
    manifest = tmp_path / "splits.csv"
    manifest.write_text(
        "split,patient_id,study_id,image_path,pneumonia_label\n"
        "train,p1,s1,train.png,1\n"
        "validation,p2,s2,validation.png,0\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "out"
    received: list[tuple[object, object, object]] = []

    def training_setup(*args: object, **kwargs: object) -> Path:
        received.append((args[3], args[4], args[6]))
        return output_root / "original_run"

    monkeypatch.setattr(runner, "create_model", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(runner, "_masking_dependencies", lambda *args, **kwargs: pytest.fail("original mode must not initialise masking dependencies"))
    monkeypatch.setattr(runner, "_run_experiment_after_preflight", training_setup)

    runner.run_experiment(
        OptimisationConfig(experiment="original_run", input_mode="original"),
        manifest, image_root, output_root=output_root, device_name=device,
    )

    output = output_root / "original_run"
    (output / "last_checkpoint.pt").write_bytes(b"interrupted")
    runner.run_experiment(
        OptimisationConfig(experiment="original_run", input_mode="original"),
        manifest, image_root, output_root=output_root, device_name=device, resume=True,
    )
    runner.run_experiment(
        OptimisationConfig(experiment="original_run", input_mode="original"),
        manifest, image_root, output_root=output_root, device_name=device, restart=True,
    )
    assert received == [(None, None, device)] * 3
    assert not (output_root / "shared_mask_cache").exists()


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_parser_accepts_supported_devices(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    import scripts.classification.run_optimisation_experiment as runner

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_optimisation_experiment.py", "--config", "config.yaml",
            "--splits-csv", "splits.csv", "--image-root", "images",
            "--device", device,
        ],
    )
    assert runner.parse_args().device == device


def test_preflight_rejects_unavailable_cuda_with_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.classification.run_optimisation_experiment as runner

    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)
    with pytest.raises(
        RuntimeError,
        match=r"CUDA was requested but torch.cuda\.is_available\(\) is False",
    ):
        runner._preflight(
            OptimisationConfig(experiment="cuda_preflight", input_mode="original"),
            tmp_path / "missing.csv", tmp_path, None, None, "cuda",
        )


def test_hard_masked_segmenter_receives_requested_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.classification.run_optimisation_experiment as runner

    checkpoint = tmp_path / "segmenter.pt"
    checkpoint.write_bytes(b"checkpoint")
    received: list[object] = []

    class Segmenter:
        def __init__(self, path: Path, device: str) -> None:
            received.extend((path, device))

    monkeypatch.setattr(runner, "FrozenLungSegmenter", Segmenter)
    segmenter, cache = runner._masking_dependencies(
        checkpoint, None, "cuda", create_cache=False
    )
    assert isinstance(segmenter, Segmenter)
    assert cache is None
    assert received == [checkpoint, "cuda"]


def test_original_run_passes_requested_device_to_classifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.classification.run_optimisation_experiment as runner

    class TrackingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = nn.Linear(3 * 224 * 224, 1)
            self.devices: list[torch.device] = []

        def to(self, device: object, *args: object, **kwargs: object) -> "TrackingModel":
            self.devices.append(torch.device(device))
            return super().to(device, *args, **kwargs)

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return self.classifier(image.flatten(1))

    model = TrackingModel()
    monkeypatch.setattr(runner, "create_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(runner, "_plot", lambda *args: None)
    image_root = tmp_path / "images"
    image_root.mkdir()
    for name, colour in (("train0.png", 32), ("train1.png", 224), ("validation0.png", 64), ("validation1.png", 192)):
        Image.new("RGB", (12, 12), color=colour).save(image_root / name)
    manifest = tmp_path / "splits.csv"
    manifest.write_text(
        "split,patient_id,study_id,image_path,pneumonia_label\n"
        "train,p1,s1,train0.png,0\n"
        "train,p2,s2,train1.png,1\n"
        "validation,p3,s3,validation0.png,0\n"
        "validation,p4,s4,validation1.png,1\n",
        encoding="utf-8",
    )
    config = OptimisationConfig(
        experiment="device_propagation", input_mode="original", loss="bce",
        scheduler="none", epochs=1, batch_size=2, augmentation="none",
        horizontal_flip=False, rotation_degrees=0,
    )
    output = runner.run_experiment(
        config, manifest, image_root, output_root=tmp_path, device_name="cpu"
    )
    assert model.devices == [torch.device("cpu")]
    metadata = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert metadata["input_mode"] == "original"
    assert metadata["segmentation_checkpoint"] is None
    assert metadata["segmentation_checkpoint_sha256"] is None
    assert metadata["mask_cache"] is None
    checkpoint = torch.load(output / "best_checkpoint.pt", weights_only=False)
    assert checkpoint["configuration"]["input_mode"] == "original"
    assert not (tmp_path / "shared_mask_cache").exists()


def test_incomplete_empty_directory_is_reused_automatically(tmp_path: Path) -> None:
    output = tmp_path / "empty"
    output.mkdir()
    (output / "stale.txt").write_text("failed startup")
    _prepare_run_directory(output, resume=False, restart=False, force=False)
    assert output.is_dir()
    assert not (output / "stale.txt").exists()


def test_resume_without_checkpoint_has_clear_error(tmp_path: Path) -> None:
    output = tmp_path / "empty"
    output.mkdir()
    with pytest.raises(FileNotFoundError, match="last_checkpoint.pt"):
        _prepare_run_directory(output, resume=True, restart=False, force=False)


def test_failed_run_writes_failure_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.classification.run_optimisation_experiment as runner

    monkeypatch.setattr(runner, "_preflight", lambda *args: None)

    def fail(*args: object, **kwargs: object) -> Path:
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(runner, "_run_experiment_after_preflight", fail)
    with pytest.raises(RuntimeError, match="synthetic training failure"):
        runner.run_experiment(
            OptimisationConfig(experiment="failed"),
            tmp_path / "splits.csv",
            tmp_path,
            output_root=tmp_path / "out",
        )
    failure = json.loads((tmp_path / "out" / "failed" / "failure.json").read_text())
    assert failure["status"] == "failed"
    assert "synthetic training failure" in failure["message"]


def test_leaderboard_uses_completed_validation_run_only(tmp_path: Path) -> None:
    run = tmp_path / "A1"
    run.mkdir()
    (run / "config.json").write_text(
        json.dumps(OptimisationConfig(experiment="A1", pretrained=True).to_dict())
    )
    (run / "run_summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "epochs_completed": 2,
                "best_epoch": 2,
                "auroc": 0.7,
                "pr_auc": 0.8,
                "f1": 0.6,
                "sensitivity": 0.7,
                "specificity": 0.5,
                "balanced_accuracy": 0.6,
                "selected_threshold": 0.4,
                "training_time_seconds": 1.0,
            }
        )
    )
    (run / "validation_metrics.json").write_text("{}")
    (run / "validation_predictions.csv").write_text("label,probability\n0,0.1\n1,0.9\n")
    table = build_leaderboard(tmp_path)
    assert table.experiment.tolist() == ["A1"]
    assert (tmp_path / "experiment_leaderboard.csv").is_file()
