"""Optimise a classifier on development-only original or dynamically hard-masked inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback
import subprocess
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pneumonia_ai.data.chexpert_dataset import CheXpertPneumoniaDataset  # noqa: E402
from pneumonia_ai.classification.segmentation_guided import InputMode  # noqa: E402
from pneumonia_ai.data.label_strategy import apply_label_strategy  # noqa: E402
from pneumonia_ai.models.factory import create_model  # noqa: E402
from pneumonia_ai.segmentation.cache import MaskCache  # noqa: E402
from pneumonia_ai.segmentation.inference import FrozenLungSegmenter  # noqa: E402
from pneumonia_ai.training.engine import calculate_pos_weight  # noqa: E402
from pneumonia_ai.training.seed import seed_everything, seed_worker  # noqa: E402
from scripts.classification.optimisation_config import (  # noqa: E402
    OptimisationConfig,
    load_config,
    resolve_preprocessing,
    serialise_config,
    validate_controlled_a_series,
)
from scripts.classification.optimisation_losses import build_loss  # noqa: E402
from scripts.classification.optimisation_thresholds import threshold_analysis  # noqa: E402
from scripts.train_baseline import _transforms as build_classifier_transforms  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "classification_optimisation"


def build_transforms(
    config: OptimisationConfig,
) -> tuple[object, object]:
    """Reuse the classifier training transform factory without modification.

    This must remain a direct delegation: optimisation inputs must be identical to
    those used by the original classifier for the selected backbone.
    """
    return build_classifier_transforms(
        config.input_size,
        resolve_preprocessing(config),
        augmentation=config.augmentation,
        horizontal_flip=config.augmentation == "historical" and config.horizontal_flip,
        rotation_degrees=config.rotation_degrees if config.augmentation == "historical" else 0,
    )


def resolved_transform_metadata(config: OptimisationConfig) -> dict[str, object]:
    """Record the exact shared classifier transform options used by a run."""
    return {
        "horizontal_flip": config.augmentation == "historical" and config.horizontal_flip,
        "rotation_degrees": config.rotation_degrees if config.augmentation == "historical" else 0,
        "preprocessing": resolve_preprocessing(config),
        "image_size": config.input_size,
    }


def configure_fine_tuning(model: nn.Module, freeze_backbone: bool) -> None:
    """Freeze/unfreeze backbone while keeping frozen BatchNorm modules in evaluation mode."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad = not freeze_backbone or name.startswith("classifier")
    if freeze_backbone:
        for name, module in model.named_modules():
            if not name.startswith("classifier") and isinstance(
                module, nn.modules.batchnorm._BatchNorm
            ):
                module.eval()


def differential_parameter_groups(
    model: nn.Module, config: OptimisationConfig
) -> list[dict[str, object]]:
    """Return disjoint backbone and classifier parameter groups with differential rates."""
    head = [
        p
        for n, p in model.named_parameters()
        if n.startswith("classifier") and p.requires_grad
    ]
    body = [
        p
        for n, p in model.named_parameters()
        if not n.startswith("classifier") and p.requires_grad
    ]
    groups: list[dict[str, object]] = []
    if body:
        groups.append({"params": body, "lr": config.backbone_learning_rate})
    if head:
        groups.append({"params": head, "lr": config.head_learning_rate})
    if not groups:
        raise ValueError("No trainable parameters remain after freezing.")
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--splits-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--segmentation-checkpoint", type=Path)
    parser.add_argument(
        "--mask-cache",
        type=Path,
        help="Optional complete indexed cache of hard-mask probabilities.",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-test-evaluation", action="store_true")
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="Reserved explicit final-test mode; requires --allow-test-evaluation.",
    )
    return parser.parse_args()


def _development_datasets(
    config: OptimisationConfig,
    splits_csv: Path,
    image_root: Path,
    output: Path,
    segmenter: FrozenLungSegmenter | None,
    mask_cache: MaskCache | None,
) -> tuple[CheXpertPneumoniaDataset, CheXpertPneumoniaDataset]:
    """Create development datasets without materialising test rows."""
    with splits_csv.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        if reader.fieldnames is None or "split" not in reader.fieldnames:
            raise ValueError("Split manifest is missing the required split column.")
        development_rows = [
            row for row in reader if row["split"] in {"train", "validation"}
        ]
    development = pd.DataFrame(development_rows)
    if set(development.split) != {"train", "validation"}:
        raise ValueError("Manifest requires train and validation rows.")
    train = apply_label_strategy(
        development.loc[development.split == "train"], "ignore"
    )
    validation = apply_label_strategy(
        development.loc[development.split == "validation"], "ignore"
    )
    manifest = pd.concat([train, validation], ignore_index=True)
    path = output / "_development_manifest.csv"
    manifest.to_csv(path, index=False)
    train_tf, validation_tf = build_transforms(config)
    return (
        CheXpertPneumoniaDataset(
            image_root, path, "train", train_tf,
            input_mode=InputMode(config.input_mode), lung_segmenter=segmenter,
            mask_cache=mask_cache,
            classifier_image_size=config.input_size if config.input_mode == "hard_masked" else None,
        ),
        CheXpertPneumoniaDataset(
            image_root, path, "validation", validation_tf,
            input_mode=InputMode(config.input_mode), lung_segmenter=segmenter,
            mask_cache=mask_cache,
            classifier_image_size=config.input_size if config.input_mode == "hard_masked" else None,
        ),
    )


def validate_development_manifest(splits_csv: Path) -> None:
    """Confirm that development rows exist without materialising any test records."""
    with splits_csv.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        if reader.fieldnames is None or "split" not in reader.fieldnames:
            raise ValueError("Split manifest is missing the required split column.")
        splits = {
            row["split"] for row in reader if row["split"] in {"train", "validation"}
        }
    if splits != {"train", "validation"}:
        raise ValueError("Manifest requires train and validation rows.")


def _evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[pd.DataFrame, float, float]:
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            probability = (
                torch.sigmoid(model(batch["image"].to(device)).view(-1)).cpu().numpy()
            )
            for i, value in enumerate(probability):
                rows.append(
                    {
                        "patient_id": batch["patient_id"][i],
                        "study_id": batch["study_id"][i],
                        "image_path": batch["image_path"][i],
                        "label": int(batch["target"][i]),
                        "probability": float(value),
                        "split": "validation",
                    }
                )
    frame = pd.DataFrame(rows)
    y, p = frame.label.to_numpy(int), frame.probability.to_numpy(float)
    return frame, float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def _metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    y, p = frame.label.to_numpy(int), frame.probability.to_numpy(float)
    prediction = p >= threshold
    tp = int(((prediction == 1) & (y == 1)).sum())
    tn = int(((prediction == 0) & (y == 0)).sum())
    fp = int(((prediction == 1) & (y == 0)).sum())
    fn = int(((prediction == 0) & (y == 1)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "auroc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "accuracy": (tp + tn) / len(y) if len(y) else 0.0,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cache_metadata(segmenter: FrozenLungSegmenter, image_root: Path) -> dict[str, object]:
    """Identity for reusable probability masks before hard masking and RGB conversion."""
    return {
        "schema_version": 1,
        "image_root": str(image_root.resolve()),
        "segmentation_checkpoint": str(segmenter.checkpoint_path.resolve()),
        "segmentation_checkpoint_sha256": _file_sha256(segmenter.checkpoint_path),
        "segmentation_architecture": type(segmenter.model).__name__,
        "segmentation_input_size": segmenter.image_size,
        "mask_threshold": 0.5,
        "mask_postprocessing": "none",
        "classifier_masking_mode": InputMode.HARD_MASKED.value,
    }


def _plot(history: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(history.epoch, history.validation_auroc, label="AUROC")
    axes[0].plot(history.epoch, history.validation_pr_auc, label="PR-AUC")
    axes[0].legend()
    axes[1].plot(history.epoch, history.train_loss, label="train loss")
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel("Epoch")
    figure.tight_layout()
    figure.savefig(output / "learning_curves.png", dpi=300)
    figure.savefig(output / "learning_curves.pdf")
    plt.close(figure)


def _masking_dependencies(
    segmentation_checkpoint: Path | None,
    mask_cache_path: Path | None,
    device_name: str,
    *,
    create_cache: bool,
) -> tuple[FrozenLungSegmenter | None, MaskCache | None]:
    """Select segmentation inference or a complete indexed probability-mask cache."""
    if segmentation_checkpoint is not None:
        if not segmentation_checkpoint.is_file():
            raise FileNotFoundError(
                f"Segmentation checkpoint does not exist: {segmentation_checkpoint}"
            )
        return (
            FrozenLungSegmenter(segmentation_checkpoint, device_name),
            MaskCache(mask_cache_path, create=create_cache)
            if mask_cache_path is not None
            else None,
        )
    if mask_cache_path is None:
        raise ValueError(
            "Hard-masked optimisation requires --segmentation-checkpoint or a complete "
            "indexed --mask-cache; refusing to fall back to unmasked images."
        )
    return None, MaskCache(mask_cache_path, create=create_cache)


def _preflight(
    config: OptimisationConfig,
    splits_csv: Path,
    image_root: Path,
    segmentation_checkpoint: Path | None,
    mask_cache_path: Path | None,
    device_name: str,
) -> None:
    """Validate every dependency without creating an experiment output directory."""
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False."
        )
    if not splits_csv.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {splits_csv}")
    if not image_root.is_dir():
        raise NotADirectoryError(f"Image root does not exist: {image_root}")
    validate_development_manifest(splits_csv)
    if config.input_mode == "hard_masked":
        segmenter, mask_cache = _masking_dependencies(
            segmentation_checkpoint, mask_cache_path, device_name, create_cache=False
        )
    else:
        if segmentation_checkpoint is not None or mask_cache_path is not None:
            # Inputs are deliberately ignored: original mode must not read segmentation state.
            segmenter, mask_cache = None, None
    with tempfile.TemporaryDirectory() as temporary:
        train_set, validation_set = _development_datasets(
            config, splits_csv, image_root, Path(temporary), segmenter, mask_cache
        )
        if config.input_mode == "hard_masked" and segmenter is None:
            assert mask_cache is not None
            mask_cache.require_coverage(
                [*train_set._resolved_image_paths, *validation_set._resolved_image_paths]
            )
    try:
        create_model(config.backbone, pretrained=config.pretrained)
    except Exception as error:
        if config.pretrained:
            raise RuntimeError(
                "ImageNet pretrained weights could not be loaded; refusing random-initialisation fallback."
            ) from error
        raise


def _prepare_run_directory(
    output: Path, *, resume: bool, restart: bool, force: bool
) -> None:
    """Apply completed, resumable, and failed-run protections after preflight."""
    if not output.exists():
        output.mkdir(parents=True)
        return
    summary_path = output / "run_summary.json"
    status = None
    if summary_path.is_file():
        try:
            status = json.loads(summary_path.read_text(encoding="utf-8")).get("status")
        except (json.JSONDecodeError, OSError):
            status = None
    completed = status == "completed"
    last_checkpoint = output / "last_checkpoint.pt"
    if restart:
        if completed and not force:
            archive = output.with_name(output.name + ".archived_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
            output.rename(archive)
            print(f"Archived prior completed run to {archive}", flush=True)
            output.mkdir(parents=True)
            return
        shutil.rmtree(output)
        output.mkdir(parents=True)
        return
    if completed:
        raise FileExistsError(f"Refusing to overwrite completed experiment: {output}")
    if resume:
        if not last_checkpoint.is_file():
            raise FileNotFoundError(
                "--resume was requested but last_checkpoint.pt is absent. Use --restart to start fresh."
            )
        return
    if last_checkpoint.is_file():
        raise FileExistsError(
            "An interrupted run has last_checkpoint.pt. Use --resume to continue or --restart to discard it."
        )
    shutil.rmtree(output)
    output.mkdir(parents=True)


def _write_failure(output: Path, error: BaseException) -> None:
    """Persist an honest terminal failure state, including a traceback for recovery."""
    payload = {
        "status": "failed",
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }
    (output / "failure.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (output / "run_summary.json").write_text(
        json.dumps({"status": "failed", "error": str(error)}, indent=2),
        encoding="utf-8",
    )


def _run_experiment_after_preflight(
    config: OptimisationConfig,
    splits_csv: Path,
    image_root: Path,
    segmentation_checkpoint: Path | None,
    mask_cache_path: Path | None,
    output_root: Path = OUTPUT_ROOT,
    device_name: str = "cpu",
    resume: bool = False,
) -> Path:
    """Execute one validation-selected experiment without loading the test split."""
    output = output_root / config.experiment
    seed_everything(config.seed)
    device = torch.device(device_name)
    if config.input_mode == "hard_masked":
        segmenter, mask_cache = _masking_dependencies(
            segmentation_checkpoint,
            mask_cache_path or output / "mask_cache",
            device_name,
            create_cache=True,
        )
    else:
        # Original-image control has no segmentation or cache dependency by design.
        segmenter, mask_cache = None, None
    if segmenter is not None and mask_cache is not None:
        mask_cache.validate_or_initialise_metadata(_cache_metadata(segmenter, image_root))
        print(f"Using compatible shared mask cache: {mask_cache.directory}", flush=True)
        print(f"Cached masks found: {mask_cache.coverage_count()}", flush=True)
    train_set, validation_set = _development_datasets(
        config, splits_csv, image_root, output, segmenter, mask_cache
    )
    if mask_cache is not None:
        requested_masks = [*train_set._resolved_image_paths, *validation_set._resolved_image_paths]
        present = sum(mask_cache.get_for_source(path) is not None for path in requested_masks)
        print(f"Mask-cache coverage: cached masks found: {present}; missing masks: {len(requested_masks) - present}", flush=True)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
    )
    model = create_model(config.backbone, pretrained=config.pretrained)
    print(f"Pretrained-weight loading status: pretrained={config.pretrained}", flush=True)
    model.to(device)
    configure_fine_tuning(model, config.frozen_head_warmup_epochs > 0)
    optimizer = _optimizer(differential_parameter_groups(model, config), config)
    scheduler = (
        ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
        if config.scheduler == "plateau"
        else (
            CosineAnnealingLR(optimizer, config.epochs)
            if config.scheduler == "cosine"
            else None
        )
    )
    criterion = build_loss(
        config.loss,
        calculate_pos_weight(
            train_set._targets, config.loss in {"weighted_bce", "focal"}
        ),
    ).to(device)
    start, best, best_epoch, history = 1, -np.inf, 0, []
    last = output / "last_checkpoint.pt"
    best_path = output / "best_checkpoint.pt"
    if resume:
        if not last.is_file():
            raise FileNotFoundError(
                "Resume requested but last_checkpoint.pt is absent."
            )
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start, best, best_epoch = (
            state["epoch"] + 1,
            state["best_auroc"],
            state["best_epoch"],
        )
        history = (
            pd.read_csv(output / "training_history.csv").to_dict("records")
            if (output / "training_history.csv").is_file()
            else []
        )
    stalled = 0
    started = time.perf_counter()
    early_stopped = False
    for epoch in range(start, config.epochs + 1):
        print(f"Epoch {epoch}/{config.epochs}: training", flush=True)
        if (
            epoch == config.frozen_head_warmup_epochs + 1
            and config.progressive_unfreezing
        ):
            configure_fine_tuning(model, False)
            optimizer = _optimizer(differential_parameter_groups(model, config), config)
        model.train()
        if epoch <= config.frozen_head_warmup_epochs:
            configure_fine_tuning(model, True)
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, 1):
            logits = model(batch["image"].to(device)).view(-1)
            loss = (
                criterion(logits, batch["target"].to(device))
                / config.gradient_accumulation
            )
            loss.backward()
            total_loss += float(loss.item()) * config.gradient_accumulation
            if step % config.gradient_accumulation == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % max(1, len(train_loader) // 10) == 0 or step == len(train_loader):
                print(f"Epoch {epoch}: training batch {step}/{len(train_loader)}", flush=True)
        print(f"Epoch {epoch}: validation start", flush=True)
        validation, auroc, pr_auc = _evaluate(model, validation_loader, device)
        print(f"Epoch {epoch}: validation complete AUROC={auroc:.6f} PR-AUC={pr_auc:.6f}", flush=True)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(train_loader),
                "validation_auroc": auroc,
                "validation_pr_auc": pr_auc,
            }
        )
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        threshold, analysis = threshold_analysis(
            validation.label.to_numpy(),
            validation.probability.to_numpy(),
            "max_f1",
        )
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_auroc": best,
            "best_epoch": best_epoch,
            "configuration": serialise_config(config),
            "max_f1_threshold": threshold,
        }
        if auroc > best:
            best, best_epoch, stalled = auroc, epoch, 0
            state["best_auroc"] = best
            state["best_epoch"] = best_epoch
            torch.save(state, best_path)
            validation.to_csv(output / "validation_predictions.csv", index=False)
            analysis.to_csv(output / "threshold_analysis.csv", index=False)
            print(f"Epoch {epoch}: updated best checkpoint", flush=True)
        else:
            stalled += 1
        state["best_auroc"] = best
        state["best_epoch"] = best_epoch
        torch.save(state, last)
        if scheduler:
            scheduler.step(auroc) if config.scheduler == "plateau" else scheduler.step()
        logging.getLogger(config.experiment).info(
            "epoch=%s auroc=%.6f pr_auc=%.6f", epoch, auroc, pr_auc
        )
        if stalled >= config.early_stopping_patience:
            early_stopped = True
            print(f"Early stopping after epoch {epoch}; counter={stalled}", flush=True)
            break
        print(f"Epoch {epoch}: early-stopping counter={stalled}", flush=True)
    print("Finalization stage: evaluating best validation checkpoint", flush=True)
    best_state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])
    validation, _, _ = _evaluate(model, validation_loader, device)
    threshold, analysis = threshold_analysis(
        validation.label.to_numpy(),
        validation.probability.to_numpy(),
        "max_f1",
    )
    validation.to_csv(output / "validation_predictions.csv", index=False)
    analysis.to_csv(output / "threshold_analysis.csv", index=False)
    fixed_metrics = _metrics(validation, 0.5)
    max_f1_metrics = _metrics(validation, threshold)
    metrics = {
        "auroc": fixed_metrics["auroc"], "pr_auc": fixed_metrics["pr_auc"],
        "fixed_threshold": 0.5, "fixed_f1": fixed_metrics["f1"],
        "fixed_sensitivity": fixed_metrics["sensitivity"],
        "fixed_specificity": fixed_metrics["specificity"],
        "fixed_balanced_accuracy": fixed_metrics["balanced_accuracy"],
        "fixed_precision": fixed_metrics["precision"], "fixed_accuracy": fixed_metrics["accuracy"],
        "max_f1_threshold": threshold, "max_f1_f1": max_f1_metrics["f1"],
        "max_f1_sensitivity": max_f1_metrics["sensitivity"],
        "max_f1_specificity": max_f1_metrics["specificity"],
        "max_f1_balanced_accuracy": max_f1_metrics["balanced_accuracy"],
    }
    (output / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    history_frame = pd.DataFrame(history)
    _plot(history_frame, output)
    summary = {
        "status": "completed",
        "experiment": config.experiment,
        "input_mode": config.input_mode,
        "epochs_completed": len(history_frame),
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "training_time_seconds": time.perf_counter() - started,
        "end_time": datetime.now(timezone.utc).isoformat(),
        "train_sample_count": len(train_set),
        "validation_sample_count": len(validation_set),
        **metrics,
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output / "_development_manifest.csv").unlink(missing_ok=True)
    print(f"Run complete: {output}", flush=True)
    return output


def run_experiment(
    config: OptimisationConfig,
    splits_csv: Path,
    image_root: Path,
    segmentation_checkpoint: Path | None = None,
    mask_cache_path: Path | None = None,
    output_root: Path = OUTPUT_ROOT,
    device_name: str = "cpu",
    resume: bool = False,
    restart: bool = False,
    force: bool = False,
) -> Path:
    """Preflight first, then execute with safe restart and failure-state handling."""
    shared_cache = output_root / "shared_mask_cache"
    effective_mask_cache = (mask_cache_path or shared_cache) if config.input_mode == "hard_masked" else None
    # A missing shared cache is expected on the first run; segmentation will populate it.
    preflight_cache = (
        effective_mask_cache
        if effective_mask_cache is not None and effective_mask_cache.is_dir()
        else None
    )
    _preflight(
        config, splits_csv, image_root, segmentation_checkpoint, preflight_cache,
        device_name,
    )
    output = output_root / config.experiment
    _prepare_run_directory(output, resume=resume, restart=restart, force=force)
    logger = logging.getLogger(config.experiment)
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(output / "training.log", encoding="utf-8"))
    logger.setLevel(logging.INFO)
    run_metadata = {
        "transform": resolved_transform_metadata(config),
        "dataset_split_path": str(splits_csv.resolve()), "image_root": str(image_root.resolve()),
        "label_policy": "ignore",
        "input_mode": config.input_mode,
        "segmentation_checkpoint": str(segmentation_checkpoint.resolve()) if config.input_mode == "hard_masked" and segmentation_checkpoint else None,
        "segmentation_checkpoint_sha256": _file_sha256(segmentation_checkpoint) if config.input_mode == "hard_masked" and segmentation_checkpoint else None,
        "mask_cache": str(effective_mask_cache.resolve()) if effective_mask_cache else None, "torch_version": torch.__version__,
        "device": device_name, "start_time": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }
    (output / "config.json").write_text(json.dumps(serialise_config(config, run_metadata), indent=2), encoding="utf-8")
    print(f"Experiment: {config.experiment}\nResolved configuration: {json.dumps(config.to_dict(), sort_keys=True)}", flush=True)
    try:
        return _run_experiment_after_preflight(
            config,
            splits_csv,
            image_root,
            segmentation_checkpoint,
            effective_mask_cache,
            output_root,
            device_name,
            resume,
        )
    except Exception as error:
        _write_failure(output, error)
        raise


def main() -> None:
    args = parse_args()
    validate_controlled_a_series(args.config.parent)
    if args.final_test:
        if not args.allow_test_evaluation:
            raise PermissionError(
                "Final test evaluation requires --allow-test-evaluation."
            )
        raise NotImplementedError(
            "Final test evaluation is intentionally separate from development and is not enabled during optimisation."
        )
    print(
        run_experiment(
            load_config(args.config),
            args.splits_csv,
            args.image_root,
            args.segmentation_checkpoint,
            args.mask_cache,
            args.output_root,
            args.device,
            args.resume,
            args.restart,
            args.force,
        )
    )


def _optimizer(
    groups: list[dict[str, object]], config: OptimisationConfig
) -> torch.optim.Optimizer:
    """Create the configured optimizer without silently substituting another one."""
    if config.optimizer == "adamw":
        return torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, weight_decay=config.weight_decay)
    raise ValueError("optimizer must be adamw or sgd.")


if __name__ == "__main__":
    main()
