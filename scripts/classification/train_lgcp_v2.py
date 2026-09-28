"""Train LGCP-Net on the CheXpert development cohort.

LGCP-Net keeps the full radiograph intact and uses a frozen lung
probability map only as a feature-level anatomical prior.

External datasets are never used for model or threshold selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Repository root is required for imports from scripts.*
sys.path.insert(0, str(PROJECT_ROOT))

# src/ is required for imports from pneumonia_ai.*
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.classification.segmentation_guided import InputMode
from pneumonia_ai.data.chexpert_dataset import CheXpertPneumoniaDataset
from pneumonia_ai.data.label_strategy import apply_label_strategy
from pneumonia_ai.models.lgcp_v2 import LGCPv2Net
from pneumonia_ai.segmentation.cache import MaskCache
from pneumonia_ai.segmentation.inference import FrozenLungSegmenter
from pneumonia_ai.training.seed import seed_everything, seed_worker
from scripts.classification.optimisation_thresholds import threshold_analysis


IMAGENET_MEAN = (0.485, 0.456, 0.406)

IMAGENET_STD = (0.229, 0.224, 0.225)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lgcp_cache_metadata(
    segmenter: FrozenLungSegmenter,
    image_root: Path,
) -> dict[str, object]:
    """Identity of continuous lung-probability maps used by LGCP."""
    return {
        "schema_version": 1,
        "image_root": str(image_root.resolve()),
        "segmentation_checkpoint": str(
            segmenter.checkpoint_path.resolve()
        ),
        "segmentation_checkpoint_sha256": file_sha256(
            segmenter.checkpoint_path
        ),
        "segmentation_architecture": type(segmenter.model).__name__,
        "segmentation_input_size": segmenter.image_size,
        "mask_threshold": 0.5,
        "mask_postprocessing": "none",
        "classifier_masking_mode": "lgcp_probability_guidance",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-cache", type=Path, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def make_transform() -> transforms.Compose:
    """Alignment-safe transform: no rotation, crop, or horizontal flip."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def prepare_development_manifest(
    source: Path,
    destination: Path,
) -> Path:
    frame = pd.read_csv(source)

    development = frame.loc[
        frame["split"].isin(["train", "validation"])
    ].copy()

    train = apply_label_strategy(
        development.loc[development["split"] == "train"],
        "ignore",
    )
    validation = apply_label_strategy(
        development.loc[development["split"] == "validation"],
        "ignore",
    )

    prepared = pd.concat([train, validation], ignore_index=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(destination, index=False)

    return destination


def metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    y = frame["label"].to_numpy(dtype=int)
    p = frame["probability"].to_numpy(dtype=float)
    pred = p >= threshold

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    accuracy = (tp + tn) / len(y) if len(y) else 0.0

    return {
        "auroc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "accuracy": float(accuracy),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, object]] = []

    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            lung_probability = batch["lung_probability"].to(device)

            logits = model(image, lung_probability)
            probabilities = torch.sigmoid(logits).cpu().numpy()

            for i, probability in enumerate(probabilities):
                rows.append(
                    {
                        "patient_id": batch["patient_id"][i],
                        "study_id": batch["study_id"][i],
                        "image_path": batch["image_path"][i],
                        "label": int(batch["target"][i]),
                        "probability": float(probability),
                        "split": "validation",
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    args.output.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device(args.device)

    development_manifest = prepare_development_manifest(
        args.splits_csv,
        args.output / "_development_manifest.csv",
    )

    if args.mask_cache is None:
        raise ValueError(
            "LGCP training requires --mask-cache so lung probability maps "
            "can be generated once and reused safely."
        )

    # Stage 1: initialise a provenance-bound probability-map cache.
    segmenter = FrozenLungSegmenter(
        args.segmentation_checkpoint,
        args.device,
    )
    mask_cache = MaskCache(args.mask_cache, create=True)
    mask_cache.validate_or_initialise_metadata(
        lgcp_cache_metadata(segmenter, args.image_root)
    )

    print(
        f"Compatible LGCP mask cache: {mask_cache.directory}",
        flush=True,
    )
    print(
        f"Cached masks before preparation: "
        f"{mask_cache.coverage_count()}",
        flush=True,
    )

    transform = make_transform()

    train_set = CheXpertPneumoniaDataset(
        dataset_root=args.image_root,
        manifest_path=development_manifest,
        split="train",
        transform=transform,
        input_mode=InputMode.ORIGINAL,
        lung_segmenter=segmenter,
        mask_cache=mask_cache,
        classifier_image_size=None,
        return_lung_probability=True,
    )

    validation_set = CheXpertPneumoniaDataset(
        dataset_root=args.image_root,
        manifest_path=development_manifest,
        split="validation",
        transform=transform,
        input_mode=InputMode.ORIGINAL,
        lung_segmenter=segmenter,
        mask_cache=mask_cache,
        classifier_image_size=None,
        return_lung_probability=True,
    )

    if args.max_train_samples is not None:
        train_set.records = train_set.records.iloc[: args.max_train_samples].reset_index(drop=True)
        train_set._raw_labels = train_set._raw_labels.iloc[: args.max_train_samples].reset_index(drop=True)
        train_set._targets = train_set._targets.iloc[: args.max_train_samples].reset_index(drop=True)
        train_set._sample_loss_weights = train_set._sample_loss_weights.iloc[: args.max_train_samples].reset_index(drop=True)
        train_set._resolved_image_paths = train_set._resolved_image_paths[: args.max_train_samples]
        train_set._returned_image_paths = train_set._returned_image_paths[: args.max_train_samples]

    if args.max_validation_samples is not None:
        validation_set.records = validation_set.records.iloc[: args.max_validation_samples].reset_index(drop=True)
        validation_set._raw_labels = validation_set._raw_labels.iloc[: args.max_validation_samples].reset_index(drop=True)
        validation_set._targets = validation_set._targets.iloc[: args.max_validation_samples].reset_index(drop=True)
        validation_set._sample_loss_weights = validation_set._sample_loss_weights.iloc[: args.max_validation_samples].reset_index(drop=True)
        validation_set._resolved_image_paths = validation_set._resolved_image_paths[: args.max_validation_samples]
        validation_set._returned_image_paths = validation_set._returned_image_paths[: args.max_validation_samples]

    # --------------------------------------------------------
    # Stage 1: generate only missing probability maps once.
    # --------------------------------------------------------
    required_paths = [
        *train_set._resolved_image_paths,
        *validation_set._resolved_image_paths,
    ]

    missing_before = sum(
        mask_cache.get_for_source(source) is None
        for source in required_paths
    )

    print(
        f"LGCP mask-cache coverage before preparation: "
        f"{len(required_paths) - missing_before}/{len(required_paths)}",
        flush=True,
    )

    if missing_before:
        print(
            f"Generating {missing_before} missing lung probability maps...",
            flush=True,
        )

        # Accessing samples causes the dataset to use the frozen U-Net
        # only for missing cache entries.
        with torch.inference_mode():
            for dataset_name, dataset in (
                ("train", train_set),
                ("validation", validation_set),
            ):
                total = len(dataset)
                for index in range(total):
                    _ = dataset[index]

                    processed = index + 1
                    if processed == total or processed % 500 == 0:
                        print(
                            f"Cache preparation {dataset_name}: "
                            f"{processed}/{total}",
                            flush=True,
                        )

    mask_cache.require_coverage(required_paths)

    print(
        f"LGCP mask-cache coverage after preparation: "
        f"{len(required_paths)}/{len(required_paths)}",
        flush=True,
    )

    # --------------------------------------------------------
    # Stage 2: train from cache only.
    # No live U-Net remains inside DataLoader datasets.
    # --------------------------------------------------------
    del train_set
    del validation_set
    del segmenter

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_set = CheXpertPneumoniaDataset(
        dataset_root=args.image_root,
        manifest_path=development_manifest,
        split="train",
        transform=transform,
        input_mode=InputMode.ORIGINAL,
        lung_segmenter=None,
        mask_cache=mask_cache,
        classifier_image_size=None,
        return_lung_probability=True,
    )

    validation_set = CheXpertPneumoniaDataset(
        dataset_root=args.image_root,
        manifest_path=development_manifest,
        split="validation",
        transform=transform,
        input_mode=InputMode.ORIGINAL,
        lung_segmenter=None,
        mask_cache=mask_cache,
        classifier_image_size=None,
        return_lung_probability=True,
    )

    # Preserve smoke-test subsetting after rebuilding cache-only datasets.
    if args.max_train_samples is not None:
        train_set.records = train_set.records.iloc[
            : args.max_train_samples
        ].reset_index(drop=True)
        train_set._raw_labels = train_set._raw_labels.iloc[
            : args.max_train_samples
        ].reset_index(drop=True)
        train_set._targets = train_set._targets.iloc[
            : args.max_train_samples
        ].reset_index(drop=True)
        train_set._sample_loss_weights = (
            train_set._sample_loss_weights.iloc[
                : args.max_train_samples
            ].reset_index(drop=True)
        )
        train_set._resolved_image_paths = (
            train_set._resolved_image_paths[: args.max_train_samples]
        )
        train_set._returned_image_paths = (
            train_set._returned_image_paths[: args.max_train_samples]
        )

    if args.max_validation_samples is not None:
        validation_set.records = validation_set.records.iloc[
            : args.max_validation_samples
        ].reset_index(drop=True)
        validation_set._raw_labels = validation_set._raw_labels.iloc[
            : args.max_validation_samples
        ].reset_index(drop=True)
        validation_set._targets = validation_set._targets.iloc[
            : args.max_validation_samples
        ].reset_index(drop=True)
        validation_set._sample_loss_weights = (
            validation_set._sample_loss_weights.iloc[
                : args.max_validation_samples
            ].reset_index(drop=True)
        )
        validation_set._resolved_image_paths = (
            validation_set._resolved_image_paths[
                : args.max_validation_samples
            ]
        )
        validation_set._returned_image_paths = (
            validation_set._returned_image_paths[
                : args.max_validation_samples
            ]
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed),
    )

    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
    )

    model = LGCPv2Net(
        pretrained=not args.no_pretrained,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
    ).to(device)

    positive = float((train_set._targets == 1).sum())
    negative = float((train_set._targets == 0).sum())

    if positive <= 0:
        raise RuntimeError("Training cohort contains no positive examples.")

    pos_weight = torch.tensor(
        [negative / positive],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_auroc = -np.inf
    best_epoch = 0
    stalled = 0
    history: list[dict[str, float | int]] = []

    print("=== LGCP TRAINING ===", flush=True)
    print("Device:", device, flush=True)
    print("Train samples:", len(train_set), flush=True)
    print("Validation samples:", len(validation_set), flush=True)
    print("Positive weight:", float(pos_weight.item()), flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            image = batch["image"].to(device)
            lung_probability = batch["lung_probability"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(image, lung_probability)
            loss = criterion(logits, target)

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

            if step % max(1, len(train_loader) // 10) == 0:
                print(
                    f"Epoch {epoch}: batch {step}/{len(train_loader)}",
                    flush=True,
                )

        validation = evaluate(model, validation_loader, device)

        auroc = float(
            roc_auc_score(
                validation["label"],
                validation["probability"],
            )
        )
        pr_auc = float(
            average_precision_score(
                validation["label"],
                validation["probability"],
            )
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            "validation_auroc": auroc,
            "validation_pr_auc": pr_auc,
        }
        history.append(epoch_record)

        pd.DataFrame(history).to_csv(
            args.output / "training_history.csv",
            index=False,
        )

        print(
            f"Epoch {epoch}: "
            f"loss={epoch_record['train_loss']:.6f} "
            f"AUROC={auroc:.6f} "
            f"PR-AUC={pr_auc:.6f}",
            flush=True,
        )

        if auroc > best_auroc:
            best_auroc = auroc
            best_epoch = epoch
            stalled = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_auroc": best_auroc,
                    "seed": args.seed,
                    "projection_dim": args.projection_dim,
                    "dropout": args.dropout,
                    "architecture": "LGCPv2Net",
                },
                args.output / "best_checkpoint.pt",
            )

            validation.to_csv(
                args.output / "validation_predictions.csv",
                index=False,
            )

            print("Updated best checkpoint.", flush=True)

        else:
            stalled += 1

        if stalled >= args.patience:
            print(
                f"Early stopping after epoch {epoch}.",
                flush=True,
            )
            break

    # -----------------------------------------------------
    # Final validation using best-AUROC checkpoint.
    # -----------------------------------------------------
    state = torch.load(
        args.output / "best_checkpoint.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(state["model_state_dict"])

    validation = evaluate(model, validation_loader, device)

    threshold, analysis = threshold_analysis(
        validation["label"].to_numpy(),
        validation["probability"].to_numpy(),
        "max_f1",
    )

    validation.to_csv(
        args.output / "validation_predictions.csv",
        index=False,
    )
    analysis.to_csv(
        args.output / "threshold_analysis.csv",
        index=False,
    )

    fixed = metrics(validation, 0.5)
    selected = metrics(validation, float(threshold))

    summary = {
        "status": "completed",
        "architecture": "LGCPv2Net",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_auroc": best_auroc,
        "validation_auroc": fixed["auroc"],
        "validation_pr_auc": fixed["pr_auc"],
        "fixed_threshold": 0.5,
        "fixed_threshold_metrics": fixed,
        "max_f1_threshold": float(threshold),
        "max_f1_metrics": selected,
        "train_samples": len(train_set),
        "validation_samples": len(validation_set),
        "label_policy": "ignore",
        "external_data_used_for_selection": False,
        "input": "full_cxr_plus_frozen_unet_probability_map",
        "segmentation_checkpoint": str(args.segmentation_checkpoint),
    }

    (args.output / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== LGCP COMPLETE ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
