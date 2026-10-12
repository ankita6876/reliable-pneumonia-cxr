"""BPEC CheXpert pilot, source-domain only.

Reuse the historical LCDR data and augmentation utilities,
but train an ordinary ResNet50 with BPEC consistency.
"""
import argparse
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'analysis'))
from shortcut_protocol import shortcut_image

import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix
)

from pneumonia_ai.models.bpec import (
    perturb_exterior, bpec_consistency_loss
)
from pneumonia_ai.data.chexpert_dataset import CheXpertPneumoniaDataset
from pneumonia_ai.classification.segmentation_guided import InputMode
from pneumonia_ai.segmentation.inference import FrozenLungSegmenter
from pneumonia_ai.segmentation.cache import MaskCache


def load_lcdr_helpers(repo):
    path = repo / "scripts/classification/train_lcdr.py"
    spec = importlib.util.spec_from_file_location("lcd r_helpers".replace(" ", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(
            labels, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            labels, predictions, zero_division=0
        )),
        "specificity": float(tn / max(tn + fp, 1)),
        "accuracy": float((tp + tn) / len(labels)),
        "threshold": float(threshold),
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
    }


def evaluate(model, loader, device):
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["image"].to(device)).flatten()
            probs = torch.sigmoid(logits).cpu().numpy()

            for i, p in enumerate(probs):
                rows.append({
                    "patient_id": str(batch["patient_id"][i]),
                    "study_id": str(batch["study_id"][i]),
                    "image_path": str(batch["image_path"][i]),
                    "label": int(batch["target"][i].item()),
                    "probability": float(p),
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--splits-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--mask-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-bpec", type=float, default=1.0)
    parser.add_argument("--shortcut-region", default="exterior")
    parser.add_argument("--shortcut-correlation", type=float, default=0.90)
    parser.add_argument("--shortcut-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "Enable Kaggle GPU first."
    assert args.lambda_bpec >= 0
    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    helpers = load_lcdr_helpers(args.repo)

    manifest = helpers.prepare_manifest(
        args.splits_csv,
        args.output / "_development_manifest.csv"
    )

    # Use the historical mask-cache metadata and cohort construction.
    segmenter = FrozenLungSegmenter(
        args.segmentation_checkpoint, device
    )
    cache = MaskCache(args.mask_cache, create=True)
    cache.validate_or_initialise_metadata(
        helpers.cache_metadata(segmenter, args.image_root)
    )

    def make_dataset(split, with_segmenter):
        return CheXpertPneumoniaDataset(
            dataset_root=args.image_root,
            manifest_path=manifest,
            split=split,
            transform=(
                helpers.make_train_transform()
                if split == "train"
                else helpers.make_validation_transform()
            ),
            input_mode=InputMode.ORIGINAL,
            lung_segmenter=segmenter if with_segmenter else None,
            mask_cache=cache,
            classifier_image_size=None,
            return_lung_probability=True,
        )

    train_set = make_dataset("train", True)
    val_set = make_dataset("validation", True)

    paths = (
        list(train_set._resolved_image_paths)
        + list(val_set._resolved_image_paths)
    )
    missing = sum(
        cache.get_for_source(p) is None for p in paths
    )
    print("Missing lung maps:", missing, flush=True)

    if missing:
        with torch.inference_mode():
            for name, dataset in [
                ("train", train_set), ("validation", val_set)
            ]:
                for i in range(len(dataset)):
                    _ = dataset[i]
                    if (i + 1) % 500 == 0:
                        print(
                            f"Cache {name}: {i+1}/{len(dataset)}",
                            flush=True
                        )

    cache.require_coverage(paths)

    del train_set, val_set, segmenter
    torch.cuda.empty_cache()

    train_set = make_dataset("train", False)
    val_set = make_dataset("validation", False)

    # Ensure all targets are binary after ignore strategy.
    for dataset in (train_set, val_set):
        targets = np.asarray(dataset._targets, dtype=float)
        assert np.isin(targets, [0.0, 1.0]).all(), (
            "Non-binary targets remain after label strategy"
        )

    generator = torch.Generator().manual_seed(args.seed)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    model = resnet50(
        weights=ResNet50_Weights.IMAGENET1K_V2
    )
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    positive = float((train_set._targets == 1).sum())
    negative = float((train_set._targets == 0).sum())
    pos_weight = torch.tensor(
        [negative / positive], device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    backbone_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if name.startswith("fc."):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 1e-5},
            {"params": head_params, "lr": 1e-5},
        ],
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max"
    )

    history = []
    best_auroc = -float("inf")
    best_epoch = 0

    config = {
        "architecture": "ResNet50_shortcut_training",
        "initialization": "IMAGENET1K_V2",
        "seed": args.seed,
        "lambda_bpec": args.lambda_bpec,
        "shortcut_region": args.shortcut_region,
        "shortcut_correlation": args.shortcut_correlation,
        "shortcut_size": args.shortcut_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "backbone_lr": 1e-5,
        "head_lr": 1e-5,
        "weight_decay": 1e-5,
        "patch_size": 8,
        "patches_per_image": 2,
        "boundary_radius": 12,
        "classification_loss": "weighted_BCE",
        "selection": "CheXpert validation AUROC",
        "threshold": 0.5,
        "external_labels_used": False,
    }
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2)
    )

    print("\n=== BPEC TRAINING PILOT ===", flush=True)
    print("Train:", len(train_set), flush=True)
    print("Validation:", len(val_set), flush=True)
    print("Positive weight:", pos_weight.item(), flush=True)
    print("Lambda:", args.lambda_bpec, flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_cls = total_cons = total_loss = 0.0
        selected_pixels = 0
        start = time.time()

        for batch in train_loader:
            images = batch["image"].to(device)
            lung = batch["lung_probability"].to(device)
            labels = batch["target"].to(device).float()

            images, lung = helpers.paired_random_rotation(
                images, lung, degrees=7.0
            )

            # Apply deterministic label-correlated markers.
            # Convert ImageNet-normalized tensors to uint8 RGB.
            mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            raw = ((images * std + mean).clamp(0, 1) * 255).round()
            marked = []
            for j in range(len(images)):
                rgb = raw[j].byte().permute(1, 2, 0).cpu().numpy()
                mask = lung[j].detach().cpu().numpy()
                changed, meta = shortcut_image(
                    rgb, mask, int(labels[j].item()),
                    str(batch["image_path"][j]), args.shortcut_region, "train",
                    correlation=args.shortcut_correlation,
                    size=args.shortcut_size
                )
                if meta["marker"] == "unavailable":
                    raise RuntimeError("Marker placement failed")
                marked.append(
                    torch.from_numpy(changed.copy()).permute(2, 0, 1)
                )
            images = torch.stack(marked).to(device).float() / 255
            images = (images - mean) / std
            selected = torch.zeros_like(images[:, :1])

            optimizer.zero_grad(set_to_none=True)

            logits = model(images).flatten()
            cls_loss = criterion(logits, labels)

            if False:
                perturbed_logits = model(perturbed).flatten()
                consistency = bpec_consistency_loss(
                    logits, perturbed_logits
                )
            else:
                consistency = logits.new_zeros(())

            loss = (
                cls_loss + args.lambda_bpec * consistency
            )
            loss.backward()
            optimizer.step()

            total_cls += float(cls_loss.detach())
            total_cons += float(consistency.detach())
            total_loss += float(loss.detach())
            selected_pixels += int(selected.sum().item())

        validation = evaluate(model, val_loader, device)
        scores = metrics(
            validation["label"],
            validation["probability"],
            threshold=0.5,
        )

        n = len(train_loader)
        record = {
            "epoch": epoch,
            "train_classification_loss": total_cls / n,
            "train_consistency_loss": total_cons / n,
            "train_total_loss": total_loss / n,
            "selected_pixels": selected_pixels,
            "seconds": time.time() - start,
            **{f"validation_{k}": v for k, v in scores.items()},
        }
        history.append(record)

        pd.DataFrame(history).to_csv(
            args.output / "training_history.csv", index=False
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"class={record['train_classification_loss']:.5f} | "
            f"cons={record['train_consistency_loss']:.5f} | "
            f"AUROC={scores['auroc']:.5f} | "
            f"AUPRC={scores['auprc']:.5f} | "
            f"F1@0.5={scores['f1']:.5f}",
            flush=True,
        )

        scheduler.step(scores["auroc"])

        if scores["auroc"] > best_auroc:
            best_auroc = scores["auroc"]
            best_epoch = epoch

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_auroc": best_auroc,
                "architecture": "resnet50",
                "training_method": "artificial_shortcut",
                "lambda_bpec": args.lambda_bpec,
                "seed": args.seed,
            }, args.output / "best_checkpoint.pt")

            validation.to_csv(
                args.output / "validation_predictions.csv",
                index=False,
            )
            (args.output / "best_metrics.json").write_text(
                json.dumps({
                    "best_epoch": epoch,
                    **scores
                }, indent=2)
            )

    print("\n=== PILOT COMPLETE ===")
    print("Best epoch:", best_epoch)
    print("Best validation AUROC:", best_auroc)
    print("Results:", args.output)
    print("Checkpoint:", args.output / "best_checkpoint.pt")


if __name__ == "__main__":
    main()
