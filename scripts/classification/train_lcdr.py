"""Train LCDR on the CheXpert development cohort.

Frozen recovery protocol:
- ResNet50 / ImageNet weights
- full CXR classifier input
- layer2 context representation
- linear 512->1 context adversary
- GRL alpha = 0.03
- weighted BCE for both classifier and adversary
- P7 classifier recipe otherwise
- selection by CheXpert validation AUROC only
- no external data used for selection
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.classification.segmentation_guided import InputMode
from pneumonia_ai.data.chexpert_dataset import CheXpertPneumoniaDataset
from pneumonia_ai.data.label_strategy import apply_label_strategy
from pneumonia_ai.models.lcdr import LCDRResNet50
from pneumonia_ai.segmentation.cache import MaskCache
from pneumonia_ai.segmentation.inference import FrozenLungSegmenter
from pneumonia_ai.training.seed import seed_everything, seed_worker
from scripts.classification.optimisation_thresholds import threshold_analysis


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--splits-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)

    parser.add_argument(
        "--segmentation-checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument("--mask-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--head-lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--adversary-lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--grl-alpha",
        type=float,
        default=0.03,
    )

    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
    )

    return parser.parse_args()



def paired_random_rotation(
    image: torch.Tensor,
    lung_probability: torch.Tensor,
    *,
    degrees: float = 7.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate image and lung map by one identical random angle.

    The full radiograph and continuous frozen U-Net probability
    map must remain spatially aligned during LCDR training.

    The image uses bilinear interpolation. The lung probability
    map is continuous, so it also uses bilinear interpolation
    rather than nearest-neighbour interpolation.
    """

    angle = float(
        torch.empty(1).uniform_(
            -float(degrees),
            float(degrees),
        ).item()
    )

    image = TF.rotate(
        image,
        angle=angle,
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    )

    lung_probability = TF.rotate(
        lung_probability,
        angle=angle,
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    ).clamp(0.0, 1.0)

    return image, lung_probability


def make_train_transform():
    # IMPORTANT:
    # Geometric rotation is NOT performed here.
    #
    # LCDR requires the image and its frozen lung-probability
    # map to remain anatomically aligned. Paired rotation is
    # therefore applied jointly in the training loop after
    # both tensors have been loaded.
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                IMAGENET_MEAN,
                IMAGENET_STD,
            ),
        ]
    )


def make_validation_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                IMAGENET_MEAN,
                IMAGENET_STD,
            ),
        ]
    )


def prepare_manifest(
    source: Path,
    destination: Path,
) -> Path:

    frame = pd.read_csv(source)

    development = frame.loc[
        frame["split"].isin(
            ["train", "validation"]
        )
    ].copy()

    train = apply_label_strategy(
        development.loc[
            development["split"] == "train"
        ],
        "ignore",
    )

    validation = apply_label_strategy(
        development.loc[
            development["split"] == "validation"
        ],
        "ignore",
    )

    prepared = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared.to_csv(
        destination,
        index=False,
    )

    return destination


def subset_dataset(dataset, limit):
    if limit is None:
        return

    dataset.records = (
        dataset.records.iloc[:limit]
        .reset_index(drop=True)
    )

    dataset._raw_labels = (
        dataset._raw_labels.iloc[:limit]
        .reset_index(drop=True)
    )

    dataset._targets = (
        dataset._targets.iloc[:limit]
        .reset_index(drop=True)
    )

    dataset._sample_loss_weights = (
        dataset._sample_loss_weights.iloc[:limit]
        .reset_index(drop=True)
    )

    dataset._resolved_image_paths = (
        dataset._resolved_image_paths[:limit]
    )

    dataset._returned_image_paths = (
        dataset._returned_image_paths[:limit]
    )


def cache_metadata(
    segmenter,
    image_root: Path,
):
    return {
        "schema_version": 1,
        "image_root": str(image_root.resolve()),
        "segmentation_checkpoint":
            str(segmenter.checkpoint_path.resolve()),
        "segmentation_checkpoint_sha256":
            file_sha256(segmenter.checkpoint_path),
        "segmentation_architecture":
            type(segmenter.model).__name__,
        "segmentation_input_size":
            segmenter.image_size,
        "mask_threshold": 0.5,
        "mask_postprocessing": "none",
        "classifier_masking_mode":
            "lcdr_probability_guidance",
    }


def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    rows = []

    with torch.inference_mode():

        for batch in loader:

            image = batch["image"].to(device)

            logits = model(image)

            probabilities = (
                torch.sigmoid(logits)
                .cpu()
                .numpy()
            )

            for i, probability in enumerate(
                probabilities
            ):
                rows.append(
                    {
                        "patient_id":
                            batch["patient_id"][i],
                        "study_id":
                            batch["study_id"][i],
                        "image_path":
                            batch["image_path"][i],
                        "label":
                            int(batch["target"][i]),
                        "probability":
                            float(probability),
                        "split":
                            "validation",
                    }
                )

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    if args.grl_alpha <= 0:
        raise ValueError(
            "LCDR requires grl_alpha > 0."
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_everything(args.seed)

    device = torch.device(args.device)

    manifest = prepare_manifest(
        args.splits_csv,
        args.output / "_development_manifest.csv",
    )

    # --------------------------------------------------------
    # Lung-probability cache preparation
    # --------------------------------------------------------

    segmenter = FrozenLungSegmenter(
        args.segmentation_checkpoint,
        args.device,
    )

    mask_cache = MaskCache(
        args.mask_cache,
        create=True,
    )

    mask_cache.validate_or_initialise_metadata(
        cache_metadata(
            segmenter,
            args.image_root,
        )
    )

    train_set = CheXpertPneumoniaDataset(
        dataset_root=args.image_root,
        manifest_path=manifest,
        split="train",
        transform=make_train_transform(),
        input_mode=InputMode.ORIGINAL,
        lung_segmenter=segmenter,
        mask_cache=mask_cache,
        classifier_image_size=None,
        return_lung_probability=True,
    )

    validation_set = CheXpertPneumoniaDataset(
        dataset_root=args.image_root,
        manifest_path=manifest,
        split="validation",
        transform=make_validation_transform(),
        input_mode=InputMode.ORIGINAL,
        lung_segmenter=segmenter,
        mask_cache=mask_cache,
        classifier_image_size=None,
        return_lung_probability=True,
    )

    subset_dataset(
        train_set,
        args.max_train_samples,
    )

    subset_dataset(
        validation_set,
        args.max_validation_samples,
    )

    required_paths = [
        *train_set._resolved_image_paths,
        *validation_set._resolved_image_paths,
    ]

    missing = sum(
        mask_cache.get_for_source(path) is None
        for path in required_paths
    )

    print(
        "Missing lung maps:",
        missing,
        flush=True,
    )

    if missing:

        with torch.inference_mode():

            for name, dataset in (
                ("train", train_set),
                ("validation", validation_set),
            ):

                for index in range(len(dataset)):
                    _ = dataset[index]

                    if (
                        (index + 1) % 500 == 0
                        or index + 1 == len(dataset)
                    ):
                        print(
                            f"Cache {name}: "
                            f"{index + 1}/{len(dataset)}",
                            flush=True,
                        )

    mask_cache.require_coverage(
        required_paths
    )

    # --------------------------------------------------------
    # Rebuild cache-only datasets.
    # No live U-Net during classifier training.
    # --------------------------------------------------------

    del train_set
    del validation_set
    del segmenter

    torch.cuda.empty_cache()

    train_set = CheXpertPneumoniaDataset(
        dataset_root=args.image_root,
        manifest_path=manifest,
        split="train",
        transform=make_train_transform(),
        input_mode=InputMode.ORIGINAL,
        lung_segmenter=None,
        mask_cache=mask_cache,
        classifier_image_size=None,
        return_lung_probability=True,
    )

    validation_set = CheXpertPneumoniaDataset(
        dataset_root=args.image_root,
        manifest_path=manifest,
        split="validation",
        transform=make_validation_transform(),
        input_mode=InputMode.ORIGINAL,
        lung_segmenter=None,
        mask_cache=mask_cache,
        classifier_image_size=None,
        return_lung_probability=True,
    )

    subset_dataset(
        train_set,
        args.max_train_samples,
    )

    subset_dataset(
        validation_set,
        args.max_validation_samples,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
    )

    model = LCDRResNet50(
        pretrained=True,
        grl_alpha=args.grl_alpha,
    ).to(device)

    positive = float(
        (train_set._targets == 1).sum()
    )

    negative = float(
        (train_set._targets == 0).sum()
    )

    pos_weight = torch.tensor(
        [negative / positive],
        dtype=torch.float32,
        device=device,
    )

    classification_criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    )

    context_criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    )

    # --------------------------------------------------------
    # Separate optimizer groups.
    # Same LR in frozen recovery recipe.
    # --------------------------------------------------------

    backbone_parameters = []
    classifier_parameters = []

    for name, parameter in (
        model.backbone.named_parameters()
    ):
        if name.startswith("fc."):
            classifier_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": args.backbone_lr,
            },
            {
                "params": classifier_parameters,
                "lr": args.head_lr,
            },
            {
                "params":
                    model.context_adversary.parameters(),
                "lr": args.adversary_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
        )
    )

    best_auroc = -np.inf
    best_epoch = 0
    stalled = 0

    history = []

    print("\n=== LCDR RECOVERY TRAINING ===")
    print("Device:", device)
    print("Seed:", args.seed)
    print("GRL alpha:", args.grl_alpha)
    print("Feature layer: layer2")
    print("Train samples:", len(train_set))
    print("Validation samples:", len(validation_set))
    print("Positive weight:", pos_weight.item())

    start_time = time.time()

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        running_total = 0.0
        running_classification = 0.0
        running_context = 0.0

        context_targets = []
        context_probabilities = []

        for batch in train_loader:

            image = batch["image"].to(device)

            lung_probability = (
                batch["lung_probability"]
                .to(device)
            )

            # ------------------------------------------------
            # Paired geometric augmentation
            #
            # The same sampled angle MUST be applied to the
            # radiograph and continuous lung-probability map.
            # This fixes the historical image/mask alignment
            # problem while preserving the P7 ±7° recipe.
            # ------------------------------------------------
            image, lung_probability = paired_random_rotation(
                image,
                lung_probability,
                degrees=7.0,
            )

            target = batch["target"].to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(
                image,
                lung_probability,
                return_aux=True,
            )

            classification_loss = (
                classification_criterion(
                    output["logit"],
                    target,
                )
            )

            context_loss = (
                context_criterion(
                    output["context_logit"],
                    target,
                )
            )

            # IMPORTANT:
            # Do NOT multiply context loss by alpha.
            # GRL already scales/reverses only the
            # backbone-side context gradient.
            loss = (
                classification_loss
                + context_loss
            )

            loss.backward()
            optimizer.step()

            running_total += float(
                loss.item()
            )

            running_classification += float(
                classification_loss.item()
            )

            running_context += float(
                context_loss.item()
            )

            context_targets.extend(
                target.detach()
                .cpu()
                .numpy()
                .tolist()
            )

            context_probabilities.extend(
                torch.sigmoid(
                    output["context_logit"]
                )
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

        validation = evaluate(
            model,
            validation_loader,
            device,
        )

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

        context_auroc = float(
            roc_auc_score(
                context_targets,
                context_probabilities,
            )
        )

        n_batches = len(train_loader)

        record = {
            "epoch": epoch,
            "train_loss":
                running_total / n_batches,
            "train_classification_loss":
                running_classification / n_batches,
            "train_context_adversary_loss":
                running_context / n_batches,
            "train_context_adversary_auroc":
                context_auroc,
            "validation_auroc":
                auroc,
            "validation_pr_auc":
                pr_auc,
        }

        history.append(record)

        pd.DataFrame(history).to_csv(
            args.output /
            "training_history.csv",
            index=False,
        )

        print(
            f"Epoch {epoch:02d}: "
            f"class={record['train_classification_loss']:.6f} "
            f"context={record['train_context_adversary_loss']:.6f} "
            f"context_auc={context_auroc:.6f} "
            f"val_auc={auroc:.6f} "
            f"val_pr={pr_auc:.6f}",
            flush=True,
        )

        scheduler.step(auroc)

        if auroc > best_auroc:

            best_auroc = auroc
            best_epoch = epoch
            stalled = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "epoch":
                        epoch,
                    "best_auroc":
                        best_auroc,
                    "seed":
                        args.seed,
                    "grl_alpha":
                        args.grl_alpha,
                    "feature_layer":
                        "layer2",
                    "architecture":
                        "LCDRResNet50",
                    "inference_mode":
                        "ordinary_resnet50_mask_free",
                },
                args.output /
                "best_checkpoint.pt",
            )

            validation.to_csv(
                args.output /
                "validation_predictions.csv",
                index=False,
            )

        else:
            stalled += 1

        if stalled >= args.patience:
            print(
                "Early stopping.",
                flush=True,
            )
            break

    elapsed = time.time() - start_time

    state = torch.load(
        args.output / "best_checkpoint.pt",
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        state["model_state_dict"]
    )

    validation = evaluate(
        model,
        validation_loader,
        device,
    )

    final_auroc = float(
        roc_auc_score(
            validation["label"],
            validation["probability"],
        )
    )

    final_pr_auc = float(
        average_precision_score(
            validation["label"],
            validation["probability"],
        )
    )

    threshold, analysis = threshold_analysis(
        validation["label"].to_numpy(),
        validation["probability"].to_numpy(),
        "max_f1",
    )

    validation.to_csv(
        args.output /
        "validation_predictions.csv",
        index=False,
    )

    analysis.to_csv(
        args.output /
        "threshold_analysis.csv",
        index=False,
    )

    summary = {
        "status": "completed",
        "recovery_retrain": True,
        "architecture": "LCDRResNet50",
        "seed": args.seed,
        "grl_alpha": args.grl_alpha,
        "feature_layer": "layer2",
        "best_epoch": best_epoch,
        "validation_auroc": final_auroc,
        "validation_pr_auc": final_pr_auc,
        "max_f1_threshold":
            float(threshold),
        "train_samples":
            len(train_set),
        "validation_samples":
            len(validation_set),
        "elapsed_seconds":
            elapsed,
        "external_data_used_for_selection":
            False,
        "inference":
            "ordinary_resnet50_mask_free",
        "segmentation_checkpoint":
            str(args.segmentation_checkpoint),
        "segmentation_checkpoint_sha256":
            file_sha256(
                args.segmentation_checkpoint
            ),
    }

    (
        args.output /
        "run_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== LCDR RECOVERY COMPLETE ==="
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
