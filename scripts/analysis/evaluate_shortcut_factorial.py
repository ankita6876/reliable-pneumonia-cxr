
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
)

from pneumonia_ai.data.chexpert_dataset import (
    CheXpertPneumoniaDataset
)
from pneumonia_ai.classification.segmentation_guided import (
    InputMode
)
from pneumonia_ai.segmentation.cache import MaskCache
from pneumonia_ai.segmentation.inference import (
    FrozenLungSegmenter
)

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)

from contrast_controlled_shortcut import (
    contrast_controlled_shortcut
)


REGIONS = ("interior", "boundary", "exterior")
SEEDS = (42, 123, 2026)
CORRELATIONS = (0.50, 0.75, 1.00)
CONDITIONS = ("clean", "matching", "reversed")


def checkpoint_path(repo, correlation, region, seed):
    root = repo / "results/annotation_bias_audit"

    if correlation == 0.75:
        folder = root / "correlation_075_training"
    elif correlation == 0.50:
        folder = root / "correlation_500_training"
    elif correlation == 1.00:
        folder = (
            root / "contrast_controlled_training"
            if seed == 42
            else root / "correlation_1000_training"
        )

    return (
        folder /
        f"{region}_seed{seed}/best_checkpoint.pt"
    )


def compute_metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    return {
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
        "F1_at_0.5": f1_score(
            y, p >= 0.5, zero_division=0
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--mask-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(
        0, str(args.repo / "scripts/classification")
    )
    from train_shortcut_contrast_controlled import (
        load_lcdr_helpers
    )

    helpers = load_lcdr_helpers(args.repo)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    manifest = (
        args.repo /
        "results/annotation_bias_audit/"
        "matched_cohort/matched_development_manifest.csv"
    )

    segmenter = FrozenLungSegmenter(
        args.segmentation_checkpoint, device
    )

    cache = MaskCache(args.mask_cache, create=True)
    cache.validate_or_initialise_metadata(
        helpers.cache_metadata(
            segmenter, args.image_root
        )
    )

    dataset = CheXpertPneumoniaDataset(
        dataset_root=args.image_root,
        manifest_path=manifest,
        split="validation",
        transform=helpers.make_validation_transform(),
        input_mode=InputMode.ORIGINAL,
        lung_segmenter=None,
        mask_cache=cache,
        classifier_image_size=None,
        return_lung_probability=True,
    )

    assert len(dataset) == 679

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    args.output.mkdir(parents=True, exist_ok=True)

    # Store one output per model, so an interrupted run
    # can resume without repeating completed inference.
    model_dir = args.output / "model_predictions"
    model_dir.mkdir(exist_ok=True)

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )[:, None, None]

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )[:, None, None]

    for correlation in CORRELATIONS:
        for seed in SEEDS:
            for region in REGIONS:

                identifier = (
                    f"c{int(correlation*100):03d}_"
                    f"{region}_seed{seed}"
                )

                output_csv = (
                    model_dir / f"{identifier}.csv"
                )

                if output_csv.exists():
                    previous = pd.read_csv(output_csv)

                    coverage = previous.groupby(
                        "condition"
                    ).size().to_dict()

                    if (
                        len(previous) == 679 * 3
                        and all(
                            coverage.get(c) == 679
                            for c in CONDITIONS
                        )
                    ):
                        print(
                            "SKIP:", identifier,
                            flush=True
                        )
                        continue

                checkpoint = checkpoint_path(
                    args.repo,
                    correlation,
                    region,
                    seed
                )

                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)

                print(
                    "\nEVALUATING:", identifier,
                    flush=True
                )

                model = resnet50(weights=None)
                model.fc = torch.nn.Linear(
                    model.fc.in_features, 1
                )

                state = torch.load(
                    checkpoint,
                    map_location="cpu",
                    weights_only=False
                )

                model.load_state_dict(
                    state["model_state_dict"]
                )

                model = model.to(device).eval()

                rows = []

                for condition in CONDITIONS:
                    with torch.inference_mode():
                        for batch in loader:
                            images = batch["image"].numpy()
                            masks = (
                                batch["lung_probability"].numpy()
                            )
                            labels = batch["target"].numpy()

                            modified = []

                            for i in range(len(images)):
                                rgb = np.clip(
                                    (
                                        images[i] * std + mean
                                    ) * 255,
                                    0, 255
                                ).round().astype(np.uint8)

                                rgb = np.transpose(
                                    rgb, (1, 2, 0)
                                )

                                altered, _ = (
                                    contrast_controlled_shortcut(
                                        rgb,
                                        masks[i],
                                        int(labels[i]),
                                        str(
                                            batch["image_path"][i]
                                        ),
                                        region,
                                        condition,
                                        correlation=correlation,
                                        size=24,
                                        amplitude=45,
                                    )
                                )

                                x = (
                                    altered.astype(np.float32)
                                    / 255
                                )
                                x = np.transpose(
                                    x, (2, 0, 1)
                                )
                                x = (x - mean) / std
                                modified.append(x)

                            x = torch.tensor(
                                np.stack(modified),
                                dtype=torch.float32,
                                device=device,
                            )

                            probabilities = torch.sigmoid(
                                model(x).flatten()
                            ).cpu().numpy()

                            for i, p in enumerate(
                                probabilities
                            ):
                                rows.append({
                                    "correlation": correlation,
                                    "region": region,
                                    "seed": seed,
                                    "condition": condition,
                                    "patient_id": str(
                                        batch["patient_id"][i]
                                    ),
                                    "image_path": str(
                                        batch["image_path"][i]
                                    ),
                                    "label": int(labels[i]),
                                    "probability": float(p),
                                })

                    print(
                        "  completed:", condition,
                        flush=True
                    )

                frame = pd.DataFrame(rows)

                assert len(frame) == 679 * 3

                frame.to_csv(
                    output_csv,
                    index=False
                )

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # =====================================================
    # AGGREGATE RESULTS
    # =====================================================

    predictions = pd.concat(
        [
            pd.read_csv(p)
            for p in sorted(
                model_dir.glob("c*_seed*.csv")
            )
        ],
        ignore_index=True
    )

    assert len(predictions) == 27 * 679 * 3

    predictions.to_csv(
        args.output / "all_paired_predictions.csv",
        index=False
    )

    performance = []

    for keys, group in predictions.groupby(
        ["correlation", "region", "seed", "condition"]
    ):
        performance.append({
            "correlation": keys[0],
            "region": keys[1],
            "seed": keys[2],
            "condition": keys[3],
            "n": len(group),
            **compute_metrics(
                group["label"],
                group["probability"]
            )
        })

    performance = pd.DataFrame(performance)
    performance.to_csv(
        args.output / "performance_summary.csv",
        index=False
    )

    effects = []

    for keys, group in predictions.groupby(
        ["correlation", "region", "seed"]
    ):
        wide = group.pivot(
            index="image_path",
            columns="condition",
            values="probability"
        )

        matching = group[
            group["condition"] == "matching"
        ].set_index("image_path")

        reversed_ = group[
            group["condition"] == "reversed"
        ].set_index("image_path")

        matching_auc = roc_auc_score(
            matching["label"],
            matching["probability"]
        )
        reversed_auc = roc_auc_score(
            reversed_["label"],
            reversed_["probability"]
        )

        effects.append({
            "correlation": keys[0],
            "region": keys[1],
            "seed": keys[2],
            "AUROC_gap":
                matching_auc - reversed_auc,
            "mean_absolute_probability_change":
                np.abs(
                    wide["matching"] -
                    wide["reversed"]
                ).mean(),
            "decision_flip_rate":
                (
                    (wide["matching"] >= 0.5)
                    != (wide["reversed"] >= 0.5)
                ).mean(),
        })

    effects = pd.DataFrame(effects)

    effects.to_csv(
        args.output / "shortcut_effects.csv",
        index=False
    )

    seed_summary = effects.groupby(
        ["correlation", "region"]
    ).agg(
        AUROC_gap_mean=("AUROC_gap", "mean"),
        AUROC_gap_sd=("AUROC_gap", "std"),
        probability_change_mean=(
            "mean_absolute_probability_change",
            "mean"
        ),
        decision_flip_mean=(
            "decision_flip_rate",
            "mean"
        ),
    ).reset_index()

    seed_summary.to_csv(
        args.output / "seed_summary.csv",
        index=False
    )

    print("\n=== SEED-AVERAGED SHORTCUT EFFECTS ===")
    print(
        seed_summary.round(4).to_string(
            index=False
        )
    )

    print("\nSaved results to:", args.output)


if __name__ == "__main__":
    main()
