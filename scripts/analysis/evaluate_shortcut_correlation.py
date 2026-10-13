from pneumonia_ai.data.chexpert_dataset import CheXpertPneumoniaDataset
from pneumonia_ai.classification.segmentation_guided import InputMode
from pneumonia_ai.segmentation.inference import FrozenLungSegmenter
from pneumonia_ai.segmentation.cache import MaskCache


import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score
)


sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)
from contrast_controlled_shortcut import (
    contrast_controlled_shortcut
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--mask-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Import existing training helpers.
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
        "results/annotation_bias_audit/matched_cohort/"
        "matched_development_manifest.csv"
    )

    segmenter = FrozenLungSegmenter(
        args.segmentation_checkpoint, device
    )
    cache = MaskCache(args.mask_cache, create=True)
    cache.validate_or_initialise_metadata(
        helpers.cache_metadata(segmenter, args.image_root)
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

    assert len(dataset) == 679, len(dataset)

    loader = DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=0
    )

    rows = []

    for region in ("interior", "boundary", "exterior"):
        checkpoint = (
            args.repo /
            "results/annotation_bias_audit/"
            "correlation_075_training" /
            f"{region}_seed42/best_checkpoint.pt"
        )

        model = resnet50(weights=None)
        model.fc = torch.nn.Linear(
            model.fc.in_features, 1
        )

        state = torch.load(
            checkpoint, map_location="cpu",
            weights_only=False
        )
        model.load_state_dict(state["model_state_dict"])
        model = model.to(device).eval()

        for condition in ("clean", "matching", "reversed"):
            with torch.inference_mode():
                for batch in loader:
                    images = batch["image"].numpy()
                    masks = batch["lung_probability"].numpy()
                    labels = batch["target"].numpy()

                    modified = []

                    for i in range(len(images)):
                        # Undo ImageNet normalization.
                        mean = np.array(
                            [0.485, 0.456, 0.406]
                        )[:, None, None]
                        std = np.array(
                            [0.229, 0.224, 0.225]
                        )[:, None, None]

                        rgb = np.clip(
                            (images[i] * std + mean) * 255,
                            0, 255
                        ).round().astype(np.uint8)

                        rgb = np.transpose(rgb, (1, 2, 0))

                        altered, _ = contrast_controlled_shortcut(
                            rgb,
                            masks[i],
                            int(labels[i]),
                            str(batch["image_path"][i]),
                            region,
                            condition,
                            correlation=0.75,
                            size=24,
                            amplitude=45,
                        )

                        tensor = (
                            altered.astype(np.float32) / 255
                        )
                        tensor = np.transpose(
                            tensor, (2, 0, 1)
                        )
                        tensor = (
                            tensor - mean
                        ) / std

                        modified.append(tensor)

                    x = torch.tensor(
                        np.stack(modified),
                        dtype=torch.float32,
                        device=device
                    )

                    probs = torch.sigmoid(
                        model(x).flatten()
                    ).cpu().numpy()

                    for i, probability in enumerate(probs):
                        rows.append({
                            "model": region,
                            "condition": condition,
                            "patient_id": str(
                                batch["patient_id"][i]
                            ),
                            "image_path": str(
                                batch["image_path"][i]
                            ),
                            "label": int(labels[i]),
                            "probability": float(probability),
                        })

            print(
                f"{region}: {condition} complete",
                flush=True
            )

    args.output.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(rows)

    predictions.to_csv(
        args.output / "paired_predictions.csv",
        index=False
    )

    summary = []

    for (region, condition), group in predictions.groupby(
        ["model", "condition"]
    ):
        y = group["label"].to_numpy()
        p = group["probability"].to_numpy()

        summary.append({
            "model": region,
            "condition": condition,
            "n": len(group),
            "AUROC": roc_auc_score(y, p),
            "AUPRC": average_precision_score(y, p),
            "F1_at_0.5": f1_score(
                y, p >= 0.5, zero_division=0
            ),
        })

    pd.DataFrame(summary).to_csv(
        args.output / "performance_summary.csv",
        index=False
    )

    effects = []

    for region, group in predictions.groupby("model"):
        wide = group.pivot(
            index="image_path",
            columns="condition",
            values="probability"
        )

        perf = pd.DataFrame(summary)
        perf = perf[perf["model"] == region]

        matching_auc = float(
            perf.loc[
                perf["condition"] == "matching",
                "AUROC"
            ].iloc[0]
        )
        reversed_auc = float(
            perf.loc[
                perf["condition"] == "reversed",
                "AUROC"
            ].iloc[0]
        )

        effects.append({
            "model": region,
            "AUROC_matching_minus_reversed":
                matching_auc - reversed_auc,
            "mean_absolute_probability_change":
                float(
                    np.abs(
                        wide["matching"] - wide["reversed"]
                    ).mean()
                ),
            "decision_flip_rate":
                float(
                    (
                        (wide["matching"] >= 0.5)
                        != (wide["reversed"] >= 0.5)
                    ).mean()
                ),
        })

    pd.DataFrame(effects).to_csv(
        args.output / "shortcut_effects.csv",
        index=False
    )

    print("\n=== PERFORMANCE SUMMARY ===")
    print(pd.DataFrame(summary).to_string(index=False))

    print("\n=== SHORTCUT EFFECTS ===")
    print(pd.DataFrame(effects).to_string(index=False))


if __name__ == "__main__":
    main()
