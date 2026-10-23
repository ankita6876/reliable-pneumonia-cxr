
import gc
import json
import random
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score
)

from tqdm.auto import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("/kaggle/working/reliable-pneumonia-cxr")
EXP = ROOT / "experiments/oxford_pet_spatial"

OUT = EXP / "coordinate_counterfactual"
RESULTS = OUT / "results"
FIGURES = OUT / "figures"
CONFIGS = OUT / "configs"

for folder in [RESULTS, FIGURES, CONFIGS]:
    folder.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path(
    "/kaggle/input/datasets/"
    "julinmaloof/the-oxfordiiit-pet-dataset"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

assert DEVICE.type == "cuda", "Enable Kaggle GPU."

SEEDS = [42, 123, 2026]

MODELS = [
    "clean",
    "interior",
    "background",
    "center",
    "bottom_right"
]

POSITIONS = [
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
    "center",
    "interior",
    "background"
]

IMAGE_SIZE = 224
PATCH = 24
BATCH_SIZE = 32
BOOTSTRAP_REPS = 2000

torch.backends.cudnn.benchmark = True

print("Device:", DEVICE)

# ============================================================
# LOAD LOCKED VALIDATION COHORT
# ============================================================

manifest = pd.read_csv(
    EXP /
    "multiseed_spatial/results/"
    "locked_validation_manifest.csv"
)

assert len(manifest) == 722
assert manifest.image_id.is_unique

manifest["image_path"] = manifest.image_id.apply(
    lambda image_id: str(
        DATA_ROOT / "images" / f"{image_id}.jpg"
    )
)

manifest["mask_path"] = manifest.image_id.apply(
    lambda image_id: str(
        DATA_ROOT / "annotations/trimaps" /
        f"{image_id}.png"
    )
)

ABS_POSITIONS = {
    "top_left": (8, 8),
    "top_right": (192, 8),
    "bottom_left": (8, 192),
    "bottom_right": (192, 192),
    "center": (100, 100)
}

for name, (x, y) in ABS_POSITIONS.items():
    manifest[f"{name}_x"] = x
    manifest[f"{name}_y"] = y

for position in POSITIONS:
    assert f"{position}_x" in manifest.columns
    assert f"{position}_y" in manifest.columns

    assert (
        manifest[f"{position}_x"]
        .between(0, IMAGE_SIZE - PATCH)
    ).all()

    assert (
        manifest[f"{position}_y"]
        .between(0, IMAGE_SIZE - PATCH)
    ).all()

manifest.to_csv(
    RESULTS / "locked_validation_manifest.csv",
    index=False
)

# ============================================================
# PRECOMPUTE IMAGES AND MASKS
# ============================================================

print("Caching 722 resized images in RAM...")

images = []
masks = []

for _, row in tqdm(
    manifest.iterrows(),
    total=len(manifest)
):
    with Image.open(row.image_path) as im:
        image = np.asarray(
            im.convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            )
        ).copy()

    with Image.open(row.mask_path) as im:
        mask = np.asarray(
            im.resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.NEAREST
            )
        ).copy()

    images.append(image)
    masks.append(mask)

images = np.stack(images)
masks = np.stack(masks)

labels = manifest.label.to_numpy(dtype=np.int64)
image_ids = manifest.image_id.astype(str).to_numpy()

print("Images:", images.shape)
print("Masks:", masks.shape)

# ============================================================
# MARKER PATTERNS
# ============================================================

yy, xx = np.indices((PATCH, PATCH))

PATCHES = {}

for marker_type in [0, 1]:
    stripes = (
        (xx // 3) % 2
        if marker_type == 0
        else (yy // 3) % 2
    )

    gray = np.where(
        stripes == 1,
        240,
        15
    ).astype(np.uint8)

    PATCHES[marker_type] = np.repeat(
        gray[..., None],
        3,
        axis=2
    )

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
).reshape(3, 1, 1)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
).reshape(3, 1, 1)

# ============================================================
# DATASET
# ============================================================

class CounterfactualDataset(Dataset):

    def __init__(self, position, marker_type):
        self.position = position
        self.marker_type = marker_type

    def __len__(self):
        return len(labels)

    def __getitem__(self, index):
        image = images[index].copy()

        if self.position != "clean":
            row = manifest.iloc[index]

            x = int(
                row[f"{self.position}_x"]
            )

            y = int(
                row[f"{self.position}_y"]
            )

            image[
                y:y+PATCH,
                x:x+PATCH
            ] = PATCHES[self.marker_type]

        array = (
            image.astype(np.float32)
            .transpose(2, 0, 1) / 255.0
        )

        array = (array - MEAN) / STD

        return (
            torch.from_numpy(array.copy()),
            index
        )

def loader(position, marker_type):
    return DataLoader(
        CounterfactualDataset(
            position, marker_type
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

# ============================================================
# CHECKPOINT DISCOVERY
# ============================================================

def checkpoint_path(model_name, seed):

    if seed == 42:

        if model_name in [
            "clean", "interior", "background"
        ]:
            folder = (
                EXP /
                "shortcut_pressure/checkpoints"
            )
        else:
            folder = (
                EXP /
                "absolute_positions/checkpoints"
            )

        candidates = [
            folder / f"{model_name}_final.pt",
            folder / f"{model_name}_best.pt"
        ]

    else:
        folder = (
            EXP /
            "multiseed_spatial/checkpoints"
        )

        candidates = [
            folder /
            f"{model_name}_seed{seed}_final.pt"
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


available = []
missing = []

for seed in SEEDS:
    for model_name in MODELS:
        ckpt = checkpoint_path(
            model_name, seed
        )

        if ckpt is None:
            missing.append(
                (model_name, seed)
            )
        else:
            available.append(
                (model_name, seed, ckpt)
            )

print("\nAvailable checkpoints:", len(available))
print("Missing checkpoints:", missing)

if not available:
    raise RuntimeError(
        "No checkpoints available. "
        "Restore the saved checkpoint ZIPs."
    )

# Do not silently present incomplete seeds as a
# complete 15-model experiment.
if missing:
    print(
        "\nWARNING: Some models are missing. "
        "Results will clearly report checkpoint coverage."
    )

# ============================================================
# MODEL
# ============================================================

def make_model():
    model = models.resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features, 2
    )

    return model.to(DEVICE)


@torch.inference_mode()
def predict(model, position, marker_type):

    probabilities = np.zeros(
        len(labels), dtype=np.float32
    )

    model.eval()

    for batch, indices in loader(
        position, marker_type
    ):
        batch = batch.to(
            DEVICE, non_blocking=True
        )

        with torch.autocast(
            "cuda", dtype=torch.float16
        ):
            logits = model(batch)

        probs = torch.softmax(
            logits.float(), dim=1
        )[:, 1]

        probabilities[
            indices.numpy()
        ] = probs.cpu().numpy()

    return probabilities


# ============================================================
# POSITION OVERLAP
# ============================================================

print("\nComputing animal overlap...")

overlap_rows = []

for position in POSITIONS:
    for i, row in manifest.iterrows():

        x = int(row[f"{position}_x"])
        y = int(row[f"{position}_y"])

        region = masks[
            i,
            y:y+PATCH,
            x:x+PATCH
        ]

        overlap_rows.append({
            "image_id": str(row.image_id),
            "position": position,
            "foreground_fraction": float(
                np.mean(region == 1)
            ),
            "boundary_fraction": float(
                np.mean(region == 3)
            ),
            "background_fraction": float(
                np.mean(region == 2)
            ),
            "x": x,
            "y": y
        })

overlap = pd.DataFrame(overlap_rows)

overlap.to_csv(
    RESULTS / "position_overlap.csv",
    index=False
)

# ============================================================
# COUNTERFACTUAL INFERENCE
# ============================================================

summary_rows = []
prediction_rows = []

for model_name, seed, ckpt in available:

    print("\n" + "=" * 70)
    print(
        "MODEL:", model_name,
        "| SEED:", seed
    )
    print("=" * 70)

    model = make_model()

    checkpoint = torch.load(
        ckpt,
        map_location="cpu",
        weights_only=True
    )

    state = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state)

    clean_probs = predict(
        model, "clean", 0
    )

    clean_predictions = (
        clean_probs >= 0.5
    ).astype(int)

    for position in POSITIONS:

        p0 = predict(
            model, position, 0
        )

        p1 = predict(
            model, position, 1
        )

        abs_change = np.abs(p1 - p0)

        flips = (
            (p0 >= 0.5)
            != (p1 >= 0.5)
        )

        matching = np.where(
            labels == 0, p0, p1
        )

        opposing = np.where(
            labels == 0, p1, p0
        )

        true_class_confidence_drop = np.where(
            labels == 1,
            matching - opposing,
            opposing - matching
        )

        matching_pred = (
            matching >= 0.5
        ).astype(int)

        opposing_pred = (
            opposing >= 0.5
        ).astype(int)

        position_overlap = (
            overlap[
                overlap.position == position
            ]
            .set_index("image_id")
            .loc[image_ids]
        )

        summary_rows.append({
            "trained_region": model_name,
            "seed": seed,
            "tested_position": position,
            "n": len(labels),
            "mean_abs_probability_change": float(
                abs_change.mean()
            ),
            "decision_flip_pct": float(
                100 * flips.mean()
            ),
            "mean_true_class_confidence_drop": float(
                true_class_confidence_drop.mean()
            ),
            "matching_macro_f1": float(
                f1_score(
                    labels,
                    matching_pred,
                    average="macro"
                )
            ),
            "opposing_macro_f1": float(
                f1_score(
                    labels,
                    opposing_pred,
                    average="macro"
                )
            ),
            "matching_auroc": float(
                roc_auc_score(
                    labels, matching
                )
            ),
            "opposing_auroc": float(
                roc_auc_score(
                    labels, opposing
                )
            ),
            "mean_foreground_overlap": float(
                position_overlap.foreground_fraction.mean()
            )
        })

        prediction_rows.append(
            pd.DataFrame({
                "image_id": image_ids,
                "label": labels,
                "trained_region": model_name,
                "seed": seed,
                "tested_position": position,
                "p_marker0": p0,
                "p_marker1": p1,
                "p_clean": clean_probs,
                "abs_change": abs_change,
                "decision_flip": flips,
                "true_class_confidence_drop":
                    true_class_confidence_drop,
                "foreground_fraction":
                    position_overlap.foreground_fraction.to_numpy(),
                "boundary_fraction":
                    position_overlap.boundary_fraction.to_numpy(),
                "background_fraction":
                    position_overlap.background_fraction.to_numpy()
            })
        )

        print(
            f"{position:14s} "
            f"abs={abs_change.mean():.5f} "
            f"flips={100*flips.mean():.2f}%"
        )

    del model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()

summary = pd.DataFrame(summary_rows)

predictions = pd.concat(
    prediction_rows,
    ignore_index=True
)

summary.to_csv(
    RESULTS / "counterfactual_summary.csv",
    index=False
)

predictions.to_csv(
    RESULTS / "counterfactual_predictions.csv",
    index=False
)

# ============================================================
# WITHIN-MODEL PAIRED COMPARISONS
# ============================================================

reference = "background"

paired_rows = []

for (model_name, seed), group in predictions.groupby(
    ["trained_region", "seed"]
):

    pivot = group.pivot(
        index="image_id",
        columns="tested_position",
        values="abs_change"
    )

    for position in POSITIONS:
        if position == reference:
            continue

        difference = (
            pivot[position] -
            pivot[reference]
        ).to_numpy()

        paired_rows.append({
            "trained_region": model_name,
            "seed": seed,
            "position": position,
            "reference": reference,
            "mean_paired_difference": float(
                difference.mean()
            ),
            "median_paired_difference": float(
                np.median(difference)
            ),
            "fraction_images_larger": float(
                np.mean(difference > 0)
            )
        })

paired_comparisons = pd.DataFrame(
    paired_rows
)

paired_comparisons.to_csv(
    RESULTS / "within_model_paired_comparisons.csv",
    index=False
)

# ============================================================
# BOOTSTRAP POSITIONAL DIFFERENCES
# ============================================================

# Seed is the replication unit for training variability.
# Images are paired within each seed/model.
# Bootstrap is exploratory, with few training seeds.

rng = np.random.default_rng(2026)

bootstrap_rows = []

for trained_region in MODELS:

    subset = predictions[
        predictions.trained_region == trained_region
    ]

    seeds_here = sorted(
        subset.seed.unique()
    )

    for position in POSITIONS:

        if position == reference:
            continue

        differences = []

        for seed in seeds_here:

            g = subset[
                subset.seed == seed
            ]

            pivot = g.pivot(
                index="image_id",
                columns="tested_position",
                values="abs_change"
            ).loc[image_ids]

            differences.append(
                (
                    pivot[position] -
                    pivot[reference]
                ).to_numpy()
            )

        arr = np.stack(differences)

        estimates = []

        for _ in range(BOOTSTRAP_REPS):

            selected_seeds = rng.integers(
                0,
                len(seeds_here),
                len(seeds_here)
            )

            selected_images = rng.integers(
                0,
                len(image_ids),
                len(image_ids)
            )

            estimates.append(
                arr[
                    np.ix_(
                        selected_seeds,
                        selected_images
                    )
                ].mean()
            )

        bootstrap_rows.append({
            "trained_region": trained_region,
            "position": position,
            "reference": reference,
            "n_seeds": len(seeds_here),
            "mean_difference": float(
                arr.mean()
            ),
            "ci_low": float(
                np.quantile(estimates, 0.025)
            ),
            "ci_high": float(
                np.quantile(estimates, 0.975)
            )
        })

bootstrap = pd.DataFrame(bootstrap_rows)

bootstrap.to_csv(
    RESULTS / "paired_bootstrap.csv",
    index=False
)

# ============================================================
# HEATMAP: TRAINED REGION × TESTED POSITION
# ============================================================

heatmap = (
    summary.groupby(
        ["trained_region", "tested_position"]
    )["mean_abs_probability_change"]
    .mean()
    .unstack()
    .reindex(
        index=MODELS,
        columns=POSITIONS
    )
)

heatmap.to_csv(
    RESULTS / "sensitivity_matrix.csv"
)

fig, ax = plt.subplots(
    figsize=(11, 6)
)

image = ax.imshow(
    heatmap.values,
    aspect="auto",
    cmap="viridis"
)

ax.set_xticks(
    np.arange(len(POSITIONS))
)

ax.set_xticklabels(
    POSITIONS,
    rotation=35,
    ha="right"
)

ax.set_yticks(
    np.arange(len(MODELS))
)

ax.set_yticklabels(MODELS)

ax.set_xlabel("Marker test position")
ax.set_ylabel("Model training condition")

ax.set_title(
    "Within-model marker sensitivity across spatial positions"
)

for i in range(len(MODELS)):
    for j in range(len(POSITIONS)):

        value = heatmap.iloc[i, j]

        if np.isfinite(value):
            ax.text(
                j, i,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color=(
                    "white"
                    if value < np.nanmax(heatmap.values)*0.55
                    else "black"
                )
            )

fig.colorbar(
    image,
    ax=ax,
    label="Mean absolute change in P(dog)"
)

plt.tight_layout()

plt.savefig(
    FIGURES / "spatial_sensitivity_heatmap.png",
    dpi=300
)

plt.close()

# ============================================================
# PLOT: MARKER OVERLAP VS SENSITIVITY
# ============================================================

plot_data = (
    summary.groupby("tested_position")
    .agg(
        sensitivity=(
            "mean_abs_probability_change",
            "mean"
        ),
        foreground=(
            "mean_foreground_overlap",
            "mean"
        )
    )
    .reset_index()
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.scatter(
    plot_data.foreground,
    plot_data.sensitivity,
    s=100
)

for _, row in plot_data.iterrows():
    ax.annotate(
        row.tested_position,
        (
            row.foreground,
            row.sensitivity
        ),
        xytext=(5, 5),
        textcoords="offset points"
    )

ax.set_xlabel(
    "Mean animal foreground overlap"
)

ax.set_ylabel(
    "Mean marker sensitivity"
)

ax.set_title(
    "Foreground overlap versus marker sensitivity"
)

ax.grid(alpha=0.2)

plt.tight_layout()

plt.savefig(
    FIGURES / "overlap_vs_counterfactual_sensitivity.png",
    dpi=300
)

plt.close()

# ============================================================
# SUMMARY TABLES
# ============================================================

position_summary = (
    summary.groupby("tested_position")
    .agg(
        mean_abs_change=(
            "mean_abs_probability_change",
            "mean"
        ),
        mean_flip_pct=(
            "decision_flip_pct",
            "mean"
        ),
        mean_confidence_drop=(
            "mean_true_class_confidence_drop",
            "mean"
        ),
        mean_foreground_overlap=(
            "mean_foreground_overlap",
            "mean"
        )
    )
    .reset_index()
    .sort_values(
        "mean_abs_change",
        ascending=False
    )
)

position_summary.to_csv(
    RESULTS / "position_summary.csv",
    index=False
)

# ============================================================
# REPORT
# ============================================================

report = {
    "experiment": "coordinate_counterfactual",
    "status": "completed",
    "validation_images": len(manifest),
    "model_conditions": MODELS,
    "seeds_requested": SEEDS,
    "checkpoints_available": len(available),
    "checkpoints_missing": [
        {"model": name, "seed": seed}
        for name, seed in missing
    ],
    "tested_positions": POSITIONS,
    "official_test_used": False,
    "interpretation": (
        "Inference-time counterfactual marker-orientation "
        "sensitivity at fixed positions within each model. "
        "Position comparisons are paired by image and model."
    ),
    "limitations": [
        "Models were trained under different marker placements",
        "Only three training seeds were planned",
        "Position changes alter which image pixels are occluded",
        "No causal isolation of training-time position effects",
        "Validation cohort used for exploratory analysis",
        "Official test not evaluated"
    ]
}

(RESULTS / "experiment_report.json").write_text(
    json.dumps(report, indent=2)
)

print("\n" + "=" * 75)
print("POSITION SUMMARY")
print("=" * 75)

print(
    position_summary.round(6)
    .to_string(index=False)
)

print("\n" + "=" * 75)
print("SENSITIVITY MATRIX")
print("=" * 75)

print(
    heatmap.round(5)
    .to_string()
)

print("\n" + "=" * 75)
print("PAIRED BOOTSTRAP: INTERIOR VS BACKGROUND")
print("=" * 75)

print(
    bootstrap[
        bootstrap.position == "interior"
    ].round(6).to_string(index=False)
)

print("\nArtifacts:", OUT)
print("Official test set untouched.")
