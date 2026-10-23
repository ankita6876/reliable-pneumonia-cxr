
import gc
import json
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
    accuracy_score, balanced_accuracy_score,
    f1_score, roc_auc_score
)
from tqdm.auto import tqdm

# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

ROOT = Path("/kaggle/working/reliable-pneumonia-cxr")
EXP = ROOT / "experiments/oxford_pet_spatial"

OUT = EXP / "neutral_patch_control"
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
TRAINED_REGIONS = [
    "clean", "interior", "background",
    "center", "bottom_right"
]
POSITIONS = [
    "top_left", "top_right", "bottom_left",
    "bottom_right", "center",
    "interior", "background"
]

IMAGE_SIZE = 224
PATCH_SIZE = 24
BATCH_SIZE = 32
BOOTSTRAP_REPS = 2000

torch.backends.cudnn.benchmark = True

print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# 2. LOCKED VALIDATION COHORT
# ============================================================

manifest = pd.read_csv(
    EXP / "multiseed_spatial/results/"
    "locked_validation_manifest.csv"
)

assert len(manifest) == 722
assert manifest.image_id.is_unique

manifest["image_path"] = manifest.image_id.apply(
    lambda x: str(DATA_ROOT / "images" / f"{x}.jpg")
)

manifest["mask_path"] = manifest.image_id.apply(
    lambda x: str(
        DATA_ROOT / "annotations/trimaps" / f"{x}.png"
    )
)

ABS_POSITIONS = {
    "top_left": (8, 8),
    "top_right": (192, 8),
    "bottom_left": (8, 192),
    "bottom_right": (192, 192),
    "center": (100, 100)
}

for position, (x, y) in ABS_POSITIONS.items():
    manifest[f"{position}_x"] = x
    manifest[f"{position}_y"] = y

for position in POSITIONS:
    for coordinate in ["x", "y"]:
        column = f"{position}_{coordinate}"
        assert column in manifest.columns
        assert manifest[column].notna().all()
        assert manifest[column].between(
            0, IMAGE_SIZE-PATCH_SIZE
        ).all()

manifest.to_csv(
    RESULTS / "locked_validation_manifest.csv",
    index=False
)

labels = manifest.label.to_numpy(dtype=np.int64)
image_ids = manifest.image_id.astype(str).to_numpy()

# ============================================================
# 3. CACHE IMAGES AND MASKS
# ============================================================

print("\nCaching validation images...")

images = []
masks = []

for _, row in tqdm(
    manifest.iterrows(),
    total=len(manifest)
):
    with Image.open(row.image_path) as im:
        images.append(
            np.asarray(
                im.convert("RGB").resize(
                    (IMAGE_SIZE, IMAGE_SIZE)
                )
            ).copy()
        )

    with Image.open(row.mask_path) as im:
        masks.append(
            np.asarray(
                im.resize(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    Image.Resampling.NEAREST
                )
            ).copy()
        )

images = np.stack(images)
masks = np.stack(masks)

# ============================================================
# 4. MARKER AND NEUTRAL PATCH DESIGNS
# ============================================================

yy, xx = np.indices(
    (PATCH_SIZE, PATCH_SIZE)
)

patches = {}

for marker_type in [0, 1]:
    pattern = (
        (xx // 3) % 2
        if marker_type == 0
        else (yy // 3) % 2
    )

    gray = np.where(
        pattern == 1, 240, 15
    ).astype(np.uint8)

    patches[f"marker{marker_type}"] = np.repeat(
        gray[..., None], 3, axis=2
    )

# Mid-gray patch; fixed RGB value for all images.
patches["neutral"] = np.full(
    (PATCH_SIZE, PATCH_SIZE, 3),
    128,
    dtype=np.uint8
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
# 5. DATASET
# ============================================================

class PatchDataset(Dataset):
    def __init__(self, position, patch_type):
        self.position = position
        self.patch_type = patch_type

    def __len__(self):
        return len(labels)

    def __getitem__(self, index):
        image = images[index].copy()

        if self.position != "clean":
            row = manifest.iloc[index]
            x = int(row[f"{self.position}_x"])
            y = int(row[f"{self.position}_y"])

            image[
                y:y+PATCH_SIZE,
                x:x+PATCH_SIZE
            ] = patches[self.patch_type]

        arr = (
            image.astype(np.float32)
            .transpose(2, 0, 1) / 255.0
        )

        arr = (arr-MEAN)/STD

        return (
            torch.from_numpy(arr.copy()),
            index
        )


def make_loader(position, patch_type):
    return DataLoader(
        PatchDataset(position, patch_type),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

# ============================================================
# 6. CHECKPOINT DISCOVERY
# ============================================================

def checkpoint_path(region, seed):
    if seed == 42:
        if region in [
            "clean", "interior", "background"
        ]:
            folder = EXP / "shortcut_pressure/checkpoints"
        else:
            folder = EXP / "absolute_positions/checkpoints"

        candidates = [
            folder / f"{region}_final.pt",
            folder / f"{region}_best.pt"
        ]
    else:
        folder = EXP / "multiseed_spatial/checkpoints"
        candidates = [
            folder / f"{region}_seed{seed}_final.pt"
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


available = []

for seed in SEEDS:
    for region in TRAINED_REGIONS:
        path = checkpoint_path(region, seed)

        if path is None:
            raise FileNotFoundError(
                f"Missing checkpoint: {region}, seed {seed}. "
                "Restore your saved checkpoint ZIPs."
            )

        available.append((region, seed, path))

print("Verified checkpoints:", len(available))
assert len(available) == 15

# ============================================================
# 7. MODEL AND INFERENCE
# ============================================================

def make_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(DEVICE)


@torch.inference_mode()
def predict(model, position, patch_type):
    model.eval()

    probs = np.zeros(
        len(labels), dtype=np.float32
    )

    for batch, indices in make_loader(
        position, patch_type
    ):
        batch = batch.to(
            DEVICE, non_blocking=True
        )

        with torch.autocast(
            "cuda", dtype=torch.float16
        ):
            logits = model(batch)

        p = torch.softmax(
            logits.float(), dim=1
        )[:, 1]

        probs[indices.numpy()] = (
            p.cpu().numpy()
        )

    return probs


def calculate_metrics(p):
    pred = (p >= 0.5).astype(int)

    return {
        "accuracy": float(
            accuracy_score(labels, pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, pred)
        ),
        "macro_f1": float(
            f1_score(labels, pred, average="macro")
        ),
        "auroc": float(
            roc_auc_score(labels, p)
        )
    }

# ============================================================
# 8. REUSE EXISTING COUNTERFACTUAL PREDICTIONS
# ============================================================

previous_path = (
    EXP / "coordinate_counterfactual/results/"
    "counterfactual_predictions.csv"
)

previous = pd.read_csv(previous_path)

assert len(previous) == 15 * 7 * 722
assert previous.image_id.notna().all()

# Strict image alignment.
previous["image_id"] = previous.image_id.astype(str)

old_index = previous.set_index(
    ["trained_region", "seed", "tested_position", "image_id"]
)

# ============================================================
# 9. NEUTRAL-PATCH INFERENCE
# ============================================================

all_rows = []
summary_rows = []

for trained_region, seed, checkpoint_path in available:

    print("\n" + "="*70)
    print(
        "TRAINED MODEL:", trained_region,
        "| SEED:", seed
    )
    print("="*70)

    model = make_model()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    state = checkpoint.get(
        "model_state_dict", checkpoint
    )

    model.load_state_dict(state)

    for position in POSITIONS:

        old = old_index.loc[
            (trained_region, seed, position)
        ].loc[image_ids]

        p_clean = old.p_clean.to_numpy(
            dtype=np.float32
        )

        p0 = old.p_marker0.to_numpy(
            dtype=np.float32
        )

        p1 = old.p_marker1.to_numpy(
            dtype=np.float32
        )

        # Cache neutral predictions for resumability.
        neutral_path = (
            RESULTS /
            f"neutral_{trained_region}_"
            f"seed{seed}_{position}.npy"
        )

        if neutral_path.exists():
            p_neutral = np.load(neutral_path)
        else:
            p_neutral = predict(
                model, position, "neutral"
            )
            np.save(neutral_path, p_neutral)

        assert len(p_neutral) == len(labels)

        encoding_effect = np.abs(p1-p0)
        neutral_occlusion = np.abs(
            p_neutral-p_clean
        )

        marker0_occlusion = np.abs(
            p0-p_clean
        )

        marker1_occlusion = np.abs(
            p1-p_clean
        )

        mean_marker_occlusion = (
            marker0_occlusion
            + marker1_occlusion
        ) / 2

        # Difference in absolute responses:
        # descriptive, not an additive causal decomposition.
        excess_encoding = (
            encoding_effect-neutral_occlusion
        )

        marker_vs_neutral = (
            np.abs(p0-p_neutral)
            + np.abs(p1-p_neutral)
        ) / 2

        neutral_flip = (
            (p_neutral >= 0.5)
            != (p_clean >= 0.5)
        )

        orientation_flip = (
            (p0 >= 0.5)
            != (p1 >= 0.5)
        )

        all_rows.append(
            pd.DataFrame({
                "image_id": image_ids,
                "label": labels,
                "trained_region": trained_region,
                "seed": seed,
                "tested_position": position,
                "p_clean": p_clean,
                "p_marker0": p0,
                "p_marker1": p1,
                "p_neutral": p_neutral,
                "encoding_effect": encoding_effect,
                "neutral_occlusion": neutral_occlusion,
                "mean_marker_occlusion":
                    mean_marker_occlusion,
                "marker_vs_neutral":
                    marker_vs_neutral,
                "encoding_minus_neutral_occlusion":
                    excess_encoding,
                "neutral_decision_flip": neutral_flip,
                "orientation_decision_flip":
                    orientation_flip
            })
        )

        summary_rows.append({
            "trained_region": trained_region,
            "seed": seed,
            "tested_position": position,
            "n": len(labels),
            "mean_encoding_effect": float(
                encoding_effect.mean()
            ),
            "mean_neutral_occlusion": float(
                neutral_occlusion.mean()
            ),
            "mean_marker_occlusion": float(
                mean_marker_occlusion.mean()
            ),
            "mean_marker_vs_neutral": float(
                marker_vs_neutral.mean()
            ),
            "encoding_minus_neutral_occlusion":
                float(excess_encoding.mean()),
            "neutral_flip_pct": float(
                100*neutral_flip.mean()
            ),
            "orientation_flip_pct": float(
                100*orientation_flip.mean()
            ),
            "clean_macro_f1": calculate_metrics(
                p_clean
            )["macro_f1"],
            "neutral_macro_f1": calculate_metrics(
                p_neutral
            )["macro_f1"],
            "neutral_auroc": calculate_metrics(
                p_neutral
            )["auroc"]
        })

        print(
            f"{position:14s} "
            f"encoding={encoding_effect.mean():.5f} "
            f"neutral={neutral_occlusion.mean():.5f} "
            f"marker-neutral={marker_vs_neutral.mean():.5f}"
        )

    del model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()

predictions = pd.concat(
    all_rows, ignore_index=True
)

summary = pd.DataFrame(summary_rows)

assert len(summary) == 105
assert len(predictions) == 105 * 722

predictions.to_csv(
    RESULTS / "neutral_patch_predictions.csv",
    index=False
)

summary.to_csv(
    RESULTS / "neutral_patch_summary.csv",
    index=False
)

# ============================================================
# 10. AGGREGATED POSITION SUMMARY
# ============================================================

position_summary = (
    summary.groupby("tested_position")
    .agg(
        encoding_effect=("mean_encoding_effect", "mean"),
        neutral_occlusion=("mean_neutral_occlusion", "mean"),
        marker_occlusion=("mean_marker_occlusion", "mean"),
        marker_vs_neutral=("mean_marker_vs_neutral", "mean"),
        encoding_minus_neutral=(
            "encoding_minus_neutral_occlusion", "mean"
        ),
        neutral_flip_pct=("neutral_flip_pct", "mean"),
        orientation_flip_pct=(
            "orientation_flip_pct", "mean"
        )
    )
    .reset_index()
    .sort_values(
        "encoding_effect",
        ascending=False
    )
)

position_summary.to_csv(
    RESULTS / "position_control_summary.csv",
    index=False
)

# ============================================================
# 11. WITHIN-MODEL PAIRED BOOTSTRAP
# ============================================================

rng = np.random.default_rng(2026)

bootstrap_rows = []

for trained_region in TRAINED_REGIONS:

    for position in POSITIONS:

        group = predictions[
            (predictions.trained_region == trained_region)
            & (predictions.tested_position == position)
        ]

        arrays = []

        for seed in SEEDS:
            subset = (
                group[group.seed == seed]
                .set_index("image_id")
                .loc[image_ids]
            )

            difference = (
                subset.encoding_effect.to_numpy()
                - subset.neutral_occlusion.to_numpy()
            )

            arrays.append(difference)

        arr = np.stack(arrays)

        estimates = np.empty(
            BOOTSTRAP_REPS, dtype=np.float64
        )

        for b in range(BOOTSTRAP_REPS):

            sampled_seeds = rng.integers(
                0, len(SEEDS), len(SEEDS)
            )

            sampled_images = rng.integers(
                0, len(labels), len(labels)
            )

            estimates[b] = arr[
                np.ix_(
                    sampled_seeds,
                    sampled_images
                )
            ].mean()

        bootstrap_rows.append({
            "trained_region": trained_region,
            "tested_position": position,
            "n_seeds": len(SEEDS),
            "mean_encoding_minus_neutral": float(
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
# 12. PUBLICATION-READY FIGURES
# ============================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False
})

plot = position_summary.set_index(
    "tested_position"
).loc[POSITIONS]

x = np.arange(len(POSITIONS))
width = 0.27

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    x-width,
    plot.encoding_effect,
    width,
    label="Stripe orientation change"
)

ax.bar(
    x,
    plot.neutral_occlusion,
    width,
    label="Neutral patch vs clean"
)

ax.bar(
    x+width,
    plot.marker_vs_neutral,
    width,
    label="Striped vs neutral"
)

ax.set_xticks(x)
ax.set_xticklabels(
    POSITIONS, rotation=30, ha="right"
)

ax.set_ylabel(
    "Mean absolute change in P(dog)"
)

ax.set_title(
    "Encoding sensitivity versus neutral-patch occlusion"
)

ax.legend()
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    FIGURES / "encoding_vs_occlusion.png",
    dpi=300
)

plt.savefig(
    FIGURES / "encoding_vs_occlusion.pdf"
)

plt.close()

# ============================================================
# 13. HEATMAP OF ENCODING MINUS NEUTRAL OCCLUSION
# ============================================================

matrix = (
    summary.groupby(
        ["trained_region", "tested_position"]
    )["encoding_minus_neutral_occlusion"]
    .mean()
    .unstack()
    .reindex(
        index=TRAINED_REGIONS,
        columns=POSITIONS
    )
)

matrix.to_csv(
    RESULTS / "encoding_minus_neutral_matrix.csv"
)

fig, ax = plt.subplots(figsize=(12, 6))

max_abs = max(
    abs(np.nanmin(matrix.values)),
    abs(np.nanmax(matrix.values))
)

im = ax.imshow(
    matrix.values,
    cmap="coolwarm",
    vmin=-max_abs,
    vmax=max_abs,
    aspect="auto"
)

ax.set_xticks(np.arange(len(POSITIONS)))
ax.set_xticklabels(
    POSITIONS, rotation=35, ha="right"
)

ax.set_yticks(np.arange(len(TRAINED_REGIONS)))
ax.set_yticklabels(TRAINED_REGIONS)

for i in range(len(TRAINED_REGIONS)):
    for j in range(len(POSITIONS)):
        ax.text(
            j, i,
            f"{matrix.iloc[i, j]:+.3f}",
            ha="center",
            va="center",
            fontsize=9
        )

ax.set_title(
    "Stripe-orientation sensitivity minus neutral occlusion"
)

fig.colorbar(
    im, ax=ax,
    label="Difference in mean absolute probability response"
)

plt.tight_layout()

plt.savefig(
    FIGURES / "encoding_minus_neutral_heatmap.png",
    dpi=300
)

plt.close()

# ============================================================
# 14. FIGURE: MATCHED MODEL-SPECIFIC EFFECTS
# ============================================================

focused = summary[
    summary.trained_region.isin(
        ["clean", "center", "bottom_right"]
    )
]

fig, axes = plt.subplots(
    1, 3,
    figsize=(16, 5),
    sharey=True
)

for ax, trained_region in zip(
    axes,
    ["clean", "center", "bottom_right"]
):
    g = (
        focused[
            focused.trained_region == trained_region
        ]
        .groupby("tested_position")
        .agg(
            encoding=("mean_encoding_effect", "mean"),
            neutral=("mean_neutral_occlusion", "mean")
        )
        .reindex(POSITIONS)
    )

    ax.plot(
        POSITIONS,
        g.encoding,
        marker="o",
        label="Encoding"
    )

    ax.plot(
        POSITIONS,
        g.neutral,
        marker="s",
        label="Neutral occlusion"
    )

    ax.set_title(
        f"Trained: {trained_region}"
    )

    ax.tick_params(
        axis="x", rotation=60
    )

    ax.grid(alpha=0.2)

axes[0].set_ylabel(
    "Mean absolute probability change"
)

axes[-1].legend()

fig.suptitle(
    "Within-model spatial response profiles"
)

plt.tight_layout()

plt.savefig(
    FIGURES / "model_specific_spatial_profiles.png",
    dpi=300
)

plt.close()

# ============================================================
# 15. CONFIGURATION AND METHODOLOGY REPORT
# ============================================================

config = {
    "experiment": "neutral_patch_control",
    "dataset": "Oxford-IIIT Pet",
    "validation_images": len(labels),
    "trained_regions": TRAINED_REGIONS,
    "seeds": SEEDS,
    "tested_positions": POSITIONS,
    "image_size": IMAGE_SIZE,
    "patch_size": PATCH_SIZE,
    "neutral_patch_rgb": [128, 128, 128],
    "marker_patterns": [
        "vertical stripes",
        "horizontal stripes"
    ],
    "bootstrap_repetitions": BOOTSTRAP_REPS,
    "training_performed": False,
    "official_test_used": False
}

(CONFIGS / "experiment_config.json").write_text(
    json.dumps(config, indent=2)
)

report = {
    "status": "completed",
    "research_question": (
        "How does sensitivity to synthetic marker encoding "
        "compare with the effect of occluding the same image "
        "region using a neutral patch?"
    ),
    "definitions": {
        "encoding_effect": "|p(marker1)-p(marker0)|",
        "neutral_occlusion": "|p(neutral)-p(clean)|",
        "marker_vs_neutral": (
            "0.5*(|p(marker0)-p(neutral)|"
            "+|p(marker1)-p(neutral)|)"
        ),
        "encoding_minus_neutral": (
            "encoding_effect-neutral_occlusion"
        )
    },
    "interpretation_rules": [
        "All patch comparisons are paired within image, model, and location",
        "Encoding minus neutral is descriptive, not an additive causal decomposition",
        "A fixed RGB gray patch is only one occlusion control",
        "Patch replacement also changes local image texture and contrast",
        "Three seeds provide limited information about training variability",
        "Bootstrap intervals are exploratory",
        "Validation results are not final held-out test results"
    ],
    "official_test_used": False
}

(RESULTS / "methodology_report.json").write_text(
    json.dumps(report, indent=2)
)

print("\n" + "="*75)
print("POSITION CONTROL SUMMARY")
print("="*75)

print(
    position_summary.round(6).to_string(index=False)
)

print("\n" + "="*75)
print("ENCODING MINUS NEUTRAL MATRIX")
print("="*75)

print(
    matrix.round(5).to_string()
)

print("\n" + "="*75)
print("FOCUSED BOOTSTRAP RESULTS")
print("="*75)

focused_bootstrap = bootstrap[
    (
        bootstrap.trained_region.isin(
            ["clean", "center", "bottom_right"]
        )
    )
    & (
        bootstrap.tested_position.isin(
            ["center", "bottom_right", "interior", "background"]
        )
    )
]

print(
    focused_bootstrap.round(6).to_string(index=False)
)

print("\nCompleted.")
print("Saved:", OUT)
print("Official test set untouched.")
