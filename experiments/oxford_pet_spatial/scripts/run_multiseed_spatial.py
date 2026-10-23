
import os
import json
import random
import hashlib
import gc
import copy
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
# 1. CONFIGURATION
# ============================================================

ROOT = Path("/kaggle/working/reliable-pneumonia-cxr")
EXP = ROOT / "experiments/oxford_pet_spatial"

OUT = EXP / "multiseed_spatial"
RESULTS = OUT / "results"
FIGURES = OUT / "figures"
CHECKPOINTS = OUT / "checkpoints"
CONFIGS = OUT / "configs"

for p in [RESULTS, FIGURES, CHECKPOINTS, CONFIGS]:
    p.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path(
    "/kaggle/input/datasets/"
    "julinmaloof/the-oxfordiiit-pet-dataset"
)

SEEDS = [42, 123, 2026]
NEW_SEEDS = [123, 2026]

REGIONS = [
    "clean",
    "interior",
    "background",
    "center",
    "bottom_right"
]

IMAGE_SIZE = 224
PATCH = 24
EPOCHS = 6
BATCH_SIZE = 24

BACKBONE_LR = 2e-5
HEAD_LR = 2e-4
WEIGHT_DECAY = 1e-4

BOOTSTRAP_REPS = 2000

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

assert DEVICE.type == "cuda", "Enable Kaggle GPU."

torch.backends.cudnn.benchmark = True

print("Device:", DEVICE)
print("New seeds:", NEW_SEEDS)
print("Conditions:", REGIONS)

# ============================================================
# 2. LOAD FIXED TRAIN AND VALIDATION COHORTS
# ============================================================

BASE = EXP / "shortcut_pressure/results"

train = pd.read_csv(
    BASE / "small_train_manifest.csv"
)

val = pd.read_csv(
    BASE / "common_feasible_validation_manifest.csv"
)

assert len(train) == 148
assert len(val) == 722

assert train.image_id.is_unique
assert val.image_id.is_unique

assert set(train.image_id).isdisjoint(
    set(val.image_id)
)

# Repair image paths after Kaggle session resets.

def fix_paths(df):
    df = df.copy()

    df["image_path"] = df.image_id.apply(
        lambda image_id: str(
            DATA_ROOT / "images" / f"{image_id}.jpg"
        )
    )

    df["mask_path"] = df.image_id.apply(
        lambda image_id: str(
            DATA_ROOT / "annotations/trimaps" /
            f"{image_id}.png"
        )
    )

    for col in ["image_path", "mask_path"]:
        missing = [
            p for p in df[col]
            if not Path(p).exists()
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing {col}: {missing[:3]}"
            )

    return df

train = fix_paths(train)
val = fix_paths(val)

# Fixed absolute positions.

ABS_POS = {
    "center": (100, 100),
    "bottom_right": (192, 192)
}

for region, (x, y) in ABS_POS.items():
    train[f"{region}_x"] = x
    train[f"{region}_y"] = y

    val[f"{region}_x"] = x
    val[f"{region}_y"] = y

for region in ["interior", "background"]:
    for df in [train, val]:
        assert (
            (df[f"{region}_x"] >= 0)
            & (df[f"{region}_y"] >= 0)
        ).all()

train.to_csv(
    RESULTS / "locked_train_manifest.csv",
    index=False
)

val.to_csv(
    RESULTS / "locked_validation_manifest.csv",
    index=False
)

print("Training images:", len(train))
print("Validation images:", len(val))

# ============================================================
# 3. MARKER DESIGN
# ============================================================

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
).reshape(3, 1, 1)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
).reshape(3, 1, 1)

yy, xx = np.indices((PATCH, PATCH))

PATTERNS = {}

for marker_type in [0, 1]:
    pattern = (
        (xx // 3) % 2
        if marker_type == 0
        else (yy // 3) % 2
    )

    patch = np.where(
        pattern[..., None] == 1,
        240,
        15
    ).astype(np.uint8)

    PATTERNS[marker_type] = np.repeat(
        patch, 3, axis=2
    )


def add_marker(image, x, y, marker_type):
    image = image.copy()

    image[
        y:y+PATCH,
        x:x+PATCH
    ] = PATTERNS[marker_type]

    return image


# ============================================================
# 4. DATASET
# ============================================================

class PetDataset(Dataset):
    def __init__(self, df, region, condition):
        self.df = df.reset_index(drop=True)
        self.region = region
        self.condition = condition

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        with Image.open(row.image_path) as im:
            image = np.asarray(
                im.convert("RGB").resize(
                    (IMAGE_SIZE, IMAGE_SIZE)
                )
            )

        label = int(row.label)

        if (
            self.region != "clean"
            and self.condition != "clean"
        ):
            x = int(row[f"{self.region}_x"])
            y = int(row[f"{self.region}_y"])

            assert (
                0 <= x <= IMAGE_SIZE - PATCH
                and 0 <= y <= IMAGE_SIZE - PATCH
            )

            marker_type = (
                label
                if self.condition == "matching"
                else 1 - label
            )

            image = add_marker(
                image, x, y, marker_type
            )

        arr = (
            image.astype(np.float32)
            .transpose(2, 0, 1) / 255.0
        )

        arr = (arr - MEAN) / STD

        return (
            torch.from_numpy(arr.copy()),
            label,
            str(row.image_id)
        )


def make_loader(df, region, condition, shuffle=False):
    return DataLoader(
        PetDataset(df, region, condition),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True
    )


# ============================================================
# 5. MODEL AND METRICS
# ============================================================

def make_model(pretrained=True):
    weights = (
        models.ResNet50_Weights.DEFAULT
        if pretrained else None
    )

    model = models.resnet50(weights=weights)

    model.fc = nn.Linear(
        model.fc.in_features, 2
    )

    return model.to(DEVICE)


def metrics_from_probs(y, p):
    y = np.asarray(y)
    p = np.asarray(p)

    pred = (p >= 0.5).astype(int)

    return {
        "accuracy": float(
            accuracy_score(y, pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, pred)
        ),
        "macro_f1": float(
            f1_score(y, pred, average="macro")
        ),
        "auroc": float(
            roc_auc_score(y, p)
        )
    }


@torch.inference_mode()
def predict(model, dl):
    model.eval()

    ids, labels, probs = [], [], []

    for x, y, image_ids in dl:
        x = x.to(
            DEVICE, non_blocking=True
        )

        with torch.autocast(
            "cuda", dtype=torch.float16
        ):
            logits = model(x)

        p = torch.softmax(
            logits.float(), dim=1
        )[:, 1].cpu().numpy()

        ids.extend(image_ids)
        labels.extend(y.numpy().tolist())
        probs.extend(p.tolist())

    return pd.DataFrame({
        "image_id": ids,
        "label": labels,
        "prob_dog": probs
    })


# ============================================================
# 6. TRAIN NEW SEEDS
# ============================================================

for seed in NEW_SEEDS:
    for region in REGIONS:

        print("\n" + "=" * 65)
        print("SEED:", seed, "| REGION:", region)
        print("=" * 65)

        checkpoint_path = (
            CHECKPOINTS /
            f"{region}_seed{seed}_final.pt"
        )

        if checkpoint_path.exists():
            print("Checkpoint exists. Skipping training.")
            continue

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model = make_model(pretrained=True)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        p for name, p in model.named_parameters()
                        if not name.startswith("fc.")
                    ],
                    "lr": BACKBONE_LR
                },
                {
                    "params": model.fc.parameters(),
                    "lr": HEAD_LR
                }
            ],
            weight_decay=WEIGHT_DECAY
        )

        scaler = torch.amp.GradScaler("cuda")
        criterion = nn.CrossEntropyLoss()

        train_condition = (
            "clean" if region == "clean"
            else "matching"
        )

        dl = make_loader(
            train,
            region,
            train_condition,
            shuffle=True
        )

        history = []

        for epoch in range(1, EPOCHS + 1):
            model.train()

            total_loss = 0.0
            n = 0

            for images, labels, _ in tqdm(
                dl,
                desc=f"{region} seed{seed} epoch{epoch}"
            ):
                images = images.to(
                    DEVICE, non_blocking=True
                )

                labels = labels.to(
                    DEVICE, non_blocking=True
                )

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    "cuda", dtype=torch.float16
                ):
                    logits = model(images)
                    loss = criterion(
                        logits, labels
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += (
                    float(loss.item()) * len(labels)
                )
                n += len(labels)

            avg_loss = total_loss / n

            history.append({
                "seed": seed,
                "region": region,
                "epoch": epoch,
                "train_loss": avg_loss
            })

            print(
                f"Epoch {epoch}: loss={avg_loss:.5f}"
            )

        pd.DataFrame(history).to_csv(
            RESULTS /
            f"{region}_seed{seed}_history.csv",
            index=False
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "seed": seed,
                "region": region,
                "epoch": EPOCHS
            },
            checkpoint_path
        )

        del model
        gc.collect()
        torch.cuda.empty_cache()

# ============================================================
# 7. EVALUATE ALL THREE SEEDS
# ============================================================

# Existing seed-42 checkpoints are stored separately.
OLD_CHECKPOINTS = {
    "clean": (
        EXP / "shortcut_pressure/checkpoints/"
        "clean_final.pt"
    ),
    "interior": (
        EXP / "shortcut_pressure/checkpoints/"
        "interior_final.pt"
    ),
    "background": (
        EXP / "shortcut_pressure/checkpoints/"
        "background_final.pt"
    ),
    "center": (
        EXP / "absolute_positions/checkpoints/"
        "center_final.pt"
    ),
    "bottom_right": (
        EXP / "absolute_positions/checkpoints/"
        "bottom_right_final.pt"
    )
}

all_metrics = []
all_effects = []

for seed in SEEDS:
    for region in REGIONS:

        if seed == 42:
            ckpt = OLD_CHECKPOINTS[region]
        else:
            ckpt = (
                CHECKPOINTS /
                f"{region}_seed{seed}_final.pt"
            )

        if not ckpt.exists():
            if seed == 42:
                print(
                    "Seed-42 checkpoint unavailable:",
                    region,
                    "— using saved predictions instead."
                )
                continue

            raise FileNotFoundError(ckpt)

        model = make_model(pretrained=False)

        state = torch.load(
            ckpt,
            map_location="cpu",
            weights_only=True
        )

        model.load_state_dict(
            state["model_state_dict"]
        )

        conditions = (
            ["clean"] if region == "clean"
            else ["clean", "matching", "opposing"]
        )

        prediction_sets = {}

        for condition in conditions:
            pred_path = (
                RESULTS /
                f"{region}_seed{seed}_{condition}.csv"
            )

            if pred_path.exists():
                preds = pd.read_csv(pred_path)
            else:
                preds = predict(
                    model,
                    make_loader(
                        val, region, condition
                    )
                )

                preds.to_csv(
                    pred_path,
                    index=False
                )

            prediction_sets[condition] = (
                preds.set_index("image_id")
            )

            metrics = metrics_from_probs(
                preds.label,
                preds.prob_dog
            )

            all_metrics.append({
                "seed": seed,
                "region": region,
                "condition": condition,
                **metrics
            })

        if region != "clean":
            match = prediction_sets["matching"]
            oppose = prediction_sets["opposing"]

            paired = pd.DataFrame({
                "image_id": val.image_id,
                "label": val.label
            })

            paired["p_matching"] = (
                paired.image_id.map(
                    match.prob_dog
                )
            )

            paired["p_opposing"] = (
                paired.image_id.map(
                    oppose.prob_dog
                )
            )

            paired["abs_change"] = (
                paired.p_matching
                - paired.p_opposing
            ).abs()

            paired["decision_flip"] = (
                (paired.p_matching >= 0.5)
                != (paired.p_opposing >= 0.5)
            )

            paired["confidence_drop"] = np.where(
                paired.label == 1,
                paired.p_matching
                - paired.p_opposing,
                paired.p_opposing
                - paired.p_matching
            )

            paired["region"] = region
            paired["seed"] = seed

            paired.to_csv(
                RESULTS /
                f"{region}_seed{seed}_paired.csv",
                index=False
            )

            all_effects.append({
                "seed": seed,
                "region": region,
                "n": len(paired),
                "mean_abs_change": float(
                    paired.abs_change.mean()
                ),
                "decision_flip_pct": float(
                    100 * paired.decision_flip.mean()
                ),
                "mean_confidence_drop": float(
                    paired.confidence_drop.mean()
                )
            })

        del model
        gc.collect()
        torch.cuda.empty_cache()

# ============================================================
# 8. INCLUDE EXISTING SEED-42 PREDICTIONS
# ============================================================

# When old weights are absent, recover exact seed-42 metrics
# and effects from committed prediction CSVs.
# This does not retrain or approximate their outputs.

OLD_RESULTS = {
    "clean": EXP / "shortcut_pressure/results",
    "interior": EXP / "shortcut_pressure/results",
    "background": EXP / "shortcut_pressure/results",
    "center": EXP / "absolute_positions/results",
    "bottom_right": EXP / "absolute_positions/results"
}

for region in REGIONS:
    if any(
        r["seed"] == 42 and r["region"] == region
        for r in all_metrics
    ):
        continue

    base = OLD_RESULTS[region]

    conditions = (
        ["clean"] if region == "clean"
        else ["clean", "matching", "opposing"]
    )

    pred_sets = {}

    for condition in conditions:
        path = base / f"{region}_{condition}_predictions.csv"

        assert path.exists(), (
            f"Missing seed-42 predictions: {path}"
        )

        pred = pd.read_csv(path)

        assert len(pred) == len(val)
        assert set(pred.image_id) == set(val.image_id)

        pred_sets[condition] = pred.set_index("image_id")

        all_metrics.append({
            "seed": 42,
            "region": region,
            "condition": condition,
            **metrics_from_probs(
                pred.label,
                pred.prob_dog
            )
        })

    if region != "clean":
        a = pred_sets["matching"]
        b = pred_sets["opposing"]

        paired = pd.DataFrame({
            "image_id": val.image_id,
            "label": val.label
        })

        paired["p_matching"] = (
            paired.image_id.map(a.prob_dog)
        )

        paired["p_opposing"] = (
            paired.image_id.map(b.prob_dog)
        )

        paired["abs_change"] = (
            paired.p_matching
            - paired.p_opposing
        ).abs()

        paired["decision_flip"] = (
            (paired.p_matching >= 0.5)
            != (paired.p_opposing >= 0.5)
        )

        paired["confidence_drop"] = np.where(
            paired.label == 1,
            paired.p_matching - paired.p_opposing,
            paired.p_opposing - paired.p_matching
        )

        paired["seed"] = 42
        paired["region"] = region

        paired.to_csv(
            RESULTS /
            f"{region}_seed42_paired.csv",
            index=False
        )

        all_effects.append({
            "seed": 42,
            "region": region,
            "n": len(paired),
            "mean_abs_change": float(
                paired.abs_change.mean()
            ),
            "decision_flip_pct": float(
                100 * paired.decision_flip.mean()
            ),
            "mean_confidence_drop": float(
                paired.confidence_drop.mean()
            )
        })

metrics_df = pd.DataFrame(all_metrics)
effects_df = pd.DataFrame(all_effects)

assert len(metrics_df) == 39, (
    "Expected 3 clean + 4 regions × 3 seeds × "
    "3 conditions = 39 evaluation rows."
)

assert len(effects_df) == 12

metrics_df.to_csv(
    RESULTS / "all_seed_metrics.csv",
    index=False
)

effects_df.to_csv(
    RESULTS / "all_seed_marker_effects.csv",
    index=False
)

# ============================================================
# 9. ANIMAL OVERLAP ANALYSIS
# ============================================================

# Oxford trimaps: 1=foreground, 2=background, 3=boundary.
# Foreground occupancy and boundary occupancy are separate.

overlap_rows = []

for _, row in tqdm(
    val.iterrows(),
    total=len(val),
    desc="Calculating marker overlap"
):
    with Image.open(row.mask_path) as im:
        mask = np.asarray(
            im.resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.NEAREST
            )
        )

    for region in REGIONS:
        if region == "clean":
            continue

        x = int(row[f"{region}_x"])
        y = int(row[f"{region}_y"])

        patch_mask = mask[
            y:y+PATCH,
            x:x+PATCH
        ]

        overlap_rows.append({
            "image_id": row.image_id,
            "label": int(row.label),
            "region": region,
            "foreground_fraction": float(
                np.mean(patch_mask == 1)
            ),
            "boundary_fraction": float(
                np.mean(patch_mask == 3)
            ),
            "background_fraction": float(
                np.mean(patch_mask == 2)
            ),
            "x": x,
            "y": y
        })

overlap_df = pd.DataFrame(overlap_rows)

overlap_df.to_csv(
    RESULTS / "marker_animal_overlap.csv",
    index=False
)

overlap_summary = (
    overlap_df.groupby("region")
    .agg(
        mean_foreground_fraction=(
            "foreground_fraction", "mean"
        ),
        mean_boundary_fraction=(
            "boundary_fraction", "mean"
        ),
        mean_background_fraction=(
            "background_fraction", "mean"
        ),
        mean_x=("x", "mean"),
        mean_y=("y", "mean")
    )
    .reset_index()
)

overlap_summary.to_csv(
    RESULTS / "marker_overlap_summary.csv",
    index=False
)

# ============================================================
# 10. THREE-SEED SUMMARY
# ============================================================

seed_summary = (
    effects_df.groupby("region")
    .agg(
        mean_abs_change=("mean_abs_change", "mean"),
        sd_abs_change=("mean_abs_change", "std"),
        mean_flip_pct=("decision_flip_pct", "mean"),
        sd_flip_pct=("decision_flip_pct", "std"),
        mean_confidence_drop=(
            "mean_confidence_drop", "mean"
        ),
        sd_confidence_drop=(
            "mean_confidence_drop", "std"
        )
    )
    .reset_index()
)

seed_summary = seed_summary.merge(
    overlap_summary,
    on="region",
    how="left"
)

seed_summary.to_csv(
    RESULTS / "three_seed_summary.csv",
    index=False
)

# ============================================================
# 11. HIERARCHICAL BOOTSTRAP
# ============================================================

# Resample training seeds and validation images.
# This is an exploratory two-level uncertainty estimate.

rng = np.random.default_rng(2026)

bootstrap_rows = []

for region in REGIONS:
    if region == "clean":
        continue

    arrays = []

    for seed in SEEDS:
        path = (
            RESULTS /
            f"{region}_seed{seed}_paired.csv"
        )

        df = pd.read_csv(path)

        df = (
            df.set_index("image_id")
            .loc[val.image_id]
            .reset_index()
        )

        arrays.append(
            df.abs_change.to_numpy()
        )

    arr = np.stack(arrays, axis=0)

    estimates = []

    for _ in range(BOOTSTRAP_REPS):
        seed_idx = rng.integers(
            0, len(SEEDS), len(SEEDS)
        )

        image_idx = rng.integers(
            0, len(val), len(val)
        )

        estimates.append(
            arr[np.ix_(seed_idx, image_idx)].mean()
        )

    bootstrap_rows.append({
        "region": region,
        "mean_abs_change": arr.mean(),
        "ci_low": np.quantile(
            estimates, 0.025
        ),
        "ci_high": np.quantile(
            estimates, 0.975
        )
    })

bootstrap_df = pd.DataFrame(bootstrap_rows)

bootstrap_df.to_csv(
    RESULTS / "hierarchical_bootstrap.csv",
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

order = [
    "interior",
    "background",
    "center",
    "bottom_right"
]

plot_df = (
    seed_summary.set_index("region")
    .loc[order]
    .reset_index()
)

fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(order))

ax.bar(
    x,
    plot_df.mean_abs_change,
    yerr=plot_df.sd_abs_change,
    capsize=5
)

ax.set_xticks(x)
ax.set_xticklabels(order)

ax.set_ylabel(
    "Mean absolute change in P(dog)"
)

ax.set_title(
    "Marker sensitivity across three training seeds"
)

ax.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    FIGURES / "three_seed_marker_sensitivity.png",
    dpi=300
)

plt.close()

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    seed_summary.mean_foreground_fraction,
    seed_summary.mean_abs_change,
    s=90
)

for _, row in seed_summary.iterrows():
    ax.annotate(
        row.region,
        (
            row.mean_foreground_fraction,
            row.mean_abs_change
        ),
        xytext=(5, 5),
        textcoords="offset points"
    )

ax.set_xlabel(
    "Mean foreground overlap fraction"
)

ax.set_ylabel(
    "Mean absolute probability change"
)

ax.set_title(
    "Exploratory relationship: overlap and sensitivity"
)

ax.grid(alpha=0.2)

plt.tight_layout()

plt.savefig(
    FIGURES / "overlap_vs_sensitivity.png",
    dpi=300
)

plt.close()

# ============================================================
# 13. REPORT AND CONFIGURATION
# ============================================================

config = {
    "experiment": "multiseed_spatial_replication",
    "seeds": SEEDS,
    "new_training_seeds": NEW_SEEDS,
    "regions": REGIONS,
    "training_size": len(train),
    "validation_size": len(val),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "marker_correlation": 1.0,
    "checkpoint_selection": "final_epoch",
    "official_test_used": False,
    "bootstrap_repetitions": BOOTSTRAP_REPS
}

(CONFIGS / "experiment_config.json").write_text(
    json.dumps(config, indent=2)
)

report = {
    "status": "completed",
    "objective": (
        "Replicate spatial marker sensitivity across "
        "three training seeds and characterize "
        "marker-animal overlap."
    ),
    "important_caveats": [
        "Single fixed 148-image training subset",
        "Only three training seeds",
        "Object-relative and absolute positions are not coordinate matched",
        "Foreground overlap is descriptive, not causal identification",
        "Validation set has been used for exploratory analysis",
        "Official test remains untouched"
    ],
    "future_work": [
        "Coordinate-matched intervention",
        "Within-image counterfactual placements",
        "Patient-independent clinical replication where applicable",
        "Final held-out evaluation after protocol lock"
    ]
}

(RESULTS / "experiment_report.json").write_text(
    json.dumps(report, indent=2)
)

print("\n" + "=" * 70)
print("THREE-SEED SPATIAL SUMMARY")
print("=" * 70)

print(
    seed_summary.round(6).to_string(index=False)
)

print("\n" + "=" * 70)
print("HIERARCHICAL BOOTSTRAP")
print("=" * 70)

print(
    bootstrap_df.round(6).to_string(index=False)
)

print("\n" + "=" * 70)
print("MARKER OVERLAP SUMMARY")
print("=" * 70)

print(
    overlap_summary.round(5).to_string(index=False)
)

print("\nSaved results:", OUT)
print("Official test set untouched.")
