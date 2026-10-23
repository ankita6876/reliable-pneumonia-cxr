
import json
import random
import hashlib
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torch.utils.data import Dataset, DataLoader
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
OUT = EXP / "absolute_positions"

RESULTS = OUT / "results"
FIGURES = OUT / "figures"
CHECKPOINTS = OUT / "checkpoints"
CONFIGS = OUT / "configs"

for folder in [RESULTS, FIGURES, CHECKPOINTS, CONFIGS]:
    folder.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_FRACTION = 0.05
EPOCHS = 6
BATCH_SIZE = 24
IMAGE_SIZE = 224
PATCH = 24

BACKBONE_LR = 2e-5
HEAD_LR = 2e-4
WEIGHT_DECAY = 1e-4

REGIONS = ["top_left", "top_right", "bottom_left", "bottom_right", "center"]
BOOTSTRAP_REPS = 1000

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if DEVICE.type != "cuda":
    raise RuntimeError("Enable Kaggle GPU before running.")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True

print("Device:", DEVICE)
print("Training fraction:", TRAIN_FRACTION)
print("Epochs:", EPOCHS)

# ============================================================
# 2. LOAD EXISTING MANIFESTS AND PLACEMENTS
# ============================================================

BASE_RESULTS = EXP / "results"

train = pd.read_csv(
    BASE_RESULTS / "pilot_train_manifest.csv"
)
val = pd.read_csv(
    BASE_RESULTS / "pilot_val_manifest.csv"
)

train_pos = pd.read_csv(
    BASE_RESULTS / "pilot_train_marker_positions.csv"
)
val_pos = pd.read_csv(
    BASE_RESULTS / "pilot_val_marker_positions.csv"
)

assert len(train) == 2944
assert len(val) == 736

assert train.image_id.is_unique
assert val.image_id.is_unique

assert set(train.image_id).isdisjoint(set(val.image_id))

train = train.merge(
    train_pos.drop(
        columns=["label", "marker_type"],
        errors="ignore"
    ),
    on="image_id",
    validate="one_to_one"
)

val = val.merge(
    val_pos.drop(
        columns=["label", "marker_type"],
        errors="ignore"
    ),
    on="image_id",
    validate="one_to_one"
)

# Only images where ALL THREE marker regions are feasible.
# This prevents marker absence from varying across conditions.

def common_feasible(df):
    valid = np.ones(len(df), dtype=bool)

    for region in ["interior", "boundary", "background"]:
        valid &= (
            (df[f"{region}_x"].to_numpy() >= 0)
            & (df[f"{region}_y"].to_numpy() >= 0)
        )

    return df.loc[valid].copy().reset_index(drop=True)

train = pd.read_csv(
    EXP / "shortcut_pressure/results/small_train_manifest.csv"
)
val = pd.read_csv(
    EXP / "shortcut_pressure/results/common_feasible_validation_manifest.csv"
)

assert len(train) == 148
assert len(val) == 722

assert len(train) > 0
assert len(val) > 0

# ============================================================
# 3. BREED-STRATIFIED SMALL TRAINING SUBSET
# ============================================================

# Use the official globally unique breed class identifier.
# Every breed contributes at least one sample when possible.

assert "class_id" in train.columns, (
    "class_id missing from manifest; "
    "do not substitute species or breed_id."
)

rng = np.random.default_rng(SEED)

small_train = train.copy().reset_index(drop=True)

small_train.to_csv(
    RESULTS / "small_train_manifest.csv",
    index=False
)

val.to_csv(
    RESULTS / "common_feasible_validation_manifest.csv",
    index=False
)

print("\n=== DATASET ===")
print("Small train:", len(small_train))
print("Validation:", len(val))
print("Train class counts:")
print(small_train.label.value_counts().sort_index())
print("Validation class counts:")
print(val.label.value_counts().sort_index())


# Absolute-coordinate controls:
# top-left coordinates for a 24x24 marker on 224x224 images.

ABSOLUTE_POSITIONS = {
    "top_left": (8, 8),
    "top_right": (192, 8),
    "bottom_left": (8, 192),
    "bottom_right": (192, 192),
    "center": (100, 100)
}

for region, (x, y) in ABSOLUTE_POSITIONS.items():
    small_train[f"{region}_x"] = x
    small_train[f"{region}_y"] = y
    val[f"{region}_x"] = x
    val[f"{region}_y"] = y

# ============================================================
# 4. MARKER RENDERING
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
    if marker_type == 0:
        pattern = (xx // 3) % 2
    else:
        pattern = (yy // 3) % 2

    patch = np.where(
        pattern[..., None] == 1,
        240,
        15
    ).astype(np.uint8)

    PATTERNS[marker_type] = np.repeat(
        patch, 3, axis=2
    )


def render_marker(image, x, y, marker_type):
    image = image.copy()

    image[
        y:y+PATCH,
        x:x+PATCH
    ] = PATTERNS[marker_type]

    return image


# ============================================================
# 5. DATASET
# ============================================================

class PetDataset(Dataset):
    def __init__(self, df, region, condition):
        self.df = df.reset_index(drop=True)
        self.region = region
        self.condition = condition

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        with Image.open(row.image_path) as im:
            image = np.asarray(
                im.convert("RGB").resize(
                    (IMAGE_SIZE, IMAGE_SIZE)
                )
            )

        label = int(row.label)

        if self.region != "clean" and self.condition != "clean":
            x = int(row[f"{self.region}_x"])
            y = int(row[f"{self.region}_y"])

            assert x >= 0 and y >= 0

            if self.condition == "matching":
                marker_type = label
            elif self.condition == "opposing":
                marker_type = 1 - label
            else:
                raise ValueError(self.condition)

            image = render_marker(
                image, x, y, marker_type
            )

        tensor = (
            image.astype(np.float32)
            .transpose(2, 0, 1) / 255.0
        )

        tensor = (tensor - MEAN) / STD

        return (
            torch.from_numpy(tensor.copy()),
            label,
            str(row.image_id)
        )


def loader(df, region, condition, shuffle=False):
    return DataLoader(
        PetDataset(df, region, condition),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True
    )


# ============================================================
# 6. MODEL
# ============================================================

def create_model(pretrained=True):
    weights = (
        models.ResNet50_Weights.DEFAULT
        if pretrained else None
    )

    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(
        model.fc.in_features, 2
    )

    return model.to(DEVICE)


def calculate_metrics(labels, probs):
    labels = np.asarray(labels)
    probs = np.asarray(probs)

    preds = (probs >= 0.5).astype(int)

    return {
        "accuracy": float(
            accuracy_score(labels, preds)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, preds)
        ),
        "macro_f1": float(
            f1_score(labels, preds, average="macro")
        ),
        "auroc": float(
            roc_auc_score(labels, probs)
        ),
        "cat_recall": float(
            np.mean(preds[labels == 0] == 0)
        ),
        "dog_recall": float(
            np.mean(preds[labels == 1] == 1)
        )
    }


@torch.inference_mode()
def evaluate(model, data_loader):
    model.eval()

    labels_all = []
    probs_all = []
    ids_all = []

    for images, labels, image_ids in data_loader:
        images = images.to(
            DEVICE, non_blocking=True
        )

        with torch.autocast(
            "cuda", dtype=torch.float16
        ):
            logits = model(images)

        probs = torch.softmax(
            logits.float(), dim=1
        )[:, 1].cpu().numpy()

        labels_all.extend(labels.numpy().tolist())
        probs_all.extend(probs.tolist())
        ids_all.extend(image_ids)

    metrics = calculate_metrics(
        labels_all, probs_all
    )

    predictions = pd.DataFrame({
        "image_id": ids_all,
        "label": labels_all,
        "prob_dog": probs_all
    })

    return metrics, predictions


# ============================================================
# 7. TRAIN ALL FOUR MODELS
# ============================================================

all_metrics = []
all_effects = []

for region in REGIONS:
    print("\n" + "=" * 60)
    print("EXPERIMENT:", region.upper())
    print("=" * 60)

    checkpoint_path = (
        CHECKPOINTS / f"{region}_final.pt"
    )

    history_path = (
        RESULTS / f"{region}_training_history.csv"
    )

    if checkpoint_path.exists() and history_path.exists():
        print("Using existing completed checkpoint.")

        model = create_model(pretrained=False)

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:
        # Same initial weights for every experimental condition.
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        model = create_model(pretrained=True)

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

        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler("cuda")

        train_condition = (
            "clean" if region == "clean"
            else "matching"
        )

        train_loader = loader(
            small_train,
            region,
            train_condition,
            shuffle=True
        )

        history = []

        for epoch in range(1, EPOCHS + 1):
            model.train()

            loss_total = 0.0
            n_seen = 0

            for images, labels, _ in tqdm(
                train_loader,
                desc=f"{region} epoch {epoch}/{EPOCHS}"
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
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_total += (
                    float(loss.item()) * len(labels)
                )
                n_seen += len(labels)

            mean_loss = loss_total / n_seen

            history.append({
                "epoch": epoch,
                "train_loss": mean_loss
            })

            print(
                f"Epoch {epoch}/{EPOCHS}: "
                f"train loss = {mean_loss:.5f}"
            )

        pd.DataFrame(history).to_csv(
            history_path,
            index=False
        )

        # Fixed final epoch: no validation-based selection.
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "region": region,
                "seed": SEED,
                "epoch": EPOCHS,
                "train_fraction": TRAIN_FRACTION,
                "marker_correlation": (
                    None if region == "clean" else 1.0
                )
            },
            checkpoint_path
        )

    conditions = (
        ["clean"] if region == "clean"
        else ["clean", "matching", "opposing"]
    )

    pred_by_condition = {}

    for condition in conditions:
        metrics, predictions = evaluate(
            model,
            loader(val, region, condition)
        )

        pred_by_condition[condition] = (
            predictions.set_index("image_id")["prob_dog"]
        )

        all_metrics.append({
            "model": region,
            "condition": condition,
            **metrics
        })

        predictions.to_csv(
            RESULTS / f"{region}_{condition}_predictions.csv",
            index=False
        )

        print(
            f"{condition:9s} "
            f"F1={metrics['macro_f1']:.4f} "
            f"AUROC={metrics['auroc']:.4f} "
            f"BalancedAcc={metrics['balanced_accuracy']:.4f}"
        )

    if region != "clean":
        paired = val[["image_id", "label"]].copy()

        for condition, series in pred_by_condition.items():
            paired[f"p_{condition}"] = (
                paired.image_id.map(series)
            )

        paired["abs_marker_effect"] = (
            paired.p_matching
            - paired.p_opposing
        ).abs()

        paired["decision_flip"] = (
            (paired.p_matching >= 0.5)
            != (paired.p_opposing >= 0.5)
        )

        # Probability assigned to the correct class.
        paired["true_class_prob_matching"] = np.where(
            paired.label == 1,
            paired.p_matching,
            1 - paired.p_matching
        )

        paired["true_class_prob_opposing"] = np.where(
            paired.label == 1,
            paired.p_opposing,
            1 - paired.p_opposing
        )

        paired["true_class_confidence_drop"] = (
            paired.true_class_prob_matching
            - paired.true_class_prob_opposing
        )

        paired["region"] = region

        paired.to_csv(
            RESULTS / f"{region}_paired_effects.csv",
            index=False
        )

        # Paired image bootstrap; exploratory CIs.
        values = paired.abs_marker_effect.to_numpy()
        flips = paired.decision_flip.to_numpy().astype(float)
        drops = paired.true_class_confidence_drop.to_numpy()

        bootstrap_rng = np.random.default_rng(
            SEED + REGIONS.index(region)
        )

        boot_means = []
        boot_flips = []
        boot_drops = []

        for _ in range(BOOTSTRAP_REPS):
            idx = bootstrap_rng.integers(
                0, len(values), len(values)
            )

            boot_means.append(values[idx].mean())
            boot_flips.append(flips[idx].mean())
            boot_drops.append(drops[idx].mean())

        all_effects.append({
            "region": region,
            "n": len(paired),
            "mean_abs_probability_change": values.mean(),
            "mean_abs_change_ci_low": np.quantile(
                boot_means, 0.025
            ),
            "mean_abs_change_ci_high": np.quantile(
                boot_means, 0.975
            ),
            "decision_flip_pct": 100 * flips.mean(),
            "decision_flip_ci_low_pct": 100 * np.quantile(
                boot_flips, 0.025
            ),
            "decision_flip_ci_high_pct": 100 * np.quantile(
                boot_flips, 0.975
            ),
            "mean_true_class_confidence_drop": drops.mean(),
            "confidence_drop_ci_low": np.quantile(
                boot_drops, 0.025
            ),
            "confidence_drop_ci_high": np.quantile(
                boot_drops, 0.975
            )
        })

    pd.DataFrame(all_metrics).to_csv(
        RESULTS / "evaluation_metrics.csv",
        index=False
    )

    pd.DataFrame(all_effects).to_csv(
        RESULTS / "marker_effect_summary.csv",
        index=False
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 8. FIGURES
# ============================================================

metrics_df = pd.DataFrame(all_metrics)
effects_df = pd.DataFrame(all_effects)

fig, ax = plt.subplots(figsize=(10, 5))

for region in REGIONS:
    sub = metrics_df[
        metrics_df.model == region
    ]

    ax.plot(
        sub.condition,
        sub.macro_f1,
        marker="o",
        label=region
    )

ax.set_ylim(0, 1)
ax.set_ylabel("Macro-F1")
ax.set_title(
    "Absolute marker positions: 5% training subset"
)
ax.legend()
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(
    FIGURES / "shortcut_pressure_macro_f1.png",
    dpi=300
)
plt.close()

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    effects_df.region,
    effects_df.mean_abs_probability_change
)

ax.set_ylabel("Mean absolute change in P(dog)")
ax.set_title(
    "Matching vs opposing marker sensitivity"
)

plt.tight_layout()
plt.savefig(
    FIGURES / "shortcut_pressure_marker_effect.png",
    dpi=300
)
plt.close()

# ============================================================
# 9. SAVE CONFIGURATION AND REPORT
# ============================================================

config = {
    "experiment": "absolute_position_controls_5pct",
    "seed": SEED,
    "train_fraction": TRAIN_FRACTION,
    "train_size": len(small_train),
    "validation_size": len(val),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "backbone_lr": BACKBONE_LR,
    "head_lr": HEAD_LR,
    "weight_decay": WEIGHT_DECAY,
    "marker_correlation": 1.0,
    "checkpoint_selection": "final_epoch",
    "same_initial_weights": True,
    "common_feasible_images": True,
    "absolute_position_controls": True,
    "validation_conditions": [
        "clean",
        "matching",
        "opposing"
    ],
    "official_test_used": False,
    "bootstrap_repetitions": BOOTSTRAP_REPS
}

(CONFIGS / "experiment_config.json").write_text(
    json.dumps(config, indent=2)
)

report = {
    "status": "completed",
    "research_question": (
        "Does reduced clean training evidence increase "
        "reliance on label-correlated markers?"
    ),
    "limitations": [
        "Single training seed",
        "Exploratory subset size",
        "Marker placements have different coordinates and occlusion",
        "No causal isolation of object relevance",
        "Validation data is reused across exploratory studies",
        "Bootstrap intervals quantify sampling variation, not seed variation"
    ],
    "next_steps": [
        "Compare against the full-data pilot",
        "Add coordinate-matched controls",
        "Add four-corner and center placements",
        "Replicate selected conditions with multiple seeds"
    ]
}

(RESULTS / "experiment_report.json").write_text(
    json.dumps(report, indent=2)
)

print("\n" + "=" * 70)
print("FINAL SHORTCUT-PRESSURE RESULTS")
print("=" * 70)

print(
    metrics_df[
        [
            "model",
            "condition",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "auroc"
        ]
    ].round(5).to_string(index=False)
)

print("\n" + "=" * 70)
print("PAIRED MARKER EFFECTS")
print("=" * 70)

print(
    effects_df.round(6).to_string(index=False)
)

print("\nArtifacts:", OUT)
print("All checkpoints saved locally.")
print("Official test set NOT used.")
