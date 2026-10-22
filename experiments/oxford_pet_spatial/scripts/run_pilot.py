
import os
import json
import time
import random
import hashlib
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm.auto import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

REPO = Path("/kaggle/working/reliable-pneumonia-cxr")
EXP = REPO / "experiments/oxford_pet_spatial"

RESULTS = EXP / "results"
FIGURES = EXP / "figures"
CHECKPOINTS = EXP / "checkpoints"
CONFIGS = EXP / "configs"

for p in [RESULTS, FIGURES, CHECKPOINTS, CONFIGS]:
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
SIZE = 224
PATCH = 24
EPOCHS = 3
BATCH_SIZE = 24
NUM_WORKERS = 2

BACKBONE_LR = 2e-5
HEAD_LR = 2e-4
WEIGHT_DECAY = 1e-4

CORRELATION = 0.90

# Run clean baseline and three anatomical marker conditions.
EXPERIMENTS = [
    "clean",
    "interior",
    "boundary",
    "background"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if DEVICE.type != "cuda":
    raise RuntimeError(
        "GPU is required for this training run. "
        "Enable GPU in Kaggle settings."
    )

torch.backends.cudnn.benchmark = True

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("Device:", DEVICE)
print("PyTorch:", torch.__version__)
print("Experiments:", EXPERIMENTS)

# ============================================================
# LOAD SAVED DEVELOPMENT MANIFESTS
# ============================================================

train_df = pd.read_csv(
    RESULTS / "pilot_train_manifest.csv"
)

val_df = pd.read_csv(
    RESULTS / "pilot_val_manifest.csv"
)

assert len(train_df) == 2944
assert len(val_df) == 736
assert set(train_df.image_id).isdisjoint(
    set(val_df.image_id)
)

# ============================================================
# DETERMINISTIC UTILITIES
# ============================================================

def stable_seed(*items):
    raw = "|".join(str(x) for x in items)
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return int(digest[:8], 16)


def marker_type_for_image(image_id, label, split):
    """
    Marker type 0 or 1.
    Approximately 90% of each class receives matching type.
    Assignment is independent of image pixels.
    """
    rng = random.Random(
        stable_seed(SEED, image_id, split, "marker")
    )

    match = rng.random() < CORRELATION

    return int(label) if match else 1 - int(label)


# ============================================================
# PRECOMPUTE VALID MARKER PLACEMENTS
# ============================================================

def choose_position(mask, region, image_id):
    """
    mask:
      1 = foreground
      2 = background
      3 = boundary

    Interior/background: full 24x24 patch contained in region.
    Boundary: 24x24 patch centered on boundary pixel.
    """

    rng = np.random.default_rng(
        stable_seed(SEED, image_id, region, "position")
    )

    if region in ("interior", "background"):
        target = 1 if region == "interior" else 2

        binary = (mask == target).astype(np.uint8)

        # A valid top-left coordinate has a full PATCH x PATCH
        # square belonging to the target region.
        counts = cv2.boxFilter(
            binary,
            ddepth=cv2.CV_32S,
            ksize=(PATCH, PATCH),
            normalize=False,
            anchor=(0, 0),
            borderType=cv2.BORDER_CONSTANT
        )

        candidates = np.argwhere(
            counts[:SIZE-PATCH+1, :SIZE-PATCH+1]
            == PATCH * PATCH
        )

        if len(candidates) == 0:
            return None

        y, x = candidates[
            rng.integers(len(candidates))
        ]

        return int(x), int(y)

    if region == "boundary":
        ys, xs = np.where(mask == 3)

        valid = (
            (xs >= PATCH // 2)
            & (xs <= SIZE - PATCH // 2)
            & (ys >= PATCH // 2)
            & (ys <= SIZE - PATCH // 2)
        )

        xs = xs[valid]
        ys = ys[valid]

        if len(xs) == 0:
            return None

        k = rng.integers(len(xs))

        x = int(xs[k] - PATCH // 2)
        y = int(ys[k] - PATCH // 2)

        return x, y

    raise ValueError(region)


def build_placement_manifest(df, split):
    rows = []

    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc=f"Placements {split}"
    ):
        with Image.open(row.mask_path) as m:
            mask = np.asarray(
                m.resize(
                    (SIZE, SIZE),
                    Image.Resampling.NEAREST
                )
            )

        entry = {
            "image_id": row.image_id,
            "label": int(row.label),
            "marker_type": marker_type_for_image(
                row.image_id, row.label, split
            )
        }

        for region in ["interior", "boundary", "background"]:
            pos = choose_position(
                mask, region, row.image_id
            )

            entry[f"{region}_x"] = (
                pos[0] if pos else -1
            )
            entry[f"{region}_y"] = (
                pos[1] if pos else -1
            )

        rows.append(entry)

    return pd.DataFrame(rows)


train_positions_path = (
    RESULTS / "pilot_train_marker_positions.csv"
)
val_positions_path = (
    RESULTS / "pilot_val_marker_positions.csv"
)

if train_positions_path.exists() and val_positions_path.exists():
    train_positions = pd.read_csv(train_positions_path)
    val_positions = pd.read_csv(val_positions_path)

    expected_cols = {
        "image_id", "label", "marker_type",
        "interior_x", "interior_y",
        "boundary_x", "boundary_y",
        "background_x", "background_y"
    }

    if (
        len(train_positions) != len(train_df)
        or len(val_positions) != len(val_df)
        or not expected_cols.issubset(train_positions.columns)
        or not expected_cols.issubset(val_positions.columns)
    ):
        raise RuntimeError(
            "Existing placement manifests are incompatible."
        )
else:
    train_positions = build_placement_manifest(
        train_df, "train"
    )
    val_positions = build_placement_manifest(
        val_df, "val"
    )

    train_positions.to_csv(
        train_positions_path, index=False
    )
    val_positions.to_csv(
        val_positions_path, index=False
    )

print("\n=== MARKER AVAILABILITY ===")

for region in ["interior", "boundary", "background"]:
    available = (
        train_positions[f"{region}_x"] >= 0
    ).mean()

    print(region, f"{available:.2%}")

# ============================================================
# IMAGE DATASET
# ============================================================

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
).reshape(3, 1, 1)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
).reshape(3, 1, 1)


def apply_marker(image, x, y, marker_type):
    """
    Two equally sized high-contrast striped patterns.
    Same footprint; different stripe orientation.
    """
    image = image.copy()

    yy, xx = np.indices((PATCH, PATCH))

    if marker_type == 0:
        pattern = ((xx // 3) % 2).astype(np.uint8)
    else:
        pattern = ((yy // 3) % 2).astype(np.uint8)

    patch = np.where(
        pattern[..., None] == 1,
        240,
        15
    ).astype(np.uint8)

    patch = np.repeat(patch, 3, axis=2)

    image[
        y:y+PATCH,
        x:x+PATCH
    ] = patch

    return image


class PetDataset(Dataset):
    def __init__(
        self,
        df,
        positions,
        region="clean",
        evaluation="matching"
    ):
        self.df = df.reset_index(drop=True)
        self.region = region
        self.evaluation = evaluation

        self.positions = positions.set_index(
            "image_id"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        with Image.open(row.image_path) as im:
            image = np.asarray(
                im.convert("RGB").resize(
                    (SIZE, SIZE)
                )
            )

        if self.region != "clean" and self.evaluation != "clean":
            position = self.positions.loc[row.image_id]

            x = int(position[f"{self.region}_x"])
            y = int(position[f"{self.region}_y"])

            if x >= 0 and y >= 0:
                marker_type = int(position.marker_type)

                if self.evaluation == "reversed":
                    marker_type = 1 - marker_type

                image = apply_marker(
                    image, x, y, marker_type
                )

        tensor = image.astype(np.float32).transpose(
            2, 0, 1
        ) / 255.0

        tensor = (tensor - MEAN) / STD

        return (
            torch.from_numpy(tensor.copy()),
            torch.tensor(int(row.label), dtype=torch.long),
            row.image_id
        )


def make_loader(
    df,
    positions,
    region,
    evaluation,
    shuffle=False
):
    dataset = PetDataset(
        df, positions, region, evaluation
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )


# ============================================================
# MODEL
# ============================================================

def build_model():
    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features, 2
    )

    return model.to(DEVICE)


def compute_metrics(labels, probs):
    labels = np.asarray(labels)
    probs = np.asarray(probs)

    preds = probs.argmax(axis=1)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            preds,
            labels=[0, 1],
            average=None,
            zero_division=0
        )
    )

    return {
        "accuracy": float(
            accuracy_score(labels, preds)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, preds)
        ),
        "macro_f1": float(np.mean(f1)),
        "cat_f1": float(f1[0]),
        "dog_f1": float(f1[1]),
        "cat_recall": float(recall[0]),
        "dog_recall": float(recall[1]),
        "auroc": float(
            roc_auc_score(labels, probs[:, 1])
        )
    }


@torch.no_grad()
def evaluate(model, loader, condition, region):
    model.eval()

    all_labels = []
    all_probs = []
    all_ids = []

    for images, labels, image_ids in loader:
        images = images.to(
            DEVICE, non_blocking=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            logits = model(images)

        probs = torch.softmax(
            logits.float(), dim=1
        ).cpu().numpy()

        all_labels.extend(labels.numpy())
        all_probs.extend(probs)
        all_ids.extend(image_ids)

    all_probs = np.asarray(all_probs)

    metrics = compute_metrics(
        all_labels, all_probs
    )

    predictions = pd.DataFrame({
        "image_id": all_ids,
        "label": all_labels,
        "prob_cat": all_probs[:, 0],
        "prob_dog": all_probs[:, 1],
        "prediction": all_probs.argmax(axis=1)
    })

    return metrics, predictions


# ============================================================
# TRAINING
# ============================================================

def train_experiment(region):
    checkpoint_path = (
        CHECKPOINTS / f"pilot_{region}_best.pt"
    )

    history_path = (
        RESULTS / f"pilot_{region}_history.csv"
    )

    completion_path = (
        RESULTS / f"pilot_{region}_complete.json"
    )

    if (
        checkpoint_path.exists()
        and history_path.exists()
        and completion_path.exists()
    ):
        print(f"\nSkipping completed model: {region}")
        return checkpoint_path

    print(f"\n{'='*60}")
    print(f"TRAINING: {region.upper()}")
    print(f"{'='*60}")

    model = build_model()

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not n.startswith("fc.")
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

    train_loader = make_loader(
        train_df,
        train_positions,
        region,
        "matching",
        shuffle=True
    )

    # All models are selected on clean validation performance.
    val_loader = make_loader(
        val_df,
        val_positions,
        region,
        "clean"
    )

    best_score = -1
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0
        seen = 0

        progress = tqdm(
            train_loader,
            desc=f"{region} epoch {epoch}/{EPOCHS}"
        )

        for images, labels, _ in progress:
            images = images.to(
                DEVICE, non_blocking=True
            )
            labels = labels.to(
                DEVICE, non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):
                logits = model(images)
                loss = criterion(
                    logits, labels
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += (
                loss.item() * len(labels)
            )
            seen += len(labels)

            progress.set_postfix(
                loss=f"{running_loss/seen:.4f}"
            )

        metrics, _ = evaluate(
            model, val_loader,
            "clean", region
        )

        record = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            **metrics
        }

        history.append(record)

        pd.DataFrame(history).to_csv(
            history_path,
            index=False
        )

        print(
            f"Epoch {epoch}: "
            f"loss={record['train_loss']:.4f}, "
            f"clean_val_macro_f1={record['macro_f1']:.4f}, "
            f"clean_val_auroc={record['auroc']:.4f}"
        )

        # Select by clean-validation macro-F1.
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_macro_f1": best_score,
                    "region": region,
                    "seed": SEED,
                    "correlation": (
                        None if region == "clean"
                        else CORRELATION
                    )
                },
                checkpoint_path
            )

    completion_path.write_text(
        json.dumps({
            "region": region,
            "status": "completed",
            "epochs": EPOCHS,
            "best_clean_val_macro_f1": best_score
        }, indent=2)
    )

    del model
    torch.cuda.empty_cache()

    return checkpoint_path


# ============================================================
# RUN TRAINING AND EVALUATION
# ============================================================

config = {
    "seed": SEED,
    "image_size": SIZE,
    "patch_size": PATCH,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "backbone_lr": BACKBONE_LR,
    "head_lr": HEAD_LR,
    "weight_decay": WEIGHT_DECAY,
    "marker_correlation": CORRELATION,
    "experiments": EXPERIMENTS,
    "model": "resnet50_imagenet_pretrained",
    "selection_metric": "clean_validation_macro_f1",
    "test_set_used": False
}

(CONFIGS / "pilot_training_config.json").write_text(
    json.dumps(config, indent=2)
)

all_results = []

for region in EXPERIMENTS:
    checkpoint_path = train_experiment(region)

    model = build_model()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    conditions = (
        ["clean"]
        if region == "clean"
        else ["clean", "matching", "reversed"]
    )

    for condition in conditions:
        loader = make_loader(
            val_df,
            val_positions,
            region,
            condition
        )

        metrics, predictions = evaluate(
            model, loader,
            condition, region
        )

        predictions.to_csv(
            RESULTS / (
                f"pilot_{region}_{condition}_predictions.csv"
            ),
            index=False
        )

        all_results.append({
            "model": region,
            "condition": condition,
            **metrics
        })

        print(
            f"{region:12s} "
            f"{condition:9s} "
            f"F1={metrics['macro_f1']:.4f} "
            f"AUROC={metrics['auroc']:.4f}"
        )

    pd.DataFrame(all_results).to_csv(
        RESULTS / "pilot_evaluation_results.csv",
        index=False
    )

    del model
    torch.cuda.empty_cache()


# ============================================================
# FIGURES AND SUMMARY
# ============================================================

results = pd.DataFrame(all_results)

results.to_csv(
    RESULTS / "pilot_evaluation_results.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(9, 5))

for region in EXPERIMENTS:
    subset = results[
        results.model == region
    ]

    ax.plot(
        subset.condition,
        subset.macro_f1,
        marker="o",
        label=region
    )

ax.set_title(
    "Oxford-IIIT Pet: Pilot Macro-F1"
)
ax.set_ylabel("Macro-F1")
ax.set_xlabel("Evaluation condition")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(
    FIGURES / "pilot_macro_f1_comparison.png",
    dpi=300
)
plt.close()

fig, ax = plt.subplots(figsize=(9, 5))

for region in EXPERIMENTS:
    subset = results[
        results.model == region
    ]

    ax.plot(
        subset.condition,
        subset.auroc,
        marker="o",
        label=region
    )

ax.set_title(
    "Oxford-IIIT Pet: Pilot AUROC"
)
ax.set_ylabel("AUROC")
ax.set_xlabel("Evaluation condition")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(
    FIGURES / "pilot_auroc_comparison.png",
    dpi=300
)
plt.close()

print("\n" + "="*60)
print("FINAL PILOT RESULTS")
print("="*60)

print(
    results[
        [
            "model",
            "condition",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "auroc"
        ]
    ].round(4).to_string(index=False)
)

print("\nAll artifacts saved to:", EXP)
print("OFFICIAL TEST SET NOT USED.")
