# =====================================================================
# MSc THESIS — CIFAR-10 PHASE 5B
# CONTROLLED DATASET-SIZE ABLATION + SPATIAL SHORTCUT LEARNING
#
# TRAINING:
#   3 sizes: 5,000 / 15,000 / 45,000
#   3 seeds: 42 / 123 / 2026
#   4 marker reliabilities: independent / 50% / 75% / 100%
#   3 marker positions: center / bottom_right / random
#   3 clean baselines per size
#
# TOTAL: 117 MODELS
#
# SCIENTIFIC CONTROLS:
#   - Fixed 5,000-image stratified validation split
#   - Nested, stratified training subsets
#   - Same marker assignments across nested subsets
#   - Official test set untouched
#   - Fixed ImageNet-pretrained ResNet18 backbone
#   - Class-balanced marker codes
#   - Equal-size, approximately luminance-balanced marker patterns
#   - Independent-marker negative controls
#   - Counterfactual evaluation
#   - Three training seeds
#
# ENGINEERING:
#   - GPU feature extraction
#   - Lightweight linear-head training
#   - Resume-safe individual-run JSON files
#   - Compact artifacts only
#   - No Google Drive
#   - Secure optional GitHub push
# =====================================================================

import os
import sys
import gc
import json
import time
import random
import hashlib
import getpass
import tempfile
import subprocess
import warnings

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchvision import models
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss
)

warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================================
# 1. CONFIGURATION
# =====================================================================

ROOT = Path("/kaggle/working")
REPO = ROOT / "reliable-pneumonia-cxr"

DATA_DIR = (
    ROOT
    / "thesis_benchmark_expansion"
    / "datasets"
)

BASE = (
    REPO
    / "results"
    / "benchmark_expansion"
    / "cifar10"
)

RESULTS = BASE / "training"
RUNS = RESULTS / "individual_runs"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "publication_figures"

CODE_DIR = (
    REPO
    / "experiments"
    / "cifar10_spatial"
)

for directory in [
    RESULTS,
    RUNS,
    TABLES,
    FIGURES,
    CODE_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2026]

TRAINING_SIZES = [
    5000,
    15000,
    45000
]

RELIABILITIES = [
    "independent",
    0.50,
    0.75,
    1.00
]

POSITIONS = [
    "center",
    "bottom_right",
    "random"
]

NUM_CLASSES = 10

MARKER_SIZE = 6
MARKER_MARGIN = 2

BATCH_SIZE = 256
HEAD_BATCH_SIZE = 1024

MAX_EPOCHS = 30
PATIENCE = 6

LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

assert DEVICE.type == "cuda", (
    "Please enable GPU in Kaggle."
)

assert (REPO / ".git").exists(), (
    "Thesis GitHub repository not found."
)

print("=" * 78)
print("THESIS — CIFAR-10 PHASE 5B")
print("=" * 78)

print("Device:", DEVICE)
print("PyTorch:", torch.__version__)
print("Training sizes:", TRAINING_SIZES)
print("Seeds:", SEEDS)

# =====================================================================
# 2. REPRODUCIBILITY
# =====================================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# =====================================================================
# 3. LOAD EXISTING CIFAR-10 DATA
# =====================================================================

train_dataset = CIFAR10(
    root=str(DATA_DIR),
    train=True,
    download=False
)

X = np.asarray(train_dataset.data)

Y = np.asarray(
    train_dataset.targets,
    dtype=np.int64
)

assert X.shape == (50000, 32, 32, 3)

assert np.all(
    np.bincount(Y, minlength=10) == 5000
)

print("\nCIFAR-10 loaded from existing cache.")
print("Original training images:", len(X))

# =====================================================================
# 4. FIXED VALIDATION SPLIT
# =====================================================================

SPLIT_SEED = 20261028

rng = np.random.default_rng(SPLIT_SEED)

train_pool_indices = []
validation_indices = []

for cls in range(NUM_CLASSES):

    indices = np.flatnonzero(Y == cls)

    indices = rng.permutation(indices)

    validation_indices.extend(
        indices[:500]
    )

    train_pool_indices.extend(
        indices[500:]
    )

validation_indices = np.asarray(
    sorted(validation_indices),
    dtype=np.int64
)

train_pool_indices = np.asarray(
    sorted(train_pool_indices),
    dtype=np.int64
)

assert len(validation_indices) == 5000
assert len(train_pool_indices) == 45000

assert not set(validation_indices).intersection(
    set(train_pool_indices)
)

X_VAL = X[validation_indices]
Y_VAL = Y[validation_indices]

print("\nFixed validation split:", len(Y_VAL))
print("Available training pool:", len(train_pool_indices))

np.savez_compressed(
    TABLES / "fixed_split_indices.npz",
    validation_indices=validation_indices,
    train_pool_indices=train_pool_indices
)

# =====================================================================
# 5. NESTED STRATIFIED TRAINING SUBSETS
# =====================================================================

SUBSETS = {}

subset_records = []

for seed in SEEDS:

    subset_rng = np.random.default_rng(seed)

    class_orders = {}

    for cls in range(NUM_CLASSES):

        cls_indices = train_pool_indices[
            Y[train_pool_indices] == cls
        ]

        class_orders[cls] = subset_rng.permutation(
            cls_indices
        )

    previous = set()

    for size in TRAINING_SIZES:

        per_class = size // NUM_CLASSES

        indices = np.concatenate(
            [
                class_orders[cls][:per_class]
                for cls in range(NUM_CLASSES)
            ]
        )

        indices = np.sort(indices)

        assert len(indices) == size

        assert previous.issubset(
            set(indices.tolist())
        )

        previous = set(indices.tolist())

        SUBSETS[(seed, size)] = indices

        for index in indices:

            subset_records.append({
                "seed": seed,
                "training_size": size,
                "original_index": int(index),
                "class_label": int(Y[index])
            })

pd.DataFrame(
    subset_records
).to_csv(
    TABLES / "leakage_free_nested_subsets.csv",
    index=False
)

print("Nested training subsets verified.")

# =====================================================================
# 6. CLASS-CODE MARKER DESIGN
# =====================================================================

# Ten equal-area binary patterns.
# Every pattern has exactly 18 white and 18 black pixels.
#
# This balances foreground area and average luminance.
# Distinct patterns are deterministic.

PATTERNS = []

pattern_rng = np.random.default_rng(314159)

while len(PATTERNS) < NUM_CLASSES:

    flat = np.array(
        [255] * 18 + [0] * 18,
        dtype=np.uint8
    )

    pattern_rng.shuffle(flat)

    candidate = flat.reshape(
        MARKER_SIZE,
        MARKER_SIZE
    )

    if not any(
        np.array_equal(candidate, p)
        for p in PATTERNS
    ):
        PATTERNS.append(candidate)

PATTERNS = np.stack(PATTERNS)

assert PATTERNS.shape == (10, 6, 6)

assert all(
    np.sum(p == 255) == 18
    for p in PATTERNS
)

def assign_codes(labels, reliability, seed):

    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    rng = np.random.default_rng(seed)

    codes = np.empty_like(labels)

    if reliability == "independent":

        codes = np.arange(
            len(labels)
        ) % NUM_CLASSES

        return rng.permutation(codes)

    for cls in range(NUM_CLASSES):

        indices = rng.permutation(
            np.flatnonzero(labels == cls)
        )

        n_correct = int(
            round(
                len(indices)
                *
                float(reliability)
            )
        )

        codes[indices[:n_correct]] = cls

        incorrect = indices[n_correct:]

        wrong = rng.integers(
            0,
            NUM_CLASSES - 1,
            size=len(incorrect)
        )

        wrong += (wrong >= cls)

        codes[incorrect] = wrong

    return codes


def marker_coordinates(position, rng):

    if position == "center":

        return 13, 13

    if position == "bottom_right":

        return 24, 24

    if position == "random":

        return (
            int(rng.integers(2, 25)),
            int(rng.integers(2, 25))
        )

    raise ValueError(position)


def apply_marker(image, code, row, col):

    output = image.copy()

    pattern = PATTERNS[int(code)]

    output[
        row:row + MARKER_SIZE,
        col:col + MARKER_SIZE,
        :
    ] = pattern[..., None]

    return output


def apply_neutral(image, row, col):

    output = image.copy()

    output[
        row:row + MARKER_SIZE,
        col:col + MARKER_SIZE,
        :
    ] = 127

    return output


# =====================================================================
# 7. CONSISTENT MARKER ASSIGNMENT ACROSS TRAINING SIZES
# =====================================================================

# Marker codes are generated once for each seed/reliability
# over the entire 45,000-image training pool.
#
# Smaller subsets inherit the same assignments.

CODE_MAPS = {}

for seed in SEEDS:

    for reliability in RELIABILITIES:

        full_labels = Y[train_pool_indices]

        codes = assign_codes(
            full_labels,
            reliability,
            seed + 1000
        )

        CODE_MAPS[
            (seed, str(reliability))
        ] = dict(
            zip(
                train_pool_indices.tolist(),
                codes.tolist()
            )
        )


def create_marked_images(
    indices,
    seed,
    reliability,
    position
):

    code_map = CODE_MAPS[
        (seed, str(reliability))
    ]

    output = np.empty(
        (len(indices), 32, 32, 3),
        dtype=np.uint8
    )

    for i, index in enumerate(indices):

        index = int(index)

        coordinate_rng = np.random.default_rng(
            seed * 1000003 + index
        )

        row, col = marker_coordinates(
            position,
            coordinate_rng
        )

        output[i] = apply_marker(
            X[index],
            code_map[index],
            row,
            col
        )

    return output


# =====================================================================
# 8. PREPROCESSING
# =====================================================================

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
).reshape(3, 1, 1)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
).reshape(3, 1, 1)


class ImageDataset(Dataset):

    def __init__(self, images):

        self.images = images

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image = self.images[index]

        tensor = (
            image.transpose(2, 0, 1).astype(np.float32)
            / 255.0
        )

        tensor = (tensor - MEAN) / STD

        return torch.from_numpy(tensor)


# =====================================================================
# 9. PRETRAINED RESNET18
# =====================================================================

try:

    backbone = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )

except Exception as exc:

    raise RuntimeError(
        "Pretrained ResNet18 weights unavailable."
    ) from exc

backbone.fc = nn.Identity()

backbone = backbone.to(DEVICE).eval()

for parameter in backbone.parameters():
    parameter.requires_grad = False

print("\nPretrained ResNet18 loaded.")

# =====================================================================
# 10. GPU FEATURE EXTRACTION
# =====================================================================

@torch.inference_mode()
def extract_features(images):

    loader = DataLoader(
        ImageDataset(images),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    features = []

    for batch in loader:

        batch = batch.to(
            DEVICE,
            non_blocking=True
        )

        output = backbone(batch)

        features.append(
            output.cpu().numpy()
        )

    return np.concatenate(
        features,
        axis=0
    ).astype(np.float32)


# =====================================================================
# 11. VALIDATION COUNTERFACTUALS
# =====================================================================

def build_validation_variants(position):

    variants = {
        "clean": X_VAL.copy(),
        "matching": np.empty_like(X_VAL),
        "neutral": np.empty_like(X_VAL)
    }

    for code in range(NUM_CLASSES):

        variants[f"code_{code}"] = np.empty_like(
            X_VAL
        )

    for i, image in enumerate(X_VAL):

        coordinate_rng = np.random.default_rng(
            SPLIT_SEED + int(validation_indices[i])
        )

        row, col = marker_coordinates(
            position,
            coordinate_rng
        )

        variants["matching"][i] = apply_marker(
            image,
            Y_VAL[i],
            row,
            col
        )

        variants["neutral"][i] = apply_neutral(
            image,
            row,
            col
        )

        for code in range(NUM_CLASSES):

            variants[f"code_{code}"][i] = apply_marker(
                image,
                code,
                row,
                col
            )

    return variants


# =====================================================================
# 12. MULTICLASS METRICS
# =====================================================================

def calculate_metrics(labels, probabilities):

    predictions = probabilities.argmax(axis=1)

    result = {
        "accuracy": float(
            accuracy_score(
                labels,
                predictions
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions
            )
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0
            )
        ),
        "log_loss": float(
            log_loss(
                labels,
                probabilities,
                labels=list(range(NUM_CLASSES))
            )
        )
    }

    try:

        result["macro_auroc"] = float(
            roc_auc_score(
                labels,
                probabilities,
                multi_class="ovr",
                average="macro",
                labels=list(range(NUM_CLASSES))
            )
        )

    except ValueError:

        result["macro_auroc"] = np.nan

    return result


# =====================================================================
# 13. CLASSIFICATION HEAD
# =====================================================================

class Head(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc = nn.Linear(
            512,
            NUM_CLASSES
        )

    def forward(self, x):

        return self.fc(x)


def train_head(
    train_features,
    train_labels,
    val_features,
    val_labels,
    seed
):

    set_seed(seed)

    head = Head().to(DEVICE)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    x_train = torch.tensor(
        train_features,
        device=DEVICE
    )

    y_train = torch.tensor(
        train_labels,
        dtype=torch.long,
        device=DEVICE
    )

    x_val = torch.tensor(
        val_features,
        device=DEVICE
    )

    best_score = -1
    best_epoch = 0
    best_state = None

    patience = 0
    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        head.train()

        order = torch.randperm(
            len(y_train),
            device=DEVICE
        )

        losses = []

        for batch_indices in order.split(
            HEAD_BATCH_SIZE
        ):

            logits = head(
                x_train[batch_indices]
            )

            loss = criterion(
                logits,
                y_train[batch_indices]
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()
            optimizer.step()

            losses.append(
                float(loss.item())
            )

        head.eval()

        with torch.no_grad():

            logits = head(x_val)

            probs = torch.softmax(
                logits,
                dim=1
            ).cpu().numpy()

        score = calculate_metrics(
            val_labels,
            probs
        )["macro_f1"]

        history.append({
            "epoch": epoch,
            "train_loss": float(
                np.mean(losses)
            ),
            "val_macro_f1": score
        })

        if score > best_score + 1e-6:

            best_score = score
            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }

            patience = 0

        else:

            patience += 1

        if patience >= PATIENCE:
            break

    head.load_state_dict(best_state)

    return head, best_epoch, best_score, history


@torch.inference_mode()
def predict(head, features):

    head.eval()

    x = torch.tensor(
        features,
        device=DEVICE
    )

    return torch.softmax(
        head(x),
        dim=1
    ).cpu().numpy()


# =====================================================================
# 14. COUNTERFACTUAL EVALUATION
# =====================================================================

def evaluate_counterfactuals(
    head,
    validation_features
):

    probabilities = {
        name: predict(head, features)
        for name, features in validation_features.items()
    }

    metrics = []

    for name in [
        "clean",
        "matching",
        "neutral"
    ]:

        metrics.append({
            "evaluation": name,
            **calculate_metrics(
                Y_VAL,
                probabilities[name]
            )
        })

    code_probs = np.stack(
        [
            probabilities[f"code_{code}"]
            for code in range(NUM_CLASSES)
        ]
    )

    matching = probabilities["matching"]

    alternative_mean = (
        code_probs.sum(axis=0) - matching
    ) / (NUM_CLASSES - 1)

    metrics.append({
        "evaluation": "mean_alternative",
        **calculate_metrics(
            Y_VAL,
            alternative_mean
        )
    })

    matching_predictions = matching.argmax(axis=1)

    total_variation = []
    flip_rates = []

    for code in range(NUM_CLASSES):

        mask = Y_VAL != code

        alternatives = code_probs[code, mask]

        matching_subset = matching[mask]

        total_variation.extend(
            (
                np.abs(
                    matching_subset - alternatives
                ).sum(axis=1) / 2
            ).tolist()
        )

        flip_rates.extend(
            (
                matching_predictions[mask]
                != alternatives.argmax(axis=1)
            ).tolist()
        )

    matching_metrics = calculate_metrics(
        Y_VAL,
        matching
    )

    alternative_metrics = calculate_metrics(
        Y_VAL,
        alternative_mean
    )

    reliance = {
        "macro_auroc_gap": (
            matching_metrics["macro_auroc"]
            -
            alternative_metrics["macro_auroc"]
        ),
        "macro_f1_gap": (
            matching_metrics["macro_f1"]
            -
            alternative_metrics["macro_f1"]
        ),
        "mean_total_variation": float(
            np.mean(total_variation)
        ),
        "decision_flip_rate": float(
            np.mean(flip_rates)
        )
    }

    return metrics, reliance


# =====================================================================
# 15. VALIDATION FEATURE CACHE
# =====================================================================

VALIDATION_FEATURES = {}

for position in POSITIONS:

    print(
        "\nExtracting validation counterfactual features:",
        position
    )

    variants = build_validation_variants(
        position
    )

    VALIDATION_FEATURES[position] = {
        name: extract_features(images)
        for name, images in variants.items()
    }

    del variants
    gc.collect()

print("\nValidation counterfactual features ready.")

# =====================================================================
# 16. CLEAN TRAINING FEATURE CACHE
# =====================================================================

print("\nExtracting clean training features...")

CLEAN_POOL_FEATURES = extract_features(
    X[train_pool_indices]
)

POOL_LOOKUP = {
    int(index): position
    for position, index in enumerate(
        train_pool_indices
    )
}

print(
    "Clean training feature matrix:",
    CLEAN_POOL_FEATURES.shape
)

# =====================================================================
# 17. EXPERIMENT SCHEDULE
# =====================================================================

EXPERIMENTS = []

for seed in SEEDS:

    for size in TRAINING_SIZES:

        EXPERIMENTS.append({
            "seed": seed,
            "training_size": size,
            "reliability": "clean",
            "position": "none"
        })

        for reliability in RELIABILITIES:

            for position in POSITIONS:

                EXPERIMENTS.append({
                    "seed": seed,
                    "training_size": size,
                    "reliability": reliability,
                    "position": position
                })

assert len(EXPERIMENTS) == 117

print("\nScheduled models:", len(EXPERIMENTS))

# =====================================================================
# 18. TRAINING AND RESUMPTION
# =====================================================================

all_metrics = []
all_reliance = []
all_history = []

for run_number, config in enumerate(
    EXPERIMENTS,
    start=1
):

    seed = config["seed"]
    size = config["training_size"]
    reliability = config["reliability"]
    position = config["position"]

    reliability_name = (
        str(reliability).replace(".", "p")
    )

    run_id = (
        f"n{size}_seed{seed}_"
        f"{reliability_name}_{position}"
    )

    output_path = RUNS / f"{run_id}.json"

    print(
        f"\n[{run_number}/{len(EXPERIMENTS)}]",
        run_id
    )

    if output_path.exists():

        with open(output_path) as f:
            saved = json.load(f)

        all_metrics.extend(
            saved["metrics"]
        )

        all_reliance.extend(
            saved["reliance"]
        )

        all_history.extend(
            saved["history"]
        )

        print("Resumed from saved results.")

        continue

    indices = SUBSETS[(seed, size)]

    if reliability == "clean":

        pool_positions = np.asarray([
            POOL_LOOKUP[int(index)]
            for index in indices
        ])

        train_features = (
            CLEAN_POOL_FEATURES[pool_positions]
        )

        # Clean model selection.
        val_features = (
            VALIDATION_FEATURES["center"]["clean"]
        )

    else:

        marked_images = create_marked_images(
            indices,
            seed,
            reliability,
            position
        )

        train_features = extract_features(
            marked_images
        )

        del marked_images

        # Matching-marker validation is used for
        # selecting the shortcut-trained model.
        val_features = (
            VALIDATION_FEATURES[position]["matching"]
        )

    train_labels = Y[indices]

    head, best_epoch, best_score, history = train_head(
        train_features,
        train_labels,
        val_features,
        Y_VAL,
        seed
    )

    del train_features

    run_metrics = []
    run_reliance = []

    for evaluation_position in POSITIONS:

        metrics, reliance = evaluate_counterfactuals(
            head,
            VALIDATION_FEATURES[evaluation_position]
        )

        for row in metrics:

            run_metrics.append({
                "run_id": run_id,
                "seed": seed,
                "training_size": size,
                "training_reliability": str(reliability),
                "training_position": position,
                "evaluation_position": evaluation_position,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_score,
                **row
            })

        run_reliance.append({
            "run_id": run_id,
            "seed": seed,
            "training_size": size,
            "training_reliability": str(reliability),
            "training_position": position,
            "evaluation_position": evaluation_position,
            **reliance
        })

    history_rows = [
        {
            "run_id": run_id,
            "seed": seed,
            **row
        }
        for row in history
    ]

    saved = {
        "config": config,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "metrics": run_metrics,
        "reliance": run_reliance,
        "history": history_rows
    }

    temporary = output_path.with_suffix(
        ".json.tmp"
    )

    with open(temporary, "w") as f:
        json.dump(saved, f, indent=2)

    os.replace(
        temporary,
        output_path
    )

    all_metrics.extend(run_metrics)
    all_reliance.extend(run_reliance)
    all_history.extend(history_rows)

    print(
        "Best macro-F1:",
        round(best_score, 4),
        "| Epoch:",
        best_epoch
    )

    del head

    gc.collect()
    torch.cuda.empty_cache()

# =====================================================================
# 19. CONSOLIDATE
# =====================================================================

metrics_df = pd.DataFrame(all_metrics)
reliance_df = pd.DataFrame(all_reliance)
history_df = pd.DataFrame(all_history)

metrics_df.to_csv(
    TABLES / "all_evaluation_metrics.csv",
    index=False
)

reliance_df.to_csv(
    TABLES / "all_shortcut_reliance.csv",
    index=False
)

history_df.to_csv(
    TABLES / "training_history.csv",
    index=False
)

assert metrics_df["run_id"].nunique() == 117
assert reliance_df["run_id"].nunique() == 117

# =====================================================================
# 20. SIZE-ABLATION SUMMARY
# =====================================================================

matched = reliance_df[
    reliance_df["training_position"]
    ==
    reliance_df["evaluation_position"]
].copy()

summary = (
    matched.groupby(
        [
            "training_size",
            "training_reliability",
            "training_position"
        ],
        as_index=False
    )
    .agg(
        mean_auroc_gap=(
            "macro_auroc_gap",
            "mean"
        ),
        sd_auroc_gap=(
            "macro_auroc_gap",
            "std"
        ),
        mean_macro_f1_gap=(
            "macro_f1_gap",
            "mean"
        ),
        mean_decision_flip_rate=(
            "decision_flip_rate",
            "mean"
        ),
        mean_total_variation=(
            "mean_total_variation",
            "mean"
        )
    )
)

summary.to_csv(
    TABLES / "size_ablation_summary.csv",
    index=False
)

print("\n" + "=" * 78)
print("CIFAR-10 SIZE-ABLATION RESULTS")
print("=" * 78)

print(summary.to_string(index=False))

# =====================================================================
# 21. FIGURE — SIZE VS SHORTCUT RELIANCE
# =====================================================================

fig, ax = plt.subplots(figsize=(9, 5.5))

for position in POSITIONS:

    subset = summary[
        (summary["training_reliability"] == "1.0")
        &
        (summary["training_position"] == position)
    ].sort_values("training_size")

    ax.errorbar(
        subset["training_size"],
        subset["mean_auroc_gap"],
        yerr=subset["sd_auroc_gap"],
        marker="o",
        capsize=4,
        linewidth=2,
        label=position.replace("_", " ").title()
    )

ax.set_xlabel("Number of training images")

ax.set_ylabel(
    "Matching − alternative macro-AUROC"
)

ax.set_title(
    "CIFAR-10: Dataset Size and Shortcut Reliance"
)

ax.grid(alpha=0.2)
ax.legend(title="Marker location")

plt.tight_layout()

for extension in ["png", "pdf"]:

    fig.savefig(
        FIGURES
        / f"cifar10_size_vs_shortcut_reliance.{extension}",
        dpi=250,
        bbox_inches="tight"
    )

plt.show()
plt.close(fig)

# =====================================================================
# 22. FIGURE — SIZE VS DECISION FLIP RATE
# =====================================================================

fig, ax = plt.subplots(figsize=(9, 5.5))

for position in POSITIONS:

    subset = summary[
        (summary["training_reliability"] == "1.0")
        &
        (summary["training_position"] == position)
    ].sort_values("training_size")

    ax.plot(
        subset["training_size"],
        subset["mean_decision_flip_rate"] * 100,
        marker="o",
        linewidth=2,
        label=position.replace("_", " ").title()
    )

ax.set_xlabel("Number of training images")
ax.set_ylabel("Decision flip rate (%)")

ax.set_title(
    "CIFAR-10: Dataset Size and Counterfactual Sensitivity"
)

ax.grid(alpha=0.2)
ax.legend(title="Marker location")

plt.tight_layout()

for extension in ["png", "pdf"]:

    fig.savefig(
        FIGURES
        / f"cifar10_size_vs_flip_rate.{extension}",
        dpi=250,
        bbox_inches="tight"
    )

plt.show()
plt.close(fig)

# =====================================================================
# 23. SAVE SCIENTIFIC REPORT
# =====================================================================

report = f"""# CIFAR-10 Spatial Shortcut Size-Ablation Study

## Dataset
Official CIFAR-10, 10 classes.

## Training
Nested class-stratified subsets:
{TRAINING_SIZES}

## Validation
Fixed 5,000-image stratified subset from the official
training split.

## Test
Official 10,000-image test split reserved and not accessed.

## Architecture
ImageNet-pretrained ResNet18, frozen backbone,
linear ten-class classification head.

## Conditions
Three seeds: {SEEDS}
Reliability: {RELIABILITIES}
Position: {POSITIONS}

## Completed models
{metrics_df["run_id"].nunique()}

## Evaluation
Macro-AUROC, macro-F1, balanced accuracy, accuracy,
log loss, total variation, counterfactual decision flips.

## Limitations
- Synthetic shortcut markers.
- Frozen backbone.
- Validation split used for model selection and analysis.
- Marker pattern recognition may vary between codes.
- This experiment isolates sample size within CIFAR-10,
  but does not establish universal effects across domains.
- Held-out test evaluation is still required.
"""

REPORT_PATH = (
    RESULTS
    / "cifar10_scientific_report.md"
)

REPORT_PATH.write_text(
    report,
    encoding="utf-8"
)

# =====================================================================
# 24. SAVE MANIFEST
# =====================================================================

manifest = {
    "dataset": "CIFAR-10",
    "training_sizes": TRAINING_SIZES,
    "validation_size": 5000,
    "seeds": SEEDS,
    "reliabilities": [
        str(x) for x in RELIABILITIES
    ],
    "positions": POSITIONS,
    "marker_size": MARKER_SIZE,
    "architecture": "ResNet18",
    "weights": "IMAGENET1K_V1",
    "backbone_frozen": True,
    "head": "Linear(512,10)",
    "model_selection": "validation_macro_f1",
    "completed_models": int(
        metrics_df["run_id"].nunique()
    ),
    "test_split_accessed": False,
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat()
}

with open(
    RESULTS / "training_manifest.json",
    "w"
) as f:
    json.dump(manifest, f, indent=2)

# =====================================================================
# 25. PRESERVE EXECUTABLE CODE
# =====================================================================

source_saved = False

try:

    from IPython import get_ipython

    ip = get_ipython()

    candidates = [
        source
        for source in ip.history_manager.input_hist_raw
        if (
            "CIFAR-10 PHASE 5B"
            in source
            and
            "def train_head(" in source
            and
            "def evaluate_counterfactuals(" in source
        )
    ]

    if candidates:

        source_path = (
            CODE_DIR
            / "cifar10_phase5b_training.py"
        )

        source_path.write_text(
            candidates[-1],
            encoding="utf-8"
        )

        source_saved = True

except Exception as exc:

    print(
        "Source capture unavailable:",
        type(exc).__name__
    )

# =====================================================================
# 26. COMMIT RESEARCH ARTIFACTS
# =====================================================================

def git(*args, env=None, timeout=90):

    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env
    )

git(
    "config",
    "--local",
    "user.name",
    "Ankita Roy"
)

git(
    "config",
    "--local",
    "user.email",
    "thesis-research@users.noreply.github.com"
)

allowed_extensions = {
    ".csv",
    ".json",
    ".md",
    ".png",
    ".pdf",
    ".py",
    ".npz"
}

artifact_files = []

for directory in [BASE, CODE_DIR]:

    artifact_files.extend([
        path
        for path in directory.rglob("*")
        if (
            path.is_file()
            and
            path.suffix.lower() in allowed_extensions
            and
            path.stat().st_size < 20 * 1024 * 1024
        )
    ])

for path in artifact_files:

    git(
        "add",
        "--",
        str(path.relative_to(REPO))
    )

staged = git(
    "diff",
    "--cached",
    "--name-only"
)

staged_files = [
    line.strip()
    for line in staged.stdout.splitlines()
    if line.strip()
]

allowed_prefixes = (
    "results/benchmark_expansion/cifar10/",
    "experiments/cifar10_spatial/"
)

unexpected = [
    path
    for path in staged_files
    if not path.startswith(allowed_prefixes)
]

assert not unexpected, (
    f"Unexpected staged files: {unexpected}"
)

if staged_files:

    commit = git(
        "commit",
        "-m",
        "research: add CIFAR-10 controlled size ablation"
    )

    print(
        commit.stdout.strip()
        or commit.stderr.strip()
    )

local_commit = git(
    "rev-parse",
    "HEAD"
).stdout.strip()

print("\nLocal Git commit:", local_commit)

# =====================================================================
# 27. OPTIONAL SECURE GITHUB PUSH
# =====================================================================

answer = input(
    "\nPush research artifacts to GitHub now? [y/N]: "
).strip().lower()

remote_verified = False

if answer == "y":

    token = getpass.getpass(
        "Paste GitHub Personal Access Token: "
    ).strip()

    try:

        with tempfile.TemporaryDirectory() as temp_dir:

            askpass = Path(temp_dir) / "askpass.sh"

            askpass.write_text(
                "#!/bin/sh\n"
                "case \"$1\" in\n"
                "  *Username*) printf '%s\\n' 'x-access-token' ;;\n"
                "  *Password*) printf '%s\\n' \"$THESIS_GH_TOKEN\" ;;\n"
                "  *) exit 1 ;;\n"
                "esac\n"
            )

            askpass.chmod(0o700)

            env = os.environ.copy()

            env["GIT_ASKPASS"] = str(askpass)
            env["GIT_TERMINAL_PROMPT"] = "0"
            env["THESIS_GH_TOKEN"] = token

            push = git(
                "push",
                "origin",
                "main",
                env=env
            )

            remote_result = git(
                "ls-remote",
                "origin",
                "refs/heads/main",
                env=env
            )

            remote = (
                remote_result.stdout.split()[0]
                if (
                    remote_result.returncode == 0
                    and remote_result.stdout.strip()
                )
                else None
            )

            remote_verified = (
                push.returncode == 0
                and
                remote == local_commit
            )

            print(
                "\nGitHub remote verified:",
                remote_verified
            )

    finally:

        del token

        if "env" in locals():
            del env

# =====================================================================
# 28. FINAL STATUS
# =====================================================================

print("\n" + "=" * 78)
print("CIFAR-10 PHASE 5B COMPLETE")
print("=" * 78)

print("Models completed:", metrics_df["run_id"].nunique())
print("Evaluation rows:", len(metrics_df))
print("Reliance rows:", len(reliance_df))

print("Training sizes:", TRAINING_SIZES)
print("Validation size:", len(Y_VAL))

print("Official test split accessed: NO")
print("Executable source saved:", source_saved)
print("GitHub remote verified:", remote_verified)

print("Local Git commit:", local_commit)
print("Google Drive used: NO")

print("\nNEXT:")
print(
    "Review size-ablation results, then proceed "
    "to PathMNIST and final held-out evaluation."
)

print("=" * 78)