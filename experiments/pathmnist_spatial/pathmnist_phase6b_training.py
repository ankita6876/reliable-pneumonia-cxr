# ======================================================================
# MSc THESIS — PATHMNIST PHASE 6B
# FROZEN RESNET18 + SPATIAL SHORTCUT EVALUATION
#
# 39 MODELS:
#   3 seeds × (1 clean + 4 reliability levels × 3 positions)
#
# Saves:
#   - 39 trained classification heads
#   - Validation metrics and shortcut-reliance results
#   - Training histories
#   - Publication figures
#   - Reproducibility manifest
#   - Executable source
#   - Verified GitHub backup
#
# Does NOT evaluate the official test split.
# Does NOT use Google Drive.
# ======================================================================

import os
import sys
import gc
import json
import random
import getpass
import tempfile
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score
)

# ======================================================================
# 1. CONFIGURATION
# ======================================================================

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
    / "pathmnist"
)

TRAIN_DIR = BASE / "training"
RUN_DIR = TRAIN_DIR / "individual_runs"
HEAD_DIR = TRAIN_DIR / "heads"
TABLE_DIR = TRAIN_DIR / "tables"
FIG_DIR = TRAIN_DIR / "publication_figures"

CODE_DIR = (
    REPO
    / "experiments"
    / "pathmnist_spatial"
)

for directory in [
    RUN_DIR,
    HEAD_DIR,
    TABLE_DIR,
    FIG_DIR,
    CODE_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )

SEEDS = [42, 123, 2026]

RELIABILITIES = [
    "independent",
    0.5,
    0.75,
    1.0
]

POSITIONS = [
    "center",
    "bottom_right",
    "random"
]

NUM_CLASSES = 9
IMAGE_SIZE = 64
MARKER_SIZE = 8

BATCH_SIZE = 256
HEAD_BATCH_SIZE = 2048

MAX_EPOCHS = 35
PATIENCE = 7

LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0001

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

assert DEVICE.type == "cuda", (
    "Enable GPU in Kaggle before running."
)

assert (
    REPO / ".git"
).exists(), (
    "Thesis GitHub repository not found."
)

print("=" * 78)
print("THESIS — PATHMNIST PHASE 6B")
print("=" * 78)

print("Device:", DEVICE)
print("PyTorch:", torch.__version__)

# ======================================================================
# 2. REPRODUCIBILITY
# ======================================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# ======================================================================
# 3. LOAD OFFICIAL DATASET
# ======================================================================

try:

    import medmnist
    from medmnist import INFO

except ImportError:

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "medmnist"
        ],
        check=True
    )

    import medmnist
    from medmnist import INFO


info = INFO["pathmnist"]

DataClass = getattr(
    medmnist,
    info["python_class"]
)

train_dataset = DataClass(
    split="train",
    size=64,
    root=str(DATA_DIR),
    download=False
)

val_dataset = DataClass(
    split="val",
    size=64,
    root=str(DATA_DIR),
    download=False
)

X_TRAIN = np.asarray(
    train_dataset.imgs
)

Y_TRAIN = np.asarray(
    train_dataset.labels
).reshape(-1).astype(np.int64)

X_VAL = np.asarray(
    val_dataset.imgs
)

Y_VAL = np.asarray(
    val_dataset.labels
).reshape(-1).astype(np.int64)

assert X_TRAIN.shape == (
    89996,
    64,
    64,
    3
)

assert X_VAL.shape == (
    10004,
    64,
    64,
    3
)

print("\nPathMNIST loaded.")
print("Train:", len(Y_TRAIN))
print("Validation:", len(Y_VAL))

# ======================================================================
# 4. RECONSTRUCT PHASE 6A MARKER PATTERNS
# ======================================================================

pattern_rng = np.random.default_rng(
    314159
)

PATTERNS = []

while len(PATTERNS) < NUM_CLASSES:

    flat = np.array(
        [255] * 32 + [0] * 32,
        dtype=np.uint8
    )

    pattern_rng.shuffle(flat)

    candidate = flat.reshape(
        MARKER_SIZE,
        MARKER_SIZE
    )

    if not any(
        np.array_equal(
            candidate,
            existing
        )
        for existing in PATTERNS
    ):

        PATTERNS.append(
            candidate
        )

PATTERNS = np.stack(
    PATTERNS
)

def marker_coordinates(
    position,
    rng
):

    if position == "center":
        return 28, 28

    if position == "bottom_right":
        return 52, 52

    if position == "random":

        return (
            int(rng.integers(4, 53)),
            int(rng.integers(4, 53))
        )

    raise ValueError(position)


def apply_marker(
    image,
    code,
    row,
    col
):

    output = image.copy()

    pattern = PATTERNS[
        int(code)
    ]

    output[
        row:row + MARKER_SIZE,
        col:col + MARKER_SIZE,
        :
    ] = pattern[..., None]

    return output


def apply_neutral(
    image,
    row,
    col
):

    output = image.copy()

    output[
        row:row + MARKER_SIZE,
        col:col + MARKER_SIZE,
        :
    ] = 127

    return output


def assign_codes(
    labels,
    reliability,
    seed
):

    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    rng = np.random.default_rng(
        seed
    )

    codes = np.empty_like(
        labels
    )

    if reliability == "independent":

        for cls in range(NUM_CLASSES):

            indices = np.flatnonzero(
                labels == cls
            )

            codes_for_class = np.tile(
                np.arange(NUM_CLASSES),
                int(
                    np.ceil(
                        len(indices) / NUM_CLASSES
                    )
                )
            )[:len(indices)]

            rng.shuffle(
                codes_for_class
            )

            codes[indices] = (
                codes_for_class
            )

        return codes

    for cls in range(NUM_CLASSES):

        indices = rng.permutation(
            np.flatnonzero(
                labels == cls
            )
        )

        n_correct = round(
            len(indices)
            *
            float(reliability)
        )

        codes[
            indices[:n_correct]
        ] = cls

        incorrect = indices[
            n_correct:
        ]

        wrong = rng.integers(
            0,
            NUM_CLASSES - 1,
            size=len(incorrect)
        )

        wrong += (
            wrong >= cls
        )

        codes[
            incorrect
        ] = wrong

    return codes


# ======================================================================
# 5. IMAGE PREPROCESSING
# ======================================================================

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
).reshape(3, 1, 1)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
).reshape(3, 1, 1)


class ArrayDataset(Dataset):

    def __init__(
        self,
        images,
        labels=None,
        codes=None,
        position=None,
        mode="clean",
        seed=0
    ):

        self.images = images
        self.labels = labels
        self.codes = codes

        self.position = position
        self.mode = mode
        self.seed = seed

    def __len__(self):

        return len(
            self.images
        )

    def __getitem__(self, index):

        image = self.images[index]

        if self.mode != "clean":

            coordinate_rng = np.random.default_rng(
                self.seed * 1000003 + index
            )

            row, col = marker_coordinates(
                self.position,
                coordinate_rng
            )

            if self.mode == "neutral":

                image = apply_neutral(
                    image,
                    row,
                    col
                )

            else:

                image = apply_marker(
                    image,
                    self.codes[index],
                    row,
                    col
                )

        tensor = (
            image.transpose(2, 0, 1)
            .astype(np.float32)
            / 255.0
        )

        tensor = (
            tensor - MEAN
        ) / STD

        return torch.from_numpy(
            tensor
        )


# ======================================================================
# 6. PRETRAINED BACKBONE
# ======================================================================

backbone = models.resnet18(
    weights=models.ResNet18_Weights.IMAGENET1K_V1
)

backbone.fc = nn.Identity()

backbone = backbone.to(
    DEVICE
).eval()

for parameter in backbone.parameters():
    parameter.requires_grad = False

print("\nFrozen ResNet18 loaded.")

# ======================================================================
# 7. FEATURE EXTRACTION
# ======================================================================

@torch.inference_mode()
def extract_features(
    dataset
):

    loader = DataLoader(
        dataset,
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

        output = backbone(
            batch
        )

        features.append(
            output.cpu().numpy()
        )

    return np.concatenate(
        features,
        axis=0
    ).astype(np.float32)


# ======================================================================
# 8. VALIDATION FEATURE CACHE
# ======================================================================

# All evaluation conditions share the same
# validation images and are cached once.

VALIDATION_FEATURES = {}

for position in POSITIONS:

    print(
        "\nPreparing validation features:",
        position
    )

    VALIDATION_FEATURES[
        position
    ] = {}

    variants = {
        "clean": None,
        "matching": Y_VAL,
        "neutral": None
    }

    for code in range(NUM_CLASSES):

        variants[
            f"code_{code}"
        ] = np.full(
            len(Y_VAL),
            code,
            dtype=np.int64
        )

    for name, codes in variants.items():

        mode = (
            "clean"
            if name == "clean"
            else (
                "neutral"
                if name == "neutral"
                else "marked"
            )
        )

        dataset = ArrayDataset(
            X_VAL,
            labels=Y_VAL,
            codes=codes,
            position=position,
            mode=mode,
            seed=2026
        )

        features = extract_features(
            dataset
        )

        VALIDATION_FEATURES[
            position
        ][name] = features

print("\nValidation features cached.")

# ======================================================================
# 9. CLEAN TRAINING FEATURES
# ======================================================================

print(
    "\nExtracting clean training features..."
)

CLEAN_TRAIN_FEATURES = extract_features(
    ArrayDataset(
        X_TRAIN
    )
)

print(
    "Clean training features:",
    CLEAN_TRAIN_FEATURES.shape
)

# ======================================================================
# 10. TRAINABLE CLASSIFICATION HEAD
# ======================================================================

class Head(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc = nn.Linear(
            512,
            NUM_CLASSES
        )

    def forward(
        self,
        x
    ):

        return self.fc(x)


def calculate_metrics(
    labels,
    probabilities
):

    predictions = probabilities.argmax(
        axis=1
    )

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
        )
    }

    try:

        result["macro_auroc"] = float(
            roc_auc_score(
                labels,
                probabilities,
                multi_class="ovr",
                average="macro",
                labels=list(
                    range(NUM_CLASSES)
                )
            )
        )

    except ValueError:

        result["macro_auroc"] = np.nan

    return result


def train_head(
    train_features,
    train_labels,
    val_features,
    val_labels,
    seed
):

    set_seed(seed)

    head = Head().to(
        DEVICE
    )

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

    best_score = -1.0
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
                x_train[
                    batch_indices
                ]
            )

            loss = criterion(
                logits,
                y_train[
                    batch_indices
                ]
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()
            optimizer.step()

            losses.append(
                float(
                    loss.item()
                )
            )

        head.eval()

        with torch.no_grad():

            probabilities = torch.softmax(
                head(x_val),
                dim=1
            ).cpu().numpy()

        score = calculate_metrics(
            val_labels,
            probabilities
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

    head.load_state_dict(
        best_state
    )

    return (
        head,
        best_epoch,
        best_score,
        history
    )


@torch.inference_mode()
def predict(
    head,
    features
):

    head.eval()

    x = torch.tensor(
        features,
        device=DEVICE
    )

    probabilities = torch.softmax(
        head(x),
        dim=1
    )

    return probabilities.cpu().numpy()


# ======================================================================
# 11. COUNTERFACTUAL EVALUATION
# ======================================================================

def evaluate_counterfactuals(
    head,
    feature_dict
):

    probabilities = {
        name: predict(
            head,
            features
        )
        for name, features
        in feature_dict.items()
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

    code_probabilities = np.stack(
        [
            probabilities[
                f"code_{code}"
            ]
            for code in range(
                NUM_CLASSES
            )
        ]
    )

    matching = probabilities[
        "matching"
    ]

    alternative_mean = (
        code_probabilities.sum(
            axis=0
        ) - matching
    ) / (
        NUM_CLASSES - 1
    )

    alternative_metrics = calculate_metrics(
        Y_VAL,
        alternative_mean
    )

    metrics.append({
        "evaluation": "mean_alternative",
        **alternative_metrics
    })

    matching_metrics = calculate_metrics(
        Y_VAL,
        matching
    )

    total_variation = []
    flip_rates = []

    matching_predictions = matching.argmax(
        axis=1
    )

    for code in range(
        NUM_CLASSES
    ):

        mask = Y_VAL != code

        alternatives = (
            code_probabilities[
                code,
                mask
            ]
        )

        matching_subset = (
            matching[
                mask
            ]
        )

        total_variation.extend(
            (
                np.abs(
                    matching_subset
                    -
                    alternatives
                ).sum(axis=1) / 2
            ).tolist()
        )

        flip_rates.extend(
            (
                matching_predictions[
                    mask
                ]
                !=
                alternatives.argmax(
                    axis=1
                )
            ).tolist()
        )

    reliance = {
        "macro_auroc_gap": float(
            matching_metrics[
                "macro_auroc"
            ]
            -
            alternative_metrics[
                "macro_auroc"
            ]
        ),
        "macro_f1_gap": float(
            matching_metrics[
                "macro_f1"
            ]
            -
            alternative_metrics[
                "macro_f1"
            ]
        ),
        "decision_flip_rate": float(
            np.mean(
                flip_rates
            )
        ),
        "mean_total_variation": float(
            np.mean(
                total_variation
            )
        )
    }

    return (
        metrics,
        reliance
    )


# ======================================================================
# 12. EXPERIMENT SCHEDULE
# ======================================================================

EXPERIMENTS = []

for seed in SEEDS:

    EXPERIMENTS.append({
        "seed": seed,
        "reliability": "clean",
        "position": "none"
    })

    for reliability in RELIABILITIES:

        for position in POSITIONS:

            EXPERIMENTS.append({
                "seed": seed,
                "reliability": reliability,
                "position": position
            })

assert len(
    EXPERIMENTS
) == 39

print(
    "\nScheduled models:",
    len(EXPERIMENTS)
)

# ======================================================================
# 13. TRAIN ALL 39 MODELS
# ======================================================================

all_metrics = []
all_reliance = []
all_history = []

for run_number, config in enumerate(
    EXPERIMENTS,
    start=1
):

    seed = config[
        "seed"
    ]

    reliability = config[
        "reliability"
    ]

    position = config[
        "position"
    ]

    reliability_name = str(
        reliability
    ).replace(
        ".",
        "p"
    )

    run_id = (
        f"seed{seed}_"
        f"{reliability_name}_"
        f"{position}"
    )

    json_path = (
        RUN_DIR
        / f"{run_id}.json"
    )

    head_path = (
        HEAD_DIR
        / f"{run_id}.pt"
    )

    print(
        f"\n[{run_number}/39]",
        run_id
    )

    if (
        json_path.exists()
        and
        head_path.exists()
    ):

        with open(
            json_path
        ) as f:

            saved = json.load(
                f
            )

        all_metrics.extend(
            saved["metrics"]
        )

        all_reliance.extend(
            saved["reliance"]
        )

        all_history.extend(
            saved["history"]
        )

        print(
            "Resumed from saved results."
        )

        continue

    if reliability == "clean":

        train_features = (
            CLEAN_TRAIN_FEATURES
        )

        val_features = (
            VALIDATION_FEATURES[
                "center"
            ]["clean"]
        )

    else:

        codes = assign_codes(
            Y_TRAIN,
            reliability,
            seed + 1000
        )

        dataset = ArrayDataset(
            X_TRAIN,
            labels=Y_TRAIN,
            codes=codes,
            position=position,
            mode="marked",
            seed=seed
        )

        print(
            "Extracting marked training features..."
        )

        train_features = extract_features(
            dataset
        )

        val_features = (
            VALIDATION_FEATURES[
                position
            ]["matching"]
        )

    head, best_epoch, best_score, history = train_head(
        train_features,
        Y_TRAIN,
        val_features,
        Y_VAL,
        seed
    )

    if reliability != "clean":
        del train_features

    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu()
                for key, value
                in head.state_dict().items()
            },
            "seed": seed,
            "reliability": str(
                reliability
            ),
            "position": position,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_score,
            "num_classes": NUM_CLASSES,
            "backbone": "resnet18",
            "weights": "IMAGENET1K_V1"
        },
        head_path
    )

    run_metrics = []
    run_reliance = []

    for evaluation_position in POSITIONS:

        metrics, reliance = evaluate_counterfactuals(
            head,
            VALIDATION_FEATURES[
                evaluation_position
            ]
        )

        for row in metrics:

            run_metrics.append({
                "run_id": run_id,
                "seed": seed,
                "training_reliability": str(
                    reliability
                ),
                "training_position": position,
                "evaluation_position": evaluation_position,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_score,
                **row
            })

        run_reliance.append({
            "run_id": run_id,
            "seed": seed,
            "training_reliability": str(
                reliability
            ),
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
        "metrics": run_metrics,
        "reliance": run_reliance,
        "history": history_rows
    }

    temporary = json_path.with_suffix(
        ".json.tmp"
    )

    with open(
        temporary,
        "w"
    ) as f:

        json.dump(
            saved,
            f,
            indent=2
        )

    os.replace(
        temporary,
        json_path
    )

    all_metrics.extend(
        run_metrics
    )

    all_reliance.extend(
        run_reliance
    )

    all_history.extend(
        history_rows
    )

    print(
        "Best validation macro-F1:",
        round(
            best_score,
            4
        ),
        "| Epoch:",
        best_epoch
    )

    del head

    gc.collect()
    torch.cuda.empty_cache()

# ======================================================================
# 14. CONSOLIDATE RESULTS
# ======================================================================

metrics_df = pd.DataFrame(
    all_metrics
)

reliance_df = pd.DataFrame(
    all_reliance
)

history_df = pd.DataFrame(
    all_history
)

metrics_df.to_csv(
    TABLE_DIR
    / "pathmnist_validation_metrics.csv",
    index=False
)

reliance_df.to_csv(
    TABLE_DIR
    / "pathmnist_shortcut_reliance.csv",
    index=False
)

history_df.to_csv(
    TABLE_DIR
    / "pathmnist_training_history.csv",
    index=False
)

assert (
    metrics_df["run_id"].nunique()
    == 39
)

assert (
    reliance_df["run_id"].nunique()
    == 39
)

# ======================================================================
# 15. THREE-SEED SUMMARY
# ======================================================================

matched = reliance_df[
    reliance_df[
        "training_position"
    ]
    ==
    reliance_df[
        "evaluation_position"
    ]
].copy()

summary = (
    matched.groupby(
        [
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
    TABLE_DIR
    / "pathmnist_three_seed_summary.csv",
    index=False
)

print(
    "\nPATHMNIST THREE-SEED SUMMARY"
)

print(
    summary.to_string(
        index=False
    )
)

# ======================================================================
# 16. PUBLICATION FIGURES
# ======================================================================

fig, ax = plt.subplots(
    figsize=(9, 5.5)
)

for position in POSITIONS:

    subset = summary[
        summary[
            "training_position"
        ] == position
    ].copy()

    reliability_order = {
        "independent": 0,
        "0.5": 0.5,
        "0.75": 0.75,
        "1.0": 1.0
    }

    subset["x"] = subset[
        "training_reliability"
    ].map(
        reliability_order
    )

    subset = subset.sort_values(
        "x"
    )

    ax.errorbar(
        subset["x"],
        subset["mean_auroc_gap"],
        yerr=subset["sd_auroc_gap"],
        marker="o",
        linewidth=2,
        capsize=4,
        label=position.replace(
            "_",
            " "
        ).title()
    )

ax.set_xlabel(
    "Training marker reliability"
)

ax.set_ylabel(
    "Matching − alternative macro-AUROC"
)

ax.set_title(
    "PathMNIST: Shortcut Reliance by Marker Reliability"
)

ax.set_xticks(
    [0, 0.5, 0.75, 1.0]
)

ax.set_xticklabels(
    [
        "Independent",
        "50%",
        "75%",
        "100%"
    ]
)

ax.grid(
    alpha=0.2
)

ax.legend(
    title="Marker position"
)

plt.tight_layout()

for extension in [
    "png",
    "pdf"
]:

    fig.savefig(
        FIG_DIR
        / f"pathmnist_reliability_auroc_gap.{extension}",
        dpi=250,
        bbox_inches="tight"
    )

plt.show()
plt.close(fig)

fig, ax = plt.subplots(
    figsize=(9, 5.5)
)

for position in POSITIONS:

    subset = summary[
        summary[
            "training_position"
        ] == position
    ].copy()

    subset["x"] = subset[
        "training_reliability"
    ].map(
        reliability_order
    )

    subset = subset.sort_values(
        "x"
    )

    ax.plot(
        subset["x"],
        subset["mean_decision_flip_rate"] * 100,
        marker="o",
        linewidth=2,
        label=position.replace(
            "_",
            " "
        ).title()
    )

ax.set_xlabel(
    "Training marker reliability"
)

ax.set_ylabel(
    "Counterfactual decision flip rate (%)"
)

ax.set_title(
    "PathMNIST: Counterfactual Prediction Sensitivity"
)

ax.set_xticks(
    [0, 0.5, 0.75, 1.0]
)

ax.set_xticklabels(
    [
        "Independent",
        "50%",
        "75%",
        "100%"
    ]
)

ax.grid(
    alpha=0.2
)

ax.legend(
    title="Marker position"
)

plt.tight_layout()

for extension in [
    "png",
    "pdf"
]:

    fig.savefig(
        FIG_DIR
        / f"pathmnist_reliability_flip_rate.{extension}",
        dpi=250,
        bbox_inches="tight"
    )

plt.show()
plt.close(fig)

# ======================================================================
# 17. REPRODUCIBILITY MANIFEST
# ======================================================================

manifest = {
    "dataset": "PathMNIST",
    "dataset_resolution": 64,
    "training_images": len(
        Y_TRAIN
    ),
    "validation_images": len(
        Y_VAL
    ),
    "official_test_evaluated": False,
    "classes": NUM_CLASSES,
    "seeds": SEEDS,
    "reliabilities": [
        str(x)
        for x in RELIABILITIES
    ],
    "positions": POSITIONS,
    "architecture": "ResNet18",
    "backbone_weights": "IMAGENET1K_V1",
    "backbone_frozen": True,
    "head": "Linear(512,9)",
    "model_selection": "validation_macro_f1",
    "marker_size": MARKER_SIZE,
    "models_completed": int(
        metrics_df["run_id"].nunique()
    ),
    "head_checkpoints_saved": len(
        list(
            HEAD_DIR.glob("*.pt")
        )
    ),
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat()
}

(
    TRAIN_DIR
    / "pathmnist_training_manifest.json"
).write_text(
    json.dumps(
        manifest,
        indent=2
    ),
    encoding="utf-8"
)

# ======================================================================
# 18. CAPTURE EXECUTABLE SOURCE
# ======================================================================

source_saved = False

try:

    from IPython import get_ipython

    ip = get_ipython()

    candidates = [
        source
        for source
        in ip.history_manager.input_hist_raw
        if (
            "PATHMNIST PHASE 6B"
            in source
            and
            "def train_head(" in source
            and
            "pathmnist_three_seed_summary.csv"
            in source
        )
    ]

    if candidates:

        (
            CODE_DIR
            / "pathmnist_phase6b_training.py"
        ).write_text(
            candidates[-1],
            encoding="utf-8"
        )

        source_saved = True

except Exception as exc:

    print(
        "Source capture unavailable:",
        type(exc).__name__
    )

# ======================================================================
# 19. GIT COMMIT
# ======================================================================

def git(
    *args,
    env=None,
    timeout=120
):

    return subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            *args
        ],
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

# Save compact artifacts and checkpoints.
# The 39 linear heads are small and safe to preserve.

allowed_extensions = {
    ".csv",
    ".json",
    ".md",
    ".png",
    ".pdf",
    ".py",
    ".pt"
}

artifact_files = []

for directory in [
    BASE,
    CODE_DIR
]:

    artifact_files.extend([
        path
        for path in directory.rglob("*")
        if (
            path.is_file()
            and
            path.suffix.lower()
            in allowed_extensions
            and
            path.stat().st_size
            <
            20 * 1024 * 1024
        )
    ])

for path in artifact_files:

    git(
        "add",
        "--",
        str(
            path.relative_to(
                REPO
            )
        )
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
    "results/benchmark_expansion/pathmnist/",
    "experiments/pathmnist_spatial/"
)

unexpected = [
    path
    for path in staged_files
    if not path.startswith(
        allowed_prefixes
    )
]

assert not unexpected, (
    f"Unexpected staged files: {unexpected}"
)

if staged_files:

    commit = git(
        "commit",
        "-m",
        "research: add PathMNIST three-seed shortcut training"
    )

    print(
        commit.stdout.strip()
        or
        commit.stderr.strip()
    )

local_commit = git(
    "rev-parse",
    "HEAD"
).stdout.strip()

print(
    "\nLocal Git commit:",
    local_commit
)

# ======================================================================
# 20. SECURE GITHUB PUSH
# ======================================================================

answer = input(
    "\nPush PathMNIST results to GitHub? [y/N]: "
).strip().lower()

remote_verified = False

if answer == "y":

    token = getpass.getpass(
        "Paste GitHub Personal Access Token: "
    ).strip()

    try:

        with tempfile.TemporaryDirectory() as temp_dir:

            askpass = (
                Path(temp_dir)
                / "askpass.sh"
            )

            askpass.write_text(
                "#!/bin/sh\n"
                "case \"$1\" in\n"
                "  *Username*) printf '%s\\n' 'x-access-token' ;;\n"
                "  *Password*) printf '%s\\n' \"$THESIS_GH_TOKEN\" ;;\n"
                "  *) exit 1 ;;\n"
                "esac\n"
            )

            askpass.chmod(
                0o700
            )

            env = os.environ.copy()

            env["GIT_ASKPASS"] = str(
                askpass
            )

            env["GIT_TERMINAL_PROMPT"] = "0"

            env["THESIS_GH_TOKEN"] = token

            push = git(
                "push",
                "origin",
                "main",
                env=env,
                timeout=180
            )

            remote_result = git(
                "ls-remote",
                "origin",
                "refs/heads/main",
                env=env,
                timeout=60
            )

            remote = (
                remote_result.stdout.split()[0]
                if (
                    remote_result.returncode == 0
                    and
                    remote_result.stdout.strip()
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

# ======================================================================
# 21. FINAL STATUS
# ======================================================================

print("\n" + "=" * 78)
print("PATHMNIST PHASE 6B COMPLETE")
print("=" * 78)

print(
    "Models completed:",
    metrics_df["run_id"].nunique()
)

print(
    "Validation metric rows:",
    len(metrics_df)
)

print(
    "Shortcut reliance rows:",
    len(reliance_df)
)

print(
    "Saved classification heads:",
    len(
        list(
            HEAD_DIR.glob("*.pt")
        )
    )
)

print(
    "GitHub remote verified:",
    remote_verified
)

print(
    "Local commit:",
    local_commit
)

print(
    "Official test evaluated: False"
)

print(
    "Google Drive used: False"
)

print("\nNEXT:")
print(
    "Perform final held-out evaluation and "
    "seven-dataset scientific synthesis."
)

print("=" * 78)