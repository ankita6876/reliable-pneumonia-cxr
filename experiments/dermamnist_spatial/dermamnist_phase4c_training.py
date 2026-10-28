# =====================================================================
# MSc THESIS — PHASE 4C
# DERMAMNIST MULTICLASS SPATIAL SHORTCUT TRAINING
#
# 39 MODELS:
#   3 clean baselines
#   4 reliability conditions × 3 positions × 3 seeds = 36
#
# FEATURES:
#   - Frozen pretrained ResNet18
#   - Seven-class classification
#   - Class-weighted cross-entropy
#   - Independent, 50%, 75%, 100% marker conditions
#   - Center, bottom-right, random locations
#   - Three seeds: 42, 123, 2026
#   - Validation-only model selection and evaluation
#   - Multiclass counterfactual interventions
#   - Resume support
#   - Publication figures and tables
#   - Exact executable notebook-cell source preservation
#   - GitHub commit preparation
#
# NO GOOGLE DRIVE
# NO LARGE DATASETS OR CHECKPOINTS COMMITTED
# =====================================================================

import os
import sys
import gc
import json
import time
import random
import hashlib
import subprocess
import warnings
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
    roc_auc_score,
    log_loss,
    confusion_matrix
)

warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================================
# 1. CONFIGURATION
# =====================================================================

ROOT = Path("/kaggle/working")

REPO = ROOT / "reliable-pneumonia-cxr"

DATA_FILE = (
    ROOT
    / "thesis_benchmark_expansion"
    / "datasets"
    / "dermamnist_64.npz"
)

BASE = (
    REPO
    / "results"
    / "benchmark_expansion"
    / "dermamnist"
)

RESULTS = BASE / "training"

RUNS = RESULTS / "individual_runs"

FIGURES = RESULTS / "publication_figures"

TABLES = RESULTS / "tables"

CODE_DIR = (
    REPO
    / "experiments"
    / "dermamnist_spatial"
)

for directory in [
    RESULTS,
    RUNS,
    FIGURES,
    TABLES,
    CODE_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )

SEEDS = [42, 123, 2026]

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

NUM_CLASSES = 7

MARKER_SIZE = 6
MARGIN = 2

BATCH_SIZE = 128
HEAD_BATCH_SIZE = 256

MAX_EPOCHS = 25
PATIENCE = 5

LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

assert DEVICE.type == "cuda", (
    "Enable GPU acceleration in Kaggle before running."
)

assert DATA_FILE.exists(), (
    "Verified DermaMNIST dataset not found."
)

print("=" * 78)
print("THESIS — DERMAMNIST CONTROLLED SHORTCUT TRAINING")
print("=" * 78)

print("Device:", DEVICE)
print("PyTorch:", torch.__version__)
print("Dataset:", DATA_FILE)
print("Results:", RESULTS)

# =====================================================================
# 2. REPRODUCIBILITY
# =====================================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# 3. DATASET VERIFICATION
# =====================================================================

EXPECTED_MD5 = "b70a2f5635c6199aeaa28c31d7202e1f"


def calculate_md5(path):

    digest = hashlib.md5()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(4 * 1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


assert calculate_md5(DATA_FILE) == EXPECTED_MD5, (
    "DermaMNIST checksum mismatch."
)

with np.load(DATA_FILE, allow_pickle=False) as data:

    SPLITS = {}

    for split in ["train", "val"]:

        SPLITS[split] = {
            "images": data[f"{split}_images"].copy(),
            "labels": (
                data[f"{split}_labels"]
                .reshape(-1)
                .astype(np.int64)
                .copy()
            )
        }

assert len(SPLITS["train"]["labels"]) == 7007
assert len(SPLITS["val"]["labels"]) == 1003

TRAIN_LABELS = SPLITS["train"]["labels"]
VAL_LABELS = SPLITS["val"]["labels"]

print("\nDataset checksum: VERIFIED")
print("Training images:", len(TRAIN_LABELS))
print("Validation images:", len(VAL_LABELS))

# =====================================================================
# 4. MULTICLASS MARKER PROTOCOL
# =====================================================================

MARKER_COLORS = np.asarray(
    [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 255, 255]
    ],
    dtype=np.uint8
)


def marker_coordinates(height, width, position, rng):

    if position == "center":

        return (
            (height - MARKER_SIZE) // 2,
            (width - MARKER_SIZE) // 2
        )

    if position == "bottom_right":

        return (
            height - MARKER_SIZE - MARGIN,
            width - MARKER_SIZE - MARGIN
        )

    if position == "random":

        return (
            int(
                rng.integers(
                    MARGIN,
                    height - MARKER_SIZE - MARGIN + 1
                )
            ),
            int(
                rng.integers(
                    MARGIN,
                    width - MARKER_SIZE - MARGIN + 1
                )
            )
        )

    raise ValueError(position)


def apply_marker(image, code, row, col):

    out = image.copy()

    color = MARKER_COLORS[int(code)]

    out[
        row:row + MARKER_SIZE,
        col:col + MARKER_SIZE,
        :
    ] = color

    return out


def apply_neutral(image, row, col):

    out = image.copy()

    out[
        row:row + MARKER_SIZE,
        col:col + MARKER_SIZE,
        :
    ] = 127

    return out


def assign_codes(labels, reliability, seed):

    rng = np.random.default_rng(seed)

    labels = np.asarray(labels, dtype=np.int64)

    codes = np.empty_like(labels)

    if reliability == "independent":

        # Balanced marginal code counts, randomly assigned
        # without consulting class labels.
        codes = np.arange(len(labels)) % NUM_CLASSES
        codes = rng.permutation(codes)

        return codes.astype(np.int64)

    reliability = float(reliability)

    for cls in range(NUM_CLASSES):

        indices = np.flatnonzero(labels == cls)

        indices = rng.permutation(indices)

        n_match = int(
            round(len(indices) * reliability)
        )

        codes[indices[:n_match]] = cls

        incorrect = indices[n_match:]

        if len(incorrect):

            wrong = rng.integers(
                0,
                NUM_CLASSES - 1,
                size=len(incorrect)
            )

            wrong += (wrong >= cls)

            codes[incorrect] = wrong

    return codes


def generate_training_images(
    images,
    labels,
    reliability,
    position,
    seed
):

    codes = assign_codes(
        labels,
        reliability,
        seed
    )

    rng = np.random.default_rng(seed + 10000)

    output = np.empty_like(images)

    for i, image in enumerate(images):

        row, col = marker_coordinates(
            image.shape[0],
            image.shape[1],
            position,
            rng
        )

        output[i] = apply_marker(
            image,
            codes[i],
            row,
            col
        )

    return output


# =====================================================================
# 5. COUNTERFACTUAL EVALUATION
# =====================================================================

def build_validation_variants(position, seed):

    images = SPLITS["val"]["images"]
    labels = VAL_LABELS

    rng = np.random.default_rng(seed + 500000)

    variants = {
        "clean": images.copy(),
        "matching": np.empty_like(images),
        "neutral": np.empty_like(images)
    }

    # Seven fixed-code interventions allow us to
    # examine all possible marker identities without
    # arbitrarily choosing one "reversed" class.
    for code in range(NUM_CLASSES):

        variants[f"code_{code}"] = np.empty_like(images)

    for i, image in enumerate(images):

        row, col = marker_coordinates(
            image.shape[0],
            image.shape[1],
            position,
            rng
        )

        variants["matching"][i] = apply_marker(
            image,
            int(labels[i]),
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
# 6. IMAGE PREPROCESSING
# =====================================================================

class ImageDataset(Dataset):

    def __init__(self, images):

        self.images = images

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image = self.images[index]

        if image.ndim == 2:

            image = np.repeat(
                image[..., None],
                3,
                axis=2
            )

        tensor = torch.from_numpy(
            image.transpose(2, 0, 1).copy()
        ).float() / 255.0

        # ImageNet normalization for pretrained ResNet18.
        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

        tensor = (tensor - mean) / std

        return tensor


# =====================================================================
# 7. PRETRAINED FEATURE EXTRACTOR
# =====================================================================

set_seed(42)

weights = models.ResNet18_Weights.IMAGENET1K_V1

try:

    backbone = models.resnet18(
        weights=weights
    )

except Exception as exc:

    raise RuntimeError(
        "Pretrained ResNet18 weights unavailable. "
        "Training stopped to preserve recipe consistency."
    ) from exc

backbone.fc = nn.Identity()

backbone = backbone.to(DEVICE)
backbone.eval()

for parameter in backbone.parameters():
    parameter.requires_grad = False

FEATURE_DIM = 512

print("\nPretrained backbone: VERIFIED")
print("Feature dimension:", FEATURE_DIM)

# =====================================================================
# 8. FEATURE EXTRACTION
# =====================================================================

@torch.inference_mode()
def extract_features(images):

    loader = DataLoader(
        ImageDataset(images),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    outputs = []

    backbone.eval()

    for batch in loader:

        batch = batch.to(
            DEVICE,
            non_blocking=True
        )

        features = backbone(batch)

        outputs.append(
            features.cpu().numpy()
        )

    return np.concatenate(
        outputs,
        axis=0
    ).astype(np.float32)


# Clean features are reused by all clean baseline runs.
print("\nExtracting clean training features...")

CLEAN_TRAIN_FEATURES = extract_features(
    SPLITS["train"]["images"]
)

print("Clean training features:", CLEAN_TRAIN_FEATURES.shape)

# =====================================================================
# 9. CLASS WEIGHTS
# =====================================================================

counts = np.bincount(
    TRAIN_LABELS,
    minlength=NUM_CLASSES
)

class_weights = (
    len(TRAIN_LABELS)
    /
    (
        NUM_CLASSES
        * np.maximum(counts, 1)
    )
)

# Normalize for stable loss scale.
class_weights = (
    class_weights
    /
    class_weights.mean()
)

CLASS_WEIGHTS = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)

print("\nClass counts:", counts.tolist())
print("Class weights:", class_weights.round(4).tolist())

# =====================================================================
# 10. CLASSIFIER HEAD
# =====================================================================

class ClassificationHead(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc = nn.Linear(
            FEATURE_DIM,
            NUM_CLASSES
        )

    def forward(self, features):

        return self.fc(features)


# =====================================================================
# 11. MULTICLASS METRICS
# =====================================================================

def calculate_metrics(y, probabilities):

    y = np.asarray(y)

    probabilities = np.asarray(probabilities)

    predictions = probabilities.argmax(axis=1)

    result = {
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y,
                predictions,
                average="macro",
                zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                y,
                predictions,
                average="weighted",
                zero_division=0
            )
        ),
        "log_loss": float(
            log_loss(
                y,
                probabilities,
                labels=list(range(NUM_CLASSES))
            )
        )
    }

    try:

        result["macro_auroc"] = float(
            roc_auc_score(
                y,
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
# 12. TRAIN CLASSIFIER HEAD
# =====================================================================

def train_head(
    train_features,
    train_labels,
    val_features,
    val_labels,
    seed
):

    set_seed(seed)

    head = ClassificationHead().to(DEVICE)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss(
        weight=CLASS_WEIGHTS
    )

    x_train = torch.tensor(
        train_features,
        dtype=torch.float32,
        device=DEVICE
    )

    y_train = torch.tensor(
        train_labels,
        dtype=torch.long,
        device=DEVICE
    )

    x_val = torch.tensor(
        val_features,
        dtype=torch.float32,
        device=DEVICE
    )

    best_score = -np.inf
    best_state = None
    best_epoch = 0

    patience_counter = 0

    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        head.train()

        permutation = torch.randperm(
            len(y_train),
            device=DEVICE
        )

        losses = []

        for indices in permutation.split(
            HEAD_BATCH_SIZE
        ):

            logits = head(
                x_train[indices]
            )

            loss = criterion(
                logits,
                y_train[indices]
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

            val_logits = head(x_val)

            val_probabilities = (
                torch.softmax(
                    val_logits,
                    dim=1
                )
                .cpu()
                .numpy()
            )

        val_metrics = calculate_metrics(
            val_labels,
            val_probabilities
        )

        # Macro-F1 is the prespecified selection metric.
        score = val_metrics["macro_f1"]

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_macro_f1": score,
            "val_macro_auroc": val_metrics["macro_auroc"],
            "val_balanced_accuracy": (
                val_metrics["balanced_accuracy"]
            )
        })

        if score > best_score + 1e-6:

            best_score = score
            best_epoch = epoch

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in head.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    head.load_state_dict(best_state)

    return (
        head,
        best_epoch,
        best_score,
        history
    )


# =====================================================================
# 13. FEATURE-BASED PREDICTION
# =====================================================================

@torch.inference_mode()
def predict_head(head, features):

    head.eval()

    tensor = torch.tensor(
        features,
        dtype=torch.float32,
        device=DEVICE
    )

    logits = head(tensor)

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    return probabilities.cpu().numpy()


# =====================================================================
# 14. VALIDATION FEATURE CACHE
# =====================================================================

# Cache one evaluation position at a time.
# Avoid storing all large image variants or features on disk.

def extract_validation_feature_variants(position, seed):

    variants = build_validation_variants(
        position,
        seed
    )

    features = {}

    for name, images in variants.items():

        features[name] = extract_features(
            images
        )

    return features


# =====================================================================
# 15. COUNTERFACTUAL RELIANCE
# =====================================================================

def evaluate_counterfactuals(
    head,
    feature_variants
):

    probabilities = {
        name: predict_head(
            head,
            features
        )
        for name, features in feature_variants.items()
    }

    metric_rows = []

    for name in [
        "clean",
        "matching",
        "neutral"
    ]:

        metric_rows.append({
            "evaluation": name,
            **calculate_metrics(
                VAL_LABELS,
                probabilities[name]
            )
        })

    # Counterfactual alternative:
    # For each image, average the predictions from
    # all six marker codes that are NOT its true label.

    code_probs = np.stack(
        [
            probabilities[f"code_{i}"]
            for i in range(NUM_CLASSES)
        ],
        axis=0
    )

    n = len(VAL_LABELS)

    matching = probabilities["matching"]

    alternative_sum = (
        code_probs.sum(axis=0)
        -
        matching
    )

    alternative_mean = (
        alternative_sum
        /
        (NUM_CLASSES - 1)
    )

    # AUROC under the averaged alternative probability
    # distribution is a descriptive summary, not an
    # individual realizable marker intervention.
    alternative_metrics = calculate_metrics(
        VAL_LABELS,
        alternative_mean
    )

    metric_rows.append({
        "evaluation": "mean_alternative",
        **alternative_metrics
    })

    matching_pred = matching.argmax(axis=1)

    neutral_pred = probabilities[
        "neutral"
    ].argmax(axis=1)

    # Every alternative marker code is evaluated.
    alternative_differences = []
    alternative_flips = []

    for i in range(n):

        label = int(VAL_LABELS[i])

        for code in range(NUM_CLASSES):

            if code == label:
                continue

            p_alt = code_probs[code, i]

            alternative_differences.append(
                np.abs(
                    matching[i] - p_alt
                ).sum() / 2
            )

            alternative_flips.append(
                matching_pred[i]
                !=
                np.argmax(p_alt)
            )

    reliance = {
        "matching_alternative_macro_auroc_gap": (
            calculate_metrics(
                VAL_LABELS,
                matching
            )["macro_auroc"]
            -
            alternative_metrics["macro_auroc"]
        ),
        "matching_alternative_macro_f1_gap": (
            calculate_metrics(
                VAL_LABELS,
                matching
            )["macro_f1"]
            -
            alternative_metrics["macro_f1"]
        ),
        "mean_total_variation_change": float(
            np.mean(
                alternative_differences
            )
        ),
        "mean_decision_flip_rate": float(
            np.mean(
                alternative_flips
            )
        ),
        "matching_neutral_decision_flip_rate": float(
            np.mean(
                matching_pred != neutral_pred
            )
        ),
        "matching_clean_total_variation": float(
            np.mean(
                np.abs(
                    matching
                    -
                    probabilities["clean"]
                ).sum(axis=1) / 2
            )
        )
    }

    return metric_rows, reliance


# =====================================================================
# 16. EXPERIMENT SCHEDULE
# =====================================================================

EXPERIMENTS = []

for seed in SEEDS:

    EXPERIMENTS.append({
        "condition": "clean",
        "reliability": None,
        "position": "none",
        "seed": seed
    })

    for reliability in RELIABILITIES:

        for position in POSITIONS:

            EXPERIMENTS.append({
                "condition": "shortcut",
                "reliability": reliability,
                "position": position,
                "seed": seed
            })

assert len(EXPERIMENTS) == 39

print("\nScheduled models:", len(EXPERIMENTS))

# =====================================================================
# 17. RESUME-SAFE TRAINING
# =====================================================================

all_metrics = []
all_reliance = []
all_history = []

# Fixed validation marker placements for each evaluation
# seed and location. Each model is evaluated at all three
# locations to measure spatial transfer.

for run_number, config in enumerate(
    EXPERIMENTS,
    start=1
):

    condition = config["condition"]
    reliability = config["reliability"]
    position = config["position"]
    seed = config["seed"]

    if condition == "clean":

        run_id = f"clean_seed{seed}"

    else:

        reliability_name = (
            "independent"
            if reliability == "independent"
            else f"r{int(reliability * 100)}"
        )

        run_id = (
            f"{reliability_name}_"
            f"{position}_seed{seed}"
        )

    RUN_FILE = RUNS / f"{run_id}.json"

    print("\n" + "=" * 78)
    print(
        f"RUN {run_number}/{len(EXPERIMENTS)}:",
        run_id
    )
    print("=" * 78)

    if RUN_FILE.exists():

        with open(RUN_FILE) as f:
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

        print("Previously completed. Results loaded.")

        continue

    start_time = time.time()

    if condition == "clean":

        train_features = CLEAN_TRAIN_FEATURES

    else:

        marked_train = generate_training_images(
            SPLITS["train"]["images"],
            TRAIN_LABELS,
            reliability,
            position,
            seed
        )

        train_features = extract_features(
            marked_train
        )

        del marked_train

    # Model-selection validation images are generated
    # using the same training marker protocol.
    # Test images are not accessed.

    if condition == "clean":

        val_images = SPLITS["val"]["images"]

    else:

        val_images = generate_training_images(
            SPLITS["val"]["images"],
            VAL_LABELS,
            reliability,
            position,
            seed + 100000
        )

    val_features = extract_features(
        val_images
    )

    del val_images

    head, best_epoch, best_score, history = train_head(
        train_features,
        TRAIN_LABELS,
        val_features,
        VAL_LABELS,
        seed
    )

    if condition != "clean":
        del train_features

    del val_features

    run_metrics = []
    run_reliance = []

    for evaluation_position in POSITIONS:

        feature_variants = (
            extract_validation_feature_variants(
                evaluation_position,
                seed
            )
        )

        metric_rows, reliance = (
            evaluate_counterfactuals(
                head,
                feature_variants
            )
        )

        for row in metric_rows:

            run_metrics.append({
                "run_id": run_id,
                "condition": condition,
                "train_reliability": (
                    str(reliability)
                ),
                "train_position": position,
                "seed": seed,
                "evaluation_position": (
                    evaluation_position
                ),
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_score,
                **row
            })

        run_reliance.append({
            "run_id": run_id,
            "condition": condition,
            "train_reliability": (
                str(reliability)
            ),
            "train_position": position,
            "seed": seed,
            "evaluation_position": (
                evaluation_position
            ),
            **reliance
        })

        del feature_variants
        gc.collect()

    history_rows = [
        {
            "run_id": run_id,
            "seed": seed,
            **row
        }
        for row in history
    ]

    saved = {
        "run_id": run_id,
        "config": config,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "metrics": run_metrics,
        "reliance": run_reliance,
        "history": history_rows,
        "elapsed_seconds": (
            time.time() - start_time
        )
    }

    # Atomic result save: interrupted writes do not
    # create a falsely completed run.
    temporary = RUN_FILE.with_suffix(".json.tmp")

    with open(temporary, "w") as f:
        json.dump(saved, f, indent=2)

    os.replace(temporary, RUN_FILE)

    all_metrics.extend(run_metrics)
    all_reliance.extend(run_reliance)
    all_history.extend(history_rows)

    print(
        "Best validation macro-F1:",
        round(best_score, 4)
    )

    print("Best epoch:", best_epoch)

    print(
        "Elapsed seconds:",
        round(saved["elapsed_seconds"], 1)
    )

    del head

    gc.collect()
    torch.cuda.empty_cache()

# =====================================================================
# 18. CONSOLIDATE RESULTS
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

assert metrics_df["run_id"].nunique() == 39
assert reliance_df["run_id"].nunique() == 39

# =====================================================================
# 19. THREE-SEED SUMMARY
# =====================================================================

shortcut = reliance_df[
    reliance_df["condition"] == "shortcut"
].copy()

summary = (
    shortcut.groupby(
        [
            "train_reliability",
            "train_position",
            "evaluation_position"
        ],
        as_index=False
    )
    .agg(
        mean_macro_auroc_gap=(
            "matching_alternative_macro_auroc_gap",
            "mean"
        ),
        sd_macro_auroc_gap=(
            "matching_alternative_macro_auroc_gap",
            "std"
        ),
        mean_macro_f1_gap=(
            "matching_alternative_macro_f1_gap",
            "mean"
        ),
        mean_total_variation_change=(
            "mean_total_variation_change",
            "mean"
        ),
        mean_decision_flip_rate=(
            "mean_decision_flip_rate",
            "mean"
        )
    )
)

summary.to_csv(
    TABLES / "three_seed_shortcut_summary.csv",
    index=False
)

print("\n" + "=" * 78)
print("DERMAMNIST THREE-SEED SHORTCUT SUMMARY")
print("=" * 78)

print(summary.to_string(index=False))

# =====================================================================
# 20. MATCHED-LOCATION ANALYSIS
# =====================================================================

matched = summary[
    summary["train_position"]
    ==
    summary["evaluation_position"]
].copy()

matched.to_csv(
    TABLES / "matched_location_summary.csv",
    index=False
)

print("\n" + "=" * 78)
print("MATCHED-LOCATION RESULTS")
print("=" * 78)

print(
    matched[
        [
            "train_reliability",
            "train_position",
            "mean_macro_auroc_gap",
            "mean_macro_f1_gap",
            "mean_total_variation_change",
            "mean_decision_flip_rate"
        ]
    ].to_string(index=False)
)

# =====================================================================
# 21. PUBLICATION FIGURE — RELIABILITY VS AUROC GAP
# =====================================================================

reliability_order = [
    "independent",
    "0.5",
    "0.75",
    "1.0"
]

fig, ax = plt.subplots(figsize=(9, 5.5))

for position in POSITIONS:

    subset = matched[
        matched["train_position"] == position
    ].copy()

    subset["order"] = subset[
        "train_reliability"
    ].map(
        {
            value: i
            for i, value in enumerate(
                reliability_order
            )
        }
    )

    subset = subset.sort_values("order")

    ax.errorbar(
        subset["order"],
        subset["mean_macro_auroc_gap"],
        yerr=subset["sd_macro_auroc_gap"],
        marker="o",
        capsize=4,
        linewidth=2,
        label=position.replace("_", " ").title()
    )

ax.axhline(0, linestyle="--", linewidth=1)

ax.set_xticks(range(4))

ax.set_xticklabels([
    "Independent",
    "50%",
    "75%",
    "100%"
])

ax.set_xlabel("Marker reliability")
ax.set_ylabel(
    "Matching − mean-alternative macro AUROC"
)

ax.set_title(
    "DermaMNIST: Shortcut Reliability and "
    "Spatial Dependence"
)

ax.legend(title="Marker location")
ax.grid(alpha=0.2)

plt.tight_layout()

for extension in ["png", "pdf"]:

    fig.savefig(
        FIGURES
        / f"dermamnist_reliability_auroc_gap.{extension}",
        dpi=250,
        bbox_inches="tight"
    )

plt.show()
plt.close(fig)

# =====================================================================
# 22. PUBLICATION FIGURE — DECISION FLIP RATE
# =====================================================================

fig, ax = plt.subplots(figsize=(9, 5.5))

for position in POSITIONS:

    subset = matched[
        matched["train_position"] == position
    ].copy()

    subset["order"] = subset[
        "train_reliability"
    ].map(
        {
            value: i
            for i, value in enumerate(
                reliability_order
            )
        }
    )

    subset = subset.sort_values("order")

    ax.plot(
        subset["order"],
        subset["mean_decision_flip_rate"] * 100,
        marker="o",
        linewidth=2,
        label=position.replace("_", " ").title()
    )

ax.set_xticks(range(4))

ax.set_xticklabels([
    "Independent",
    "50%",
    "75%",
    "100%"
])

ax.set_xlabel("Marker reliability")
ax.set_ylabel("Counterfactual decision flip rate (%)")

ax.set_title(
    "DermaMNIST: Multiclass Counterfactual "
    "Decision Sensitivity"
)

ax.legend(title="Marker location")
ax.grid(alpha=0.2)

plt.tight_layout()

for extension in ["png", "pdf"]:

    fig.savefig(
        FIGURES
        / f"dermamnist_decision_flip_rate.{extension}",
        dpi=250,
        bbox_inches="tight"
    )

plt.show()
plt.close(fig)

# =====================================================================
# 23. PUBLICATION FIGURE — SPATIAL TRANSFER MATRIX
# =====================================================================

subset = summary[
    summary["train_reliability"] == "1.0"
]

matrix = subset.pivot(
    index="train_position",
    columns="evaluation_position",
    values="mean_macro_auroc_gap"
)

matrix = matrix.reindex(
    index=POSITIONS,
    columns=POSITIONS
)

fig, ax = plt.subplots(figsize=(7, 5.5))

im = ax.imshow(
    matrix.values,
    aspect="auto"
)

ax.set_xticks(range(len(POSITIONS)))
ax.set_yticks(range(len(POSITIONS)))

ax.set_xticklabels(
    [
        p.replace("_", " ").title()
        for p in POSITIONS
    ],
    rotation=25,
    ha="right"
)

ax.set_yticklabels(
    [
        p.replace("_", " ").title()
        for p in POSITIONS
    ]
)

ax.set_xlabel("Evaluation marker location")
ax.set_ylabel("Training marker location")

ax.set_title(
    "DermaMNIST: Spatial Transfer at "
    "100% Marker Reliability"
)

for i in range(len(POSITIONS)):

    for j in range(len(POSITIONS)):

        ax.text(
            j,
            i,
            f"{matrix.iloc[i, j]:.3f}",
            ha="center",
            va="center"
        )

fig.colorbar(
    im,
    ax=ax,
    label="Macro AUROC gap"
)

plt.tight_layout()

for extension in ["png", "pdf"]:

    fig.savefig(
        FIGURES
        / f"dermamnist_spatial_transfer_100.{extension}",
        dpi=250,
        bbox_inches="tight"
    )

plt.show()
plt.close(fig)

# =====================================================================
# 24. CLEAN BASELINE SUMMARY
# =====================================================================

clean_metrics = metrics_df[
    (metrics_df["condition"] == "clean")
    &
    (metrics_df["evaluation"] == "clean")
    &
    (metrics_df["evaluation_position"] == "center")
].copy()

clean_metrics.to_csv(
    TABLES / "clean_baseline_metrics.csv",
    index=False
)

print("\n" + "=" * 78)
print("CLEAN BASELINE RESULTS")
print("=" * 78)

print(
    clean_metrics[
        [
            "seed",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "macro_auroc"
        ]
    ].to_string(index=False)
)

# =====================================================================
# 25. SCIENTIFIC REPORT
# =====================================================================

report = f"""# DermaMNIST Multiclass Spatial Shortcut Study

## Experimental design

- Dataset: Official DermaMNIST+ 64x64
- Dataset MD5: {EXPECTED_MD5}
- Training images: 7007
- Validation images: 1003
- Classes: 7
- Completed models: 39
- Seeds: 42, 123, 2026
- Architecture: ImageNet-pretrained ResNet18
- Backbone: frozen
- Classification head: linear, seven outputs
- Loss: class-weighted cross-entropy
- Model selection: validation macro-F1
- Reliability conditions: independent, 50%, 75%, 100%
- Locations: center, bottom-right, random

## Counterfactual evaluation

Each validation image is evaluated with:

1. No marker
2. Neutral marker
3. Marker matching its true class
4. Each of the seven possible marker codes

For each image, the six incorrect-code predictions are
averaged to construct a descriptive alternative-code
probability distribution.

The primary reliance summaries include:

- Matching-minus-alternative macro AUROC
- Matching-minus-alternative macro-F1
- Mean total variation probability change
- Counterfactual decision flip rate

The averaged alternative-code distribution is a
summary of multiple interventions, not a single
realizable marker assignment.

## Dataset imbalance

Training class counts:
{counts.tolist()}

Class-weighted training and macro metrics are used
because the class distribution is highly imbalanced.

## Interpretation limitations

- This is a controlled synthetic-shortcut study.
- Results are based on validation data.
- The official test split is reserved for final evaluation.
- Only three training seeds are used.
- The backbone is frozen.
- Spatial rankings may depend on the dataset and marker.
- Multiclass AUROC gaps should not be pooled directly
  with binary BreastMNIST AUROC gaps.
- A difference between datasets is not evidence that
  training-set size alone caused that difference.
- Independent marker controls can still show visual
  perturbation sensitivity.

## Next steps

1. Review validation results.
2. Audit the experimental implementation.
3. Preserve exact executable code.
4. Compare with BreastMNIST using compatible metrics.
5. Extend to CIFAR-10 and PathMNIST.
6. Perform within-dataset training-size ablations.
"""

REPORT_PATH = (
    RESULTS
    / "dermamnist_scientific_report.md"
)

REPORT_PATH.write_text(
    report,
    encoding="utf-8"
)

# =====================================================================
# 26. SAVE REPRODUCIBILITY MANIFEST
# =====================================================================

manifest = {
    "study": "DermaMNIST spatial shortcut expansion",
    "dataset": "DermaMNIST+",
    "dataset_md5": EXPECTED_MD5,
    "resolution": [64, 64],
    "train_size": 7007,
    "validation_size": 1003,
    "num_classes": NUM_CLASSES,
    "architecture": "ResNet18",
    "weights": "IMAGENET1K_V1",
    "backbone_frozen": True,
    "feature_dim": FEATURE_DIM,
    "head": "Linear(512,7)",
    "seeds": SEEDS,
    "reliabilities": [
        str(x)
        for x in RELIABILITIES
    ],
    "positions": POSITIONS,
    "marker_size": MARKER_SIZE,
    "marker_colors": MARKER_COLORS.tolist(),
    "batch_size": BATCH_SIZE,
    "head_batch_size": HEAD_BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "model_selection": "validation_macro_f1",
    "completed_models": int(
        metrics_df["run_id"].nunique()
    ),
    "evaluation_split": "validation",
    "test_split_accessed": False,
    "pytorch_version": torch.__version__,
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat()
}

MANIFEST_PATH = (
    RESULTS / "training_manifest.json"
)

with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, indent=2)

# =====================================================================
# 27. PRESERVE EXECUTABLE SOURCE
# =====================================================================

# In a Kaggle/IPython notebook, In[-1] usually contains
# the currently executing cell. Verify its content before
# preserving it as the exact executable source.

source_saved = False

try:

    from IPython import get_ipython

    ip = get_ipython()

    if ip is not None:

        history = ip.history_manager.input_hist_raw

        candidates = [
            text
            for text in history
            if (
                "DERMAMNIST MULTICLASS SPATIAL SHORTCUT TRAINING"
                in text
                and
                "def train_head(" in text
                and
                "def evaluate_counterfactuals(" in text
            )
        ]

        if candidates:

            source = candidates[-1]

            source_path = (
                CODE_DIR
                / "dermamnist_phase4c_training.py"
            )

            source_path.write_text(
                source,
                encoding="utf-8"
            )

            source_saved = True

            print(
                "\nExecutable source saved:",
                source_path
            )

except Exception as exc:

    print(
        "Source capture unavailable:",
        type(exc).__name__
    )

if not source_saved:

    print(
        "\nWARNING: Exact executable source was "
        "not captured automatically."
    )

# =====================================================================
# 28. ARTIFACT INVENTORY
# =====================================================================

allowed_extensions = {
    ".csv",
    ".json",
    ".md",
    ".png",
    ".pdf",
    ".py"
}

artifact_files = []

for directory in [BASE, CODE_DIR]:

    artifact_files.extend(
        [
            p
            for p in directory.rglob("*")
            if (
                p.is_file()
                and
                p.suffix.lower() in allowed_extensions
            )
        ]
    )

inventory = pd.DataFrame(
    [
        {
            "file": str(
                p.relative_to(REPO)
            ),
            "size_bytes": p.stat().st_size
        }
        for p in sorted(artifact_files)
    ]
)

inventory.to_csv(
    RESULTS / "artifact_inventory.csv",
    index=False
)

print("\nResearch artifacts:", len(inventory))

# =====================================================================
# 29. GIT COMMIT — NO LARGE FILES
# =====================================================================

print("\n" + "=" * 78)
print("GITHUB PRESERVATION")
print("=" * 78)

def git(*args, timeout=60):

    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True,
        text=True,
        timeout=timeout
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

# Stage only approved small research artifacts.
for path in artifact_files:

    if path.stat().st_size > 20 * 1024 * 1024:
        continue

    relative = str(path.relative_to(REPO))

    result = git(
        "add",
        "--",
        relative
    )

    if result.returncode != 0:

        print(
            "Could not stage:",
            relative
        )

# Include the inventory generated after enumeration.
git(
    "add",
    "--",
    str(
        (RESULTS / "artifact_inventory.csv")
        .relative_to(REPO)
    )
)

staged = git(
    "diff",
    "--cached",
    "--name-only"
)

staged_files = [
    x
    for x in staged.stdout.splitlines()
    if x.strip()
]

print("Staged files:", len(staged_files))

# Avoid accidentally committing unrelated files.
allowed_prefixes = (
    "results/benchmark_expansion/dermamnist/",
    "experiments/dermamnist_spatial/"
)

unexpected = [
    x
    for x in staged_files
    if not x.startswith(allowed_prefixes)
]

if unexpected:

    raise RuntimeError(
        "Unexpected staged files: "
        + str(unexpected[:10])
    )

if staged_files:

    commit = git(
        "commit",
        "-m",
        "research: add DermaMNIST multiclass shortcut study",
        timeout=90
    )

    print(
        commit.stdout.strip()
        or commit.stderr.strip()
    )

else:

    print("No new files to commit.")

local_head = git(
    "rev-parse",
    "HEAD"
).stdout.strip()

print("\nLocal commit:", local_head)

print(
    "Remote push not attempted automatically. "
    "Manual authentication is required."
)

# =====================================================================
# 30. FINAL STATUS
# =====================================================================

print("\n" + "=" * 78)
print("DERMAMNIST PHASE 4C COMPLETE")
print("=" * 78)

print("Models scheduled:", len(EXPERIMENTS))

print(
    "Models completed:",
    metrics_df["run_id"].nunique()
)

print("Evaluation rows:", len(metrics_df))

print("Reliance rows:", len(reliance_df))

print("Research artifacts:", len(inventory))

print("Exact source saved:", source_saved)

print("Scientific report:", REPORT_PATH)

print("Results directory:", RESULTS)

print("Local Git commit:", local_head)

print("GitHub remote backup verified: NO")

print("Google Drive used: NO")

print("\nNEXT STEP:")
print(
    "Review DermaMNIST findings, "
    "verify the scientific outputs, "
    "and push all research artifacts to GitHub."
)

print("=" * 78)