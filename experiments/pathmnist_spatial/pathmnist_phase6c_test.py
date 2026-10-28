# ======================================================================
# MSc THESIS — PATHMNIST PHASE 6C
# FINAL HELD-OUT TEST EVALUATION
#
# Uses 39 previously trained classification heads.
# No model retraining.
#
# Evaluates:
#   - Clean images
#   - Matching markers
#   - Neutral markers
#   - Alternative marker codes
#   - Cross-position marker transfer
#
# Metrics:
#   - Accuracy
#   - Balanced accuracy
#   - Macro-F1
#   - Macro-AUROC
#   - Counterfactual decision flip rate
#   - Mean total variation
#   - Matching-vs-alternative performance gaps
#
# Statistical analysis:
#   - Stratified paired bootstrap confidence intervals
#   - Three-seed aggregation
#
# Outputs:
#   - Final test metrics
#   - Shortcut reliance tables
#   - Bootstrap intervals
#   - Publication-ready figures
#   - Reproducibility manifest
#   - GitHub backup
#
# No Google Drive.
# No model retraining.
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

REPO = (
    ROOT
    / "reliable-pneumonia-cxr"
)

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

TRAIN_DIR = (
    BASE
    / "training"
)

HEAD_DIR = (
    TRAIN_DIR
    / "heads"
)

TEST_DIR = (
    BASE
    / "heldout_test"
)

TABLE_DIR = (
    TEST_DIR
    / "tables"
)

FIG_DIR = (
    TEST_DIR
    / "publication_figures"
)

PRED_DIR = (
    TEST_DIR
    / "compact_predictions"
)

CODE_DIR = (
    REPO
    / "experiments"
    / "pathmnist_spatial"
)

for directory in [
    TABLE_DIR,
    FIG_DIR,
    PRED_DIR,
    CODE_DIR
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

SEEDS = [
    42,
    123,
    2026
]

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

MARKER_SIZE = 8

BATCH_SIZE = 256

BOOTSTRAP_ITERATIONS = 1000

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

assert DEVICE.type == "cuda", (
    "Enable GPU in Kaggle."
)

assert (
    REPO / ".git"
).exists(), (
    "Thesis GitHub repository not found."
)

print("=" * 78)
print("PATHMNIST PHASE 6C — HELD-OUT TEST")
print("=" * 78)

print("Device:", DEVICE)

# ======================================================================
# 2. LOAD OFFICIAL TEST DATASET
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


info = INFO[
    "pathmnist"
]

DataClass = getattr(
    medmnist,
    info["python_class"]
)

test_dataset = DataClass(
    split="test",
    size=64,
    root=str(DATA_DIR),
    download=False
)

X_TEST = np.asarray(
    test_dataset.imgs
)

Y_TEST = np.asarray(
    test_dataset.labels
).reshape(-1).astype(np.int64)

assert X_TEST.shape == (
    7180,
    64,
    64,
    3
)

assert len(
    np.unique(Y_TEST)
) == NUM_CLASSES

print(
    "\nOfficial test images:",
    len(Y_TEST)
)

# ======================================================================
# 3. VERIFY ALL 39 CHECKPOINTS
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

assert len(EXPERIMENTS) == 39


def run_identifier(config):

    reliability_name = str(
        config["reliability"]
    ).replace(
        ".",
        "p"
    )

    return (
        f"seed{config['seed']}_"
        f"{reliability_name}_"
        f"{config['position']}"
    )


missing = []

for config in EXPERIMENTS:

    run_id = run_identifier(
        config
    )

    path = (
        HEAD_DIR
        / f"{run_id}.pt"
    )

    if not path.exists():

        missing.append(
            run_id
        )

assert not missing, (
    f"Missing checkpoints: {missing}"
)

print(
    "Verified classification heads:",
    len(EXPERIMENTS)
)

# ======================================================================
# 4. RECONSTRUCT EXACT MARKER PROTOCOL
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

    pattern_rng.shuffle(
        flat
    )

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

assert PATTERNS.shape == (
    NUM_CLASSES,
    8,
    8
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
            int(
                rng.integers(
                    4,
                    53
                )
            ),
            int(
                rng.integers(
                    4,
                    53
                )
            )
        )

    raise ValueError(
        position
    )


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


# ======================================================================
# 5. IMAGE DATASET
# ======================================================================

MEAN = np.array(
    [
        0.485,
        0.456,
        0.406
    ],
    dtype=np.float32
).reshape(
    3,
    1,
    1
)

STD = np.array(
    [
        0.229,
        0.224,
        0.225
    ],
    dtype=np.float32
).reshape(
    3,
    1,
    1
)


class ArrayDataset(Dataset):

    def __init__(
        self,
        images,
        codes=None,
        position=None,
        mode="clean",
        seed=2026
    ):

        self.images = images
        self.codes = codes

        self.position = position
        self.mode = mode
        self.seed = seed

    def __len__(self):

        return len(
            self.images
        )

    def __getitem__(
        self,
        index
    ):

        image = self.images[
            index
        ]

        if self.mode != "clean":

            coordinate_rng = (
                np.random.default_rng(
                    self.seed * 1000003
                    +
                    index
                )
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
            image.transpose(
                2,
                0,
                1
            ).astype(np.float32)
            /
            255.0
        )

        tensor = (
            tensor - MEAN
        ) / STD

        return torch.from_numpy(
            tensor
        )


# ======================================================================
# 6. LOAD FROZEN BACKBONE
# ======================================================================

backbone = models.resnet18(
    weights=(
        models.ResNet18_Weights.IMAGENET1K_V1
    )
)

backbone.fc = nn.Identity()

backbone = backbone.to(
    DEVICE
).eval()

for parameter in backbone.parameters():

    parameter.requires_grad = False


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


print(
    "\nFrozen ResNet18 loaded."
)

# ======================================================================
# 7. PRECOMPUTE TEST FEATURES
# ======================================================================

TEST_FEATURES = {}

for position in POSITIONS:

    print(
        "\nExtracting held-out test features:",
        position
    )

    TEST_FEATURES[
        position
    ] = {}

    variants = {
        "clean": None,
        "matching": Y_TEST,
        "neutral": None
    }

    for code in range(
        NUM_CLASSES
    ):

        variants[
            f"code_{code}"
        ] = np.full(
            len(Y_TEST),
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
            X_TEST,
            codes=codes,
            position=position,
            mode=mode,
            seed=2026
        )

        TEST_FEATURES[
            position
        ][name] = extract_features(
            dataset
        )

print(
    "\nAll test features extracted."
)

# ======================================================================
# 8. CLASSIFICATION HEAD
# ======================================================================

class Head(nn.Module):

    def __init__(
        self
    ):

        super().__init__()

        self.fc = nn.Linear(
            512,
            NUM_CLASSES
        )

    def forward(
        self,
        x
    ):

        return self.fc(
            x
        )


@torch.inference_mode()
def predict(
    head,
    features
):

    head.eval()

    tensor = torch.tensor(
        features,
        device=DEVICE
    )

    return torch.softmax(
        head(tensor),
        dim=1
    ).cpu().numpy()


# ======================================================================
# 9. METRICS
# ======================================================================

def calculate_metrics(
    labels,
    probabilities
):

    predictions = probabilities.argmax(
        axis=1
    )

    metrics = {
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

        metrics["macro_auroc"] = float(
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

        metrics["macro_auroc"] = np.nan

    return metrics


# ======================================================================
# 10. COUNTERFACTUAL EVALUATION
# ======================================================================

def evaluate_model(
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
                Y_TEST,
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

    alternative = (
        code_probabilities.sum(
            axis=0
        )
        -
        matching
    ) / (
        NUM_CLASSES - 1
    )

    matching_metrics = calculate_metrics(
        Y_TEST,
        matching
    )

    alternative_metrics = calculate_metrics(
        Y_TEST,
        alternative
    )

    metrics.append({
        "evaluation": "mean_alternative",
        **alternative_metrics
    })

    matching_predictions = (
        matching.argmax(
            axis=1
        )
    )

    total_variation = []
    flips = []

    for code in range(
        NUM_CLASSES
    ):

        mask = (
            Y_TEST != code
        )

        alternative_probs = (
            code_probabilities[
                code,
                mask
            ]
        )

        matching_probs = (
            matching[
                mask
            ]
        )

        total_variation.extend(
            (
                np.abs(
                    matching_probs
                    -
                    alternative_probs
                ).sum(axis=1)
                /
                2
            ).tolist()
        )

        flips.extend(
            (
                matching_predictions[
                    mask
                ]
                !=
                alternative_probs.argmax(
                    axis=1
                )
            ).tolist()
        )

    reliance = {
        "macro_auroc_gap": (
            matching_metrics[
                "macro_auroc"
            ]
            -
            alternative_metrics[
                "macro_auroc"
            ]
        ),
        "macro_f1_gap": (
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
                flips
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
        reliance,
        probabilities
    )


# ======================================================================
# 11. EVALUATE ALL 39 MODELS
# ======================================================================

all_metrics = []
all_reliance = []

prediction_records = {}

for run_number, config in enumerate(
    EXPERIMENTS,
    start=1
):

    run_id = run_identifier(
        config
    )

    checkpoint_path = (
        HEAD_DIR
        /
        f"{run_id}.pt"
    )

    print(
        f"\n[{run_number}/39]",
        run_id
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    head = Head().to(
        DEVICE
    )

    head.load_state_dict(
        checkpoint[
            "state_dict"
        ]
    )

    for evaluation_position in POSITIONS:

        metrics, reliance, probabilities = evaluate_model(
            head,
            TEST_FEATURES[
                evaluation_position
            ]
        )

        for row in metrics:

            all_metrics.append({
                "run_id": run_id,
                "seed": config[
                    "seed"
                ],
                "training_reliability": str(
                    config[
                        "reliability"
                    ]
                ),
                "training_position": config[
                    "position"
                ],
                "evaluation_position": evaluation_position,
                **row
            })

        all_reliance.append({
            "run_id": run_id,
            "seed": config[
                "seed"
            ],
            "training_reliability": str(
                config[
                    "reliability"
                ]
            ),
            "training_position": config[
                "position"
            ],
            "evaluation_position": evaluation_position,
            **reliance
        })

        if (
            config["reliability"] == 1.0
            and
            evaluation_position == config["position"]
        ):

            prediction_records[
                run_id
            ] = {
                "matching": probabilities[
                    "matching"
                ].astype(
                    np.float16
                ),
                "clean": probabilities[
                    "clean"
                ].astype(
                    np.float16
                )
            }

    del head

    gc.collect()
    torch.cuda.empty_cache()

# ======================================================================
# 12. SAVE TABLES
# ======================================================================

metrics_df = pd.DataFrame(
    all_metrics
)

reliance_df = pd.DataFrame(
    all_reliance
)

metrics_df.to_csv(
    TABLE_DIR
    / "pathmnist_test_metrics.csv",
    index=False
)

reliance_df.to_csv(
    TABLE_DIR
    / "pathmnist_test_shortcut_reliance.csv",
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
# 13. THREE-SEED SUMMARY
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
    / "pathmnist_test_three_seed_summary.csv",
    index=False
)

print(
    "\nPATHMNIST HELD-OUT TEST SUMMARY"
)

print(
    summary.to_string(
        index=False
    )
)

# ======================================================================
# 14. PAIRED BOOTSTRAP
# ======================================================================

# Stratified resampling preserves class representation.
# We calculate confidence intervals for matching-vs-clean
# macro-F1 on the 100%-reliable models.

bootstrap_rows = []

rng = np.random.default_rng(
    20261028
)

for run_id, prediction_set in prediction_records.items():

    matching = prediction_set[
        "matching"
    ].astype(
        np.float32
    )

    clean = prediction_set[
        "clean"
    ].astype(
        np.float32
    )

    differences = []

    class_indices = [
        np.flatnonzero(
            Y_TEST == cls
        )
        for cls in range(
            NUM_CLASSES
        )
    ]

    for iteration in range(
        BOOTSTRAP_ITERATIONS
    ):

        sample = np.concatenate(
            [
                rng.choice(
                    indices,
                    size=len(indices),
                    replace=True
                )
                for indices in class_indices
            ]
        )

        matching_f1 = f1_score(
            Y_TEST[
                sample
            ],
            matching[
                sample
            ].argmax(
                axis=1
            ),
            average="macro",
            zero_division=0
        )

        clean_f1 = f1_score(
            Y_TEST[
                sample
            ],
            clean[
                sample
            ].argmax(
                axis=1
            ),
            average="macro",
            zero_division=0
        )

        differences.append(
            matching_f1 - clean_f1
        )

    bootstrap_rows.append({
        "run_id": run_id,
        "comparison": "matching_minus_clean_macro_f1",
        "mean_difference": float(
            np.mean(
                differences
            )
        ),
        "ci_lower": float(
            np.percentile(
                differences,
                2.5
            )
        ),
        "ci_upper": float(
            np.percentile(
                differences,
                97.5
            )
        )
    })

bootstrap_df = pd.DataFrame(
    bootstrap_rows
)

bootstrap_df.to_csv(
    TABLE_DIR
    / "pathmnist_test_bootstrap.csv",
    index=False
)

# ======================================================================
# 15. PUBLICATION FIGURE
# ======================================================================

fig, ax = plt.subplots(
    figsize=(9, 5.5)
)

reliability_order = {
    "independent": 0,
    "0.5": 0.5,
    "0.75": 0.75,
    "1.0": 1.0
}

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

    ax.errorbar(
        subset["x"],
        subset["mean_auroc_gap"],
        yerr=subset["sd_auroc_gap"],
        marker="o",
        capsize=4,
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
    "Matching − alternative macro-AUROC"
)

ax.set_title(
    "PathMNIST: Held-Out Shortcut Reliance"
)

ax.set_xticks(
    [
        0,
        0.5,
        0.75,
        1.0
    ]
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
        / f"pathmnist_test_auroc_gap.{extension}",
        dpi=250,
        bbox_inches="tight"
    )

plt.show()
plt.close(fig)

# ======================================================================
# 16. SAVE REPRODUCIBILITY MANIFEST
# ======================================================================

manifest = {
    "dataset": "PathMNIST",
    "evaluation_split": "official_test",
    "test_images": len(
        Y_TEST
    ),
    "classes": NUM_CLASSES,
    "models_evaluated": int(
        metrics_df["run_id"].nunique()
    ),
    "seeds": SEEDS,
    "reliabilities": [
        str(x)
        for x in RELIABILITIES
    ],
    "positions": POSITIONS,
    "architecture": "ResNet18",
    "backbone_frozen": True,
    "head_checkpoints_reused": True,
    "model_retraining": False,
    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat()
}

(
    TEST_DIR
    / "pathmnist_test_manifest.json"
).write_text(
    json.dumps(
        manifest,
        indent=2
    ),
    encoding="utf-8"
)

# ======================================================================
# 17. CAPTURE EXECUTABLE SOURCE
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
            "PATHMNIST PHASE 6C"
            in source
            and
            "def evaluate_model(" in source
            and
            "pathmnist_test_three_seed_summary.csv"
            in source
        )
    ]

    if candidates:

        (
            CODE_DIR
            / "pathmnist_phase6c_test.py"
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
# 18. GITHUB BACKUP
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

allowed_extensions = {
    ".csv",
    ".json",
    ".md",
    ".png",
    ".pdf",
    ".py"
}

artifact_files = []

for directory in [
    TEST_DIR,
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
        "research: add PathMNIST held-out test evaluation"
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

answer = input(
    "\nPush held-out results to GitHub? [y/N]: "
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
# 19. FINAL STATUS
# ======================================================================

print("\n" + "=" * 78)
print("PATHMNIST PHASE 6C COMPLETE")
print("=" * 78)

print(
    "Models evaluated:",
    metrics_df["run_id"].nunique()
)

print(
    "Test metric rows:",
    len(metrics_df)
)

print(
    "Shortcut reliance rows:",
    len(reliance_df)
)

print(
    "Bootstrap comparisons:",
    len(bootstrap_df)
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
    "Model retraining:",
    False
)

print(
    "Google Drive used:",
    False
)

print("\nNEXT:")
print(
    "Integrate seven-dataset results, "
    "audit experimental comparability, "
    "and prepare final thesis figures."
)

print("=" * 78)