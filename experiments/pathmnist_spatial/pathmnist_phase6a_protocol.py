# ======================================================================
# MSc THESIS — PATHMNIST PHASE 6A
# DATASET VALIDATION + SPATIAL SHORTCUT PROTOCOL
#
# Dataset: Official MedMNIST PathMNIST
# Resolution: 64x64
# Classes: 9
# Seeds: 42, 123, 2026
# Reliabilities: independent, 0.50, 0.75, 1.00
# Positions: center, bottom_right, random
#
# Outputs:
#   - Dataset integrity and class-distribution audit
#   - Deterministic marker assignment audit
#   - Publication-quality marker examples
#   - Reproducibility protocol and manifest
#   - Executable source capture
#   - Git commit and optional verified GitHub push
#
# No Google Drive. No GPU training. No test-set model evaluation.
# ======================================================================

import os
import sys
import json
import time
import random
import hashlib
import getpass
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------

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

TABLES = BASE / "tables"
FIGURES = BASE / "publication_figures"
PROTOCOL = BASE / "protocol"

CODE_DIR = (
    REPO
    / "experiments"
    / "pathmnist_spatial"
)

for directory in [
    DATA_DIR,
    TABLES,
    FIGURES,
    PROTOCOL,
    CODE_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

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

NUM_CLASSES = 9
IMAGE_SIZE = 64
MARKER_SIZE = 8
MARKER_MARGIN = 4

print("=" * 78)
print("THESIS — PATHMNIST PHASE 6A")
print("=" * 78)

# ----------------------------------------------------------------------
# 2. Install dependencies
# ----------------------------------------------------------------------

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

# ----------------------------------------------------------------------
# 3. Load official PathMNIST
# ----------------------------------------------------------------------

dataset_name = "pathmnist"
info = INFO[dataset_name]

DataClass = getattr(
    medmnist,
    info["python_class"]
)

train_dataset = DataClass(
    split="train",
    size=IMAGE_SIZE,
    root=str(DATA_DIR),
    download=True
)

val_dataset = DataClass(
    split="val",
    size=IMAGE_SIZE,
    root=str(DATA_DIR),
    download=True
)

# The official test split is downloaded only for integrity checks.
# No test images are used for training, selection or model evaluation.
test_dataset = DataClass(
    split="test",
    size=IMAGE_SIZE,
    root=str(DATA_DIR),
    download=True
)

X_TRAIN = np.asarray(train_dataset.imgs)
Y_TRAIN = np.asarray(
    train_dataset.labels
).reshape(-1).astype(np.int64)

X_VAL = np.asarray(val_dataset.imgs)
Y_VAL = np.asarray(
    val_dataset.labels
).reshape(-1).astype(np.int64)

X_TEST = np.asarray(test_dataset.imgs)
Y_TEST = np.asarray(
    test_dataset.labels
).reshape(-1).astype(np.int64)

assert X_TRAIN.ndim == 4
assert X_TRAIN.shape[1:] == (64, 64, 3)

assert X_VAL.shape[1:] == (64, 64, 3)
assert X_TEST.shape[1:] == (64, 64, 3)

assert len(np.unique(Y_TRAIN)) == NUM_CLASSES

assert Y_TRAIN.min() >= 0
assert Y_TRAIN.max() < NUM_CLASSES

assert Y_VAL.min() >= 0
assert Y_VAL.max() < NUM_CLASSES

assert Y_TEST.min() >= 0
assert Y_TEST.max() < NUM_CLASSES

print("\nOfficial dataset verified.")

print("Train:", len(Y_TRAIN))
print("Validation:", len(Y_VAL))
print("Test:", len(Y_TEST))

print("Classes:", NUM_CLASSES)
print("Resolution:", IMAGE_SIZE)

# ----------------------------------------------------------------------
# 4. Class distribution
# ----------------------------------------------------------------------

distribution_rows = []

for split_name, labels in [
    ("train", Y_TRAIN),
    ("validation", Y_VAL),
    ("test", Y_TEST)
]:

    counts = np.bincount(
        labels,
        minlength=NUM_CLASSES
    )

    for cls in range(NUM_CLASSES):

        distribution_rows.append({
            "split": split_name,
            "class": cls,
            "count": int(counts[cls]),
            "fraction": float(
                counts[cls] / len(labels)
            )
        })

distribution = pd.DataFrame(
    distribution_rows
)

distribution.to_csv(
    TABLES / "pathmnist_class_distribution.csv",
    index=False
)

print("\nClass distribution:")

print(
    distribution.pivot(
        index="class",
        columns="split",
        values="count"
    ).to_string()
)

# ----------------------------------------------------------------------
# 5. Marker pattern construction
# ----------------------------------------------------------------------

# Nine distinct patterns with identical foreground area.
# Each 8x8 pattern has 32 white and 32 black pixels.
#
# This equalizes marker area and mean pixel intensity.
# It does not guarantee equal perceptual salience.

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
        PATTERNS.append(candidate)

PATTERNS = np.stack(
    PATTERNS
)

assert PATTERNS.shape == (
    NUM_CLASSES,
    MARKER_SIZE,
    MARKER_SIZE
)

assert all(
    np.sum(pattern == 255) == 32
    for pattern in PATTERNS
)

# ----------------------------------------------------------------------
# 6. Marker placement
# ----------------------------------------------------------------------

def marker_coordinates(position, rng):

    if position == "center":

        return 28, 28

    if position == "bottom_right":

        return 52, 52

    if position == "random":

        return (
            int(rng.integers(4, 53)),
            int(rng.integers(4, 53))
        )

    raise ValueError(
        f"Unknown position: {position}"
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


# ----------------------------------------------------------------------
# 7. Deterministic marker assignment
# ----------------------------------------------------------------------

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

        # Class-conditional balanced code assignment.
        # Every true class receives approximately equal
        # representation of each marker code.

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


# ----------------------------------------------------------------------
# 8. Audit all 36 experimental conditions
# ----------------------------------------------------------------------

audit_rows = []

for seed in SEEDS:

    for reliability in RELIABILITIES:

        codes = assign_codes(
            Y_TRAIN,
            reliability,
            seed + 1000
        )

        observed = float(
            np.mean(
                codes == Y_TRAIN
            )
        )

        expected = (
            1 / NUM_CLASSES
            if reliability == "independent"
            else float(reliability)
        )

        tolerance = (
            0.002
            if reliability == "independent"
            else 0.001
        )

        assert abs(
            observed - expected
        ) <= tolerance, (
            f"Reliability audit failed: "
            f"{seed}, {reliability}, {observed}"
        )

        for position in POSITIONS:

            audit_rows.append({
                "seed": seed,
                "reliability": str(reliability),
                "position": position,
                "expected_match_rate": expected,
                "observed_match_rate": observed,
                "absolute_error": abs(
                    observed - expected
                ),
                "n_images": len(Y_TRAIN)
            })

audit = pd.DataFrame(
    audit_rows
)

assert len(audit) == 36

audit.to_csv(
    TABLES / "pathmnist_marker_audit.csv",
    index=False
)

print("\nMarker conditions verified:", len(audit))

print(
    audit.groupby(
        "reliability",
        as_index=False
    )["observed_match_rate"]
    .mean()
    .to_string(index=False)
)

# ----------------------------------------------------------------------
# 9. Visual inspection figure
# ----------------------------------------------------------------------

rng = np.random.default_rng(
    2026
)

example_indices = []

for cls in range(NUM_CLASSES):

    indices = np.flatnonzero(
        Y_TRAIN == cls
    )

    example_indices.append(
        int(rng.choice(indices))
    )

fig, axes = plt.subplots(
    NUM_CLASSES,
    4,
    figsize=(11, 2.2 * NUM_CLASSES)
)

for row_index, index in enumerate(
    example_indices
):

    image = X_TRAIN[index]
    label = int(Y_TRAIN[index])

    examples = [
        ("Clean", image)
    ]

    for position in POSITIONS:

        coordinate_rng = np.random.default_rng(
            2026 + index
        )

        row, col = marker_coordinates(
            position,
            coordinate_rng
        )

        marked = apply_marker(
            image,
            label,
            row,
            col
        )

        examples.append(
            (
                position.replace(
                    "_",
                    " "
                ),
                marked
            )
        )

    for col_index, (
        title,
        example
    ) in enumerate(examples):

        ax = axes[
            row_index,
            col_index
        ]

        ax.imshow(example)
        ax.axis("off")

        if row_index == 0:

            ax.set_title(
                title,
                fontsize=11
            )

        if col_index == 0:

            ax.set_ylabel(
                f"Class {label}",
                rotation=90,
                fontsize=10
            )

fig.suptitle(
    "PathMNIST: Spatial Shortcut Marker Examples",
    fontsize=15
)

plt.tight_layout()

for extension in [
    "png",
    "pdf"
]:

    fig.savefig(
        FIGURES
        / f"pathmnist_marker_examples.{extension}",
        dpi=250,
        bbox_inches="tight"
    )

plt.show()
plt.close(fig)

# ----------------------------------------------------------------------
# 10. Save reproducibility protocol
# ----------------------------------------------------------------------

protocol = {
    "dataset": "PathMNIST",
    "source": "Official MedMNIST",
    "resolution": IMAGE_SIZE,
    "num_classes": NUM_CLASSES,
    "train_size": len(Y_TRAIN),
    "validation_size": len(Y_VAL),
    "test_size": len(Y_TEST),
    "seeds": SEEDS,
    "reliabilities": [
        str(x)
        for x in RELIABILITIES
    ],
    "positions": POSITIONS,
    "marker_size": MARKER_SIZE,
    "marker_codes": (
        "Nine balanced binary 8x8 patterns"
    ),
    "independent_expected_match_rate": (
        1 / NUM_CLASSES
    ),
    "architecture": (
        "ImageNet-pretrained ResNet18"
    ),
    "planned_head": (
        "Linear(512, 9)"
    ),
    "backbone_frozen": True,
    "model_selection": (
        "Validation macro-F1"
    ),
    "test_split_reserved": True,
    "training_completed": False,
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat()
}

protocol_path = (
    PROTOCOL
    / "pathmnist_protocol.json"
)

protocol_path.write_text(
    json.dumps(
        protocol,
        indent=2
    ),
    encoding="utf-8"
)

# ----------------------------------------------------------------------
# 11. Save manifest
# ----------------------------------------------------------------------

manifest = {
    "dataset": "PathMNIST",
    "dataset_file": (
        "pathmnist_64.npz"
    ),
    "medmnist_version": getattr(
        medmnist,
        "__version__",
        "unknown"
    ),
    "class_distribution_saved": True,
    "marker_audit_passed": True,
    "marker_conditions": len(audit),
    "training_started": False,
    "test_model_evaluation_started": False,
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat()
}

(
    BASE / "phase6a_manifest.json"
).write_text(
    json.dumps(
        manifest,
        indent=2
    ),
    encoding="utf-8"
)

# ----------------------------------------------------------------------
# 12. Capture executable source
# ----------------------------------------------------------------------

source_saved = False

try:

    from IPython import get_ipython

    ip = get_ipython()

    candidates = [
        source
        for source in ip.history_manager.input_hist_raw
        if (
            "PATHMNIST PHASE 6A"
            in source
            and
            "def assign_codes(" in source
            and
            "pathmnist_marker_audit.csv"
            in source
        )
    ]

    if candidates:

        source_path = (
            CODE_DIR
            / "pathmnist_phase6a_protocol.py"
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

# ----------------------------------------------------------------------
# 13. Git commit
# ----------------------------------------------------------------------

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
        env=env,
        timeout=timeout
    )


assert (
    REPO / ".git"
).exists(), (
    "Thesis repository not found."
)

branch = git(
    "branch",
    "--show-current"
).stdout.strip()

assert branch == "main", (
    f"Unexpected branch: {branch}"
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

    result = git(
        "add",
        "--",
        str(
            path.relative_to(REPO)
        )
    )

    assert result.returncode == 0, (
        result.stderr
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
        "research: add PathMNIST dataset and marker audit"
    )

    assert commit.returncode == 0, (
        commit.stderr
    )

local_commit = git(
    "rev-parse",
    "HEAD"
).stdout.strip()

print("\nLocal Git commit:", local_commit)

# ----------------------------------------------------------------------
# 14. Secure optional GitHub push
# ----------------------------------------------------------------------

answer = input(
    "\nPush PathMNIST audit to GitHub? [y/N]: "
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
                "GitHub remote verified:",
                remote_verified
            )

    finally:

        del token

        if "env" in locals():
            del env

# ----------------------------------------------------------------------
# 15. Final status
# ----------------------------------------------------------------------

print("\n" + "=" * 78)
print("PATHMNIST PHASE 6A COMPLETE")
print("=" * 78)

print("Dataset verified: True")
print("Marker audit passed: True")
print("Marker conditions:", len(audit))

print("Training completed: False")
print("Source saved:", source_saved)

print("GitHub remote verified:", remote_verified)
print("Local commit:", local_commit)

print("Google Drive used: False")

print("\nNEXT:")
print(
    "Train PathMNIST shortcut models, "
    "evaluate counterfactual reliance, "
    "and complete the seven-dataset synthesis."
)

print("=" * 78)