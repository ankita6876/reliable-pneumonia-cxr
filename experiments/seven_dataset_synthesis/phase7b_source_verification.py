# ======================================================================
# MSc THESIS — PHASE 7B
# SOURCE-LEVEL VERIFICATION AND SEVEN-DATASET NORMALIZATION
#
# No training | No dataset downloads | No Google Drive
# GitHub-first | Resumable | Source-preserving
#
# Outputs:
#   - Source-table inventory and column profiles
#   - Experiment-code provenance audit
#   - Spatial-shortcut table classification
#   - Candidate normalized three-seed results
#   - Strict comparability assessment
#   - Training-size evidence audit
#   - Seven-dataset evidence matrix
#   - Publication-ready audit figures
#   - Scientific limitations report
#   - Executable source and GitHub backup
#
# IMPORTANT:
# Numeric cross-dataset comparisons are NOT certified merely
# because columns share names. Source-level review remains required.
# ======================================================================

import os
import re
import sys
import json
import getpass
import tempfile
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================================
# 1. PATHS AND CONFIGURATION
# ======================================================================

ROOT = Path("/kaggle/working")
REPO = ROOT / "reliable-pneumonia-cxr"

PHASE7A = (
    REPO
    / "results"
    / "seven_dataset_synthesis"
    / "phase7a_evidence_audit"
)

OUT = (
    REPO
    / "results"
    / "seven_dataset_synthesis"
    / "phase7b_source_verification"
)

TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"

CODE_DIR = (
    REPO
    / "experiments"
    / "seven_dataset_synthesis"
)

for p in [TABLES, FIGURES, REPORTS, CODE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

assert (REPO / ".git").exists(), (
    "Existing thesis repository was not found."
)

DATASETS = [
    "chexpert",
    "oxford_pet",
    "nico",
    "breastmnist",
    "dermamnist",
    "cifar10",
    "pathmnist"
]

EXPECTED_SEEDS = {42, 123, 2026}

RELIABILITIES = {
    "independent", "0.5", "0.75", "1.0"
}

POSITIONS = {
    "center", "bottom_right", "random"
}

MAX_FILE_BYTES = 100 * 1024 * 1024

print("=" * 78)
print("THESIS — PHASE 7B: SOURCE VERIFICATION")
print("=" * 78)

# ======================================================================
# 2. HELPERS
# ======================================================================

def git(*args, env=None, timeout=120):

    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env
    )


def relative(path):
    return path.relative_to(REPO).as_posix()


def now():
    return datetime.now(timezone.utc).isoformat()


def save_csv(df, filename):
    path = TABLES / filename
    df.to_csv(path, index=False)
    return path


def normalize_reliability(value):

    if pd.isna(value):
        return None

    s = str(value).strip().lower()

    if s in {"independent", "uncorrelated"}:
        return "independent"

    if s in {"clean", "baseline"}:
        return "clean"

    try:
        x = float(s.replace("p", "."))

        if x in {0.5, 0.75, 1.0}:
            return str(x)

    except (TypeError, ValueError):
        pass

    return None


def normalize_position(value):

    if pd.isna(value):
        return None

    s = str(value).strip().lower().replace("-", "_")

    mapping = {
        "center": "center",
        "centre": "center",
        "bottom_right": "bottom_right",
        "bottomright": "bottom_right",
        "random": "random",
        "interior": "interior",
        "boundary": "boundary",
        "exterior": "exterior"
    }

    return mapping.get(s)


def detect_dataset(path):

    s = str(path).lower().replace("\\", "/")

    patterns = {
        "pathmnist": ["pathmnist"],
        "dermamnist": ["dermamnist"],
        "breastmnist": ["breastmnist"],
        "cifar10": ["cifar10", "cifar_10"],
        "oxford_pet": [
            "oxford_pet",
            "oxford-iiit",
            "pet_spatial"
        ],
        "nico": ["nico"],
        "chexpert": ["chexpert"]
    }

    matches = [
        name
        for name, aliases in patterns.items()
        if any(alias in s for alias in aliases)
    ]

    return matches[0] if len(matches) == 1 else None


def find_column(df, names):

    return next(
        (name for name in names if name in df.columns),
        None
    )


def safe_unique(df, column):

    if column is None:
        return []

    return sorted(
        df[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


# ======================================================================
# 3. GIT STATE
# ======================================================================

head = git("rev-parse", "HEAD")
branch = git("branch", "--show-current")

assert head.returncode == 0, head.stderr

START_HEAD = head.stdout.strip()
BRANCH = branch.stdout.strip()

print("\nGit HEAD:", START_HEAD)
print("Branch:", BRANCH)

# ======================================================================
# 4. RECOVER PHASE 7A TABLE CANDIDATES
# ======================================================================

candidate_path = (
    PHASE7A
    / "tables"
    / "primary_result_table_candidates.csv"
)

assert candidate_path.exists(), (
    "Phase 7A candidate table missing. "
    "Run Phase 7A or restore its GitHub artifacts."
)

candidates = pd.read_csv(candidate_path)

print("\nPhase 7A candidates:", len(candidates))

# ======================================================================
# 5. AUDIT PRIMARY TABLES
# ======================================================================

print("\n[1/6] Inspecting primary result tables...")

table_profiles = []
table_samples = []
errors = []

for _, item in candidates.iterrows():

    dataset = item["dataset"]
    path = REPO / item["path"]

    if not path.exists():
        continue

    try:

        if path.stat().st_size > MAX_FILE_BYTES:
            continue

        df = pd.read_csv(
            path,
            low_memory=False
        )

        cols = list(df.columns)

        seed_col = find_column(
            df,
            ["seed", "random_seed"]
        )

        reliability_col = find_column(
            df,
            [
                "training_reliability",
                "reliability",
                "correlation",
                "rho",
                "shortcut_strength"
            ]
        )

        position_col = find_column(
            df,
            [
                "training_position",
                "position",
                "marker_position",
                "location"
            ]
        )

        split_col = find_column(
            df,
            [
                "split",
                "evaluation_split",
                "dataset_split"
            ]
        )

        metric_cols = [
            c for c in cols
            if any(
                term in c.lower()
                for term in [
                    "auroc",
                    "auc",
                    "f1",
                    "flip",
                    "accuracy",
                    "variation"
                ]
            )
        ]

        path_text = relative(path).lower()

        spatial_keywords = [
            "spatial",
            "shortcut",
            "counterfactual",
            "marker",
            "position",
            "neutral_patch",
            "reliance"
        ]

        spatial_path = any(
            word in path_text
            for word in spatial_keywords
        )

        fusion_path = any(
            word in path_text
            for word in [
                "fusion",
                "pneumonia_baseline",
                "resnet50_complementary"
            ]
        )

        spatial_schema = (
            reliability_col is not None
            or position_col is not None
            or any(
                "flip" in c.lower()
                for c in cols
            )
        )

        classification = (
            "likely_spatial_shortcut"
            if (
                spatial_path or spatial_schema
            ) and not fusion_path
            else "not_verified_as_spatial_shortcut"
        )

        table_profiles.append({
            "dataset": dataset,
            "path": relative(path),
            "rows": len(df),
            "columns": json.dumps(cols),
            "seed_column": seed_col,
            "reliability_column": reliability_col,
            "position_column": position_col,
            "split_column": split_col,
            "metric_columns": json.dumps(metric_cols),
            "seed_values": json.dumps(
                safe_unique(df, seed_col)
            ),
            "reliability_values": json.dumps(
                safe_unique(df, reliability_col)
            ),
            "position_values": json.dumps(
                safe_unique(df, position_col)
            ),
            "split_values": json.dumps(
                safe_unique(df, split_col)
            ),
            "classification": classification,
            "contains_numeric_metrics": bool(metric_cols),
            "has_three_seed_schema": seed_col is not None
        })

        for index, row in df.head(3).iterrows():

            table_samples.append({
                "dataset": dataset,
                "path": relative(path),
                "sample_row": index,
                "record": json.dumps(
                    row.to_dict(),
                    default=str
                )
            })

    except Exception as exc:

        errors.append({
            "dataset": dataset,
            "path": relative(path),
            "error": repr(exc)
        })

profiles = pd.DataFrame(table_profiles)

save_csv(
    profiles,
    "source_table_profiles.csv"
)

save_csv(
    pd.DataFrame(table_samples),
    "source_table_samples.csv"
)

save_csv(
    pd.DataFrame(
        errors,
        columns=["dataset", "path", "error"]
    ),
    "source_table_read_errors.csv"
)

print("Tables profiled:", len(profiles))
print("Errors:", len(errors))

# ======================================================================
# 6. EXPERIMENT SOURCE CODE AUDIT
# ======================================================================

print("\n[2/6] Inspecting experimental source code...")

code_rows = []

for path in REPO.rglob("*.py"):

    if (
        path.is_relative_to(OUT)
        or path.is_relative_to(CODE_DIR)
    ):
        continue

    dataset = detect_dataset(
        relative(path)
    )

    if dataset is None:
        continue

    try:

        if path.stat().st_size > 3 * 1024 * 1024:
            continue

        text = path.read_text(
            encoding="utf-8",
            errors="replace"
        )

        lower = text.lower()

        code_rows.append({
            "dataset": dataset,
            "path": relative(path),
            "lines": len(text.splitlines()),
            "contains_seed": "seed" in lower,
            "contains_marker": (
                "marker" in lower
                or "shortcut" in lower
            ),
            "contains_counterfactual": (
                "counterfactual" in lower
                or "alternative" in lower
            ),
            "contains_auroc": (
                "roc_auc_score" in lower
                or "auroc" in lower
            ),
            "contains_f1": "f1_score" in lower,
            "contains_bootstrap": "bootstrap" in lower,
            "contains_test_split": (
                'split="test"' in lower
                or "split='test'" in lower
                or "official_test" in lower
            ),
            "contains_size_ablation": any(
                word in lower
                for word in [
                    "train_sizes",
                    "training_sizes",
                    "size_ablation",
                    "subset_sizes"
                ]
            ),
            "contains_checkpoint_save": (
                "torch.save" in lower
                or "save_model" in lower
            )
        })

    except Exception as exc:

        errors.append({
            "dataset": dataset,
            "path": relative(path),
            "error": repr(exc)
        })

code_df = pd.DataFrame(code_rows)

save_csv(
    code_df,
    "experiment_source_code_audit.csv"
)

print("Experiment scripts inspected:", len(code_df))

# ======================================================================
# 7. EXTRACT PROVISIONAL NORMALIZED RESULTS
# ======================================================================

print("\n[3/6] Building provisional normalized results...")

normalized_rows = []

METRIC_ALIASES = {
    "macro_auroc_gap": [
        "mean_auroc_gap",
        "macro_auroc_gap",
        "auroc_gap"
    ],
    "macro_f1_gap": [
        "mean_macro_f1_gap",
        "macro_f1_gap",
        "f1_gap"
    ],
    "decision_flip_rate": [
        "mean_decision_flip_rate",
        "decision_flip_rate",
        "flip_rate"
    ],
    "mean_total_variation": [
        "mean_total_variation",
        "total_variation"
    ]
}

for _, profile in profiles.iterrows():

    if profile["classification"] != "likely_spatial_shortcut":
        continue

    path = REPO / profile["path"]

    try:

        df = pd.read_csv(
            path,
            low_memory=False
        )

        reliability_col = find_column(
            df,
            [
                "training_reliability",
                "reliability",
                "correlation",
                "rho"
            ]
        )

        position_col = find_column(
            df,
            [
                "training_position",
                "position",
                "marker_position"
            ]
        )

        if reliability_col is None or position_col is None:
            continue

        metric_map = {
            metric: find_column(df, aliases)
            for metric, aliases in METRIC_ALIASES.items()
        }

        if metric_map["macro_auroc_gap"] is None:
            continue

        seed_col = find_column(
            df,
            ["seed", "random_seed"]
        )

        for _, row in df.iterrows():

            reliability = normalize_reliability(
                row[reliability_col]
            )

            position = normalize_position(
                row[position_col]
            )

            if (
                reliability not in RELIABILITIES
                or position is None
            ):
                continue

            record = {
                "dataset": profile["dataset"],
                "source": profile["path"],
                "training_reliability": reliability,
                "training_position": position,
                "seed": (
                    row[seed_col]
                    if seed_col is not None
                    else np.nan
                ),
                "split_status": "requires_verification",
                "contrast_definition": "requires_verification",
                "comparability_status": "provisional"
            }

            for metric, column in metric_map.items():

                record[metric] = (
                    pd.to_numeric(
                        row[column],
                        errors="coerce"
                    )
                    if column is not None
                    else np.nan
                )

            normalized_rows.append(record)

    except Exception as exc:

        errors.append({
            "dataset": profile["dataset"],
            "path": profile["path"],
            "error": repr(exc)
        })

normalized_df = pd.DataFrame(normalized_rows)

save_csv(
    normalized_df,
    "provisional_normalized_shortcut_results.csv"
)

print(
    "Provisional normalized rows:",
    len(normalized_df)
)

# ======================================================================
# 8. STRICT PATHMNIST REFERENCE
# ======================================================================

pathmnist_source = (
    REPO
    / "results"
    / "benchmark_expansion"
    / "pathmnist"
    / "heldout_test"
    / "tables"
    / "pathmnist_test_three_seed_summary.csv"
)

strict_rows = []

if pathmnist_source.exists():

    path_df = pd.read_csv(
        pathmnist_source
    )

    required = {
        "training_reliability",
        "training_position",
        "mean_auroc_gap",
        "mean_macro_f1_gap",
        "mean_decision_flip_rate"
    }

    if required.issubset(path_df.columns):

        for _, row in path_df.iterrows():

            strict_rows.append({
                "dataset": "pathmnist",
                "split": "official_test",
                "reliability": normalize_reliability(
                    row["training_reliability"]
                ),
                "position": normalize_position(
                    row["training_position"]
                ),
                "auroc_gap": row["mean_auroc_gap"],
                "macro_f1_gap": row["mean_macro_f1_gap"],
                "decision_flip_rate": row[
                    "mean_decision_flip_rate"
                ],
                "contrast": (
                    "matching_minus_mean_alternative"
                ),
                "source": relative(
                    pathmnist_source
                ),
                "status": "protocol_known"
            })

strict_df = pd.DataFrame(strict_rows)

save_csv(
    strict_df,
    "protocol_known_reference_results.csv"
)

# ======================================================================
# 9. TRAINING-SIZE EVIDENCE
# ======================================================================

print("\n[4/6] Auditing training-size experiments...")

size_rows = []

for dataset in DATASETS:

    relevant = profiles[
        profiles["dataset"] == dataset
    ]

    for _, profile in relevant.iterrows():

        path = REPO / profile["path"]

        try:

            df = pd.read_csv(
                path,
                low_memory=False
            )

            size_col = find_column(
                df,
                [
                    "train_size",
                    "training_size",
                    "n_train",
                    "subset_size",
                    "sample_size"
                ]
            )

            if size_col is None:
                continue

            sizes = sorted(
                pd.to_numeric(
                    df[size_col],
                    errors="coerce"
                ).dropna().unique().tolist()
            )

            if not sizes:
                continue

            size_rows.append({
                "dataset": dataset,
                "source": profile["path"],
                "size_column": size_col,
                "training_sizes": json.dumps(sizes),
                "number_of_sizes": len(sizes),
                "rows": len(df),
                "status": "candidate_requires_protocol_review"
            })

        except Exception:
            continue

size_df = pd.DataFrame(
    size_rows,
    columns=[
        "dataset",
        "source",
        "size_column",
        "training_sizes",
        "number_of_sizes",
        "rows",
        "status"
    ]
)

save_csv(
    size_df,
    "training_size_evidence.csv"
)

# ======================================================================
# 10. SEVEN-DATASET COMPARABILITY MATRIX
# ======================================================================

print("\n[5/6] Creating comparability matrix...")

matrix_rows = []

for dataset in DATASETS:

    dataset_profiles = profiles[
        profiles["dataset"] == dataset
    ]

    dataset_code = code_df[
        code_df["dataset"] == dataset
    ] if len(code_df) else pd.DataFrame()

    dataset_normalized = (
        normalized_df[
            normalized_df["dataset"] == dataset
        ]
        if len(normalized_df)
        else pd.DataFrame()
    )

    dataset_size = size_df[
        size_df["dataset"] == dataset
    ]

    has_protocol_known = (
        dataset == "pathmnist"
        and len(strict_df) == 12
    )

    matrix_rows.append({
        "dataset": dataset,
        "source_tables_inspected": len(dataset_profiles),
        "experiment_scripts_inspected": len(dataset_code),
        "spatial_table_candidates": int(
            (
                dataset_profiles["classification"]
                == "likely_spatial_shortcut"
            ).sum()
        ),
        "provisional_normalized_rows": len(
            dataset_normalized
        ),
        "training_size_table_candidates": len(
            dataset_size
        ),
        "protocol_known_test_reference": has_protocol_known,
        "fully_cross_dataset_comparable": False,
        "next_action": (
            "Verify counterfactual contrast, evaluation "
            "split, marker protocol and seed aggregation"
        )
    })

matrix_df = pd.DataFrame(matrix_rows)

save_csv(
    matrix_df,
    "seven_dataset_comparability_matrix.csv"
)

print(
    "\nSEVEN-DATASET COMPARABILITY MATRIX"
)

print(
    matrix_df.to_string(
        index=False
    )
)

# ======================================================================
# 11. AUDIT FIGURES
# ======================================================================

fig, ax = plt.subplots(
    figsize=(10, 5.5)
)

x = np.arange(
    len(matrix_df)
)

ax.bar(
    x,
    matrix_df[
        "spatial_table_candidates"
    ]
)

ax.set_xticks(x)

ax.set_xticklabels(
    matrix_df["dataset"],
    rotation=35,
    ha="right"
)

ax.set_ylabel(
    "Candidate spatial-shortcut tables"
)

ax.set_title(
    "Seven-Dataset Source Verification Coverage"
)

ax.grid(
    axis="y",
    alpha=0.2
)

plt.tight_layout()

for ext in ["png", "pdf"]:

    fig.savefig(
        FIGURES
        / f"source_verification_coverage.{ext}",
        dpi=250,
        bbox_inches="tight"
    )

plt.show()
plt.close(fig)

if len(strict_df):

    subset = strict_df[
        strict_df["reliability"] == "1.0"
    ].copy()

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        subset["position"],
        subset["decision_flip_rate"] * 100
    )

    ax.set_ylabel(
        "Decision flip rate (%)"
    )

    ax.set_title(
        "PathMNIST: Verified Held-Out Counterfactual Sensitivity"
    )

    ax.grid(
        axis="y",
        alpha=0.2
    )

    plt.tight_layout()

    for ext in ["png", "pdf"]:

        fig.savefig(
            FIGURES
            / f"pathmnist_heldout_flip_rate.{ext}",
            dpi=250,
            bbox_inches="tight"
        )

    plt.show()
    plt.close(fig)

# ======================================================================
# 12. SCIENTIFIC REPORT
# ======================================================================

report = [
    "# Phase 7B — Seven-Dataset Source Verification",
    "",
    f"Generated: {now()}",
    f"Starting Git commit: `{START_HEAD}`",
    "",
    "## Purpose",
    "",
    "Recover primary shortcut-learning results and establish "
    "whether their experimental protocols are comparable.",
    "",
    "## Evidence Matrix",
    "",
    matrix_df.to_markdown(index=False),
    "",
    "## Interpretation",
    "",
    "The provisional normalized table is a discovery artifact, "
    "not a final cross-dataset statistical comparison.",
    "",
    "PathMNIST held-out test results have a documented "
    "matching-minus-mean-alternative contrast.",
    "",
    "Other datasets require verification of their original "
    "counterfactual definitions, data splits, and aggregation.",
    "",
    "## Important Scientific Limitations",
    "",
    "1. Identical column names do not guarantee identical metrics.",
    "2. Training seeds are not independent datasets.",
    "3. Matching-marker validation performance is not clean "
    "generalization performance.",
    "4. Marker position effects may reflect background interference.",
    "5. Missing checkpoints do not invalidate previously saved "
    "predictions or results, but can limit reproducibility.",
    "6. Test-set model selection must be avoided.",
    "7. The PathMNIST bootstrap intervals currently concern "
    "matching-minus-clean macro-F1, not the principal "
    "matching-minus-alternative AUROC gap.",
    "",
    "## Recommended Next Stage",
    "",
    "Verify each dataset's primary table and source code, "
    "then build a strictly defined cross-dataset results matrix. "
    "Recover missing held-out predictions where possible.",
    ""
]

report_path = (
    REPORTS
    / "phase7b_scientific_verification.md"
)

report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)

# ======================================================================
# 13. REPRODUCIBILITY MANIFEST
# ======================================================================

manifest = {
    "phase": "7B",
    "created_utc": now(),
    "starting_commit": START_HEAD,
    "datasets": DATASETS,
    "tables_profiled": len(profiles),
    "scripts_inspected": len(code_df),
    "provisional_normalized_rows": len(normalized_df),
    "protocol_known_reference_rows": len(strict_df),
    "training_size_candidates": len(size_df),
    "training_performed": False,
    "datasets_downloaded": False,
    "google_drive_used": False,
    "cross_dataset_results_certified": False
}

(
    OUT
    / "phase7b_manifest.json"
).write_text(
    json.dumps(
        manifest,
        indent=2
    ),
    encoding="utf-8"
)

# ======================================================================
# 14. CAPTURE EXECUTABLE SOURCE
# ======================================================================

source_saved = False

try:

    from IPython import get_ipython

    ip = get_ipython()

    candidates = [
        s
        for s in ip.history_manager.input_hist_raw
        if (
            "PHASE 7B" in s
            and
            "seven_dataset_comparability_matrix.csv" in s
            and
            "phase7b_manifest.json" in s
        )
    ]

    if candidates:

        source_path = (
            CODE_DIR
            / "phase7b_source_verification.py"
        )

        source_path.write_text(
            candidates[-1],
            encoding="utf-8"
        )

        source_saved = True

except Exception as exc:

    print(
        "Source capture warning:",
        repr(exc)
    )

# ======================================================================
# 15. GITHUB COMMIT
# ======================================================================

print("\n[6/6] Preserving Phase 7B artifacts...")

allowed_extensions = {
    ".csv",
    ".json",
    ".md",
    ".png",
    ".pdf",
    ".py"
}

artifacts = [
    p
    for directory in [OUT, CODE_DIR]
    for p in directory.rglob("*")
    if (
        p.is_file()
        and p.suffix.lower() in allowed_extensions
        and p.stat().st_size < 20 * 1024 * 1024
    )
]

previously_staged = git(
    "diff",
    "--cached",
    "--name-only"
).stdout.splitlines()

if previously_staged:

    raise RuntimeError(
        "Other staged changes detected. "
        "Audit files are saved, but committing was stopped "
        "to avoid including unrelated research files."
    )

for path in artifacts:

    result = git(
        "add",
        "--",
        relative(path)
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

staged = git(
    "diff",
    "--cached",
    "--name-only"
).stdout.splitlines()

allowed_prefixes = (
    "results/seven_dataset_synthesis/phase7b_source_verification/",
    "experiments/seven_dataset_synthesis/"
)

assert all(
    p.startswith(allowed_prefixes)
    for p in staged
), "Unexpected staged path."

if staged:

    author_name = git(
        "config", "--get", "user.name"
    ).stdout.strip()

    author_email = git(
        "config", "--get", "user.email"
    ).stdout.strip()

    if not author_name or not author_email:

        git(
            "config",
            "--local",
            "user.name",
            "Thesis Research"
        )

        git(
            "config",
            "--local",
            "user.email",
            "thesis-research@users.noreply.github.com"
        )

    commit = git(
        "commit",
        "-m",
        "research: verify seven-dataset shortcut result sources"
    )

    if commit.returncode != 0:
        raise RuntimeError(commit.stderr)

LOCAL_HEAD = git(
    "rev-parse",
    "HEAD"
).stdout.strip()

print(
    "\nLocal commit:",
    LOCAL_HEAD
)

# ======================================================================
# 16. SECURE GITHUB PUSH
# ======================================================================

answer = input(
    "\nPush Phase 7B results to GitHub? [y/N]: "
).strip().lower()

remote_verified = False

if answer == "y":

    token = getpass.getpass(
        "Paste GitHub Personal Access Token: "
    ).strip()

    try:

        with tempfile.TemporaryDirectory() as td:

            askpass = (
                Path(td)
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

            askpass.chmod(0o700)

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

            remote = git(
                "ls-remote",
                "origin",
                "refs/heads/main",
                env=env,
                timeout=60
            )

            remote_sha = (
                remote.stdout.split()[0]
                if (
                    remote.returncode == 0
                    and remote.stdout.strip()
                )
                else None
            )

            remote_verified = (
                push.returncode == 0
                and remote_sha == LOCAL_HEAD
            )

            if push.returncode != 0:

                print(
                    "Push warning:",
                    push.stderr[:1000]
                )

    finally:

        del token

        if "env" in locals():
            env.pop(
                "THESIS_GH_TOKEN",
                None
            )
            del env

# ======================================================================
# 17. FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 78)
print("PHASE 7B SOURCE VERIFICATION COMPLETE")
print("=" * 78)

print(
    "Datasets:",
    len(DATASETS)
)

print(
    "Source tables profiled:",
    len(profiles)
)

print(
    "Experiment scripts inspected:",
    len(code_df)
)

print(
    "Provisional normalized rows:",
    len(normalized_df)
)

print(
    "Protocol-known reference rows:",
    len(strict_df)
)

print(
    "Training-size evidence candidates:",
    len(size_df)
)

print(
    "Source saved:",
    source_saved
)

print(
    "Model training performed:",
    False
)

print(
    "Google Drive used:",
    False
)

print(
    "GitHub remote verified:",
    remote_verified
)

print(
    "Local commit:",
    LOCAL_HEAD
)

print("\nNEXT:")
print(
    "Review the source-level evidence matrix, "
    "verify counterfactual definitions, and prepare "
    "the strictly comparable seven-dataset synthesis."
)

print("=" * 78)