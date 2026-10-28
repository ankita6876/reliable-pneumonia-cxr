# ======================================================================
# MSc THESIS — PHASE 7A
# SEVEN-DATASET GITHUB-FIRST EVIDENCE AUDIT
#
# PURPOSE
#   1. Recover and inventory existing research artifacts.
#   2. Inspect actual CSV schemas and provenance.
#   3. Identify validated, held-out and training-size evidence.
#   4. Compare results ONLY where metric semantics are compatible.
#   5. Create provisional cross-dataset synthesis and audit figures.
#   6. Preserve all audit outputs in GitHub.
#
# NO MODEL TRAINING
# NO DATASET DOWNLOADS
# NO GOOGLE DRIVE
# NO CHANGES TO EXISTING EXPERIMENTAL RESULTS
# ======================================================================

import os
import sys
import re
import json
import csv
import hashlib
import getpass
import tempfile
import subprocess
import traceback
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------

ROOT = Path("/kaggle/working")
REPO = ROOT / "reliable-pneumonia-cxr"

OUT = (
    REPO
    / "results"
    / "seven_dataset_synthesis"
    / "phase7a_evidence_audit"
)

TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"

SOURCE_DIR = (
    REPO
    / "experiments"
    / "seven_dataset_synthesis"
)

for p in [TABLES, FIGURES, REPORTS, SOURCE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "chexpert": ["chexpert"],
    "oxford_pet": ["oxford_pet", "oxford-iiit", "pet_spatial", "pet_"],
    "nico": ["nico"],
    "breastmnist": ["breastmnist"],
    "dermamnist": ["dermamnist"],
    "cifar10": ["cifar10", "cifar_10", "cifar-10"],
    "pathmnist": ["pathmnist"],
}

EXPECTED_SEEDS = {42, 123, 2026}

EXPECTED_RELIABILITIES = {
    "independent", "0.5", "0.75", "1.0"
}

EXPECTED_POSITIONS = {
    "center", "bottom_right", "random"
}

EXPECTED_METRICS = {
    "macro_auroc_gap",
    "macro_f1_gap",
    "decision_flip_rate",
    "mean_total_variation"
}

MAX_CSV_BYTES = 200 * 1024 * 1024
MAX_SCHEMA_ROWS = 5000

assert (REPO / ".git").exists(), (
    "Git repository missing. Restore the existing thesis "
    "repository before running this audit."
)

print("=" * 78)
print("THESIS — PHASE 7A: SEVEN-DATASET EVIDENCE AUDIT")
print("=" * 78)
print("Repository:", REPO)
print("Output:", OUT)

# ----------------------------------------------------------------------
# 2. HELPERS
# ----------------------------------------------------------------------

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run_command(args, env=None, timeout=120):
    return subprocess.run(
        args,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout
    )


def git(*args, env=None, timeout=120):
    return run_command(
        ["git", *args],
        env=env,
        timeout=timeout
    )


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def dataset_from_path(path):
    text = str(path).lower().replace("\\", "/")

    matches = [
        name
        for name, aliases in DATASETS.items()
        if any(alias in text for alias in aliases)
    ]

    return matches[0] if len(matches) == 1 else None


def canonical_reliability(value):
    if pd.isna(value):
        return None

    s = str(value).strip().lower()

    if s in {"clean", "baseline", "none"}:
        return "clean"

    if s in {"independent", "neutral", "uncorrelated"}:
        return "independent"

    try:
        v = float(s.replace("p", "."))
        if v in {0.5, 0.75, 1.0}:
            return str(v)
    except (ValueError, TypeError):
        pass

    return None


def canonical_position(value):
    if pd.isna(value):
        return None

    s = str(value).strip().lower().replace("-", "_")

    aliases = {
        "center": "center",
        "centre": "center",
        "interior": "center",
        "bottom_right": "bottom_right",
        "bottomright": "bottom_right",
        "exterior": "bottom_right",
        "random": "random",
        "none": "none"
    }

    return aliases.get(s)


def safe_read_csv(path, nrows=None):
    if path.stat().st_size > MAX_CSV_BYTES:
        raise ValueError("CSV exceeds audit size limit")

    return pd.read_csv(
        path,
        nrows=nrows,
        low_memory=False
    )


def save_csv(df, name):
    path = TABLES / name
    df.to_csv(path, index=False)
    return path


def choose_column(df, names):
    return next((c for c in names if c in df.columns), None)


def unique_values(df, column, limit=30):
    if column is None:
        return []

    vals = df[column].dropna().astype(str).unique()
    return sorted(vals.tolist())[:limit]


# ----------------------------------------------------------------------
# 3. RECORD GIT PROVENANCE
# ----------------------------------------------------------------------

head = git("rev-parse", "HEAD")
branch = git("branch", "--show-current")
status = git("status", "--porcelain")

assert head.returncode == 0, head.stderr

START_HEAD = head.stdout.strip()
START_BRANCH = branch.stdout.strip()

print("\nGit HEAD:", START_HEAD)
print("Branch:", START_BRANCH)

if status.stdout.strip():
    print(
        "NOTE: Existing uncommitted files detected. "
        "The audit will not discard or overwrite them."
    )

# ----------------------------------------------------------------------
# 4. INVENTORY ALL RESEARCH FILES
# ----------------------------------------------------------------------

print("\n[1/7] Inventorying repository files...")

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    "node_modules"
}

inventory = []

for path in REPO.rglob("*"):

    if not path.is_file():
        continue

    relative = path.relative_to(REPO)

    if any(part in SKIP_DIRS for part in relative.parts):
        continue

    # Do not recursively inventory this audit's own outputs.
    if path.is_relative_to(OUT):
        continue

    if path.is_relative_to(SOURCE_DIR):
        continue

    dataset = dataset_from_path(relative)

    suffix = path.suffix.lower()

    inventory.append({
        "path": relative.as_posix(),
        "dataset": dataset or "unassigned",
        "extension": suffix,
        "size_bytes": path.stat().st_size,
        "is_csv": suffix == ".csv",
        "is_checkpoint": suffix in {
            ".pt", ".pth", ".ckpt", ".safetensors"
        },
        "is_code": suffix in {
            ".py", ".ipynb"
        },
        "is_figure": suffix in {
            ".png", ".pdf", ".svg", ".jpg"
        },
        "is_manifest": (
            suffix == ".json"
            and "manifest" in path.name.lower()
        )
    })

inventory_df = pd.DataFrame(inventory)

save_csv(
    inventory_df,
    "repository_artifact_inventory.csv"
)

print("Research files found:", len(inventory_df))

# ----------------------------------------------------------------------
# 5. CSV SCHEMA AUDIT
# ----------------------------------------------------------------------

print("\n[2/7] Auditing research table schemas...")

schema_rows = []
errors = []
loaded_tables = {}

for item in inventory:

    if not item["is_csv"]:
        continue

    if item["dataset"] == "unassigned":
        continue

    path = REPO / item["path"]

    try:
        sample = safe_read_csv(
            path,
            nrows=MAX_SCHEMA_ROWS
        )

        columns = list(sample.columns)

        # Read the complete table only if reasonably sized.
        if item["size_bytes"] <= 30 * 1024 * 1024:
            full = safe_read_csv(path)
            row_count = len(full)
            loaded_tables[item["path"]] = full
        else:
            row_count = None

        schema_rows.append({
            "dataset": item["dataset"],
            "path": item["path"],
            "rows": row_count,
            "sample_rows": len(sample),
            "columns": json.dumps(columns),
            "has_seed": any(
                c in columns
                for c in ["seed", "random_seed"]
            ),
            "has_reliability": any(
                c in columns
                for c in [
                    "training_reliability",
                    "reliability",
                    "correlation",
                    "rho"
                ]
            ),
            "has_position": any(
                c in columns
                for c in [
                    "training_position",
                    "position",
                    "marker_position"
                ]
            ),
            "has_auroc": any(
                "auroc" in c.lower()
                for c in columns
            ),
            "has_f1": any(
                "f1" in c.lower()
                for c in columns
            ),
            "has_flip": any(
                "flip" in c.lower()
                for c in columns
            )
        })

    except Exception as exc:

        errors.append({
            "path": item["path"],
            "error": repr(exc)
        })

schema_df = pd.DataFrame(schema_rows)

save_csv(
    schema_df,
    "csv_schema_audit.csv"
)

save_csv(
    pd.DataFrame(errors, columns=["path", "error"]),
    "csv_read_errors.csv"
)

print("CSV schemas inspected:", len(schema_df))
print("Complete CSV tables loaded:", len(loaded_tables))
print("CSV read errors:", len(errors))

# ----------------------------------------------------------------------
# 6. DATASET EVIDENCE MATRIX
# ----------------------------------------------------------------------

print("\n[3/7] Building evidence-completeness matrix...")

evidence_rows = []

for dataset in DATASETS:

    subset = inventory_df[
        inventory_df["dataset"] == dataset
    ]

    paths = subset["path"].tolist()

    csv_paths = [
        p for p in paths
        if p.lower().endswith(".csv")
    ]

    test_paths = [
        p for p in csv_paths
        if any(
            marker in p.lower()
            for marker in [
                "heldout_test",
                "held_out_test",
                "test_metrics",
                "test_shortcut"
            ]
        )
    ]

    validation_paths = [
        p for p in csv_paths
        if any(
            marker in p.lower()
            for marker in [
                "validation",
                "val_metrics",
                "val_reliance"
            ]
        )
    ]

    size_paths = [
        p for p in paths
        if any(
            marker in p.lower()
            for marker in [
                "train_size",
                "training_size",
                "size_ablation",
                "sample_size"
            ]
        )
    ]

    evidence_rows.append({
        "dataset": dataset,
        "total_artifacts": len(subset),
        "csv_tables": len(csv_paths),
        "checkpoint_files": int(
            subset["is_checkpoint"].sum()
        ),
        "code_files": int(
            subset["is_code"].sum()
        ),
        "figure_files": int(
            subset["is_figure"].sum()
        ),
        "manifest_files": int(
            subset["is_manifest"].sum()
        ),
        "validation_table_candidates": len(
            validation_paths
        ),
        "heldout_test_table_candidates": len(
            test_paths
        ),
        "size_ablation_artifact_candidates": len(
            size_paths
        ),
        "heldout_test_verified": False,
        "size_ablation_verified": False,
        "comparability_verified": False,
        "status": "requires_schema_and_provenance_review"
    })

evidence_df = pd.DataFrame(evidence_rows)

# PathMNIST has explicitly verified outputs from the completed run.
# This verification is conditional on finding those exact artifacts.
path_test = (
    REPO
    / "results/benchmark_expansion/pathmnist/"
    "heldout_test/tables/pathmnist_test_metrics.csv"
)

path_reliance = (
    REPO
    / "results/benchmark_expansion/pathmnist/"
    "heldout_test/tables/"
    "pathmnist_test_shortcut_reliance.csv"
)

path_heads = list(
    (
        REPO
        / "results/benchmark_expansion/pathmnist/"
        "training/heads"
    ).glob("*.pt")
)

if path_test.exists() and path_reliance.exists():

    test_df = safe_read_csv(path_test)
    reliance_test_df = safe_read_csv(path_reliance)

    if (
        "run_id" in test_df
        and "run_id" in reliance_test_df
        and test_df["run_id"].nunique() == 39
        and reliance_test_df["run_id"].nunique() == 39
    ):

        idx = evidence_df["dataset"] == "pathmnist"

        evidence_df.loc[
            idx, "heldout_test_verified"
        ] = True

        evidence_df.loc[
            idx, "status"
        ] = "heldout_tables_verified"

        print(
            "PathMNIST held-out tables verified:",
            len(test_df),
            "metric rows,",
            len(reliance_test_df),
            "reliance rows"
        )

        print(
            "PathMNIST checkpoints present:",
            len(path_heads)
        )

save_csv(
    evidence_df,
    "seven_dataset_evidence_matrix.csv"
)

# ----------------------------------------------------------------------
# 7. EXTRACT STRICTLY COMPARABLE PATHMNIST RESULTS
# ----------------------------------------------------------------------

print("\n[4/7] Verifying metric semantics...")

verified_rows = []
comparison_warnings = []

# A matching-vs-alternative AUROC gap is not automatically
# equivalent to a clean-vs-marked gap or a probability shift.
#
# Only this known PathMNIST table is normalized automatically.
# Other datasets are flagged for explicit semantic review.

path_summary = (
    REPO
    / "results/benchmark_expansion/pathmnist/"
    "heldout_test/tables/"
    "pathmnist_test_three_seed_summary.csv"
)

if path_summary.exists():

    df = safe_read_csv(path_summary)

    required = {
        "training_reliability",
        "training_position",
        "mean_auroc_gap",
        "mean_macro_f1_gap",
        "mean_decision_flip_rate"
    }

    if required.issubset(df.columns):

        for _, row in df.iterrows():

            verified_rows.append({
                "dataset": "pathmnist",
                "split": "official_test",
                "training_reliability": canonical_reliability(
                    row["training_reliability"]
                ),
                "training_position": canonical_position(
                    row["training_position"]
                ),
                "metric_definition": (
                    "matching_minus_mean_alternative"
                ),
                "macro_auroc_gap": row["mean_auroc_gap"],
                "macro_f1_gap": row["mean_macro_f1_gap"],
                "decision_flip_rate": row[
                    "mean_decision_flip_rate"
                ],
                "source": path_summary.relative_to(
                    REPO
                ).as_posix(),
                "evidence_level": "verified_schema_and_known_protocol"
            })

for dataset in DATASETS:

    if dataset == "pathmnist":
        continue

    dataset_tables = schema_df[
        schema_df["dataset"] == dataset
    ] if len(schema_df) else pd.DataFrame()

    comparison_warnings.append({
        "dataset": dataset,
        "reason": (
            "Cross-dataset metric definitions and split "
            "provenance require explicit verification."
        ),
        "candidate_csv_tables": len(dataset_tables),
        "included_in_numeric_comparison": False
    })

verified_df = pd.DataFrame(verified_rows)

save_csv(
    verified_df,
    "verified_normalized_results.csv"
)

save_csv(
    pd.DataFrame(comparison_warnings),
    "cross_dataset_comparability_warnings.csv"
)

# ----------------------------------------------------------------------
# 8. DATASET-SPECIFIC SOURCE DISCOVERY
# ----------------------------------------------------------------------

print("\n[5/7] Discovering likely primary result tables...")

candidate_rows = []

for _, row in schema_df.iterrows():

    path = row["path"]
    name = path.lower()

    score = 0

    for keyword, points in {
        "three_seed": 5,
        "summary": 3,
        "heldout": 5,
        "test": 3,
        "reliance": 4,
        "bootstrap": 3,
        "matrix": 2,
        "size": 2,
        "counterfactual": 3,
        "metrics": 2
    }.items():

        if keyword in name:
            score += points

    if row["has_seed"]:
        score += 2

    if row["has_auroc"]:
        score += 2

    if row["has_f1"]:
        score += 1

    if row["has_flip"]:
        score += 2

    candidate_rows.append({
        "dataset": row["dataset"],
        "path": path,
        "priority_score": score,
        "rows": row["rows"],
        "columns": row["columns"]
    })

candidate_df = pd.DataFrame(candidate_rows)

if len(candidate_df):

    candidate_df = candidate_df.sort_values(
        ["dataset", "priority_score"],
        ascending=[True, False]
    )

save_csv(
    candidate_df,
    "primary_result_table_candidates.csv"
)

# ----------------------------------------------------------------------
# 9. FIGURES
# ----------------------------------------------------------------------

print("\n[6/7] Generating audit figures...")

fig, ax = plt.subplots(figsize=(10, 5.5))

plot_df = evidence_df.sort_values("dataset")

x = np.arange(len(plot_df))

ax.bar(
    x,
    plot_df["csv_tables"].values,
    label="CSV tables"
)

ax.set_xticks(x)
ax.set_xticklabels(
    plot_df["dataset"].values,
    rotation=35,
    ha="right"
)

ax.set_ylabel("Number of discovered tables")
ax.set_title(
    "Seven-Dataset Evidence Inventory\n"
    "(Artifact counts are not scientific results)"
)

ax.grid(axis="y", alpha=0.2)
ax.legend()

plt.tight_layout()

for ext in ["png", "pdf"]:
    fig.savefig(
        FIGURES / f"dataset_evidence_inventory.{ext}",
        dpi=250,
        bbox_inches="tight"
    )

plt.show()
plt.close(fig)

if len(verified_df):

    subset = verified_df[
        verified_df["training_reliability"] == "1.0"
    ].copy()

    if len(subset):

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(
            subset["training_position"],
            subset["macro_f1_gap"]
        )

        ax.set_ylabel(
            "Matching − alternative macro-F1"
        )

        ax.set_title(
            "PathMNIST Held-Out Shortcut Reliance\n"
            "100% Marker Reliability"
        )

        ax.grid(axis="y", alpha=0.2)

        plt.tight_layout()

        for ext in ["png", "pdf"]:
            fig.savefig(
                FIGURES
                / f"pathmnist_verified_test_f1_gap.{ext}",
                dpi=250,
                bbox_inches="tight"
            )

        plt.show()
        plt.close(fig)

# ----------------------------------------------------------------------
# 10. SCIENTIFIC AUDIT REPORT
# ----------------------------------------------------------------------

report = []

report.append("# Seven-Dataset Scientific Evidence Audit")
report.append("")
report.append(f"Generated: {utc_now()}")
report.append(f"Starting Git commit: `{START_HEAD}`")
report.append("")
report.append("## Scope")
report.append("")
report.append(
    "This audit inventories available research artifacts "
    "without retraining models or modifying experimental data."
)
report.append("")
report.append("## Evidence Inventory")
report.append("")
report.append(
    evidence_df.to_markdown(index=False)
)
report.append("")
report.append("## Interpretation Rules")
report.append("")
report.append(
    "1. Artifact presence does not establish experimental validity."
)
report.append(
    "2. Validation results must not be represented as held-out test results."
)
report.append(
    "3. Matching-minus-alternative, matching-minus-clean, "
    "and other counterfactual gaps must not be pooled without "
    "verifying their definitions."
)
report.append(
    "4. Marker position comparisons may be affected by "
    "background interference and marker visibility."
)
report.append(
    "5. Three training seeds are repeated model runs, "
    "not three independent datasets."
)
report.append(
    "6. Bootstrap confidence intervals must be labeled "
    "with the precise contrast and resampling unit."
)
report.append(
    "7. The current PathMNIST bootstrap table concerns "
    "matching-minus-clean macro-F1; it is not a confidence "
    "interval for matching-minus-alternative AUROC."
)
report.append("")
report.append("## Pending Verification")
report.append("")
report.append(
    "The other six datasets require explicit source-level "
    "review before a unified numeric meta-analysis."
)
report.append("")
report.append(
    "The presence of checkpoints, size-ablation artifacts, "
    "or test-named files is not sufficient to certify "
    "their evaluation protocol."
)
report.append("")
report.append("## Primary Table Candidates")
report.append("")

if len(candidate_df):

    for dataset in DATASETS:

        top = candidate_df[
            candidate_df["dataset"] == dataset
        ].head(5)

        report.append(f"### {dataset}")
        report.append("")

        if len(top):

            for _, item in top.iterrows():
                report.append(
                    f"- `{item['path']}` "
                    f"(priority {item['priority_score']})"
                )

        else:

            report.append(
                "- No automatically identified candidate."
            )

        report.append("")

report.append("## Required Next Steps")
report.append("")
report.append(
    "1. Inspect primary result tables and original "
    "metric-generation code for all seven datasets."
)
report.append(
    "2. Establish a common comparison dictionary "
    "and distinguish validation from official test."
)
report.append(
    "3. Audit marker generation, sampling, class balance, "
    "seed handling, and training-size subsets."
)
report.append(
    "4. Calculate paired confidence intervals for "
    "the principal shortcut-reliance contrasts."
)
report.append(
    "5. Produce final cross-dataset synthesis only "
    "for verified compatible metrics."
)

report_path = (
    REPORTS
    / "seven_dataset_evidence_audit.md"
)

report_path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

# ----------------------------------------------------------------------
# 11. SAVE REPRODUCIBILITY MANIFEST
# ----------------------------------------------------------------------

manifest = {
    "phase": "7A",
    "description": "Seven-dataset evidence and schema audit",
    "created_utc": utc_now(),
    "starting_git_head": START_HEAD,
    "branch": START_BRANCH,
    "datasets": list(DATASETS),
    "research_files_found": len(inventory_df),
    "csv_schemas_inspected": len(schema_df),
    "complete_csv_tables_loaded": len(loaded_tables),
    "csv_read_errors": len(errors),
    "training_performed": False,
    "datasets_downloaded": False,
    "google_drive_used": False,
    "cross_dataset_meta_analysis_performed": False,
    "comparison_policy": (
        "No cross-dataset numerical pooling until "
        "metric semantics and provenance are verified."
    )
}

manifest_path = (
    OUT
    / "phase7a_manifest.json"
)

manifest_path.write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8"
)

# ----------------------------------------------------------------------
# 12. CAPTURE EXECUTABLE SOURCE
# ----------------------------------------------------------------------

source_saved = False

try:
    from IPython import get_ipython

    ip = get_ipython()

    if ip is not None:

        candidates = [
            s for s in ip.history_manager.input_hist_raw
            if (
                "SEVEN-DATASET GITHUB-FIRST EVIDENCE AUDIT" in s
                and "primary_result_table_candidates.csv" in s
                and "phase7a_manifest.json" in s
            )
        ]

        if candidates:

            source_path = (
                SOURCE_DIR
                / "phase7a_evidence_audit.py"
            )

            source_path.write_text(
                candidates[-1],
                encoding="utf-8"
            )

            source_saved = True

except Exception as exc:
    print("Source capture warning:", repr(exc))

# ----------------------------------------------------------------------
# 13. COMMIT ONLY PHASE 7A OUTPUTS
# ----------------------------------------------------------------------

print("\n[7/7] Preserving audit artifacts...")

allowed_suffixes = {
    ".csv", ".json", ".md", ".png", ".pdf", ".py"
}

artifact_paths = [
    p
    for base in [OUT, SOURCE_DIR]
    for p in base.rglob("*")
    if (
        p.is_file()
        and p.suffix.lower() in allowed_suffixes
        and p.stat().st_size < 20 * 1024 * 1024
    )
]

# Do not stage unrelated work.
previously_staged = git(
    "diff", "--cached", "--name-only"
).stdout.splitlines()

if previously_staged:
    raise RuntimeError(
        "Unrelated staged changes detected. "
        "Audit outputs are saved, but commit was skipped "
        "to avoid including other work. "
        f"Staged paths: {previously_staged[:10]}"
    )

for p in artifact_paths:

    result = git(
        "add",
        "--",
        p.relative_to(REPO).as_posix()
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

staged = git(
    "diff", "--cached", "--name-only"
).stdout.splitlines()

allowed_prefixes = (
    "results/seven_dataset_synthesis/phase7a_evidence_audit/",
    "experiments/seven_dataset_synthesis/"
)

assert all(
    p.startswith(allowed_prefixes)
    for p in staged
), "Unexpected staged file detected."

if staged:

    author_name = git(
        "config", "--get", "user.name"
    ).stdout.strip()

    author_email = git(
        "config", "--get", "user.email"
    ).stdout.strip()

    if not author_name or not author_email:

        git(
            "config", "--local",
            "user.name", "Thesis Research"
        )

        git(
            "config", "--local",
            "user.email",
            "thesis-research@users.noreply.github.com"
        )

    commit = git(
        "commit",
        "-m",
        "research: audit seven-dataset evidence and provenance"
    )

    if commit.returncode != 0:
        raise RuntimeError(commit.stderr)

local_head = git(
    "rev-parse", "HEAD"
).stdout.strip()

print("Local commit:", local_head)

# ----------------------------------------------------------------------
# 14. OPTIONAL SECURE GITHUB PUSH
# ----------------------------------------------------------------------

remote_verified = False

answer = input(
    "\nPush Phase 7A audit to GitHub? [y/N]: "
).strip().lower()

if answer == "y":

    token = getpass.getpass(
        "Paste GitHub Personal Access Token: "
    ).strip()

    try:

        with tempfile.TemporaryDirectory() as td:

            askpass = Path(td) / "askpass.sh"

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
                "push", "origin", "main",
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
                if remote.returncode == 0
                and remote.stdout.strip()
                else None
            )

            remote_verified = (
                push.returncode == 0
                and remote_sha == local_head
            )

            if push.returncode != 0:
                print(
                    "Push error:",
                    push.stderr[:1000]
                )

    finally:
        del token

        if "env" in locals():
            env.pop("THESIS_GH_TOKEN", None)
            del env

# ----------------------------------------------------------------------
# 15. FINAL OUTPUT
# ----------------------------------------------------------------------

print("\n" + "=" * 78)
print("PHASE 7A EVIDENCE AUDIT COMPLETE")
print("=" * 78)

print("Datasets targeted:", len(DATASETS))
print("Repository files inventoried:", len(inventory_df))
print("CSV schemas inspected:", len(schema_df))
print("Primary table candidates:", len(candidate_df))
print("CSV read errors:", len(errors))
print("Verified normalized result rows:", len(verified_df))
print("Source saved:", source_saved)
print("Model training performed: False")
print("Google Drive used: False")
print("GitHub remote verified:", remote_verified)
print("Local commit:", local_head)

print("\nSEVEN-DATASET EVIDENCE MATRIX")
print(
    evidence_df[
        [
            "dataset",
            "csv_tables",
            "checkpoint_files",
            "heldout_test_table_candidates",
            "size_ablation_artifact_candidates",
            "heldout_test_verified"
        ]
    ].to_string(index=False)
)

print("\nTOP PRIMARY RESULT TABLES")

if len(candidate_df):

    print(
        candidate_df.groupby(
            "dataset",
            group_keys=False
        ).head(3)[
            [
                "dataset",
                "priority_score",
                "path"
            ]
        ].to_string(index=False)
    )

print("\nNEXT:")
print(
    "Use the audited source tables to verify metric "
    "definitions, recover missing held-out evidence, "
    "and build the final seven-dataset comparison."
)

print("=" * 78)