"""Validate the frozen thesis lung segmenter on explicitly mapped JSRT/SCR cases.

This script is intentionally inference-only.  It calls ``FrozenLungSegmenter``
directly so preprocessing, sigmoid activation, and resizing remain identical to
the pipeline used for the historical hard-masked classifier inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pneumonia_ai.segmentation.inference import FrozenLungSegmenter
from pneumonia_ai.segmentation.mask_utils import (
    MaskValidationError,
    merge_lung_masks,
    read_mask,
)

HISTORICAL_CHECKPOINT_SHA256 = "bdcbb77d4872292721a5bce8328401274d1b7d917e5e413f173873c8cf886a1d"
MASK_THRESHOLD = 0.5
CASE_ID_RE = re.compile(r"(?<![A-Z0-9])(JPCLN\d{3})(?![A-Z0-9])", re.IGNORECASE)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class CaseMapping:
    """One auditable JSRT image to SCR left/right mask mapping."""

    case_id: str
    image_path: Path | None
    left_mask_path: Path | None
    right_mask_path: Path | None
    mapping_status: str
    notes: str = ""


def parse_args() -> argparse.Namespace:
    """Parse command-line options without selecting any data-dependent setting."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--masks-root", type=Path, required=True)
    parser.add_argument("--left-masks-root", type=Path)
    parser.add_argument("--right-masks-root", type=Path)
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        help="Optional reviewed CSV with case_id,image_path,left_mask_path,right_mask_path.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--expected-checkpoint-sha256", default=HISTORICAL_CHECKPOINT_SHA256)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return a checkpoint SHA-256 without loading or modifying it."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint_sha256(path: Path, expected: str) -> str:
    """Verify the exact historical checkpoint identity, failing on mismatch."""

    if not path.is_file():
        raise FileNotFoundError(f"Segmentation checkpoint does not exist: {path}")
    actual = sha256_file(path)
    expected_normalized = expected.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_normalized):
        raise ValueError("--expected-checkpoint-sha256 must be a 64-character hexadecimal SHA-256.")
    if actual != expected_normalized:
        raise ValueError(
            "Segmentation checkpoint SHA-256 mismatch: "
            f"expected {expected_normalized}, observed {actual}. Inference was not started."
        )
    return actual


def case_id_from_path(path: Path) -> str | None:
    """Extract a canonical JSRT ID only when its explicit JPCLN form is present."""

    match = CASE_ID_RE.search(path.stem)
    return match.group(1).upper() if match else None


def _files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise NotADirectoryError(f"Directory does not exist: {root}")
    return sorted((path for path in root.rglob("*") if path.is_file()), key=lambda path: str(path).lower())


def _index_paths(paths: Iterable[Path]) -> tuple[dict[str, list[Path]], list[Path]]:
    indexed: dict[str, list[Path]] = {}
    unrecognized: list[Path] = []
    for path in paths:
        case_id = case_id_from_path(path)
        if case_id is None:
            unrecognized.append(path)
        else:
            indexed.setdefault(case_id, []).append(path)
    return indexed, unrecognized


def _find_mask_roots(masks_root: Path, left: Path | None, right: Path | None) -> tuple[Path, Path]:
    if (left is None) != (right is None):
        raise ValueError("Supply both --left-masks-root and --right-masks-root, or neither.")
    if left is not None and right is not None:
        return left, right
    candidates = (("left", "right"), ("leftMask", "rightMask"), ("left_lung", "right_lung"))
    found = [(masks_root / left_name, masks_root / right_name) for left_name, right_name in candidates
             if (masks_root / left_name).is_dir() and (masks_root / right_name).is_dir()]
    if len(found) != 1:
        raise ValueError(
            "Could not identify exactly one separate left/right mask directory pair below "
            f"{masks_root}. Supply --left-masks-root and --right-masks-root explicitly."
        )
    return found[0]


def _mapping_from_csv(mapping_csv: Path, images_root: Path, masks_root: Path) -> list[CaseMapping]:
    if not mapping_csv.is_file():
        raise FileNotFoundError(f"Mapping CSV does not exist: {mapping_csv}")
    rows = list(csv.DictReader(mapping_csv.open(newline="", encoding="utf-8")))
    required = {"case_id", "image_path", "left_mask_path", "right_mask_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Mapping CSV requires columns: {', '.join(sorted(required))}")
    mappings: list[CaseMapping] = []
    seen: set[str] = set()
    for row in rows:
        case_id = str(row["case_id"]).upper().strip()
        if not CASE_ID_RE.fullmatch(case_id):
            raise ValueError(f"Mapping CSV has non-canonical JSRT case ID: {case_id!r}")
        if case_id in seen:
            raise ValueError(f"Mapping CSV contains duplicate case ID: {case_id}")
        seen.add(case_id)
        image = _resolve_csv_path(row["image_path"], images_root)
        left = _resolve_csv_path(row["left_mask_path"], masks_root)
        right = _resolve_csv_path(row["right_mask_path"], masks_root)
        missing = [name for name, path in (("image", image), ("left mask", left), ("right mask", right)) if not path.is_file()]
        mappings.append(CaseMapping(case_id, image, left, right, "mapped" if not missing else "missing_file", "; ".join(missing)))
    return mappings


def _resolve_csv_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def discover_mappings(
    images_root: Path,
    masks_root: Path,
    *,
    left_masks_root: Path | None = None,
    right_masks_root: Path | None = None,
    mapping_csv: Path | None = None,
) -> list[CaseMapping]:
    """Create auditable mappings without trusting arbitrary filename similarity.

    Automatic discovery uses only canonical JSRT IDs embedded in each path and
    requires exactly one image and one mask from each separately supplied side.
    A reviewed mapping CSV is available for a differently structured mirror.
    """

    if mapping_csv is not None:
        return _mapping_from_csv(mapping_csv, images_root, masks_root)
    image_index, unrecognized_images = _index_paths([path for path in _files(images_root) if path.suffix.lower() in IMAGE_SUFFIXES])
    left_root, right_root = _find_mask_roots(masks_root, left_masks_root, right_masks_root)
    left_index, unrecognized_left = _index_paths(
        [path for path in _files(left_root) if path.suffix.lower() in IMAGE_SUFFIXES]
    )
    right_index, unrecognized_right = _index_paths(
        [path for path in _files(right_root) if path.suffix.lower() in IMAGE_SUFFIXES]
    )
    mappings: list[CaseMapping] = []
    all_ids = sorted(set(image_index) | set(left_index) | set(right_index))
    for case_id in all_ids:
        images, lefts, rights = image_index.get(case_id, []), left_index.get(case_id, []), right_index.get(case_id, [])
        issues: list[str] = []
        if len(images) != 1:
            issues.append("missing image" if not images else f"duplicate images ({len(images)})")
        if len(lefts) != 1:
            issues.append("missing left mask" if not lefts else f"duplicate left masks ({len(lefts)})")
        if len(rights) != 1:
            issues.append("missing right mask" if not rights else f"duplicate right masks ({len(rights)})")
        status = "mapped" if not issues else ("ambiguous" if any("duplicate" in issue for issue in issues) else "missing_file")
        mappings.append(CaseMapping(case_id, images[0] if len(images) == 1 else None, lefts[0] if len(lefts) == 1 else None, rights[0] if len(rights) == 1 else None, status, "; ".join(issues)))
    for path in unrecognized_images:
        mappings.append(CaseMapping("", path, None, None, "unrecognized_case_id", "image path lacks canonical JPCLN### ID"))
    for path in unrecognized_left:
        mappings.append(CaseMapping("", None, path, None, "unrecognized_case_id", "left-mask path lacks canonical JPCLN### ID"))
    for path in unrecognized_right:
        mappings.append(CaseMapping("", None, None, path, "unrecognized_case_id", "right-mask path lacks canonical JPCLN### ID"))
    return sorted(mappings, key=lambda item: (item.case_id, str(item.image_path or item.left_mask_path or item.right_mask_path)))


def mapping_frame(mappings: list[CaseMapping]) -> pd.DataFrame:
    """Render the mapping audit in the required stable schema."""

    return pd.DataFrame([{
        "case_id": item.case_id,
        "image_path": str(item.image_path) if item.image_path else "",
        "left_mask_path": str(item.left_mask_path) if item.left_mask_path else "",
        "right_mask_path": str(item.right_mask_path) if item.right_mask_path else "",
        "mapping_status": item.mapping_status,
        "notes": item.notes,
    } for item in mappings])


def require_resolved_mappings(mappings: list[CaseMapping]) -> list[CaseMapping]:
    """Reject any unresolved mapping before the checkpoint is loaded/inference begins."""

    resolved = [item for item in mappings if item.mapping_status == "mapped"]
    unresolved = [item for item in mappings if item.mapping_status != "mapped"]
    if unresolved:
        counts = pd.Series([item.mapping_status for item in unresolved]).value_counts().to_dict()
        raise ValueError(f"Unresolved JSRT/SCR mappings prevent inference: {counts}")
    if not resolved:
        raise ValueError("No mapped JSRT/SCR cases were discovered.")
    return resolved


def merge_masks_for_case(left_path: Path, right_path: Path) -> np.ndarray:
    """Merge separate SCR annotations; no cleanup or morphology is applied."""

    return merge_lung_masks(read_mask(left_path), read_mask(right_path)) > 0


def binary_dice_iou(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """Return Dice and IoU with explicit convention for two empty masks."""

    if prediction.shape != truth.shape:
        raise ValueError(f"Prediction/truth shape mismatch: {prediction.shape} and {truth.shape}.")
    prediction = prediction.astype(bool, copy=False)
    truth = truth.astype(bool, copy=False)
    intersection = int(np.logical_and(prediction, truth).sum())
    predicted_pixels = int(prediction.sum())
    truth_pixels = int(truth.sum())
    denominator = predicted_pixels + truth_pixels
    union = int(np.logical_or(prediction, truth).sum())
    dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
    iou = 1.0 if union == 0 else intersection / union
    return float(dice), float(iou)


def deterministic_sample(mappings: list[CaseMapping], max_samples: int | None, seed: int) -> list[CaseMapping]:
    """Select a reproducible case subset only for a declared smoke test."""

    if max_samples is None:
        return list(mappings)
    if max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    if max_samples >= len(mappings):
        return list(mappings)
    indices = np.sort(np.random.default_rng(seed).choice(len(mappings), size=max_samples, replace=False))
    return [mappings[int(index)] for index in indices]


def bootstrap_summary(values: np.ndarray, iterations: int, seed: int) -> dict[str, float | int]:
    """Compute a case-level deterministic percentile bootstrap for the mean."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be a non-empty one-dimensional finite array.")
    if iterations <= 0:
        raise ValueError("--bootstrap-iterations must be positive.")
    generator = np.random.default_rng(seed)
    sampled_means = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled_means[index] = values[generator.integers(0, len(values), size=len(values))].mean()
    return {
        "mean": float(values.mean()),
        "sample_standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "bootstrap_ci_95_low": float(np.percentile(sampled_means, 2.5)),
        "bootstrap_ci_95_high": float(np.percentile(sampled_means, 97.5)),
        "count": len(values),
    }


def evaluate_mappings(
    mappings: list[CaseMapping], segmenter: Any, checkpoint_sha256: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform frozen inference per mapped case and record recoverable failures."""

    metrics: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for item in mappings:
        assert item.image_path is not None and item.left_mask_path is not None and item.right_mask_path is not None
        try:
            with Image.open(item.image_path) as opened:
                image = opened.copy()
            truth = merge_masks_for_case(item.left_mask_path, item.right_mask_path)
            if image.size != (truth.shape[1], truth.shape[0]):
                raise ValueError(f"Image/mask shape mismatch: image {(image.height, image.width)}, mask {truth.shape}.")
            probability = segmenter.predict_proba(image)
            prediction = np.asarray(probability.detach().cpu().numpy()) >= MASK_THRESHOLD
            if prediction.shape != truth.shape:
                raise ValueError(f"Segmenter output/mask shape mismatch: {prediction.shape} and {truth.shape}.")
            dice, iou = binary_dice_iou(prediction, truth)
            metrics.append({
                "case_id": item.case_id,
                "dice": dice,
                "iou": iou,
                "image_height": int(image.height),
                "image_width": int(image.width),
                "predicted_lung_pixels": int(prediction.sum()),
                "ground_truth_lung_pixels": int(truth.sum()),
                "checkpoint_sha256": checkpoint_sha256,
            })
        except (OSError, ValueError, MaskValidationError, RuntimeError) as error:
            failures.append({"case_id": item.case_id, "image_path": str(item.image_path), "failure": str(error)})
    return pd.DataFrame(metrics), pd.DataFrame(failures, columns=("case_id", "image_path", "failure"))


def qualitative_case_ids(metrics: pd.DataFrame) -> list[str]:
    """Choose low, median, and high Dice examples deterministically, without cherry-picking."""

    if metrics.empty:
        return []
    ordered = metrics.sort_values(["dice", "case_id"], kind="stable").reset_index(drop=True)
    desired = list(ordered.index[:2])
    median = float(ordered["dice"].median())
    desired.extend(ordered.assign(distance=(ordered["dice"] - median).abs()).sort_values(["distance", "case_id"], kind="stable").index[:2])
    desired.extend(ordered.index[-2:])
    selected: list[str] = []
    for index in desired:
        case_id = str(ordered.loc[int(index), "case_id"])
        if case_id not in selected:
            selected.append(case_id)
    return selected


def save_qualitative_panel(metrics: pd.DataFrame, mappings: list[CaseMapping], segmenter: Any, output_dir: Path) -> list[str]:
    """Save deterministic representative image/mask/prediction/overlay panels."""

    selected = qualitative_case_ids(metrics)
    if not selected:
        return selected
    lookup = {item.case_id: item for item in mappings}
    figure, axes = plt.subplots(len(selected), 4, figsize=(12, 3 * len(selected)), squeeze=False)
    for row, case_id in enumerate(selected):
        item = lookup[case_id]
        assert item.image_path is not None and item.left_mask_path is not None and item.right_mask_path is not None
        with Image.open(item.image_path) as opened:
            image = np.asarray(opened.convert("L"))
            probability = segmenter.predict_proba(opened.copy()).detach().cpu().numpy()
        truth = merge_masks_for_case(item.left_mask_path, item.right_mask_path)
        prediction = probability >= MASK_THRESHOLD
        panels = ((image, "Chest X-ray", "gray"), (truth, "Ground-truth merged lung mask", "gray"), (prediction, "Predicted binary lung mask", "gray"))
        for column, (array, title, cmap) in enumerate(panels):
            axes[row, column].imshow(array, cmap=cmap)
            axes[row, column].set_title(title if row == 0 else "")
            axes[row, column].axis("off")
        axes[row, 0].set_ylabel(case_id, rotation=0, ha="right", va="center")
        axes[row, 3].imshow(image, cmap="gray")
        axes[row, 3].imshow(np.ma.masked_where(~prediction, prediction), cmap="autumn", alpha=0.45, vmin=0, vmax=1)
        axes[row, 3].set_title("Prediction overlay" if row == 0 else "")
        axes[row, 3].axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "segmentation_qualitative_panel.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / "segmentation_qualitative_panel.pdf", bbox_inches="tight")
    plt.close(figure)
    return selected


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is non-empty: {output_dir}. Use --overwrite only for this new evaluation output.")
    output_dir.mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run mapping audit then (only if safe) exact frozen segmentation inference."""

    if args.bootstrap_iterations <= 0:
        raise ValueError("--bootstrap-iterations must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
    _prepare_output_dir(args.output_dir, args.overwrite)
    mappings = discover_mappings(args.images_root, args.masks_root, left_masks_root=args.left_masks_root,
                                 right_masks_root=args.right_masks_root, mapping_csv=args.mapping_csv)
    mapping_frame(mappings).to_csv(args.output_dir / "segmentation_mapping_audit.csv", index=False)
    resolved = require_resolved_mappings(mappings)
    selected = deterministic_sample(resolved, args.max_samples, args.seed)
    if args.audit_only:
        return {"audit_only": True, "number_discovered": len(mappings), "number_mapped": len(resolved), "number_selected": len(selected)}
    checkpoint_sha = verify_checkpoint_sha256(args.checkpoint, args.expected_checkpoint_sha256)
    segmenter = FrozenLungSegmenter(args.checkpoint, device=args.device)
    metrics, failures = evaluate_mappings(selected, segmenter, checkpoint_sha)
    metrics.to_csv(args.output_dir / "segmentation_case_metrics.csv", index=False)
    failures.to_csv(args.output_dir / "segmentation_failures.csv", index=False)
    summary = {"dice": bootstrap_summary(metrics["dice"].to_numpy(), args.bootstrap_iterations, args.seed),
               "iou": bootstrap_summary(metrics["iou"].to_numpy(), args.bootstrap_iterations, args.seed)} if not metrics.empty else {}
    summary_rows = [{"metric": metric, **values} for metric, values in summary.items()]
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "segmentation_summary.csv", index=False)
    selected_panels = save_qualitative_panel(metrics, selected, segmenter, args.output_dir)
    model_config = segmenter.metadata.get("model_config", {})
    training_config = segmenter.metadata.get("training_config", {})
    metadata = {
        "dataset": "JSRT",
        "annotation_source": "SCR-derived separate left/right lung masks",
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "expected_checkpoint_sha256": args.expected_checkpoint_sha256.lower(),
        "model_config": model_config,
        "segmentation_input_size": segmenter.image_size,
        "in_channels": model_config.get("in_channels"),
        "out_channels": model_config.get("out_channels"),
        "base_channels": model_config.get("base_channels"),
        "depth": model_config.get("depth"),
        "training_config": training_config,
        "mask_threshold": MASK_THRESHOLD,
        "postprocessing": "none",
        "device": args.device,
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "number_discovered": len(mappings),
        "number_mapped": len(resolved),
        "number_evaluated": len(metrics),
        "failure_count": len(failures),
        "qualitative_panel_case_ids": selected_panels,
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    (args.output_dir / "segmentation_summary.json").write_text(json.dumps(summary, indent=2, default=_json_safe), encoding="utf-8")
    (args.output_dir / "segmentation_metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_safe), encoding="utf-8")
    return metadata


def main() -> None:
    """Execute the evaluator and print a concise machine-readable summary."""

    print(json.dumps(run(parse_args()), indent=2, default=_json_safe))


if __name__ == "__main__":
    main()
