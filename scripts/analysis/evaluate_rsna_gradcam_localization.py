"""Resource-aware Grad-CAM localization evaluation on boxed RSNA positives.

This evaluates frozen classifiers only; it never changes model weights or training data.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import pydicom
import torch
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(PROJECT_ROOT / "src"))
from scripts.predict_external import _classifier_configuration, _git_commit, _sha256, load_dicom_as_pil, resolve_image_path  # noqa: E402
from scripts.train_baseline import _transforms  # noqa: E402
from pneumonia_ai.classification.segmentation_guided import InputMode, prepare_classifier_image  # noqa: E402
from pneumonia_ai.models.factory import create_model  # noqa: E402
from pneumonia_ai.segmentation.inference import FrozenLungSegmenter  # noqa: E402

MANIFEST_COLUMNS = ("patient_id", "image_path", "binary_target", "has_bounding_box", "bounding_box_count", "split", "dataset")
BOX_COLUMNS = ("patient_id", "x", "y", "width", "height", "Target")
CASE_COLUMNS = ("patient_id", "model", "input_mode", "probability", "predicted_class", "correct", "pointing_game_hit", "energy_inside_boxes", "heatmap_iou_0_5", "top10_energy_inside", "top20_energy_inside", "lesion_coverage_0_5", "activation_area_ratio_0_5", "bounding_box_count", "lesion_area_pixels", "lesion_area_ratio", "image_width", "image_height")
METRICS = ("pointing_game_hit", "energy_inside_boxes", "heatmap_iou_0_5", "top10_energy_inside", "top20_energy_inside", "lesion_coverage_0_5", "activation_area_ratio_0_5")


def filter_positive_boxed_manifest(manifest: Path, image_root: Path) -> pd.DataFrame:
    if not manifest.is_file(): raise FileNotFoundError(f"Manifest does not exist: {manifest}")
    frame = pd.read_csv(manifest)
    missing = [c for c in MANIFEST_COLUMNS if c not in frame]
    if missing: raise ValueError("Manifest missing required columns: " + ", ".join(missing))
    if frame.patient_id.isna().any() or frame.patient_id.astype(str).duplicated().any(): raise ValueError("Manifest patient_id values must be unique and non-missing")
    frame = frame.copy(); frame.patient_id = frame.patient_id.astype(str)
    frame.binary_target = pd.to_numeric(frame.binary_target, errors="raise").astype(int)
    boxed = frame.has_bounding_box.astype(str).str.lower().isin(("true", "1", "yes")) if frame.has_bounding_box.dtype != bool else frame.has_bounding_box
    frame = frame.loc[(frame.binary_target == 1) & boxed].copy()
    frame["_source_path"] = frame.image_path.map(lambda p: resolve_image_path(p, image_root))
    missing_paths = [str(p) for p in frame._source_path if not p.is_file()]
    if missing_paths: raise FileNotFoundError(f"Selected manifest references missing DICOM: {missing_paths[0]}")
    return frame


def deterministic_patient_sample(frame: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    if max_samples is None: return frame.copy()
    if not 1 <= max_samples <= len(frame): raise ValueError("max_samples must be between 1 and selected positive boxed case count")
    ids = np.asarray(sorted(frame.patient_id.astype(str).unique()))
    selected = set(np.random.default_rng(seed).choice(ids, max_samples, replace=False).tolist())
    return frame.loc[frame.patient_id.isin(selected)].sort_values("patient_id").copy()


def read_boxes(path: Path, selected_ids: set[str]) -> dict[str, list[tuple[float, float, float, float]]]:
    if not path.is_file(): raise FileNotFoundError(f"Bounding-box file does not exist: {path}")
    frame = pd.read_csv(path); missing = [c for c in BOX_COLUMNS if c not in frame]
    if missing: raise ValueError("Bounding-box file missing required columns: " + ", ".join(missing))
    frame.patient_id = frame.patient_id.astype(str)
    boxes: dict[str, list[tuple[float, float, float, float]]] = {}
    positive = pd.to_numeric(frame.Target, errors="coerce").fillna(0).eq(1)
    for _, row in frame.loc[frame.patient_id.isin(selected_ids) & positive].iterrows():
        try: x, y, w, h = map(float, (row.x, row.y, row.width, row.height))
        except (ValueError, TypeError): raise ValueError(f"Invalid bounding-box values for patient {row.patient_id}")
        if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0: raise ValueError(f"Bounding-box dimensions must be positive for patient {row.patient_id}")
        boxes.setdefault(row.patient_id, []).append((x, y, w, h))
    absent = selected_ids.difference(boxes)
    if absent: raise ValueError(f"Selected patient has no valid box: {sorted(absent)[0]}")
    return boxes


def union_box_mask(boxes: list[tuple[float, float, float, float]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for x, y, w, h in boxes:
        if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0: raise ValueError("Bounding-box dimensions must be finite and positive")
        left, top = max(0, math.floor(x)), max(0, math.floor(y))
        right, bottom = min(width, math.ceil(x + w)), min(height, math.ceil(y + h))
        if right <= left or bottom <= top: raise ValueError("Bounding box does not overlap image after clipping")
        mask[top:bottom, left:right] = True
    return mask


def localization_metrics(heatmap: np.ndarray, lesion: np.ndarray) -> dict[str, float]:
    heatmap = np.asarray(heatmap, dtype=np.float64)
    if heatmap.shape != lesion.shape: raise ValueError("Heatmap and lesion mask dimensions differ")
    if not np.isfinite(heatmap).all() or heatmap.max() <= heatmap.min(): raise ValueError("Grad-CAM activation map is constant or non-finite")
    total = heatmap.sum()
    if total <= 0: raise ValueError("Grad-CAM activation map has no positive energy")
    activated = heatmap >= .5
    maximum = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    intersection = np.logical_and(activated, lesion).sum(); union = np.logical_or(activated, lesion).sum()
    def top_fraction(frac: float) -> float:
        k = max(1, int(math.ceil(heatmap.size * frac)))
        index = np.argpartition(heatmap.ravel(), -k)[-k:]
        return float(lesion.ravel()[index].mean())
    return {"pointing_game_hit": float(lesion[maximum]), "energy_inside_boxes": float(heatmap[lesion].sum() / total), "heatmap_iou_0_5": float(intersection / union) if union else 0.0, "top10_energy_inside": top_fraction(.10), "top20_energy_inside": top_fraction(.20), "lesion_coverage_0_5": float(intersection / lesion.sum()) if lesion.any() else 0.0, "activation_area_ratio_0_5": float(activated.mean())}


def resolve_gradcam_target_layer(model: nn.Module) -> nn.Module:
    features = getattr(model, "features", None)
    if isinstance(features, nn.Module): return features
    convolutions = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convolutions: raise ValueError("Unable to resolve a convolutional Grad-CAM target layer")
    return convolutions[-1]


class GradCAM:
    def __init__(self, model: nn.Module, layer: nn.Module) -> None:
        self.model, self.layer, self.activations, self.gradients = model, layer, None, None
        self._forward = layer.register_forward_hook(self._save_activation)
        self._backward = layer.register_full_backward_hook(self._save_gradient)
    def _save_activation(self, _m: nn.Module, _i: tuple[torch.Tensor, ...], output: torch.Tensor) -> None: self.activations = output
    def _save_gradient(self, _m: nn.Module, _gi: tuple[torch.Tensor | None, ...], go: tuple[torch.Tensor | None, ...]) -> None: self.gradients = go[0]
    def close(self) -> None: self._forward.remove(); self._backward.remove()
    def __enter__(self) -> "GradCAM": return self
    def __exit__(self, *_: object) -> None: self.close()
    def __call__(self, tensor: torch.Tensor, output_size: tuple[int, int]) -> tuple[float, np.ndarray]:
        self.activations = self.gradients = None; self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor).view(-1); logit = logits[0]; logit.backward()
        if self.activations is None or self.gradients is None: raise ValueError("Grad-CAM hooks did not receive activations and gradients")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True); cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)[0, 0].detach().float().cpu().numpy()
        low, high = float(cam.min()), float(cam.max())
        if not np.isfinite(cam).all() or high <= low: raise ValueError("Grad-CAM activation map is constant or unavailable")
        return float(torch.sigmoid(logit.detach()).cpu()), (cam - low) / (high - low)


def paired_frame(cases: pd.DataFrame) -> pd.DataFrame:
    left = cases.loc[cases.model == "original", ["patient_id", *METRICS]].set_index("patient_id")
    right = cases.loc[cases.model == "hard_masked", ["patient_id", *METRICS]].set_index("patient_id")
    return left.join(right, lsuffix="_original", rsuffix="_hard_masked", how="inner").reset_index()


def bootstrap_comparison(paired: pd.DataFrame, iterations: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed); rows = []
    for metric in METRICS:
        differences = (paired[f"{metric}_hard_masked"] - paired[f"{metric}_original"]).to_numpy(float)
        if len(differences) == 0: continue
        samples = np.array([rng.choice(differences, len(differences), replace=True).mean() for _ in range(iterations)])
        # Bootstrap sign probability is a transparent two-sided directional test.
        p = min(1., 2 * min(float((samples <= 0).mean()), float((samples >= 0).mean())))
        rows.append({"metric": metric, "point_difference": float(differences.mean()), "ci_2_5": float(np.percentile(samples, 2.5)), "ci_97_5": float(np.percentile(samples, 97.5)), "two_sided_bootstrap_p_value": p, "iterations": iterations})
    return pd.DataFrame(rows)


def exact_mcnemar(original: np.ndarray, masked: np.ndarray) -> dict[str, float | int]:
    b = int(np.sum((original == 1) & (masked == 0))); c = int(np.sum((original == 0) & (masked == 1))); n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n)
    return {"original_only_hit": b, "masked_only_hit": c, "exact_mcnemar_p_value": p}


def _append_csv(path: Path, row: dict[str, Any], columns: tuple[str, ...]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if not exists: writer.writeheader()
        writer.writerow(row)


def _metadata_base(**values: Any) -> dict[str, Any]: return values


def _plots_and_qualitative(output: Path, cases: pd.DataFrame, paired: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    for metric, name in (("pointing_game_hit", "pointing_game_comparison"), ("energy_inside_boxes", "energy_inside_boxes_comparison"), ("heatmap_iou_0_5", "heatmap_iou_comparison")):
        fig, ax = plt.subplots(figsize=(4, 4)); values = [cases.loc[cases.model == m, metric] for m in ("original", "hard_masked")]
        ax.boxplot(values, tick_labels=["Original", "Hard-masked"]); ax.set_ylabel(metric); fig.tight_layout()
        for suffix in ("png", "pdf"): fig.savefig(output / f"{name}.{suffix}", dpi=300)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); means = [paired[f"{m}_hard_masked"].sub(paired[f"{m}_original"]).mean() for m in METRICS]
    ax.errorbar(means, range(len(METRICS)), xerr=0, fmt="o"); ax.axvline(0, color="black"); ax.set_yticks(range(len(METRICS)), METRICS); fig.tight_layout()
    for suffix in ("png", "pdf"): fig.savefig(output / f"paired_localization_difference_forest.{suffix}", dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4)); cases.boxplot(column=list(METRICS), by="model", ax=ax, rot=45); plt.suptitle(""); fig.tight_layout()
    for suffix in ("png", "pdf"): fig.savefig(output / f"localization_metric_distributions.{suffix}", dpi=300)
    plt.close(fig)


def _save_qualitative_panel(output: Path, candidates: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]]) -> None:
    """Save at most twelve display-sized overlays; no per-case image files are retained."""
    import matplotlib.pyplot as plt
    chosen = [item for group in candidates.values() for item in group[:3]]
    if not chosen: return
    fig, axes = plt.subplots(len(chosen), 3, figsize=(9, 3 * len(chosen)))
    axes = np.atleast_2d(axes)
    for axrow, (image, lesion, original, masked, label) in zip(axes, chosen):
        axrow[0].imshow(image, cmap="gray"); axrow[0].contour(lesion, colors="lime", linewidths=.8); axrow[0].set_title(label + "\nboxes")
        for ax, heatmap, title in ((axrow[1], original, "Original Grad-CAM"), (axrow[2], masked, "Hard-masked Grad-CAM")):
            ax.imshow(image, cmap="gray"); ax.imshow(heatmap, cmap="jet", alpha=.45, vmin=0, vmax=1); ax.contour(lesion, colors="lime", linewidths=.8); ax.set_title(title)
        for ax in axrow: ax.axis("off")
    fig.tight_layout()
    for suffix in ("png", "pdf"): fig.savefig(output / f"qualitative_localization_panel.{suffix}", dpi=300)
    plt.close(fig)


def create_archive(output: Path) -> Path:
    archive = output.parent / f"{output.name}_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in output.iterdir():
            if path.suffix.lower() in {".csv", ".json", ".png", ".pdf"}: z.write(path, path.name)
    return archive


def evaluate_rsna_gradcam_localization(*, manifest: Path, bounding_boxes: Path, original_checkpoint: Path, hard_masked_checkpoint: Path, segmentation_checkpoint: Path | None, image_root: Path, output_directory: Path, device: str = "cpu", batch_size: int = 1, num_workers: int = 0, max_samples: int | None = None, seed: int = 42, overwrite: bool = False, resume: bool = False) -> pd.DataFrame:
    if device not in {"cpu", "cuda"}: raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available(): raise ValueError("CUDA requested but is not available")
    if batch_size < 1 or num_workers < 0: raise ValueError("batch_size must be positive and num_workers non-negative")
    for path in (original_checkpoint, hard_masked_checkpoint):
        if not path.is_file(): raise FileNotFoundError(f"Classifier checkpoint does not exist: {path}")
    if segmentation_checkpoint is None or not segmentation_checkpoint.is_file(): raise ValueError("--segmentation-checkpoint is required for the hard-masked model")
    output_directory.mkdir(parents=True, exist_ok=True); metrics_path = output_directory / "rsna_gradcam_case_metrics.csv"; metadata_path = output_directory / "rsna_gradcam_metadata.json"
    if metrics_path.exists() and not (resume or overwrite): raise FileExistsError("Results already exist; use --resume or --overwrite")
    if overwrite and not resume:
        for p in output_directory.glob("rsna_gradcam_*"):
            if p.is_file(): p.unlink()
    frame = deterministic_patient_sample(filter_positive_boxed_manifest(manifest, image_root), max_samples, seed)
    boxes = read_boxes(bounding_boxes, set(frame.patient_id))
    identity = {"manifest_path": str(manifest.resolve()), "bounding_box_path": str(bounding_boxes.resolve()), "original_checkpoint_sha256": _sha256(original_checkpoint), "hard_masked_checkpoint_sha256": _sha256(hard_masked_checkpoint), "segmentation_checkpoint_sha256": _sha256(segmentation_checkpoint), "seed": seed, "max_samples": max_samples, "selected_patient_ids": frame.patient_id.tolist()}
    completed: set[tuple[str, str]] = set()
    if resume and metrics_path.exists():
        old = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if any(old.get(k) != v for k, v in identity.items()): raise ValueError("Resume metadata is incompatible with requested analysis")
        existing = pd.read_csv(metrics_path); completed = set(zip(existing.patient_id.astype(str), existing.model.astype(str)))
        if len(completed) != len(existing): raise ValueError("Existing case metrics contain duplicate patient-model rows")
    # Persist identity before expensive model work: an interrupted Kaggle run is resumable.
    if not resume:
        metadata_path.write_text(json.dumps({**identity, "status": "in_progress"}, indent=2), encoding="utf-8")
    started = datetime.now(timezone.utc)
    states = {}
    for name, checkpoint in (("original", original_checkpoint), ("hard_masked", hard_masked_checkpoint)):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(state, Mapping) or not isinstance(state.get("model_state_dict"), Mapping): raise ValueError(f"{name} checkpoint lacks model_state_dict")
        config = _classifier_configuration(checkpoint, state); mode = InputMode(config["input_mode"])
        if name == "original" and mode is not InputMode.ORIGINAL: raise ValueError("Original checkpoint does not declare original input mode")
        if name == "hard_masked" and mode is not InputMode.HARD_MASKED: raise ValueError("Hard-masked checkpoint does not declare hard_masked input mode")
        model = create_model(config["backbone"], pretrained=False); model.load_state_dict(state["model_state_dict"], strict=True); model.to(device).eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        _, transform = _transforms(int(config["classifier_image_size"]), str(config["preprocessing"]))
        states[name] = (model, config, mode, transform, resolve_gradcam_target_layer(model))
    segmenter = FrozenLungSegmenter(segmentation_checkpoint, device)
    failures_path = output_directory / "rsna_gradcam_failures.csv"; processed = 0
    qualitative: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]] = {"improved": [], "worsened": [], "both_well": [], "both_poor": []}
    for _, record in frame.iterrows():
        patient = str(record.patient_id)
        try:
            original = load_dicom_as_pil(record._source_path); width, height = original.size; lesion = union_box_mask(boxes[patient], height, width)
        except Exception as error:
            _append_csv(failures_path, {"patient_id": patient, "model": "both", "reason": str(error)}, ("patient_id", "model", "reason")); continue
        probability_mask = None; current: dict[str, tuple[dict[str, float], np.ndarray]] = {}
        for name, (model, config, mode, transform, layer) in states.items():
            if (patient, name) in completed: continue
            try:
                if mode is InputMode.HARD_MASKED:
                    if probability_mask is None: probability_mask = segmenter.predict_proba(original)  # exactly once per case
                    prepared = prepare_classifier_image(original, mode, probability_mask=probability_mask, threshold=float(config["mask_threshold"]), crop_padding=int(config["lung_crop_padding"]), output_size=int(config["classifier_image_size"]))
                else: prepared = prepare_classifier_image(original, mode, output_size=int(config["classifier_image_size"]))
                tensor = transform(prepared.convert("RGB")).unsqueeze(0).to(device)
                with GradCAM(model, layer) as gradcam: probability, heatmap = gradcam(tensor, (height, width))
                values = localization_metrics(heatmap, lesion); predicted = int(probability >= .5)
                current[name] = (values, heatmap)
                row = {"patient_id": patient, "model": name, "input_mode": mode.value, "probability": probability, "predicted_class": predicted, "correct": int(predicted == 1), **values, "bounding_box_count": len(boxes[patient]), "lesion_area_pixels": int(lesion.sum()), "lesion_area_ratio": float(lesion.mean()), "image_width": width, "image_height": height}
                _append_csv(metrics_path, row, CASE_COLUMNS); completed.add((patient, name)); processed += 1
            except Exception as error: _append_csv(failures_path, {"patient_id": patient, "model": name, "reason": str(error)}, ("patient_id", "model", "reason"))
        if set(current) == {"original", "hard_masked"}:
            original_values, original_heatmap = current["original"]; masked_values, masked_heatmap = current["hard_masked"]
            delta = masked_values["heatmap_iou_0_5"] - original_values["heatmap_iou_0_5"]
            group = "improved" if delta > .05 else "worsened" if delta < -.05 else "both_well" if original_values["pointing_game_hit"] and masked_values["pointing_game_hit"] else "both_poor"
            if len(qualitative[group]) < 3:
                display_size = (512, max(1, round(height * 512 / width)))
                display_image = np.asarray(original.convert("L").resize(display_size, Image.Resampling.BILINEAR))
                display_lesion = np.asarray(Image.fromarray(lesion.astype(np.uint8)).resize(display_size, Image.Resampling.NEAREST), dtype=bool)
                display_original = np.asarray(Image.fromarray(original_heatmap).resize(display_size, Image.Resampling.BILINEAR))
                display_masked = np.asarray(Image.fromarray(masked_heatmap).resize(display_size, Image.Resampling.BILINEAR))
                qualitative[group].append((display_image, display_lesion, display_original, display_masked, f"{patient}: {group}"))
        if processed and processed % 100 == 0: print(f"Saved {processed} case-model results", flush=True)
    cases = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame(columns=CASE_COLUMNS)
    if cases.duplicated(["patient_id", "model"]).any(): raise ValueError("Duplicate patient-model rows detected")
    paired = paired_frame(cases)
    paired_rows = []
    for _, row in paired.iterrows():
        for metric in METRICS:
            original_value, masked_value = row[f"{metric}_original"], row[f"{metric}_hard_masked"]
            paired_rows.append({"patient_id": row.patient_id, "metric": metric, "original": original_value, "hard_masked": masked_value, "difference_masked_minus_original": masked_value - original_value})
    pd.DataFrame(paired_rows, columns=("patient_id", "metric", "original", "hard_masked", "difference_masked_minus_original")).to_csv(output_directory / "rsna_gradcam_paired_comparison.csv", index=False)
    boot = bootstrap_comparison(paired); mcnemar = exact_mcnemar(paired.pointing_game_hit_original.to_numpy(), paired.pointing_game_hit_hard_masked.to_numpy()) if len(paired) else {}
    boot.to_csv(output_directory / "rsna_gradcam_bootstrap_comparison.csv", index=False)
    summary = cases.groupby("model")[list(METRICS) + ["probability", "correct"]].agg(["mean", "count"]); summary.to_csv(output_directory / "rsna_gradcam_model_summary.csv")
    report = {"population": "RSNA pneumonia-positive images with bounding boxes only", "completed_case_model_pairs": len(cases), "paired_patients": len(paired), "model_metric_means": {model: {metric: float(group[metric].mean()) for metric in METRICS} for model, group in cases.groupby("model")}, "bootstrap": boot.to_dict(orient="records"), "mcnemar_pointing_game": mcnemar}
    (output_directory / "rsna_gradcam_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plots_and_qualitative(output_directory, cases, paired); _save_qualitative_panel(output_directory, qualitative)
    metadata = _metadata_base(**identity, original_checkpoint_path=str(original_checkpoint.resolve()), hard_masked_checkpoint_path=str(hard_masked_checkpoint.resolve()), segmentation_checkpoint_path=str(segmentation_checkpoint.resolve()), model_configurations={n: c for n, (_, c, _, _, _) in states.items()}, gradcam_target_layer={n: type(layer).__name__ for n, (_, _, _, _, layer) in states.items()}, device=device, batch_size=batch_size, num_workers=num_workers, completed_case_count=len(cases), failure_count=sum(1 for _ in open(failures_path, encoding="utf-8")) - 1 if failures_path.exists() else 0, mcnemar=mcnemar, localization_population="RSNA pneumonia-positive images with bounding boxes only", start_time=started.isoformat(), end_time=datetime.now(timezone.utc).isoformat(), python_version=platform.python_version(), pytorch_version=torch.__version__, pydicom_version=pydicom.__version__, git_commit=_git_commit())
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8"); create_archive(output_directory)
    return cases


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("manifest", "bounding_boxes", "original_checkpoint", "hard_masked_checkpoint", "segmentation_checkpoint", "image_root", "output_directory"): p.add_argument("--" + arg.replace("_", "-"), required=True, type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--batch-size", type=int, default=1); p.add_argument("--num-workers", type=int, default=0); p.add_argument("--max-samples", type=int); p.add_argument("--seed", type=int, default=42); p.add_argument("--overwrite", action="store_true"); p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__": evaluate_rsna_gradcam_localization(**vars(parse_args()))
