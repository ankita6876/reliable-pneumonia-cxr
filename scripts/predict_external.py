"""Frozen external inference for CheXpert-trained A4 classifiers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image
import pydicom
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pneumonia_ai.classification.segmentation_guided import InputMode, prepare_classifier_image  # noqa: E402
from pneumonia_ai.models.factory import create_model  # noqa: E402
from pneumonia_ai.segmentation.cache import MaskCache  # noqa: E402
from pneumonia_ai.segmentation.inference import FrozenLungSegmenter  # noqa: E402
from scripts.classification.checkpoint_compatibility import (  # noqa: E402
    classify_checkpoint_configuration, load_adjacent_experiment_configuration,
    normalize_checkpoint_configuration, normalize_nested_original_baseline_configuration,
)
from scripts.train_baseline import _transforms  # noqa: E402

MANIFEST_COLUMNS = ("patient_id", "image_path", "binary_target", "has_bounding_box", "bounding_box_count", "split", "dataset")
OUTPUT_COLUMNS = ("patient_id", "case_id", "image_path", "projection", "binary_target", "probability", "predicted_class", "logit", "split", "dataset", "model_name", "run_id", "input_mode", "has_bounding_box", "bounding_box_count")


def _first_valid_number(value: object) -> float | None:
    values = value if isinstance(value, (list, tuple)) or hasattr(value, "__iter__") and not isinstance(value, (str, bytes)) else [value]
    for item in values:
        try:
            number = float(item)
            if np.isfinite(number): return number
        except (TypeError, ValueError):
            pass
    return None


def load_dicom_as_pil(path: Path) -> Image.Image:
    """Decode a grayscale DICOM to RGB without writing an intermediate image."""
    try:
        dataset = pydicom.dcmread(path)
        pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
    except Exception as error:
        raise ValueError(f"Unable to read DICOM {path}: {error}") from error
    if pixels.ndim != 2 or pixels.size == 0:
        raise ValueError(f"Unsupported DICOM pixel data in {path}: expected one non-empty 2D image, got shape {pixels.shape}")
    slope = _first_valid_number(getattr(dataset, "RescaleSlope", None)); intercept = _first_valid_number(getattr(dataset, "RescaleIntercept", None))
    pixels = pixels * (1.0 if slope is None else slope) + (0.0 if intercept is None else intercept)
    centre = _first_valid_number(getattr(dataset, "WindowCenter", None)); width = _first_valid_number(getattr(dataset, "WindowWidth", None))
    if centre is not None and width is not None and width > 1.0:
        lower, upper = centre - 0.5 - (width - 1.0) / 2.0, centre - 0.5 + (width - 1.0) / 2.0
    else:
        lower, upper = np.nanpercentile(pixels, [0.5, 99.5])
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            lower, upper = float(np.nanmin(pixels)), float(np.nanmax(pixels))
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(f"DICOM {path} has non-finite pixel data")
    if upper <= lower:  # Constant images are valid; their neutral black representation is deterministic.
        normalized = np.zeros_like(pixels, dtype=np.float32)
    else:
        normalized = np.clip((pixels - lower) / (upper - lower), 0.0, 1.0)
    interpretation = str(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2")).upper()
    if interpretation == "MONOCHROME1": normalized = 1.0 - normalized
    elif interpretation != "MONOCHROME2": raise ValueError(f"Unsupported DICOM photometric interpretation {interpretation!r}: {path}")
    gray = np.rint(normalized * 255.0).astype(np.uint8)
    return Image.fromarray(gray, mode="L").convert("RGB")


def load_external_image_as_pil(path: Path) -> Image.Image:
    """Load an external image without changing the established DICOM path.

    Raster images are decoded by Pillow and converted to RGB; a 16-bit grayscale
    raster is mapped from its native [0, 65535] range to the existing 8-bit RGB
    contract before the frozen classifier transform. DICOM images retain the
    historical RSNA windowing and MONOCHROME handling above.
    """

    if path.suffix.lower() == ".dcm":
        return load_dicom_as_pil(path)
    try:
        with Image.open(path) as source:
            if source.mode.startswith("I;16"):
                pixels = np.asarray(source, dtype=np.uint16)
                # The frozen validation transform receives an 8-bit RGB PIL image
                # and applies ToTensor() (/255). Scale the complete native 16-bit
                # range explicitly rather than allowing Pillow to clip values >255.
                grayscale = np.rint(pixels.astype(np.float32) * (255.0 / 65535.0)).astype(np.uint8)
                return Image.fromarray(grayscale, mode="L").convert("RGB")
            return source.convert("RGB")
    except (OSError, ValueError) as error:
        raise ValueError(f"Unable to read raster image {path}: {error}") from error


def resolve_image_path(value: str | Path, image_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else image_root / path


def validate_manifest(manifest: Path, image_root: Path) -> pd.DataFrame:
    if not manifest.is_file(): raise FileNotFoundError(f"Manifest does not exist: {manifest}")
    frame = pd.read_csv(manifest)
    missing = [column for column in MANIFEST_COLUMNS if column not in frame.columns]
    if missing: raise ValueError("Manifest missing required columns: " + ", ".join(missing))
    if frame.patient_id.isna().any(): raise ValueError("Manifest patient_id values must be non-missing")
    if frame.image_path.isna().any() or frame.image_path.astype(str).duplicated().any(): raise ValueError("Manifest image_path values must be unique and non-missing")
    labels = pd.to_numeric(frame.binary_target, errors="raise")
    if not labels.isin([0, 1]).all(): raise ValueError("Manifest binary_target must contain only 0 and 1")
    frame = frame.copy(); frame["binary_target"] = labels.astype(int); frame["_source_path"] = frame.image_path.map(lambda value: resolve_image_path(value, image_root))
    absent = [str(path) for path in frame["_source_path"] if not path.is_file()]
    if absent: raise FileNotFoundError(f"Manifest references {len(absent)} missing image paths; first: {absent[0]}")
    if frame.binary_target.nunique() != 2: raise ValueError("External manifest must contain both binary target classes")
    return frame


def deterministic_sample(frame: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    if max_samples is None: return frame.copy()
    if max_samples < 1 or max_samples > len(frame): raise ValueError("max_samples must be between 1 and the manifest row count")
    if max_samples == len(frame): return frame.copy()
    rng = np.random.default_rng(seed); groups = {label: group.index.to_numpy() for label, group in frame.groupby("binary_target", sort=True)}
    if max_samples >= 2 and len(groups) == 2:
        n0 = max(1, min(len(groups[0]), round(max_samples * len(groups[0]) / len(frame))))
        n1 = max_samples - n0
        if n1 < 1: n1 = 1; n0 = max_samples - 1
        n0 = min(n0, len(groups[0])); n1 = min(n1, len(groups[1]))
        remaining = max_samples - n0 - n1
        for label in (0, 1):
            add = min(remaining, len(groups[label]) - (n0 if label == 0 else n1))
            if label == 0: n0 += add
            else: n1 += add
            remaining -= add
        selected = np.concatenate([rng.choice(groups[0], n0, replace=False), rng.choice(groups[1], n1, replace=False)])
    else: selected = rng.choice(frame.index.to_numpy(), max_samples, replace=False)
    return frame.loc[np.sort(selected)].copy()


def _sha256(path: Path | None) -> str | None:
    if path is None: return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError): return None


def _classifier_configuration(checkpoint: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    raw = state.get("configuration")
    if not isinstance(raw, Mapping): raise ValueError("Classifier checkpoint lacks a mapping configuration")
    schema = classify_checkpoint_configuration(raw)
    if schema == "nested_original_baseline": config = normalize_nested_original_baseline_configuration(raw)
    else: config = normalize_checkpoint_configuration(raw, load_adjacent_experiment_configuration(checkpoint))
    backbone = config.get("backbone", config.get("model"))
    if not isinstance(backbone, str): raise ValueError("Checkpoint configuration lacks a usable backbone")
    config["backbone"] = backbone
    config["preprocessing"] = str(config.get("preprocessing", "imagenet"))
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path); parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=("cpu", "cuda"))
    parser.add_argument("--segmentation-checkpoint", type=Path); parser.add_argument("--expected-segmentation-sha256")
    parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def predict_external(*, manifest: Path, image_root: Path, checkpoint: Path, output: Path, device: str = "cpu", segmentation_checkpoint: Path | None = None, expected_segmentation_sha256: str | None = None, batch_size: int = 16, num_workers: int = 0, max_samples: int | None = None, seed: int = 42, overwrite: bool = False) -> pd.DataFrame:
    if device not in {"cpu", "cuda"}: raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available(): raise ValueError("CUDA requested but is not available")
    if batch_size < 1 or num_workers < 0: raise ValueError("batch_size must be positive and num_workers non-negative")
    if not checkpoint.is_file(): raise FileNotFoundError(f"Classifier checkpoint does not exist: {checkpoint}")
    if output.exists() and not overwrite: raise FileExistsError(f"Output already exists (use --overwrite): {output}")
    frame = deterministic_sample(validate_manifest(manifest, image_root), max_samples, seed)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or not isinstance(state.get("model_state_dict"), Mapping): raise ValueError("Classifier checkpoint lacks model_state_dict")
    config = _classifier_configuration(checkpoint, state); mode = InputMode(config["input_mode"])
    if mode is not InputMode.ORIGINAL and segmentation_checkpoint is None: raise ValueError("--segmentation-checkpoint is required for non-original input modes")
    if segmentation_checkpoint is not None and not segmentation_checkpoint.is_file(): raise FileNotFoundError(f"Segmentation checkpoint does not exist: {segmentation_checkpoint}")
    segmentation_sha256 = _sha256(segmentation_checkpoint)
    if expected_segmentation_sha256 and segmentation_sha256 != expected_segmentation_sha256.lower():
        raise ValueError("Segmentation checkpoint SHA-256 does not match --expected-segmentation-sha256")
    model = create_model(config["backbone"], pretrained=False); model.load_state_dict(state["model_state_dict"], strict=True); model.to(device).eval()
    size, threshold, padding = int(config["classifier_image_size"]), float(config["mask_threshold"]), int(config["lung_crop_padding"])
    if mode is not InputMode.ORIGINAL and threshold != 0.5:
        raise ValueError(f"Historical hard-mask threshold must be 0.5; checkpoint specifies {threshold}")
    _, transform = _transforms(size, config["preprocessing"])
    segmenter = FrozenLungSegmenter(segmentation_checkpoint, device) if segmentation_checkpoint else None
    cache = None
    if segmenter:
        cache = MaskCache(output.parent / "mask_cache")
        cache.validate_or_initialise_metadata({"schema_version": 1, "segmentation_checkpoint_sha256": segmentation_sha256, "segmentation_input_size": segmenter.image_size, "mask_threshold": threshold, "postprocessing": "none"})
    started = datetime.now(timezone.utc); rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for offset in range(0, len(frame), batch_size):
            batch = frame.iloc[offset:offset + batch_size]; images = []
            for _, record in batch.iterrows():
                source = record["_source_path"]; image = load_external_image_as_pil(source)
                if segmenter and cache:
                    key = cache.key(source, segmentation_checkpoint, threshold, segmenter.image_size); probability = cache.get(key)
                    if probability is None: probability = segmenter.predict_proba(image); cache.set(key, probability, source_path=source)
                    image = prepare_classifier_image(image, mode, probability_mask=probability, threshold=threshold, crop_padding=padding, output_size=size)
                else: image = prepare_classifier_image(image, mode, output_size=size)
                images.append(transform(image.convert("RGB")))
            logits = model(torch.stack(images).to(device)).view(-1).detach().cpu(); probabilities = torch.sigmoid(logits)
            for (_, record), logit, probability in zip(batch.iterrows(), logits, probabilities):
                rows.append({"patient_id": str(record["patient_id"]), "case_id": str(record.get("case_id", record["image_path"])), "image_path": str(record["image_path"]), "projection": str(record.get("projection", "")), "binary_target": int(record["binary_target"]), "probability": float(probability), "predicted_class": int(probability >= .5), "logit": float(logit), "split": record["split"], "dataset": record["dataset"], "model_name": config["backbone"], "run_id": str(checkpoint), "input_mode": mode.value, "has_bounding_box": record["has_bounding_box"], "bounding_box_count": record["bounding_box_count"]})
            processed = offset + len(batch)
            if processed == len(frame) or processed % 500 == 0: print(f"Processed {processed}/{len(frame)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True); predictions = pd.DataFrame(rows, columns=OUTPUT_COLUMNS); predictions.to_csv(output, index=False)
    metadata = {"manifest_path": str(manifest.resolve()), "image_root": str(image_root.resolve()), "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": _sha256(checkpoint), "segmentation_checkpoint_path": str(segmentation_checkpoint.resolve()) if segmentation_checkpoint else None, "segmentation_checkpoint_sha256": segmentation_sha256, "expected_segmentation_sha256": expected_segmentation_sha256, "input_mode": mode.value, "backbone": config["backbone"], "preprocessing": config["preprocessing"], "classifier_image_size": size, "mask_threshold": threshold, "lung_crop_padding": padding, "device": device, "batch_size": batch_size, "num_workers": num_workers, "seed": seed, "max_samples": max_samples, "selected_patient_ids": frame.patient_id.astype(str).tolist(), "selected_case_ids": predictions.case_id.astype(str).tolist(), "row_count": len(predictions), "positive_count": int(predictions.binary_target.sum()), "negative_count": int((predictions.binary_target == 0).sum()), "start_time": started.isoformat(), "end_time": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version(), "pytorch_version": torch.__version__, "pydicom_version": pydicom.__version__, "git_commit": _git_commit()}
    output.with_name(output.stem + "_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return predictions


def main(argv: list[str] | None = None) -> None:
    predict_external(**vars(parse_args(argv)))

if __name__ == "__main__": main()
