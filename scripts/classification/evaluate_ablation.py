"""Evaluate a trained classification ablation on a labelled CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.classification.segmentation_guided import InputMode  # noqa: E402
from pneumonia_ai.data.chexpert_dataset import CheXpertPneumoniaDataset  # noqa: E402
from pneumonia_ai.evaluation.core import calibration_metrics, metrics_at_threshold  # noqa: E402
from pneumonia_ai.models.factory import create_model  # noqa: E402
from pneumonia_ai.segmentation.cache import MaskCache  # noqa: E402
from pneumonia_ai.segmentation.inference import FrozenLungSegmenter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--segmentation-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_ablation(**vars(args))


def evaluate_ablation(
    csv: Path,
    checkpoint: Path,
    output_directory: Path,
    segmentation_checkpoint: Path | None = None,
    device: str = "cpu",
    batch_size: int = 16,
    num_workers: int = 0,
) -> dict[str, float | int]:
    """Write predictions, metrics, and a confusion matrix for one frozen run."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    configuration = state.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("Classifier checkpoint lacks an ablation configuration.")
    configuration = normalize_checkpoint_configuration(
        configuration, _load_adjacent_experiment_configuration(checkpoint)
    )
    mode = InputMode(configuration["input_mode"])
    if mode is not InputMode.ORIGINAL and segmentation_checkpoint is None:
        raise ValueError("Guided ablation evaluation requires --segmentation-checkpoint.")
    size = int(configuration["classifier_image_size"])
    segmenter = FrozenLungSegmenter(segmentation_checkpoint, device) if segmentation_checkpoint else None
    output_directory.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(csv).copy()
    frame["split"] = "validation"
    with tempfile.TemporaryDirectory() as temporary:
        manifest = Path(temporary) / "evaluation.csv"
        frame.to_csv(manifest, index=False)
        transform = transforms.Compose([
            transforms.Resize((size, size)), transforms.Lambda(_to_rgb),
            transforms.ToTensor(), transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        dataset = CheXpertPneumoniaDataset(
            csv.parent.resolve(), manifest, "validation", transform, mode, segmenter,
            MaskCache(output_directory / "mask_cache") if segmenter else None,
            float(configuration["mask_threshold"]), int(configuration["lung_crop_padding"]), size,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        model = create_model("densenet121", pretrained=False)
        model.load_state_dict(state["model_state_dict"])
        model.to(device).eval()
        rows: list[dict[str, object]] = []
        with torch.inference_mode():
            for batch in loader:
                logits = model(batch["image"].to(device)).view(-1).cpu()
                probabilities = torch.sigmoid(logits)
                for index, probability in enumerate(probabilities):
                    if batch["raw_label"][index].item() not in (0.0, 1.0):
                        continue
                    rows.append(_prediction_row(batch, index, probability, logits, checkpoint))
    predictions = pd.DataFrame(rows)
    predictions.to_csv(output_directory / "predictions.csv", index=False)
    if predictions.empty or predictions.binary_target.nunique() != 2:
        raise ValueError("Evaluation requires definite examples from both classes.")
    y, p = predictions.binary_target.to_numpy(), predictions.probability.to_numpy()
    metrics = {
        "auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
        **metrics_at_threshold(y, p, .5), **calibration_metrics(predictions),
    }
    metrics["ece"] = metrics.pop("expected_calibration_error")
    metrics["brier"] = metrics.pop("brier_score")
    (output_directory / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    (output_directory / "confusion_matrix.json").write_text(json.dumps(confusion_matrix(y, p >= .5, labels=[0, 1]).tolist()))
    print(f"Evaluated {len(predictions)} images: {output_directory}")
    return metrics


_LEGACY_EVALUATOR_DEFAULTS: dict[str, object] = {
    "input_mode": InputMode.HARD_MASKED.value,
    "classifier_image_size": 224,
    "mask_threshold": 0.5,
    "lung_crop_padding": 0,
}


def _load_adjacent_experiment_configuration(checkpoint: Path) -> Mapping[str, Any] | None:
    """Load the run's config.json when it is co-located with its checkpoint."""
    path = checkpoint.parent / "config.json"
    if not path.is_file():
        return None
    try:
        configuration = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Adjacent experiment configuration is unreadable: {path}") from error
    if not isinstance(configuration, Mapping):
        raise ValueError(f"Adjacent experiment configuration must be a mapping: {path}")
    experiment = configuration.get("experiment")
    if experiment is not None and str(experiment) != checkpoint.parent.name:
        raise ValueError(
            "Adjacent experiment configuration does not match checkpoint directory: "
            f"{path}"
        )
    return configuration


def normalize_checkpoint_configuration(
    configuration: Mapping[str, Any],
    adjacent_configuration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill the legacy evaluator schema without changing checkpoint metadata."""
    normalized = dict(configuration)
    sources: dict[str, tuple[str, Any]] = {}
    for field in _LEGACY_EVALUATOR_DEFAULTS:
        if field in normalized:
            continue
        if field == "classifier_image_size" and "input_size" in configuration:
            sources[field] = ("checkpoint input_size", configuration["input_size"])
        elif adjacent_configuration is not None and field in adjacent_configuration:
            sources[field] = ("adjacent config.json", adjacent_configuration[field])
        elif (
            field == "classifier_image_size"
            and adjacent_configuration is not None
            and "input_size" in adjacent_configuration
        ):
            sources[field] = ("adjacent config.json input_size", adjacent_configuration["input_size"])
        else:
            sources[field] = ("verified legacy default", _LEGACY_EVALUATOR_DEFAULTS[field])
    for field, (source, value) in sources.items():
        normalized[field] = value
        print(f"WARNING: Checkpoint configuration lacks {field!r}; using {value!r} from {source}.")

    errors: list[str] = []
    try:
        InputMode(str(normalized["input_mode"]))
    except ValueError:
        errors.append("input_mode (expected original, hard_masked, or lung_crop)")
    if not _is_positive_integer(normalized["classifier_image_size"]):
        errors.append("classifier_image_size (expected a positive integer)")
    if not _is_valid_mask_threshold(normalized["mask_threshold"]):
        errors.append("mask_threshold (expected a number strictly between 0 and 1)")
    if not _is_non_negative_integer(normalized["lung_crop_padding"]):
        errors.append("lung_crop_padding (expected a non-negative integer)")
    if errors:
        raise ValueError(
            "Checkpoint configuration has unresolved evaluator-required fields: "
            + "; ".join(errors)
        )
    return normalized


def _is_positive_integer(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        integer = int(value)
        return integer > 0 and float(value) == integer
    except (TypeError, ValueError, OverflowError):
        return False


def _is_valid_mask_threshold(value: object) -> bool:
    try:
        return 0.0 < float(value) < 1.0
    except (TypeError, ValueError):
        return False


def _is_non_negative_integer(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        integer = int(value)
        return integer >= 0 and float(value) == integer
    except (TypeError, ValueError, OverflowError):
        return False


def _to_rgb(image: object) -> object:
    return image.convert("RGB") if hasattr(image, "convert") else image


def _prediction_row(
    batch: dict[str, object], index: int, probability: torch.Tensor,
    logits: torch.Tensor, checkpoint: Path,
) -> dict[str, object]:
    return {
        "patient_id": batch["patient_id"][index], "study_id": batch["study_id"][index],
        "image_path": batch["image_path"][index],
        "original_label": int(batch["raw_label"][index]),
        "binary_target": int(batch["target"][index]), "logit": float(logits[index]),
        "probability": float(probability), "predicted_class": int(probability >= .5),
        "split": "validation", "model_name": "densenet121", "run_id": checkpoint.parent.name,
    }


if __name__ == "__main__":
    main()
