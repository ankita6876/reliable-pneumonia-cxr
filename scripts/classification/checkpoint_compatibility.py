"""Compatibility helpers for historical classification-ablation checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from pneumonia_ai.classification.segmentation_guided import InputMode


_LEGACY_EVALUATOR_DEFAULTS: dict[str, object] = {
    "input_mode": InputMode.HARD_MASKED.value,
    "classifier_image_size": 224,
    "mask_threshold": 0.5,
    "lung_crop_padding": 0,
}


def load_adjacent_experiment_configuration(checkpoint: Path) -> Mapping[str, Any] | None:
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
