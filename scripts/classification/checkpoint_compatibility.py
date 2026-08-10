"""Compatibility helpers for historical classification-ablation checkpoints."""

from __future__ import annotations

import hashlib
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

# These fields are written by the optimisation runner.  A legacy A4 checkpoint
# predates ``input_mode``, but its training configuration is otherwise intact.
_OPTIMISATION_SCHEMA_MARKERS = frozenset({
    "experiment", "backbone", "pretrained", "input_size", "loss", "optimizer",
})
_NESTED_ORIGINAL_MODEL_MARKERS = frozenset({"name", "image_size", "preprocessing"})
_VERIFIED_OPTIMISATION_LABEL_POLICY = "ignore"


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


def classify_checkpoint_configuration(configuration: Mapping[str, Any]) -> str:
    """Classify a supported embedded checkpoint-configuration schema.

    Explicit ``input_mode`` is authoritative for modern optimisation checkpoints.
    Earlier optimisation checkpoints are identified by their independent training
    metadata, while old baseline checkpoints have a nested model configuration.
    Unknown schemas must not be guessed to be original-image checkpoints.
    """
    if "input_mode" in configuration:
        return "optimisation"
    present_optimisation_markers = _OPTIMISATION_SCHEMA_MARKERS.intersection(configuration)
    if len(present_optimisation_markers) >= 3 and {
        "loss", "optimizer"
    }.intersection(present_optimisation_markers):
        return "legacy_optimisation"
    model = configuration.get("model")
    if isinstance(model, Mapping) and _NESTED_ORIGINAL_MODEL_MARKERS.intersection(model):
        return "nested_original_baseline"
    keys = ", ".join(sorted(map(str, configuration))) or "(none)"
    model_keys = (
        "; nested model keys: " + ", ".join(sorted(map(str, model)))
        if isinstance(model, Mapping) else ""
    )
    raise ValueError(
        "Ambiguous checkpoint configuration schema; cannot infer input treatment. "
        f"Detected keys: {keys}{model_keys}."
    )


def normalize_nested_original_baseline_configuration(
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Adapt the known nested original-image baseline schema for evaluation."""
    model = configuration.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Nested original baseline checkpoint lacks a model mapping.")
    return {
        "input_mode": InputMode.ORIGINAL.value,
        "classifier_image_size": int(model.get("image_size", 224)),
        "mask_threshold": 0.5,
        "lung_crop_padding": 0,
        "preprocessing": model.get("preprocessing", "imagenet"),
        "backbone": model.get("name", "densenet121"),
    }


def resolve_methodological_metadata(
    configuration: Mapping[str, Any], *, supplied_splits_csv: Path,
) -> dict[str, object]:
    """Resolve audit-comparability metadata against the supplied manifest.

    Checkpoint locations are machine-specific, so the effective split identity is
    the SHA-256 of the manifest explicitly selected for this audit.  A recorded
    path is checked when it can be inspected, but is never compared as an
    absolute path.
    """
    schema = classify_checkpoint_configuration(configuration)
    supplied = Path(supplied_splits_csv).resolve()
    if not supplied.is_file():
        raise FileNotFoundError(f"Supplied split manifest does not exist: {supplied}")
    supplied_hash = _file_sha256(supplied)
    saved_path = configuration.get("dataset_split_path")
    if saved_path is not None:
        if not isinstance(saved_path, (str, Path)):
            raise ValueError("Checkpoint dataset_split_path must be a path string.")
        recorded = Path(saved_path)
        recorded_name = _portable_path_name(saved_path)
        if recorded_name != supplied.name:
            raise ValueError(
                "Checkpoint dataset_split_path filename does not match the supplied "
                f"manifest: {recorded_name!r} != {supplied.name!r}."
            )
        if recorded.is_file() and _file_sha256(recorded) != supplied_hash:
            raise ValueError(
                "Checkpoint dataset_split_path content does not match the supplied manifest."
            )
    label_policy = configuration.get("label_policy")
    if label_policy is None:
        if schema not in {"optimisation", "legacy_optimisation"}:
            raise ValueError(
                "Checkpoint lacks label_policy and is not a recognised optimisation schema."
            )
        label_policy = _VERIFIED_OPTIMISATION_LABEL_POLICY
    if not isinstance(label_policy, str):
        raise ValueError("Checkpoint label_policy must be a string.")
    resolved = dict(configuration)
    resolved["dataset_split_path"] = f"sha256:{supplied_hash}"
    resolved["label_policy"] = label_policy
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path_name(value: str | Path) -> str:
    """Extract a filename from either Windows or POSIX checkpoint metadata."""
    return str(value).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


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

    # This Phase-5 setting is meaningful only for soft-masked classifiers.
    # Do not alter the normalized shape of historical original, hard-masked,
    # or lung-crop checkpoint configurations.
    if normalized.get("input_mode") == InputMode.SOFT_MASKED.value and "soft_mask_outside_factor" not in normalized:
        normalized["soft_mask_outside_factor"] = 0.20
        print("WARNING: Checkpoint configuration lacks 'soft_mask_outside_factor'; using 0.2 from verified Phase-5 default.")

    errors: list[str] = []
    try:
        InputMode(str(normalized["input_mode"]))
    except ValueError:
        errors.append("input_mode (expected original, hard_masked, soft_masked, or lung_crop)")
    if not _is_positive_integer(normalized["classifier_image_size"]):
        errors.append("classifier_image_size (expected a positive integer)")
    if not _is_valid_mask_threshold(normalized["mask_threshold"]):
        errors.append("mask_threshold (expected a number strictly between 0 and 1)")
    if not _is_non_negative_integer(normalized["lung_crop_padding"]):
        errors.append("lung_crop_padding (expected a non-negative integer)")
    if (
        normalized.get("input_mode") == InputMode.SOFT_MASKED.value
        and not _is_unit_interval(normalized.get("soft_mask_outside_factor"))
    ):
        errors.append("soft_mask_outside_factor (expected a number between 0 and 1 inclusive)")
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


def _is_unit_interval(value: object) -> bool:
    try:
        return 0.0 <= float(value) <= 1.0
    except (TypeError, ValueError, OverflowError):
        return False
