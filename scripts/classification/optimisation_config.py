"""Configuration parsing for controlled classification optimisation experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class OptimisationConfig:
    experiment: str
    input_mode: str = "hard_masked"
    mask_threshold: float = 0.5
    soft_mask_outside_factor: float = 0.20
    context_dilation_radius: int = 12
    context_feather_radius: int = 8
    context_background_factor: float = 0.20
    seed: int = 42
    backbone: str = "densenet121"
    pretrained: bool = False
    preprocessing: str | None = None
    input_size: int = 224
    augmentation: str = "historical"
    horizontal_flip: bool = True
    rotation_degrees: int = 7
    loss: str = "weighted_bce"
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-5
    head_learning_rate: float = 1e-4
    weight_decay: float = 0.0
    scheduler: str = "plateau"
    batch_size: int = 8
    gradient_accumulation: int = 1
    epochs: int = 20
    early_stopping_patience: int = 5
    frozen_head_warmup_epochs: int = 0
    progressive_unfreezing: bool = False
    num_workers: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_preprocessing(config: OptimisationConfig) -> str:
    """Return the only classifier preprocessing compatible with ``config.backbone``."""
    preprocessing = config.preprocessing or (
        "torchxrayvision" if config.backbone == "xrv_densenet121_all" else "imagenet"
    )
    if preprocessing not in {"imagenet", "torchxrayvision"}:
        raise ValueError("preprocessing must be imagenet or torchxrayvision.")
    if (config.backbone == "xrv_densenet121_all") != (
        preprocessing == "torchxrayvision"
    ):
        raise ValueError(
            "xrv_densenet121_all requires torchxrayvision preprocessing; all other "
            "supported optimisation backbones require imagenet preprocessing."
        )
    return preprocessing


def load_config(path: Path) -> OptimisationConfig:
    """Load and validate a compact YAML optimisation configuration."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Optimisation YAML must contain a mapping.")
    config = OptimisationConfig(**payload)
    preprocessing = resolve_preprocessing(config)
    if config.preprocessing is None:
        config_values = config.to_dict()
        config_values["preprocessing"] = preprocessing
        config = OptimisationConfig(**config_values)
    if config.augmentation not in {"none", "historical"}:
        raise ValueError("augmentation must be none or historical.")
    if config.input_mode not in {"original", "hard_masked", "soft_masked", "context_preserving"}:
        raise ValueError(
            "input_mode must be original, hard_masked, soft_masked, "
            "or context_preserving."
        )
    if isinstance(config.mask_threshold, bool) or float(config.mask_threshold) != 0.5:
        raise ValueError("mask_threshold must be the frozen value 0.5.")
    if isinstance(config.soft_mask_outside_factor, bool) or not _is_unit_interval(config.soft_mask_outside_factor):
        raise ValueError("soft_mask_outside_factor must be between 0 and 1 inclusive.")
    if isinstance(config.context_dilation_radius, bool) or not isinstance(config.context_dilation_radius, int) or config.context_dilation_radius < 0:
        raise ValueError("context_dilation_radius must be a non-negative integer.")
    if isinstance(config.context_feather_radius, bool) or not isinstance(config.context_feather_radius, int) or config.context_feather_radius < 0:
        raise ValueError("context_feather_radius must be a non-negative integer.")
    if isinstance(config.context_background_factor, bool) or not _is_unit_interval(config.context_background_factor):
        raise ValueError("context_background_factor must be between 0 and 1 inclusive.")
    if config.input_mode == "context_preserving":
        if config.context_dilation_radius not in {12, 20}:
            raise ValueError(
                "context_dilation_radius must be one of the predefined Phase-7 values: 12 or 20."
            )
        if config.context_feather_radius != 8:
            raise ValueError(
                "context_feather_radius must be the predefined Phase-7 value 8."
            )
        if float(config.context_background_factor) != 0.20:
            raise ValueError(
                "context_background_factor must be the predefined Phase-7 value 0.20."
            )
    if config.rotation_degrees < 0:
        raise ValueError("rotation_degrees must be non-negative.")
    if config.augmentation == "none" and (config.horizontal_flip or config.rotation_degrees):
        raise ValueError("augmentation=none requires horizontal_flip=false and rotation_degrees=0.")
    if config.loss not in {"bce", "weighted_bce", "focal", "asymmetric_focal"}:
        raise ValueError("Unsupported loss.")
    if config.scheduler not in {"plateau", "cosine", "none"}:
        raise ValueError("scheduler must be plateau, cosine, or none.")
    if config.epochs < 1 or config.batch_size < 1 or config.gradient_accumulation < 1:
        raise ValueError(
            "epochs, batch_size, and gradient_accumulation must be positive."
        )
    return config


def _is_unit_interval(value: object) -> bool:
    try:
        return 0.0 <= float(value) <= 1.0
    except (TypeError, ValueError, OverflowError):
        return False


def configuration_differences(left: OptimisationConfig, right: OptimisationConfig) -> set[str]:
    """Return resolved differences, excluding the experiment identity."""
    first, second = left.to_dict(), right.to_dict()
    return {key for key in first if key != "experiment" and first[key] != second[key]}


def validate_controlled_a_series(config_directory: Path) -> None:
    """Fail clearly if the repository A0--A5 configs contain a confounded ablation."""
    names = ("A0_reproduce_current.yaml", "A1_pretrained.yaml", "A2_pretrained_longer.yaml", "A3_pretrained_light_aug.yaml", "A4_pretrained_progressive.yaml", "A5_pretrained_progressive_focal.yaml")
    configs = [load_config(config_directory / name) for name in names]
    contracts = (("A0 vs A1", {"pretrained"}), ("A1 vs A2", {"epochs", "early_stopping_patience"}), ("A2 vs A3", {"horizontal_flip"}), ("A3 vs A4", {"learning_rate", "backbone_learning_rate", "head_learning_rate", "weight_decay"}), ("A4 vs A5", {"loss"}))
    for (label, allowed), first, second in zip(contracts, configs, configs[1:]):
        actual = configuration_differences(first, second)
        if actual != allowed:
            raise ValueError(f"Controlled configuration violation for {label}: unexpected differences={sorted(actual - allowed)}; missing required differences={sorted(allowed - actual)}.")
    original = config_directory / "A4_original_control.yaml"
    if original.is_file():
        control = load_config(original)
        actual = configuration_differences(configs[4], control)
        if actual != {"input_mode"}:
            raise ValueError("Controlled configuration violation for A4 vs A4_original_control: "
                             f"unexpected differences={sorted(actual - {'input_mode'})}; "
                             f"missing required differences={sorted({'input_mode'} - actual)}.")


def serialise_config(
    config: OptimisationConfig, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return JSON-safe resolved configuration with optional runtime metadata."""
    result: dict[str, Any] = config.to_dict()
    if extra:
        result.update(extra)
    return result
