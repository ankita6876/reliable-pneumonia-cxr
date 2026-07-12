"""Reproducible, privacy-safe metadata for training experiments."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any
from uuid import uuid4

import torch
import yaml


def create_run_directory(output_dir: Path | str, model_name: str) -> tuple[Path, str, str]:
    """Create a unique ``model_timestamp_runid`` directory inside ``output_dir``."""
    output_path = Path(output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    while True:
        run_id = uuid4().hex[:8]
        run_directory = output_path / f"{model_name}_{timestamp}_{run_id}"
        try:
            run_directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return run_directory, run_id, timestamp


def write_resolved_config(
    run_directory: Path | str, config: dict[str, object], dry_run: bool
) -> None:
    """Write the effective configuration without adding runtime filesystem paths."""
    resolved_config = _redact_absolute_paths(deepcopy(config))
    model_config = resolved_config.get("model")
    training_config = resolved_config.get("training")
    if not isinstance(model_config, dict) or not isinstance(training_config, dict):
        raise ValueError("Configuration must contain model and training mappings.")
    resolved_config["dry_run"] = dry_run
    if dry_run:
        model_config["pretrained"] = False
        training_config["epochs"] = 1
        training_config["batch_size"] = min(int(training_config["batch_size"]), 2)
    with (Path(run_directory) / "config.yaml").open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(resolved_config, config_file, sort_keys=False)


def _redact_absolute_paths(value: object) -> object:
    """Replace absolute paths before a configuration becomes a research artifact."""
    if isinstance(value, dict):
        return {key: _redact_absolute_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_absolute_paths(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return "<redacted-absolute-path>"
    return value


def write_run_manifest(
    run_directory: Path | str,
    *,
    run_id: str,
    timestamp: str,
    model_name: str,
    pretrained: bool,
    uncertain_label_strategy: str,
    seed: int,
    image_size: int,
    batch_size: int,
    learning_rate: float,
    epoch_count: int,
    device: torch.device,
    split_manifest_path: Path | str,
    train_sample_count: int,
    validation_sample_count: int,
    dry_run: bool,
    amp_enabled: bool,
    pos_weight: float | None,
) -> dict[str, Any]:
    """Write a run manifest that excludes dataset and absolute filesystem paths."""
    cuda_available = torch.cuda.is_available()
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "utc_timestamp": timestamp,
        "model_name": model_name,
        "pretrained": pretrained,
        "uncertain_label_strategy": uncertain_label_strategy,
        "random_seed": seed,
        "image_size": image_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epoch_count": epoch_count,
        "device": str(device),
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "split_manifest_filename": Path(split_manifest_path).name,
        "train_sample_count": train_sample_count,
        "validation_sample_count": validation_sample_count,
        "dry_run": dry_run,
        "amp_enabled": amp_enabled,
        "pos_weight": pos_weight,
    }
    if cuda_available:
        metadata["cuda_device_name"] = torch.cuda.get_device_name(device)
    metadata.update(_git_metadata())
    with (Path(run_directory) / "run_manifest.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(metadata, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
    return metadata


def _git_metadata() -> dict[str, str | bool | None]:
    """Return repository revision state when Git metadata is available."""
    repository_root = Path(__file__).resolve().parents[3]
    commit_hash = _run_git_command(repository_root, "rev-parse", "HEAD")
    dirty_status = _run_git_command(repository_root, "status", "--porcelain")
    return {
        "git_commit_hash": commit_hash or None,
        "git_dirty_working_tree": None if dirty_status is None else bool(dirty_status),
    }


def _run_git_command(repository_root: Path, *args: str) -> str | None:
    """Run a read-only Git command without requiring global safe-directory changes."""
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository_root.as_posix()}",
                "-C",
                str(repository_root),
                *args,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None
