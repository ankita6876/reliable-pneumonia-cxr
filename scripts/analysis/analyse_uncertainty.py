"""Run guarded MC-dropout uncertainty analysis for the hard-masked classifier.

This script deliberately refuses to fabricate MC-dropout uncertainty when the
loaded checkpoint has no active learned dropout.  It never changes checkpoint
weights or inserts layers into the architecture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.classification.segmentation_guided import InputMode  # noqa: E402
from pneumonia_ai.models.factory import create_model  # noqa: E402
from scripts.analysis.uncertainty_utils import active_dropout_modules, dropout_modules  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT.parent / "outputs" / "classification_ablation"
DEFAULT_CHECKPOINT = OUTPUT_ROOT / "hard_masked" / "best_checkpoint.pt"
DEFAULT_OUTPUT_DIRECTORY = OUTPUT_ROOT / "uncertainty" / "hard_masked"


def parse_args() -> argparse.Namespace:
    """Parse analysis options; MC settings are retained for compatible checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--mc-passes", type=int, default=30, help="Number of stochastic forward passes (default: 30).")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_checkpoint_model(checkpoint: Path) -> tuple[torch.nn.Module, dict[str, object]]:
    """Recreate the exact saved architecture and load its state strictly."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Classifier checkpoint does not exist: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    configuration = state.get("configuration") if isinstance(state, dict) else None
    if not isinstance(configuration, dict) or not isinstance(state.get("model_state_dict"), dict):
        raise ValueError("Classifier checkpoint lacks configuration or model_state_dict.")
    if configuration.get("input_mode") != InputMode.HARD_MASKED.value:
        raise ValueError("MC-dropout analysis supports only the hard_masked classifier checkpoint.")
    model = create_model(str(configuration.get("model", "densenet121")), pretrained=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model, configuration


def write_unavailable_report(
    output_directory: Path,
    checkpoint: Path,
    configuration: dict[str, object],
    active_modules: list[tuple[str, float]],
    all_modules: list[tuple[str, float]],
) -> Path:
    """Write an auditable report explaining why genuine MC dropout cannot run."""
    output_directory.mkdir(parents=True, exist_ok=True)
    report = output_directory / "mc_dropout_unavailable_report.md"
    declared = ", ".join(f"`{name}` (p={probability:g})" for name, probability in all_modules) or "none"
    report.write_text(
        "# MC Dropout unavailable\n\n"
        f"Checkpoint: `{checkpoint}`\n\n"
        f"Architecture: `{configuration.get('model')}`; input mode: `{configuration.get('input_mode')}`.\n\n"
        "The reconstructed checkpoint contains no active dropout module (a module with p > 0), so repeated "
        "forward passes would be deterministic. The architecture was not modified and no dropout was inserted; "
        "therefore genuine Monte Carlo Dropout uncertainty estimates, statistical tests, selective-prediction "
        "outputs, and figures were not generated.\n\n"
        f"All explicit dropout modules: {declared}.\n\n"
        "Active dropout modules (p > 0): none.\n",
        encoding="utf-8",
    )
    (output_directory / "mc_dropout_architecture_inspection.json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "configuration": configuration, "dropout_modules": all_modules,
        "active_dropout_modules": active_modules, "mc_dropout_available": bool(active_modules),
    }, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """Inspect the checkpoint and stop safely when MC dropout is unavailable."""
    args = parse_args()
    if args.mc_passes < 2 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("mc-passes must be at least 2; batch-size must be positive; num-workers non-negative.")
    model, configuration = load_checkpoint_model(args.classifier_checkpoint)
    modules = active_dropout_modules(model)
    all_modules = dropout_modules(model)
    if not modules:
        report = write_unavailable_report(
            args.output_directory, args.classifier_checkpoint, configuration, modules, all_modules
        )
        raise SystemExit(f"Genuine MC Dropout is unavailable for this checkpoint. Report: {report}")
    raise NotImplementedError("Compatible active-dropout checkpoint detected; implement the MC execution workflow before use.")


if __name__ == "__main__":
    main()
