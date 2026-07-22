"""Generate portable prediction CSVs for exactly one CheXpert split and saved checkpoint."""
# ruff: noqa: E501, E701, E702
import argparse
import tempfile
from pathlib import Path
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
from pneumonia_ai.data.chexpert_dataset import CheXpertPneumoniaDataset  # noqa: E402
from pneumonia_ai.data.label_strategy import apply_label_strategy  # noqa: E402
from pneumonia_ai.models.factory import create_model  # noqa: E402
from scripts.train_baseline import _load_config, _transforms  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", required=True); parser.add_argument("--manifest", required=True)
parser.add_argument("--checkpoint", required=True); parser.add_argument("--split", required=True, choices=("validation", "test")); parser.add_argument("--output", required=True)


def _has_compatible_configuration(config: object) -> bool:
    """Return whether a checkpoint or run configuration can drive prediction."""
    return (
        isinstance(config, dict)
        and isinstance(config.get("model"), dict)
        and isinstance(config.get("training"), dict)
    )


def _resolve_checkpoint_configuration(
    state: dict[str, object], checkpoint_path: Path | str
) -> dict[str, object]:
    """Load a checkpoint's saved config, including the original baseline format.

    Early baseline checkpoints predate embedded configurations.  Their immutable
    run-local ``config.yaml`` is the canonical record of the training pipeline,
    so use it only for that documented legacy state shape rather than guessing
    architecture or preprocessing from model weights.
    """
    saved_config = state.get("configuration")
    if _has_compatible_configuration(saved_config):
        return saved_config
    if {"epoch", "model_state_dict", "auroc"}.issubset(state) and "configuration" not in state:
        config_path = Path(checkpoint_path).parent / "config.yaml"
        if not config_path.is_file():
            raise ValueError(
                "Legacy checkpoint has no embedded configuration and its canonical "
                f"run configuration is missing: {config_path}"
            )
        config = _load_config(config_path)
        if _has_compatible_configuration(config):
            return config
        raise ValueError(
            "Legacy checkpoint's canonical run configuration is incompatible: "
            f"{config_path}"
        )
    raise ValueError("Checkpoint does not contain a compatible saved configuration.")


def _prepare_prediction_manifest(manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    """Select one split and use package-local image paths when supplied.

    Test packages retain the source-manifest path for provenance and provide a
    ``packaged_image_path`` pointing at their copied, root-local image.  The
    dataset contract consumes ``image_path``, so make that substitution only in
    the temporary prediction manifest.
    """
    selected = manifest.loc[manifest.split == split].copy()
    if "packaged_image_path" in selected and selected["packaged_image_path"].notna().all():
        selected["image_path"] = selected["packaged_image_path"]
    return selected


def main() -> None:
    args = parser.parse_args(); state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = _resolve_checkpoint_configuration(state, args.checkpoint)
    model_config, training = config["model"], config["training"]; model_name = model_config["name"]
    manifest = pd.read_csv(args.manifest); selected = _prepare_prediction_manifest(manifest, args.split)
    strategy = config.get("label_strategy", {}).get("name", "ignore")
    selected = apply_label_strategy(selected, strategy, config.get("label_strategy", {}).get("uncertain_soft_target", .5), config.get("label_strategy", {}).get("uncertain_sample_weight", .5))
    if not selected.raw_pneumonia_label.isin([0, 1]).all(): raise ValueError("Evaluation split contains uncertain labels; use a definite-label evaluation manifest.")
    with tempfile.TemporaryDirectory() as temporary:
        portable = Path(temporary) / "split.csv"; selected.to_csv(portable, index=False)
        _, transform = _transforms(int(training["image_size"]), str(model_config.get("preprocessing", "imagenet")))
        dataset = CheXpertPneumoniaDataset(args.root, portable, args.split, transform); loader = DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=False)
        model = create_model(str(model_name), pretrained=False); model.load_state_dict(state["model_state_dict"]); model.eval(); rows=[]
        with torch.no_grad():
            for batch in loader:
                logits=model(batch["image"]).view(-1); probabilities=torch.sigmoid(logits)
                for index in range(len(logits)): rows.append({"patient_id":batch["patient_id"][index],"study_id":batch["study_id"][index],"image_path":batch["image_path"][index],"original_label":int(batch["raw_label"][index]),"binary_target":int(batch["target"][index]),"logit":float(logits[index]),"probability":float(probabilities[index]),"predicted_class":int(probabilities[index]>=.5),"split":args.split,"model_name":model_name,"run_id":Path(args.checkpoint).parent.name})
    pd.DataFrame(rows).to_csv(args.output,index=False)
if __name__ == "__main__": main()
