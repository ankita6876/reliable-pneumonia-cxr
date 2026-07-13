"""Generate one portable deep-ensemble prediction CSV from saved best checkpoints."""
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
from pneumonia_ai.evaluation.core import aggregate_ensemble  # noqa: E402
from pneumonia_ai.models.factory import create_model  # noqa: E402
from scripts.train_baseline import _transforms  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", required=True)
parser.add_argument("--manifest", required=True)
parser.add_argument("--checkpoints", nargs="+", required=True)
parser.add_argument("--split", choices=("validation", "test"), required=True)
parser.add_argument("--output", required=True)


def _member_predictions(checkpoint: str, root: str, manifest_path: str, split: str) -> pd.DataFrame:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state.get("configuration")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError(f"Checkpoint has no compatible configuration: {checkpoint}")
    model_config, training = config["model"], config["training"]
    records = pd.read_csv(manifest_path)
    records = records.loc[records["split"] == split].copy()
    label_config = config.get("label_strategy", {})
    records = apply_label_strategy(records, label_config.get("name", "ignore"), label_config.get("uncertain_soft_target", .5), label_config.get("uncertain_sample_weight", .5))
    if not records["raw_pneumonia_label"].isin([0, 1]).all():
        raise ValueError("Ensemble evaluation requires definite binary labels.")
    with tempfile.TemporaryDirectory() as temporary:
        portable = Path(temporary) / "split.csv"; records.to_csv(portable, index=False)
        _, transform = _transforms(int(training["image_size"]), str(model_config.get("preprocessing", "imagenet")))
        dataset = CheXpertPneumoniaDataset(root, portable, split, transform)
        loader = DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=False)
        model = create_model(str(model_config["name"]), pretrained=False)
        model.load_state_dict(state["model_state_dict"]); model.eval(); rows = []
        with torch.no_grad():
            for batch in loader:
                logits = model(batch["image"]).view(-1); probabilities = torch.sigmoid(logits)
                for index in range(len(logits)):
                    rows.append({"patient_id": batch["patient_id"][index], "study_id": batch["study_id"][index], "image_path": batch["image_path"][index], "original_label": int(batch["raw_label"][index]), "binary_target": int(batch["target"][index]), "logit": float(logits[index]), "probability": float(probabilities[index]), "predicted_class": int(probabilities[index] >= .5), "split": split, "model_name": str(model_config["name"]), "run_id": Path(checkpoint).parent.name})
    return pd.DataFrame(rows)


def main() -> None:
    args = parser.parse_args()
    if len(args.checkpoints) < 2: raise ValueError("Deep ensembles require at least two checkpoints.")
    ensemble = aggregate_ensemble([_member_predictions(path, args.root, args.manifest, args.split) for path in args.checkpoints])
    ensemble["model_name"] = "deep_ensemble"; ensemble["run_id"] = "ensemble"
    ensemble.to_csv(args.output, index=False)


if __name__ == "__main__": main()
