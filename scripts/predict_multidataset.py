"""Run a saved CheXpert-trained checkpoint on any shared-manifest dataset split."""
# ruff: noqa: E501, E701, E702
import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
from pneumonia_ai.data.multidataset import UnifiedPneumoniaDataset  # noqa: E402
from pneumonia_ai.models.factory import create_model  # noqa: E402
from scripts.train_baseline import _transforms  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="external")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["configuration"]
    model_config, training = config["model"], config["training"]
    _, transform = _transforms(int(training["image_size"]), str(model_config.get("preprocessing", "imagenet")))
    dataset = UnifiedPneumoniaDataset(args.root, args.manifest, args.split, transform)
    loader = DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=False)
    model = create_model(str(model_config["name"]), pretrained=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"]).view(-1)
            probabilities = torch.sigmoid(logits)
            for index in range(len(logits)): rows.append({"patient_id": batch["patient_id"][index], "study_id": batch["study_id"][index], "image_path": batch["image_path"][index], "original_label": int(batch["raw_label"][index]), "binary_target": int(batch["target"][index]), "logit": float(logits[index]), "probability": float(probabilities[index]), "predicted_class": int(probabilities[index] >= .5), "split": args.split, "model_name": model_config["name"], "run_id": Path(args.checkpoint).parent.name, "dataset": batch["dataset"][index]})
    pd.DataFrame(rows).to_csv(args.output, index=False)
