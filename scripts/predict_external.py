"""Run one CheXpert-trained checkpoint deterministically on an external manifest."""
# ruff: noqa: E501, E701, E702
import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src")); sys.path.insert(0, str(PROJECT_ROOT))
from pneumonia_ai.data.multidataset import UnifiedPneumoniaDataset  # noqa: E402
from pneumonia_ai.models.factory import create_model  # noqa: E402
from scripts.train_baseline import _transforms  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True); parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state.get("configuration", {}); model_config, training = config["model"], config["training"]
    _, transform = _transforms(int(training["image_size"]), str(model_config.get("preprocessing", "imagenet")))
    dataset = UnifiedPneumoniaDataset(args.image_root, args.manifest, "external_test", transform)
    loader = DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=False)
    model = create_model(str(model_config["name"]), pretrained=False).to(device)
    model.load_state_dict(state["model_state_dict"]); model.eval(); rows = []
    torch.use_deterministic_algorithms(True, warn_only=True)
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device)).view(-1).cpu(); probabilities = torch.sigmoid(logits)
            for index in range(len(logits)):
                patient_id = str(batch["patient_id"][index]); image_path = str(batch["image_path"][index]); label = int(batch["target"][index])
                rows.append({"patient_id": patient_id, "image_id": patient_id, "study_id": patient_id, "image_path": image_path, "label": label, "original_label": label, "binary_target": label, "probability": float(probabilities[index]), "logit": float(logits[index]), "predicted_class": int(probabilities[index] >= .5), "split": "external_test", "dataset": "rsna", "model": model_config["name"], "model_name": model_config["name"], "checkpoint": str(Path(args.checkpoint)), "run_id": str(Path(args.checkpoint))})
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
