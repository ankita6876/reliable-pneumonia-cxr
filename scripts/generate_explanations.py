"""Generate selected binary-model explanations from a checkpoint and prediction CSV."""
import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
from pneumonia_ai.explainability.core import generate_attribution, save_explanation_figure, select_cases  # noqa: E402
from pneumonia_ai.models.factory import create_model  # noqa: E402
from scripts.train_baseline import _transforms  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", nargs="+", default=["gradcam"])
    parser.add_argument("--samples-per-group", type=int, default=5)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["configuration"]
    model_config = config["model"]
    training = config["training"]
    model = create_model(model_config["name"], pretrained=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    _, transform = _transforms(int(training["image_size"]), model_config.get("preprocessing", "imagenet"))
    predictions = pd.read_csv(args.predictions)
    selected = select_cases(predictions.loc[predictions["split"] == args.split], args.threshold, args.samples_per_group)
    rows = []
    for index, case in selected.iterrows():
        image_path = Path(args.root) / Path(case["image_path"])
        image = transform(Image.open(image_path).convert("RGB"))
        for method in args.methods:
            attribution = generate_attribution(model, image, method)
            filename = f"{index}_{case['outcome_group']}_{method}.png"
            save_explanation_figure(image, attribution, f"{case['outcome_group']} p={case['probability']:.3f}", output / filename)
            rows.append({"patient_id": case["patient_id"], "study_id": case["study_id"], "image_path": case["image_path"], "true_label": case["binary_target"], "predicted_probability": case["probability"], "predicted_class": int(case["probability"] >= args.threshold), "outcome_group": case["outcome_group"], "uncertainty": case["uncertainty"], "attribution_method": method, "output_filename": filename})
    pd.DataFrame(rows).to_csv(output / "explanations.csv", index=False)


if __name__ == "__main__":
    main()
