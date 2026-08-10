"""Generate reproducible Grad-CAM explanations for the hard-masked classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.classification.segmentation_guided import InputMode, prepare_classifier_image  # noqa: E402
from pneumonia_ai.data.chexpert_dataset import CheXpertPneumoniaDataset  # noqa: E402
from pneumonia_ai.models.factory import create_model  # noqa: E402
from pneumonia_ai.segmentation.cache import MaskCache  # noqa: E402
from pneumonia_ai.segmentation.inference import FrozenLungSegmenter  # noqa: E402
from pneumonia_ai.training.seed import seed_everything  # noqa: E402
from scripts.classification.checkpoint_compatibility import (  # noqa: E402
    load_adjacent_experiment_configuration,
    normalize_checkpoint_configuration,
)

try:
    from .gradcam_utils import (
        GradCAM,
        activation_localisation,
        save_case_figure,
        select_representative_cases,
        select_target_layer,
    )
except ImportError:  # Direct script execution has no package context.
    from gradcam_utils import (  # noqa: E402
        GradCAM,
        activation_localisation,
        save_case_figure,
        select_representative_cases,
        select_target_layer,
    )


OUTPUT_ROOT = PROJECT_ROOT.parent / "outputs" / "classification_ablation"
DEFAULT_CLASSIFIER_CHECKPOINT = OUTPUT_ROOT / "hard_masked" / "best_checkpoint.pt"
DEFAULT_PREDICTIONS = OUTPUT_ROOT / "hard_masked" / "predictions.csv"
DEFAULT_SEGMENTATION_CHECKPOINT = PROJECT_ROOT.parent / "outputs" / "montgomery_unet_v1" / "best_checkpoint.pt"
DEFAULT_OUTPUT_DIRECTORY = OUTPUT_ROOT / "explainability" / "hard_masked"


def parse_args() -> argparse.Namespace:
    """Parse reproducible Grad-CAM analysis options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-checkpoint", type=Path, default=DEFAULT_CLASSIFIER_CHECKPOINT)
    parser.add_argument("--predictions-csv", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--segmentation-checkpoint", type=Path, default=DEFAULT_SEGMENTATION_CHECKPOINT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--image-root", type=Path, help="Overrides the image root saved in the classifier checkpoint.")
    parser.add_argument("--cases-per-category", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    return parser.parse_args()


def _resolve_device(device: str) -> torch.device:
    """Return the requested device after validating CUDA availability."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False.")
    return torch.device(device)


def _load_classifier(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, object]]:
    """Recreate and load the exact classifier architecture saved in a checkpoint."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Classifier checkpoint does not exist: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not isinstance(state.get("configuration"), dict):
        raise ValueError("Classifier checkpoint lacks its required configuration dictionary.")
    if not isinstance(state.get("model_state_dict"), dict):
        raise ValueError("Classifier checkpoint lacks model_state_dict.")
    configuration = normalize_checkpoint_configuration(
        state["configuration"], load_adjacent_experiment_configuration(checkpoint)
    )
    if configuration.get("input_mode") not in {InputMode.HARD_MASKED.value, InputMode.SOFT_MASKED.value}:
        raise ValueError("This analysis supports only a hard_masked or soft_masked classifier checkpoint.")
    model_name = str(configuration.get("model", "densenet121"))
    model = create_model(model_name, pretrained=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, configuration


def _validation_transform(image_size: int) -> transforms.Compose:
    """Return the same deterministic preprocessing transform used at evaluation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def _case_manifest(cases: list[object], directory: Path) -> Path:
    """Write a minimal test manifest accepted by CheXpertPneumoniaDataset."""
    records = [
        {
            "split": "test", "patient_id": case.patient_id, "study_id": case.study_id,
            "image_path": case.image_path, "pneumonia_label": case.label,
        }
        for case in cases
    ]
    path = directory / "selected_cases.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def _save_montage(category: str, image_paths: list[Path], destination: Path) -> None:
    """Create a deterministic two-column montage from individual PNG figures."""
    if not image_paths:
        return
    images = [Image.open(path).convert("RGB") for path in image_paths]
    try:
        width = max(image.width for image in images)
        height = max(image.height for image in images)
        rows = (len(images) + 1) // 2
        montage = Image.new("RGB", (width * 2, height * rows), "white")
        for index, image in enumerate(images):
            x, y = (index % 2) * width, (index // 2) * height
            montage.paste(image, (x, y))
        destination.parent.mkdir(parents=True, exist_ok=True)
        montage.save(destination.with_suffix(".png"), dpi=(300, 300))
        montage.save(destination.with_suffix(".pdf"), resolution=300.0)
    finally:
        for image in images:
            image.close()


def generate_gradcam_analysis(
    classifier_checkpoint: Path,
    predictions_csv: Path,
    segmentation_checkpoint: Path,
    output_directory: Path,
    image_root: Path | None = None,
    cases_per_category: int = 5,
    threshold: float = 0.5,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Generate explanation figures, manifests, localisation summaries, and montages."""
    resolved_device = _resolve_device(device)
    if not predictions_csv.is_file():
        raise FileNotFoundError(f"Held-out predictions CSV does not exist: {predictions_csv}")
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between 0 and 1.")
    seed_everything(seed)
    model, configuration = _load_classifier(classifier_checkpoint, resolved_device)
    resolved_root = image_root or Path(str(configuration.get("image_root", "")))
    if not resolved_root.is_dir():
        raise NotADirectoryError("Image root is unavailable. Supply --image-root with the CheXpert extraction directory.")
    image_size = int(configuration.get("classifier_image_size", 224))
    mask_threshold = float(configuration.get("mask_threshold", threshold))
    predictions = pd.read_csv(predictions_csv)
    cases = select_representative_cases(predictions, cases_per_category, threshold)
    if not cases:
        raise ValueError("No TP, TN, FP, or FN examples were found in the prediction file.")

    output_directory.mkdir(parents=True, exist_ok=True)
    segmenter = FrozenLungSegmenter(segmentation_checkpoint, resolved_device)
    cache = MaskCache(output_directory / "mask_cache")
    with tempfile.TemporaryDirectory() as temporary:
        dataset = CheXpertPneumoniaDataset(
            resolved_root, _case_manifest(cases, Path(temporary)), "test", _validation_transform(image_size),
            input_mode=InputMode(configuration["input_mode"]), lung_segmenter=segmenter, mask_cache=cache,
            mask_threshold=mask_threshold, lung_crop_padding=int(configuration.get("lung_crop_padding", 0)),
            soft_mask_outside_factor=float(configuration.get("soft_mask_outside_factor", 0.20)),
            classifier_image_size=image_size, allow_absolute_image_paths=True,
        )
        probe = dataset[0]["image"].unsqueeze(0).to(resolved_device)
        layer_name, layer = select_target_layer(model, probe)
        gradcam = GradCAM(model, layer)
        manifest_rows: list[dict[str, object]] = []
        localisation_rows: list[dict[str, object]] = []
        category_figures: dict[str, list[Path]] = {category: [] for category in ("TP", "TN", "FP", "FN")}
        try:
            for index, case in enumerate(cases, start=1):
                sample = dataset[index - 1]
                source = dataset._resolved_image_paths[index - 1]
                with Image.open(source) as opened:
                    original = opened.convert("L").copy()
                mask = segmenter.predict(original, threshold=mask_threshold).numpy()
                guided_image = prepare_classifier_image(
                    original, InputMode(configuration["input_mode"]), segmenter=segmenter, threshold=mask_threshold,
                    soft_mask_outside_factor=float(configuration.get("soft_mask_outside_factor", 0.20)),
                    probability_mask=segmenter.predict_proba(original),
                )
                cam = gradcam.generate(sample["image"].unsqueeze(0).to(resolved_device), case.prediction)
                resized_mask = np.asarray(
                    Image.fromarray(mask.astype(np.uint8)).resize((image_size, image_size), Image.Resampling.NEAREST), dtype=bool
                )
                focus = activation_localisation(cam, resized_mask)
                stem = f"{case.category}_{index:02d}_{Path(case.image_path).stem}"
                png_path, pdf_path = output_directory / "cases" / f"{stem}.png", output_directory / "cases" / f"{stem}.pdf"
                save_case_figure(png_path, original, mask, guided_image, cam, case, input_mode=configuration["input_mode"])
                save_case_figure(pdf_path, original, mask, guided_image, cam, case, input_mode=configuration["input_mode"])
                category_figures[case.category].append(png_path)
                manifest_rows.append({
                    "image_path": case.image_path, "category": case.category, "label": case.label,
                    "prediction": case.prediction, "probability": case.probability, "patient_id": case.patient_id,
                    "study_id": case.study_id, "figure_path": str(png_path), "pdf_figure_path": str(pdf_path),
                    "gradcam_target_layer": layer_name,
                })
                localisation_rows.append({"image_path": case.image_path, "category": case.category, **focus})
        finally:
            gradcam.close()
    for category, paths in category_figures.items():
        _save_montage(category, paths, output_directory / "montages" / f"{category.lower()}_montage")
    manifest = pd.DataFrame(manifest_rows)
    localisation = pd.DataFrame(localisation_rows)
    manifest.to_csv(output_directory / "gradcam_manifest.csv", index=False)
    localisation.to_csv(output_directory / "gradcam_localisation_cases.csv", index=False)
    summary = localisation.groupby("category", sort=False)[["activation_inside_lungs", "activation_outside_lungs", "lung_focus_ratio"]].agg(["count", "mean", "std"]).reset_index()
    summary.columns = ["_".join(filter(None, map(str, column))).rstrip("_") for column in summary.columns]
    summary.to_csv(output_directory / "gradcam_localisation_summary.csv", index=False)
    (output_directory / "analysis_metadata.json").write_text(json.dumps({
        "classifier_checkpoint": str(classifier_checkpoint), "segmentation_checkpoint": str(segmentation_checkpoint),
        "target_layer": layer_name, "cases_per_category": cases_per_category, "threshold": threshold,
        "mask_threshold": mask_threshold, "input_mode": configuration["input_mode"], "seed": seed, "device": device,
        **({"soft_mask_outside_factor": float(configuration.get("soft_mask_outside_factor", 0.20))} if configuration["input_mode"] == InputMode.SOFT_MASKED.value else {}),
    }, indent=2))
    return manifest, summary, layer_name


def main() -> None:
    """Run Grad-CAM generation and print the verified output summary."""
    args = parse_args()
    manifest, summary, layer_name = generate_gradcam_analysis(**vars(args))
    print(f"Grad-CAM target layer: {layer_name}")
    print(manifest.groupby("category", sort=False).size().to_string())
    print(summary.to_string(index=False))
    print(f"Outputs: {args.output_directory}")


if __name__ == "__main__":
    main()
