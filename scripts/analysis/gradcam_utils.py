"""Reusable Grad-CAM, selection, localisation, and rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class SelectedCase:
    """One representative prediction selected for visual explanation."""

    category: str
    image_path: str
    label: int
    prediction: int
    probability: float
    patient_id: str
    study_id: str


class GradCAM:
    """Grad-CAM implementation for a validated convolutional target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: Tensor | None = None
        self.gradients: Tensor | None = None
        self._hook = target_layer.register_forward_hook(self._capture_activation)

    def _capture_activation(self, _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        self.activations = output
        output.register_hook(self._capture_gradient)

    def _capture_gradient(self, gradient: Tensor) -> None:
        self.gradients = gradient

    def generate(self, image: Tensor, predicted_class: int) -> np.ndarray:
        """Return a unit-scaled CAM supporting the predicted binary class."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image).reshape(-1)
        if logits.numel() != 1:
            raise ValueError("Grad-CAM expects exactly one image and one binary logit.")
        score = logits[0] if predicted_class == 1 else -logits[0]
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = functional.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = functional.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        result = cam[0, 0].detach().cpu().numpy()
        maximum = float(result.max())
        return result / maximum if maximum > 0 else np.zeros_like(result)

    def close(self) -> None:
        """Remove the registered forward hook."""
        self._hook.remove()


def select_target_layer(model: nn.Module, probe_image: Tensor) -> tuple[str, nn.Module]:
    """Find the final convolutional layer that yields valid spatial CAM gradients.

    Candidates are considered from the end of the model. Each is validated with
    a real forward and backward pass, ensuring both a spatial activation and
    its gradient are available for Grad-CAM.
    """
    candidates = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    if not candidates:
        raise ValueError("The classifier contains no Conv2d layers for Grad-CAM.")
    for name, layer in reversed(candidates):
        cam = GradCAM(model, layer)
        try:
            _ = cam.generate(probe_image, predicted_class=1)
            if cam.activations is not None and min(cam.activations.shape[-2:]) > 1:
                return name, layer
        except (RuntimeError, ValueError):
            pass
        finally:
            cam.close()
    raise RuntimeError("No convolutional layer produced usable spatial Grad-CAM activations.")


def select_representative_cases(predictions: pd.DataFrame, cases_per_category: int = 5, threshold: float = 0.5) -> list[SelectedCase]:
    """Select high-confidence TP, TN, FP, and FN examples deterministically."""
    required = {"image_path", "binary_target", "probability", "predicted_class"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing required columns: {', '.join(sorted(missing))}")
    if cases_per_category <= 0:
        raise ValueError("cases_per_category must be positive.")
    records = predictions.copy()
    records["binary_target"] = pd.to_numeric(records["binary_target"], errors="raise").astype(int)
    records["probability"] = pd.to_numeric(records["probability"], errors="raise").astype(float)
    records["predicted_class"] = pd.to_numeric(records["predicted_class"], errors="raise").astype(int)
    records["selection_prediction"] = (records["probability"] >= threshold).astype(int)
    categories: Mapping[str, tuple[int, int, bool]] = {
        "TP": (1, 1, True), "TN": (0, 0, False), "FP": (0, 1, True), "FN": (1, 0, False),
    }
    selected: list[SelectedCase] = []
    for category, (label, predicted, descending) in categories.items():
        subset = records.loc[(records.binary_target == label) & (records.selection_prediction == predicted)].copy()
        subset = subset.sort_values(["probability", "image_path"], ascending=[not descending, True]).head(cases_per_category)
        for row in subset.itertuples(index=False):
            selected.append(SelectedCase(
                category=category, image_path=str(row.image_path), label=int(row.binary_target),
                prediction=int(row.selection_prediction), probability=float(row.probability),
                patient_id=str(getattr(row, "patient_id", "")), study_id=str(getattr(row, "study_id", "")),
            ))
    return selected


def activation_localisation(cam: np.ndarray, lung_mask: np.ndarray) -> dict[str, float]:
    """Calculate activation mass inside and outside a binary lung mask."""
    if cam.shape != lung_mask.shape:
        raise ValueError("CAM and lung mask must share the same dimensions.")
    activation = np.clip(np.asarray(cam, dtype=float), 0.0, None)
    mask = np.asarray(lung_mask, dtype=bool)
    total = float(activation.sum())
    inside = float(activation[mask].sum())
    outside = float(activation[~mask].sum())
    if total == 0:
        return {"activation_inside_lungs": 0.0, "activation_outside_lungs": 0.0, "lung_focus_ratio": float("nan")}
    return {
        "activation_inside_lungs": inside / total,
        "activation_outside_lungs": outside / total,
        "lung_focus_ratio": inside / outside if outside > 0 else float("inf"),
    }


def save_case_figure(
    output_path: Path,
    original: Image.Image,
    mask: np.ndarray,
    hard_masked: Image.Image,
    cam: np.ndarray,
    case: SelectedCase,
) -> None:
    """Save one annotated six-panel explanation as a 300-DPI image or PDF."""
    original_array = np.asarray(original.convert("L"))
    heatmap = Image.fromarray(np.uint8(np.clip(cam, 0, 1) * 255)).resize(original.size, Image.Resampling.BILINEAR)
    heatmap_array = np.asarray(heatmap) / 255.0
    figure, axes = plt.subplots(1, 5, figsize=(16, 3.8))
    axes[0].imshow(original_array, cmap="gray")
    axes[0].set_title("Original chest X-ray")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Lung mask")
    axes[2].imshow(np.asarray(hard_masked.convert("L")), cmap="gray")
    axes[2].set_title("Hard-masked input")
    axes[3].imshow(cam, cmap="magma", vmin=0, vmax=1)
    axes[3].set_title("Grad-CAM")
    axes[4].imshow(original_array, cmap="gray")
    axes[4].imshow(heatmap_array, cmap="magma", vmin=0, vmax=1, alpha=0.45)
    axes[4].set_title("Grad-CAM overlay")
    for axis in axes:
        axis.axis("off")
    identifier = case.patient_id or case.image_path
    figure.suptitle(
        f"{case.category} | {identifier} | label={case.label}, predicted={case.prediction}, probability={case.probability:.3f}",
        fontsize=11,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
