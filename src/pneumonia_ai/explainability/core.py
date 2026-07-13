"""Headless, checkpoint-compatible attribution methods for binary-logit models."""

from __future__ import annotations

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as functional

from pneumonia_ai.evaluation.core import add_deterministic_uncertainty, validate_predictions


def target_layer_for_model(model: nn.Module) -> nn.Module:
    """Return the final convolutional feature layer, or reject unsupported models clearly."""
    layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    if not layers:
        raise ValueError("Unsupported architecture: no convolutional target layer was found.")
    return layers[-1]


def normalize_attribution(values: torch.Tensor) -> torch.Tensor:
    """Normalize a per-image map to [0, 1] without inventing signal in constant maps."""
    values = values.detach().float()
    low, high = values.amin(dim=(-2, -1), keepdim=True), values.amax(dim=(-2, -1), keepdim=True)
    return torch.where(high > low, (values - low) / (high - low), torch.zeros_like(values))


def generate_attribution(model: nn.Module, image: torch.Tensor, method: str, target_layer: nn.Module | None = None) -> torch.Tensor:
    """Generate a normalized [H, W] attribution for one binary-logit image tensor."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Explainability requires exactly one CHW image.")
    model.eval()
    device = next(model.parameters()).device
    image = image.to(device)
    if method in {"gradcam", "gradcam++"}:
        return _gradcam(model, image, target_layer or target_layer_for_model(model), method == "gradcam++")
    if method == "integrated_gradients":
        return _integrated_gradients(model, image)
    if method == "occlusion":
        return _occlusion(model, image)
    raise ValueError("Unsupported attribution method. Use gradcam, gradcam++, integrated_gradients, or occlusion.")


def select_cases(predictions: pd.DataFrame, threshold: float, samples_per_group: int) -> pd.DataFrame:
    """Return predefined outcome/confidence/uncertainty selections from an existing CSV."""
    frame = add_deterministic_uncertainty(validate_predictions(predictions))
    if samples_per_group <= 0:
        raise ValueError("samples_per_group must be positive.")
    predicted = (frame["probability"] >= threshold).astype(int)
    target = frame["binary_target"].astype(int)
    groups = {"true_positive": (target == 1) & (predicted == 1), "true_negative": (target == 0) & (predicted == 0), "false_positive": (target == 0) & (predicted == 1), "false_negative": (target == 1) & (predicted == 0)}
    correct = predicted == target
    error = ~correct
    groups.update({"highest_confidence_correct": correct, "highest_confidence_error": error, "highest_uncertainty": pd.Series(True, index=frame.index), "lowest_uncertainty": pd.Series(True, index=frame.index)})
    rows = []
    for name, mask in groups.items():
        subset = frame.loc[mask].copy()
        ascending = name == "lowest_uncertainty"
        column = "uncertainty" if "uncertainty" in name else "confidence"
        subset = subset.sort_values(column, ascending=ascending).head(samples_per_group)
        subset["outcome_group"] = name
        rows.append(subset)
    return pd.concat(rows, ignore_index=True) if rows else frame.iloc[0:0].copy()


def attribution_quality(attribution: torch.Tensor, comparison: torch.Tensor | None = None) -> dict[str, float]:
    """Describe concentration/sparsity and optional method agreement; not localization accuracy."""
    values = normalize_attribution(attribution.unsqueeze(0) if attribution.ndim == 2 else attribution).flatten().cpu().numpy()
    probability = values / values.sum() if values.sum() else np.zeros_like(values)
    result = {"concentration": float(np.square(probability).sum()), "sparsity": float(np.mean(values <= .05))}
    if comparison is not None:
        other = normalize_attribution(comparison.unsqueeze(0) if comparison.ndim == 2 else comparison).flatten().cpu().numpy()
        result["agreement_correlation"] = float(np.corrcoef(values, other)[0, 1]) if values.std() and other.std() else float("nan")
    return result


def save_explanation_figure(image: torch.Tensor, attribution: torch.Tensor, title: str, output: Path) -> None:
    """Save original, heatmap, and overlay at 300 DPI, closing the figure unconditionally."""
    original = image.detach().cpu().mean(dim=0).numpy()
    heatmap = attribution.detach().cpu().numpy()
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    try:
        axes[0].imshow(original, cmap="gray")
        axes[0].set_title("Chest X-ray")
        axes[1].imshow(heatmap, cmap="magma")
        axes[1].set_title("Attribution")
        axes[2].imshow(original, cmap="gray")
        axes[2].imshow(heatmap, cmap="magma", alpha=.5)
        axes[2].set_title("Overlay")
        for axis in axes:
            axis.axis("off")
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(output, dpi=300)
    finally:
        plt.close(figure)


def _gradcam(model: nn.Module, image: torch.Tensor, layer: nn.Module, plus_plus: bool) -> torch.Tensor:
    activation: list[torch.Tensor] = []
    gradient: list[torch.Tensor] = []
    forward = layer.register_forward_hook(lambda _, __, output: activation.append(output))
    backward = layer.register_full_backward_hook(lambda _, __, output: gradient.append(output[0]))
    try:
        model.zero_grad(set_to_none=True)
        model(image).view(-1)[0].backward()
        activations, gradients = activation[0], gradient[0]
        if plus_plus:
            squared, cubed = gradients.square(), gradients.pow(3)
            alpha = squared / (2 * squared + (activations * cubed).sum(dim=(-2, -1), keepdim=True) + 1e-8)
            weights = (alpha * functional.relu(gradients)).sum(dim=(-2, -1), keepdim=True)
        else:
            weights = gradients.mean(dim=(-2, -1), keepdim=True)
        cam = functional.relu((weights * activations).sum(dim=1))
        return normalize_attribution(functional.interpolate(cam.unsqueeze(1), image.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)).squeeze(0)
    finally:
        forward.remove()
        backward.remove()


def _integrated_gradients(model: nn.Module, image: torch.Tensor, steps: int = 32) -> torch.Tensor:
    baseline = torch.zeros_like(image)
    total = torch.zeros_like(image)
    for alpha in torch.linspace(1 / steps, 1, steps, device=image.device):
        point = (baseline + alpha * (image - baseline)).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        model(point).view(-1)[0].backward()
        total += point.grad
    return normalize_attribution(((image - baseline) * total / steps).abs().sum(dim=1)).squeeze(0)


def _occlusion(model: nn.Module, image: torch.Tensor, window: int = 16) -> torch.Tensor:
    with torch.no_grad():
        baseline = model(image).view(-1)[0]
    result = torch.zeros(image.shape[-2:], device=image.device)
    for row in range(0, image.shape[-2], window):
        for column in range(0, image.shape[-1], window):
            occluded = image.clone()
            occluded[:, :, row:row + window, column:column + window] = 0
            with torch.no_grad():
                result[row:row + window, column:column + window] = functional.relu(baseline - model(occluded).view(-1)[0])
    return normalize_attribution(result.unsqueeze(0)).squeeze(0)
