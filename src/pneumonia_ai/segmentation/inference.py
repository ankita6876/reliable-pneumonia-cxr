"""Frozen, deterministic inference for trained lung U-Net checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from pneumonia_ai.segmentation.model import UNet


class FrozenLungSegmenter:
    """Load a U-Net checkpoint and expose read-only lung-mask prediction.

    Inputs are converted to grayscale floats in ``[0, 1]``.  Returned masks are
    CPU tensors at the input image's original ``(height, width)``.
    """

    def __init__(self, checkpoint_path: Path | str, device: str | torch.device = "cpu") -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Segmentation checkpoint does not exist: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("Segmentation checkpoint must be a dictionary.")
        model_config = checkpoint.get("model_config")
        model_state = checkpoint.get("model_state")
        if not isinstance(model_config, dict) or not isinstance(model_state, dict):
            raise ValueError("Checkpoint must contain dictionary fields 'model_config' and 'model_state'.")
        try:
            self.model = UNet(**model_config)
            self.model.load_state_dict(model_state, strict=True)
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid segmentation checkpoint model data: {error}") from error
        training_config = checkpoint.get("training_config", {})
        if not isinstance(training_config, dict):
            raise ValueError("Checkpoint training_config must be a dictionary when present.")
        image_size = training_config.get("image_size", 128)
        if not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("Checkpoint training_config.image_size must be a positive integer.")
        self.image_size = image_size
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.metadata: dict[str, Any] = checkpoint

    @torch.inference_mode()
    def predict_proba(self, image: Image.Image | np.ndarray | Tensor) -> Tensor:
        """Return a ``float32`` probability mask aligned to the input dimensions."""

        tensor = _as_grayscale_tensor(image)
        original_size = tensor.shape[-2:]
        resized = transform_functional.resize(
            tensor, [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR, antialias=True,
        )
        probabilities = torch.sigmoid(self.model(resized.unsqueeze(0).to(self.device)))[0].cpu()
        return transform_functional.resize(
            probabilities, list(original_size), interpolation=InterpolationMode.BILINEAR, antialias=True
        ).squeeze(0).clamp(0.0, 1.0).to(dtype=torch.float32)

    def predict(self, image: Image.Image | np.ndarray | Tensor, threshold: float = 0.5) -> Tensor:
        """Return a boolean binary lung mask using ``threshold``."""

        _validate_threshold(threshold)
        return self.predict_proba(image) >= threshold


def _as_grayscale_tensor(image: Image.Image | np.ndarray | Tensor) -> Tensor:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)
    if isinstance(image, np.ndarray):
        tensor = torch.from_numpy(image)
    elif isinstance(image, Tensor):
        tensor = image.detach().cpu()
    else:
        raise TypeError("image must be a PIL image, NumPy array, or torch Tensor.")
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and tensor.shape[0] == 1:
        pass
    elif tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor.permute(2, 0, 1)
    elif tensor.ndim == 3 and tensor.shape[0] in (3, 4):
        tensor = (tensor[:3].to(torch.float32) * torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)).sum(0, keepdim=True)
    elif tensor.ndim == 3 and tensor.shape[-1] in (3, 4):
        tensor = tensor[..., :3].permute(2, 0, 1).to(torch.float32)
        tensor = (tensor * torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)).sum(0, keepdim=True)
    else:
        raise ValueError("Image arrays/tensors must have shape HxW, 1xHxW, HxWx1, or RGB(A).")
    tensor = tensor.to(dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError("Image contains non-finite values.")
    if tensor.numel() == 0:
        raise ValueError("Image must not be empty.")
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


def _validate_threshold(threshold: float) -> None:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1.")
