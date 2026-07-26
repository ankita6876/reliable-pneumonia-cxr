"""Filesystem cache for deterministic frozen-segmentation masks."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
import torch


class MaskCache:
    """Cache float probability masks as compressed ``.npz`` files.

    Keys include image path/stat data, checkpoint file content identity, threshold,
    and segmentation input size, preventing cross-model or cross-threshold reuse.
    """

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)

    def key(self, source_path: Path | str, checkpoint_path: Path | str, threshold: float, image_size: int) -> str:
        source, checkpoint = Path(source_path), Path(checkpoint_path)
        if not source.is_file() or not checkpoint.is_file():
            raise FileNotFoundError("Cache keys require existing source image and checkpoint files.")
        digest = hashlib.sha256()
        for path in (source.resolve(), checkpoint.resolve()):
            stat = path.stat()
            digest.update(str(path).encode()); digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
        digest.update(f"{threshold:.17g}:{image_size}".encode())
        return digest.hexdigest()

    def get(self, key: str) -> torch.Tensor | None:
        path = self.directory / f"{key}.npz"
        try:
            with np.load(path, allow_pickle=False) as archive:
                array = archive["probability"]
            if array.ndim != 2 or not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
                raise ValueError("invalid mask")
            return torch.from_numpy(array.astype(np.float32, copy=False))
        except (OSError, KeyError, ValueError):
            return None

    def set(self, key: str, probability: torch.Tensor) -> Path:
        array = probability.detach().cpu().numpy().astype(np.float32, copy=False)
        if array.ndim != 2:
            raise ValueError("Cached probability mask must have shape HxW.")
        target = self.directory / f"{key}.npz"
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{key}-", suffix=".npz", dir=self.directory)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            np.savez_compressed(temporary, probability=array)
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return target
