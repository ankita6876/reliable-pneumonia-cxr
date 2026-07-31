"""Filesystem cache for deterministic frozen-segmentation masks."""

from __future__ import annotations

import hashlib
import json
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

    def __init__(self, directory: Path | str, *, create: bool = True) -> None:
        self.directory = Path(directory).expanduser()
        if create:
            self.directory.mkdir(parents=True, exist_ok=True)
        elif not self.directory.is_dir():
            raise FileNotFoundError(f"Mask cache does not exist: {self.directory}")

    @property
    def _index_path(self) -> Path:
        return self.directory / "coverage.json"

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

    def set(
        self, key: str, probability: torch.Tensor, *, source_path: Path | str | None = None
    ) -> Path:
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
        if source_path is not None:
            index = self._read_index()
            index[str(Path(source_path).resolve())] = key
            self._index_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
        return target

    def get_for_source(self, source_path: Path | str) -> torch.Tensor | None:
        """Get an indexed mask without requiring the generating checkpoint."""
        key = self._read_index().get(str(Path(source_path).resolve()))
        return None if not isinstance(key, str) else self.get(key)

    def require_coverage(self, source_paths: list[Path]) -> None:
        """Reject cache-only use unless every requested source has a valid mask."""
        missing = [path for path in source_paths if self.get_for_source(path) is None]
        if missing:
            raise ValueError(
                "Mask cache is incomplete for hard-masked training: "
                f"{len(missing)} of {len(source_paths)} required images lack valid masks."
            )

    @property
    def metadata_path(self) -> Path:
        return self.directory / "cache_metadata.json"

    def validate_or_initialise_metadata(self, expected: dict[str, object]) -> None:
        """Bind a cache to deterministic mask-generation inputs before reuse.

        A populated cache without this metadata is deliberately rejected: its
        provenance cannot be established safely.
        """
        try:
            existing = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            existing = None
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Mask cache metadata is unreadable: {self.metadata_path}") from error
        if existing is None:
            if any(self.directory.glob("*.npz")) or self._index_path.exists():
                raise ValueError("Mask cache has entries but no compatibility metadata; refusing unsafe reuse.")
            self.metadata_path.write_text(json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8")
            return
        if existing != expected:
            differing = sorted(set(existing if isinstance(existing, dict) else {}) | set(expected))
            differing = [key for key in differing if not isinstance(existing, dict) or existing.get(key) != expected.get(key)]
            raise ValueError("Mask cache metadata is incompatible; differing fields: " + ", ".join(differing))

    def coverage_count(self) -> int:
        """Return the number of valid indexed masks currently available."""
        return sum(self.get(key) is not None for key in self._read_index().values())

    def _read_index(self) -> dict[str, str]:
        try:
            payload = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return (
            {key: value for key, value in payload.items() if isinstance(value, str)}
            if isinstance(payload, dict)
            else {}
        )
