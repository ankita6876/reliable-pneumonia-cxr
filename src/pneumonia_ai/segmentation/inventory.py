"""Inventory utilities for lung-segmentation datasets."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
}


@dataclass(frozen=True)
class DatasetInventory:
    """Summary of files found under a dataset directory."""

    root: str
    exists: bool
    total_files: int
    image_files: int
    other_files: int
    extensions: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def iter_files(root: Path) -> Iterable[Path]:
    """Yield all regular files recursively in deterministic order."""

    if not root.exists() or not root.is_dir():
        return

    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def inventory_dataset(root: str | Path) -> DatasetInventory:
    """Inventory a lung-segmentation dataset directory."""

    root_path = Path(root).expanduser().resolve()

    if not root_path.exists() or not root_path.is_dir():
        return DatasetInventory(
            root=str(root_path),
            exists=False,
            total_files=0,
            image_files=0,
            other_files=0,
            extensions={},
        )

    files = list(iter_files(root_path))
    extension_counts = Counter(
        path.suffix.lower() if path.suffix else "<no_extension>"
        for path in files
    )

    image_count = sum(
        count
        for extension, count in extension_counts.items()
        if extension in SUPPORTED_IMAGE_EXTENSIONS
    )

    return DatasetInventory(
        root=str(root_path),
        exists=True,
        total_files=len(files),
        image_files=image_count,
        other_files=len(files) - image_count,
        extensions=dict(sorted(extension_counts.items())),
    )
