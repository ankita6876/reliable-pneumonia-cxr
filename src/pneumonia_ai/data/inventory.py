"""Filesystem-only inventory utilities for a CheXpert dataset directory."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MAX_REPRESENTATIVE_PATHS = 20


@dataclass(frozen=True)
class CheXpertInventory:
    """Summary of files and directories found below a CheXpert root directory."""

    root_path: Path
    total_files: int
    image_files_by_extension: dict[str, int]
    csv_files: tuple[Path, ...]
    directory_count: int
    representative_paths: tuple[Path, ...]

    @property
    def total_image_files(self) -> int:
        """Return the total across all supported image extensions."""
        return sum(self.image_files_by_extension.values())


def inventory_chexpert(root: Path | str) -> CheXpertInventory:
    """Recursively inventory a CheXpert directory without opening any files.

    Args:
        root: Root directory to scan. No particular CheXpert layout is assumed.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
        NotADirectoryError: If ``root`` is not a directory.
        ValueError: If no supported image files are discovered.
    """
    root_path = Path(root).expanduser()
    if not root_path.exists():
        raise FileNotFoundError(f"CheXpert root path does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"CheXpert root path is not a directory: {root_path}")

    resolved_root = root_path.resolve()
    image_counts: Counter[str] = Counter()
    csv_files: list[Path] = []
    top_level_paths: list[Path] = []
    nested_paths: list[Path] = []
    total_files = 0
    directory_count = 0

    for path in sorted(resolved_root.rglob("*")):
        relative_path = path.relative_to(resolved_root)
        if len(relative_path.parts) == 1:
            top_level_paths.append(path)
        else:
            nested_paths.append(path)

        if path.is_dir():
            directory_count += 1
            continue

        if not path.is_file():
            continue

        total_files += 1
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS and not path.name.startswith("._"):
            image_counts[suffix] += 1
        if suffix == ".csv":
            csv_files.append(path)

    image_files_by_extension = {
        extension: image_counts[extension] for extension in IMAGE_EXTENSIONS
    }
    if not any(image_files_by_extension.values()):
        raise ValueError(
            "No image files found under CheXpert root. "
            f"Expected extensions: {', '.join(IMAGE_EXTENSIONS)}."
        )

    representative_paths = _representative_paths(top_level_paths, nested_paths)
    return CheXpertInventory(
        root_path=resolved_root,
        total_files=total_files,
        image_files_by_extension=image_files_by_extension,
        csv_files=tuple(csv_files),
        directory_count=directory_count,
        representative_paths=representative_paths,
    )


def _representative_paths(
    top_level_paths: list[Path], nested_paths: list[Path]
) -> tuple[Path, ...]:
    """Select up to 20 paths while retaining both layout depths when present."""
    top_level_limit = min(10, len(top_level_paths))
    selected = top_level_paths[:top_level_limit]
    selected.extend(nested_paths[:MAX_REPRESENTATIVE_PATHS - len(selected)])
    if len(selected) < MAX_REPRESENTATIVE_PATHS:
        remaining_slots = MAX_REPRESENTATIVE_PATHS - len(selected)
        selected.extend(top_level_paths[top_level_limit : top_level_limit + remaining_slots])
    return tuple(selected)
