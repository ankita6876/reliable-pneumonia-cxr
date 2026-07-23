"""Build a validated Montgomery lung-segmentation dataset manifest."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from pneumonia_ai.segmentation.inventory import SUPPORTED_IMAGE_EXTENSIONS
from pneumonia_ai.segmentation.mask_utils import (
    MaskValidationError,
    merge_lung_masks,
    read_mask,
    save_mask,
)
from pneumonia_ai.segmentation.metadata import (
    MontgomeryMetadataRecord,
    build_metadata_dataframe,
)


class MontgomeryDatasetError(ValueError):
    """Raised when the Montgomery dataset layout or files are inconsistent."""


def enumerate_chest_xrays(cxr_directory: Path | str) -> list[Path]:
    """Return the chest x-rays in deterministic filename order."""

    directory = Path(cxr_directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Chest x-ray directory does not exist: {directory}")

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def validate_filename_matching(
    image_paths: Iterable[Path],
    left_mask_paths: Iterable[Path],
    right_mask_paths: Iterable[Path],
) -> None:
    """Ensure that every source image has exactly one mask of each type."""

    image_names = {path.name for path in image_paths}
    left_names = {path.name for path in left_mask_paths}
    right_names = {path.name for path in right_mask_paths}

    problems: list[str] = []
    for label, names in (("left", left_names), ("right", right_names)):
        missing = sorted(image_names - names)
        unexpected = sorted(names - image_names)
        if missing:
            problems.append(f"missing {label} mask(s): {', '.join(missing)}")
        if unexpected:
            problems.append(f"unexpected {label} mask(s): {', '.join(unexpected)}")

    if problems:
        raise MontgomeryDatasetError("Filename mismatch: " + "; ".join(problems))


def locate_corresponding_masks(
    image_path: Path,
    left_mask_directory: Path,
    right_mask_directory: Path,
) -> tuple[Path, Path]:
    """Locate the exact-name left and right masks for one chest x-ray."""

    left_mask_path = left_mask_directory / image_path.name
    right_mask_path = right_mask_directory / image_path.name
    missing = [
        str(path) for path in (left_mask_path, right_mask_path) if not path.is_file()
    ]
    if missing:
        raise MontgomeryDatasetError(
            f"Missing mask(s) for {image_path.name}: {', '.join(missing)}"
        )
    return left_mask_path, right_mask_path


def _image_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise MontgomeryDatasetError(f"Unable to read chest x-ray image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def _validate_pair_resolution(
    image_path: Path,
    left_mask_path: Path,
    right_mask_path: Path,
) -> tuple[int, int]:
    width, height = _image_size(image_path)
    try:
        left_mask = read_mask(left_mask_path)
        right_mask = read_mask(right_mask_path)
        merge_lung_masks(left_mask, right_mask)
    except MaskValidationError as error:
        raise MontgomeryDatasetError(
            f"Invalid masks for {image_path.name}: {error}"
        ) from error

    if left_mask.shape != (height, width):
        raise MontgomeryDatasetError(
            f"Mask resolution does not match image for {image_path.name}: "
            f"image {(height, width)}, mask {left_mask.shape}."
        )
    return width, height


def build_montgomery_dataset(dataset_root: Path | str) -> pd.DataFrame:
    """Merge Montgomery masks and write ``processed/metadata.csv``.

    The source layout must contain ``CXR_png``, ``ManualMask/leftMask``, and
    ``ManualMask/rightMask``.  Merged masks retain their source image filename
    and pixel resolution under ``merged_masks``.
    """

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root does not exist: {root}")

    cxr_directory = root / "CXR_png"
    left_mask_directory = root / "ManualMask" / "leftMask"
    right_mask_directory = root / "ManualMask" / "rightMask"
    for directory in (left_mask_directory, right_mask_directory):
        if not directory.is_dir():
            raise NotADirectoryError(f"Mask directory does not exist: {directory}")

    image_paths = enumerate_chest_xrays(cxr_directory)
    if not image_paths:
        raise MontgomeryDatasetError(f"No chest x-rays found in: {cxr_directory}")
    left_mask_paths = enumerate_chest_xrays(left_mask_directory)
    right_mask_paths = enumerate_chest_xrays(right_mask_directory)
    validate_filename_matching(image_paths, left_mask_paths, right_mask_paths)

    pairs = [
        (
            image_path,
            *locate_corresponding_masks(
                image_path, left_mask_directory, right_mask_directory
            ),
        )
        for image_path in image_paths
    ]
    dimensions = {
        image_path: _validate_pair_resolution(
            image_path, left_mask_path, right_mask_path
        )
        for image_path, left_mask_path, right_mask_path in tqdm(
            pairs, desc="Validating Montgomery masks", unit="image"
        )
    }

    merged_mask_directory = root / "merged_masks"
    records: list[MontgomeryMetadataRecord] = []
    for image_path, left_mask_path, right_mask_path in tqdm(
        pairs, desc="Writing merged masks", unit="image"
    ):
        merged_mask_path = merged_mask_directory / image_path.name
        save_mask(
            merge_lung_masks(read_mask(left_mask_path), read_mask(right_mask_path)),
            merged_mask_path,
        )
        width, height = dimensions[image_path]
        records.append(
            MontgomeryMetadataRecord.from_paths(
                image_path=image_path,
                left_mask_path=left_mask_path,
                right_mask_path=right_mask_path,
                merged_mask_path=merged_mask_path,
                width=width,
                height=height,
            )
        )

    metadata = build_metadata_dataframe(records)
    metadata_path = root / "processed" / "metadata.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_path, index=False)
    return metadata
