"""Deterministic train/validation/test split generation for segmentation data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pneumonia_ai.segmentation.dataset import load_segmentation_metadata


DEFAULT_SEED = 42
SPLIT_NAMES = ("train", "validation", "test")


def create_segmentation_splits(
    metadata_path: Path | str,
    output_directory: Path | str,
    seed: int = DEFAULT_SEED,
) -> dict[str, pd.DataFrame]:
    """Create non-overlapping 70/15/15 splits and persist them as CSV files."""

    metadata = load_segmentation_metadata(metadata_path)
    indices = np.random.default_rng(seed).permutation(len(metadata))
    train_count, validation_count, _ = _split_counts(len(metadata))
    split_indices = {
        "train": indices[:train_count],
        "validation": indices[train_count : train_count + validation_count],
        "test": indices[train_count + validation_count :],
    }
    splits = {
        name: metadata.iloc[index].reset_index(drop=True)
        for name, index in split_indices.items()
    }
    _validate_splits(splits, len(metadata))

    directory = Path(output_directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_csv(directory / f"{name}.csv", index=False)
    return splits


def _split_counts(sample_count: int) -> tuple[int, int, int]:
    """Return rounded 70/15/15 split counts that sum to ``sample_count``."""

    train_count = round(sample_count * 0.70)
    validation_count = round(sample_count * 0.15)
    test_count = sample_count - train_count - validation_count
    return train_count, validation_count, test_count


def _validate_splits(splits: dict[str, pd.DataFrame], sample_count: int) -> None:
    """Defend against accidental omissions or overlap during split generation."""

    identifiers = [
        image_id for split in splits.values() for image_id in split["image_id"].tolist()
    ]
    if len(identifiers) != sample_count or len(set(identifiers)) != sample_count:
        raise ValueError("Generated splits must contain every sample exactly once.")
