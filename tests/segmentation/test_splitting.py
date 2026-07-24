from __future__ import annotations

from pathlib import Path

import pandas as pd

from pneumonia_ai.segmentation.splitting import create_segmentation_splits


def _write_metadata(tmp_path: Path, sample_count: int) -> Path:
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {
                "image_id": f"sample-{index:03d}",
                "image_path": tmp_path / f"image-{index:03d}.png",
                "merged_mask_path": tmp_path / f"mask-{index:03d}.png",
            }
            for index in range(sample_count)
        ]
    ).to_csv(metadata_path, index=False)
    return metadata_path


def test_splits_are_deterministic_sized_and_non_overlapping(tmp_path: Path) -> None:
    metadata_path = _write_metadata(tmp_path, sample_count=20)

    first = create_segmentation_splits(metadata_path, tmp_path / "first")
    second = create_segmentation_splits(metadata_path, tmp_path / "second")

    assert {name: len(split) for name, split in first.items()} == {
        "train": 14,
        "validation": 3,
        "test": 3,
    }
    for name in first:
        assert first[name].equals(second[name])
        assert (tmp_path / "first" / f"{name}.csv").is_file()

    identifiers = [
        image_id for split in first.values() for image_id in split["image_id"].tolist()
    ]
    assert len(identifiers) == len(set(identifiers)) == 20
