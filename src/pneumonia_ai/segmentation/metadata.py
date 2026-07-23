"""Metadata structures for the Montgomery lung-segmentation dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


METADATA_COLUMNS = (
    "image_id",
    "image_path",
    "left_mask_path",
    "right_mask_path",
    "merged_mask_path",
    "width",
    "height",
)


@dataclass(frozen=True)
class MontgomeryMetadataRecord:
    """One image and its paired lung-segmentation masks."""

    image_id: str
    image_path: str
    left_mask_path: str
    right_mask_path: str
    merged_mask_path: str
    width: int
    height: int

    @classmethod
    def from_paths(
        cls,
        *,
        image_path: Path,
        left_mask_path: Path,
        right_mask_path: Path,
        merged_mask_path: Path,
        width: int,
        height: int,
    ) -> "MontgomeryMetadataRecord":
        """Create a record with normalized absolute path strings."""

        return cls(
            image_id=image_path.stem,
            image_path=str(image_path.resolve()),
            left_mask_path=str(left_mask_path.resolve()),
            right_mask_path=str(right_mask_path.resolve()),
            merged_mask_path=str(merged_mask_path.resolve()),
            width=width,
            height=height,
        )


def build_metadata_dataframe(
    records: Iterable[MontgomeryMetadataRecord],
) -> pd.DataFrame:
    """Build a consistently ordered Montgomery metadata dataframe."""

    dataframe = pd.DataFrame(
        (asdict(record) for record in records), columns=METADATA_COLUMNS
    )
    return dataframe.sort_values("image_id", kind="stable").reset_index(drop=True)
