"""PyTorch dataset for split CheXpert pneumonia cohort manifests."""

from pathlib import Path, PurePosixPath
from typing import Callable

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


VALID_SPLITS = frozenset({"train", "validation", "test"})
VALID_LABELS = frozenset({1, 0, -1})
REQUIRED_MANIFEST_COLUMNS = (
    "split",
    "patient_id",
    "study_id",
    "image_path",
    "pneumonia_label",
)


class DatasetValidationError(ValueError):
    """Raised when a split manifest cannot be used as a CheXpert dataset."""


class CheXpertPneumoniaDataset(Dataset[dict[str, object]]):
    """Read one CheXpert split without changing its original pneumonia labels."""

    def __init__(
        self,
        dataset_root: Path | str,
        manifest_path: Path | str,
        split: str,
        transform: Callable[[Image.Image], object] | None = None,
    ) -> None:
        """Load one manifest split and validate its image references.

        The dataset applies only the caller-provided transform. It intentionally
        performs no label remapping, augmentation, batching, or data loading.
        """
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"CheXpert dataset root does not exist: {self.dataset_root}")
        if not self.dataset_root.is_dir():
            raise NotADirectoryError(
                f"CheXpert dataset root is not a directory: {self.dataset_root}"
            )

        self.manifest_path = Path(manifest_path).expanduser()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Split manifest does not exist: {self.manifest_path}")
        if split not in VALID_SPLITS:
            expected = ", ".join(sorted(VALID_SPLITS))
            raise DatasetValidationError(
                f"Invalid split {split!r}. Expected one of: {expected}."
            )

        manifest = pd.read_csv(self.manifest_path)
        missing_columns = [
            column for column in REQUIRED_MANIFEST_COLUMNS if column not in manifest
        ]
        if missing_columns:
            raise DatasetValidationError(
                "Split manifest is missing required column(s): "
                f"{', '.join(missing_columns)}"
            )

        records = manifest.loc[manifest["split"] == split].copy()
        if records.empty:
            raise DatasetValidationError(f"Requested split {split!r} contains no rows.")
        labels = pd.to_numeric(records["pneumonia_label"], errors="coerce")
        invalid_labels = labels.isna() | ~labels.isin(VALID_LABELS)
        if invalid_labels.any():
            invalid_values = records.loc[invalid_labels, "pneumonia_label"].unique().tolist()
            raise DatasetValidationError(
                "Split manifest contains invalid Pneumonia labels; expected only 1, 0, or -1. "
                f"Found: {invalid_values}"
            )

        self.records = records.reset_index(drop=True)
        self._labels = labels.astype("int64").reset_index(drop=True)
        self._resolved_image_paths = [
            _resolve_manifest_image_path(self.dataset_root, image_path)
            for image_path in self.records["image_path"]
        ]
        for image_path in self._resolved_image_paths:
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image file referenced by manifest does not exist: {image_path}"
                )
        self._returned_image_paths = [
            image_path.relative_to(self.dataset_root).as_posix()
            for image_path in self._resolved_image_paths
        ]
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of rows in the requested split."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        """Load one image, convert it to RGB, and return its manifest metadata."""
        row = self.records.iloc[index]
        with Image.open(self._resolved_image_paths[index]) as opened_image:
            image = opened_image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": torch.tensor(float(self._labels.iloc[index]), dtype=torch.float32),
            "patient_id": row["patient_id"],
            "study_id": row["study_id"],
            "image_path": self._returned_image_paths[index],
        }


def _resolve_manifest_image_path(dataset_root: Path, image_path: object) -> Path:
    """Resolve a portable root-relative manifest path using native filesystem paths."""
    manifest_path = PurePosixPath(str(image_path))
    if (
        manifest_path.is_absolute()
        or not manifest_path.parts
        or ".." in manifest_path.parts
    ):
        raise DatasetValidationError(
            "Manifest image_path must be a relative POSIX path: " f"{image_path!r}"
        )
    return dataset_root.joinpath(*manifest_path.parts)
