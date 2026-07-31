"""PyTorch dataset for split CheXpert pneumonia cohort manifests."""

from pathlib import Path, PurePosixPath
from typing import Callable

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from pneumonia_ai.classification.segmentation_guided import InputMode, prepare_classifier_image
from pneumonia_ai.segmentation.cache import MaskCache
from pneumonia_ai.segmentation.inference import FrozenLungSegmenter


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
        input_mode: InputMode | str = InputMode.ORIGINAL,
        lung_segmenter: FrozenLungSegmenter | None = None,
        mask_cache: MaskCache | None = None,
        mask_threshold: float = 0.5,
        lung_crop_padding: int = 0,
        classifier_image_size: int | tuple[int, int] | None = None,
        allow_absolute_image_paths: bool = False,
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
        raw_label_column = (
            "raw_pneumonia_label" if "raw_pneumonia_label" in records else "pneumonia_label"
        )
        raw_labels = pd.to_numeric(records[raw_label_column], errors="coerce")
        invalid_labels = raw_labels.isna() | ~raw_labels.isin(VALID_LABELS)
        if invalid_labels.any():
            invalid_values = records.loc[invalid_labels, raw_label_column].unique().tolist()
            raise DatasetValidationError(
                "Split manifest contains invalid Pneumonia labels; expected only 1, 0, or -1. "
                f"Found: {invalid_values}"
            )

        self.records = records.reset_index(drop=True)
        self._raw_labels = raw_labels.astype("int64").reset_index(drop=True)
        has_training_targets = "training_target" in records
        target_values = records["training_target"] if has_training_targets else records["pneumonia_label"]
        targets = pd.to_numeric(target_values, errors="coerce")
        if targets.isna().any() or (has_training_targets and not targets.between(0.0, 1.0).all()):
            raise DatasetValidationError("Training targets must be numeric values between 0 and 1.")
        weight_values = (
            records["sample_loss_weight"]
            if "sample_loss_weight" in records
            else pd.Series(1.0, index=records.index)
        )
        weights = pd.to_numeric(weight_values, errors="coerce")
        if weights.isna().any() or (weights <= 0.0).any():
            raise DatasetValidationError("Sample loss weights must be positive numeric values.")
        self._targets = targets.astype("float32").reset_index(drop=True)
        self._sample_loss_weights = weights.astype("float32").reset_index(drop=True)
        self._resolved_image_paths = [
            _resolve_manifest_image_path(
                self.dataset_root, image_path, allow_absolute=allow_absolute_image_paths
            )
            for image_path in self.records["image_path"]
        ]
        for image_path in self._resolved_image_paths:
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image file referenced by manifest does not exist: {image_path}"
                )
        self._returned_image_paths = [
            _returned_image_path(image_path, self.dataset_root)
            for image_path in self._resolved_image_paths
        ]
        self.transform = transform
        try:
            self.input_mode = InputMode(input_mode)
        except ValueError as error:
            raise DatasetValidationError("input_mode must be original, hard_masked, or lung_crop.") from error
        if self.input_mode is not InputMode.ORIGINAL and lung_segmenter is None and mask_cache is None:
            raise DatasetValidationError(
                "hard_masked and lung_crop input modes require lung_segmenter or mask_cache."
            )
        self.lung_segmenter = lung_segmenter
        self.mask_cache = mask_cache
        self.mask_threshold = mask_threshold
        self.lung_crop_padding = lung_crop_padding
        self.classifier_image_size = classifier_image_size

    def __len__(self) -> int:
        """Return the number of rows in the requested split."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        """Load one image, convert it to RGB, and return its manifest metadata."""
        row = self.records.iloc[index]
        with Image.open(self._resolved_image_paths[index]) as opened_image:
            image = opened_image.convert("RGB")
        if self.input_mode is not InputMode.ORIGINAL or self.classifier_image_size is not None:
            probability_mask = self._probability_mask(index, image)
            image = prepare_classifier_image(
                image, self.input_mode, segmenter=self.lung_segmenter,
                threshold=self.mask_threshold, crop_padding=self.lung_crop_padding,
                output_size=self.classifier_image_size, probability_mask=probability_mask,
            )
        if self.input_mode is not InputMode.ORIGINAL:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": torch.tensor(float(self._targets.iloc[index]), dtype=torch.float32),
            "target": torch.tensor(float(self._targets.iloc[index]), dtype=torch.float32),
            "sample_weight": torch.tensor(
                float(self._sample_loss_weights.iloc[index]), dtype=torch.float32
            ),
            "raw_label": torch.tensor(float(self._raw_labels.iloc[index]), dtype=torch.float32),
            "patient_id": row["patient_id"],
            "study_id": row["study_id"],
            "image_path": self._returned_image_paths[index],
        }

    def _probability_mask(self, index: int, image: Image.Image) -> torch.Tensor | None:
        if self.input_mode is InputMode.ORIGINAL:
            return None
        if self.mask_cache is None:
            assert self.lung_segmenter is not None
            return self.lung_segmenter.predict_proba(image)
        if self.lung_segmenter is None:
            mask = self.mask_cache.get_for_source(self._resolved_image_paths[index])
            if mask is None:
                raise DatasetValidationError(
                    "Mask cache lacks a valid mask for the requested hard-masked image."
                )
            return mask
        key = self.mask_cache.key(
            self._resolved_image_paths[index],
            self.lung_segmenter.checkpoint_path,
            self.mask_threshold,
            self.lung_segmenter.image_size,
        )
        mask = self.mask_cache.get(key)
        if mask is None:
            mask = self.lung_segmenter.predict_proba(image)
            self.mask_cache.set(key, mask, source_path=self._resolved_image_paths[index])
        return mask


def _resolve_manifest_image_path(
    dataset_root: Path, image_path: object, *, allow_absolute: bool = False
) -> Path:
    """Resolve a portable root-relative manifest path using native filesystem paths."""
    native_path = Path(str(image_path)).expanduser()
    if native_path.is_absolute():
        if allow_absolute:
            return native_path.resolve()
        raise DatasetValidationError(
            "Manifest image_path must be a relative POSIX path: " f"{image_path!r}"
        )
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


def _returned_image_path(image_path: Path, dataset_root: Path) -> str:
    """Keep portable paths when possible, retaining permitted absolute paths otherwise."""

    try:
        return image_path.relative_to(dataset_root).as_posix()
    except ValueError:
        return str(image_path)
