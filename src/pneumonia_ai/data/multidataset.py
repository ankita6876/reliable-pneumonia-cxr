"""Portable manifest adapters for CheXpert, NIH ChestX-ray14, and MIMIC-CXR."""

from pathlib import Path, PurePosixPath
from typing import Callable

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from pneumonia_ai.data.dicom import dicom_to_pil

STANDARD_COLUMNS = ("dataset", "split", "patient_id", "study_id", "image_path", "pneumonia_label")
SUPPORTED_DATASETS = frozenset({"chexpert", "nih", "mimic"})
SUPPORTED_SPLITS = frozenset({"train", "validation", "test", "external", "external_test"})


class MultiDatasetValidationError(ValueError):
    """Raised when a portable multi-dataset manifest is invalid."""


def build_standard_manifest(dataset: str, metadata_path: Path | str, root: Path | str, split: str = "external") -> pd.DataFrame:
    """Parse one supported source into a privacy-safe, root-relative binary manifest."""
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}; expected chexpert, nih, or mimic.")
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported split {split!r}.")
    metadata = pd.read_csv(metadata_path)
    if dataset == "chexpert":
        result = _parse_chexpert(metadata)
    elif dataset == "nih":
        result = _parse_nih(metadata)
    else:
        result = _parse_mimic(metadata)
    result.insert(0, "dataset", dataset)
    result.insert(1, "split", split)
    _validate_standard_manifest(result, root)
    return result.loc[:, STANDARD_COLUMNS]


class UnifiedPneumoniaDataset(Dataset[dict[str, object]]):
    """Read any standard manifest with the exact sample keys used by CheXpert training."""

    def __init__(self, root: Path | str, manifest_path: Path | str, split: str, transform: Callable[[Image.Image], object] | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest = pd.read_csv(manifest_path)
        if split not in SUPPORTED_SPLITS:
            raise MultiDatasetValidationError(f"Unsupported split {split!r}.")
        _validate_standard_manifest(manifest, self.root)
        self.records = manifest.loc[manifest["split"] == split].reset_index(drop=True)
        if self.records.empty:
            raise MultiDatasetValidationError(f"Requested split {split!r} contains no rows.")
        self.paths = [self.root.joinpath(*PurePosixPath(path).parts) for path in self.records["image_path"]]
        self.transform = transform

    def __len__(self) -> int: return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.records.iloc[index]
        if self.paths[index].suffix.lower() == ".dcm":
            image = dicom_to_pil(self.paths[index])
        else:
            with Image.open(self.paths[index]) as source:
                image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = float(row["pneumonia_label"])
        return {"image": image, "label": torch.tensor(label), "target": torch.tensor(label), "raw_label": torch.tensor(label), "sample_weight": torch.tensor(1.0), "patient_id": row["patient_id"], "study_id": row["study_id"], "image_path": row["image_path"], "dataset": row["dataset"]}


def _parse_chexpert(frame: pd.DataFrame) -> pd.DataFrame:
    _require(frame, "Path", "Pneumonia")
    paths = frame["Path"].astype(str).str.replace("\\", "/", regex=False)
    return pd.DataFrame({"patient_id": paths.str.extract(r"(patient\d+)", expand=False), "study_id": paths.str.extract(r"/(study\d+)", expand=False), "image_path": paths, "pneumonia_label": pd.to_numeric(frame["Pneumonia"], errors="raise")}).dropna()


def _parse_nih(frame: pd.DataFrame) -> pd.DataFrame:
    _require(frame, "Image Index", "Finding Labels")
    patients = frame["Patient ID"].astype(str) if "Patient ID" in frame else frame["Image Index"].astype(str).str.extract(r"^(\d+)", expand=False).fillna(frame["Image Index"].astype(str))
    findings = frame["Finding Labels"].fillna("").astype(str)
    labels = findings.str.split("|").map(lambda values: int("Pneumonia" in values))
    return pd.DataFrame({"patient_id": patients, "study_id": frame["Image Index"].astype(str), "image_path": frame["Image Index"].astype(str), "pneumonia_label": labels})


def _parse_mimic(frame: pd.DataFrame) -> pd.DataFrame:
    _require(frame, "subject_id", "study_id", "Pneumonia")
    labels = pd.to_numeric(frame["Pneumonia"], errors="coerce")
    valid = labels.isin([0, 1])
    if "image_path" in frame:
        paths = frame["image_path"].astype(str)
    elif "path" in frame:
        paths = frame["path"].astype(str)
    elif "dicom_id" in frame:
        paths = "files/p" + frame["subject_id"].astype(str).str[:2] + "/p" + frame["subject_id"].astype(str) + "/s" + frame["study_id"].astype(str) + "/" + frame["dicom_id"].astype(str) + ".jpg"
    else:
        raise MultiDatasetValidationError("MIMIC metadata needs image_path, path, or dicom_id.")
    return pd.DataFrame({"patient_id": frame["subject_id"].astype(str), "study_id": frame["study_id"].astype(str), "image_path": paths.str.replace("\\", "/", regex=False), "pneumonia_label": labels}).loc[valid]


def _require(frame: pd.DataFrame, *columns: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise MultiDatasetValidationError(f"Metadata missing column(s): {', '.join(missing)}")


def _validate_standard_manifest(frame: pd.DataFrame, root: Path | str) -> None:
    missing = [column for column in STANDARD_COLUMNS if column not in frame]
    if missing:
        raise MultiDatasetValidationError(f"Manifest missing column(s): {', '.join(missing)}")
    labels = pd.to_numeric(frame["pneumonia_label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise MultiDatasetValidationError("External manifests require definite binary pneumonia labels.")
    root_path = Path(root)
    paths = frame["image_path"].astype(str)
    if paths.map(_is_unsafe_relative_path).any():
        raise MultiDatasetValidationError("image_path must be portable and root-relative.")
    missing_images = [path for path in paths if not root_path.joinpath(*PurePosixPath(path).parts).is_file()]
    if missing_images:
        raise FileNotFoundError(f"Manifest references missing image: {missing_images[0]}")


def _is_unsafe_relative_path(path: str) -> bool:
    """Return whether a manifest path is absolute or escapes its dataset root."""
    portable_path = PurePosixPath(path)
    return portable_path.is_absolute() or ".." in portable_path.parts
