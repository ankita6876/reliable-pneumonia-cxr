"""RSNA Pneumonia Detection Challenge external-validation manifest utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


class RSNAValidationError(ValueError):
    """Raised when RSNA annotations or image references are inconsistent."""


@dataclass(frozen=True)
class RSNACohortSummary:
    annotation_rows: int
    unique_patients: int
    positive_patients: int
    negative_patients: int
    prevalence: float
    duplicate_annotation_rows: int
    missing_images: int
    duplicate_image_paths: int

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def build_rsna_manifest(
    labels_path: Path | str, image_root: Path | str, *, allow_missing_images: bool = False
) -> tuple[pd.DataFrame, RSNACohortSummary]:
    """Collapse RSNA box annotations to one portable, binary-labelled row per patient."""
    labels = pd.read_csv(labels_path)
    required = {"patientId", "Target"}
    if missing := required.difference(labels.columns):
        raise RSNAValidationError(f"RSNA labels missing required column(s): {', '.join(sorted(missing))}")
    target = pd.to_numeric(labels["Target"], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise RSNAValidationError("RSNA Target must contain only binary values 0 and 1.")
    if labels["patientId"].isna().any() or (labels["patientId"].astype(str).str.strip() == "").any():
        raise RSNAValidationError("RSNA patientId must be present for every annotation row.")
    prepared = labels.assign(patientId=labels["patientId"].astype(str), _target=target.astype(int))
    conflicts = prepared.groupby("patientId")["_target"].agg(lambda values: values.nunique() > 1)
    if conflicts.any():
        raise RSNAValidationError("RSNA annotations contain conflicting Target labels for a patient.")
    collapsed = prepared.groupby("patientId", as_index=False)["_target"].max()
    root = Path(image_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"RSNA image root does not exist or is not a directory: {root}")
    paths = [root / f"{patient_id}.dcm" for patient_id in collapsed["patientId"]]
    missing_images = sum(not path.is_file() for path in paths)
    if missing_images and not allow_missing_images:
        raise FileNotFoundError(f"{missing_images} RSNA DICOM image(s) are missing under {root}.")
    manifest = pd.DataFrame({
        "patient_id": collapsed["patientId"], "image_id": collapsed["patientId"],
        "study_id": collapsed["patientId"],
        "image_path": [path.relative_to(root).as_posix() for path in paths],
        "label": collapsed["_target"].astype(int), "pneumonia_label": collapsed["_target"].astype(int),
        "split": "external_test", "dataset": "rsna",
    })
    if manifest["patient_id"].duplicated().any() or manifest["image_path"].duplicated().any():
        raise RSNAValidationError("RSNA manifest must have unique patient IDs and image paths.")
    summary = RSNACohortSummary(
        annotation_rows=len(labels), unique_patients=len(manifest),
        positive_patients=int(manifest["label"].sum()), negative_patients=int((manifest["label"] == 0).sum()),
        prevalence=float(manifest["label"].mean()), duplicate_annotation_rows=int(len(labels) - len(manifest)),
        missing_images=missing_images, duplicate_image_paths=int(manifest["image_path"].duplicated().sum()),
    )
    return manifest, summary
