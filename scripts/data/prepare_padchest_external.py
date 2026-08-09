"""Schema-driven PadChest manifest preparation; this script never runs inference."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

PNEUMONIA_TOKEN = "pneumonia"
CANDIDATE_PROJECTION_SETS = {
    "PA only": ("PA",),
    "PA + AP": ("PA", "AP"),
    "PA + AP + AP_horizontal": ("PA", "AP", "AP_horizontal"),
}
REQUIRED_MANIFEST_COLUMNS = (
    "patient_id", "image_path", "binary_target", "has_bounding_box",
    "bounding_box_count", "split", "dataset",
)


def normalize_concept(value: str) -> str:
    """Return a case/whitespace-normalized token for exact comparison."""

    return " ".join(value.casefold().split())


def parse_label_value(value: object) -> list[str] | None:
    """Safely parse a non-empty report-level Python list of label strings."""

    if value is None or (not isinstance(value, (list, tuple)) and pd.isna(value)):
        return None
    if isinstance(value, (list, tuple)):
        parsed = value
    else:
        text = str(value).strip()
        if not text or text.casefold() in {"nan", "none", "null"}:
            return None
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(parsed, (list, tuple)) or not all(isinstance(item, str) for item in parsed):
        return None
    labels = [normalize_concept(item) for item in parsed if item.strip()]
    return labels or None


def _value(record: pd.Series, column: str | None) -> str:
    if not column or column not in record or pd.isna(record[column]):
        return ""
    return str(record[column]).strip()


def _portable_relative_path(value: object) -> str:
    if value is None or pd.isna(value):
        raise ValueError("Image path is missing.")
    raw = str(value).replace("\\", "/").strip()
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Image path must be safe and root-relative: {raw!r}")
    return str(path)


def image_relative_path(
    record: pd.Series, *, image_path_column: str | None,
    image_id_column: str | None, image_dir_column: str | None,
) -> str:
    """Use a reviewed path field, or the verified ImageDir/ImageID representation."""

    if image_path_column:
        return _portable_relative_path(record[image_path_column])
    image_id, image_dir = _value(record, image_id_column), _value(record, image_dir_column)
    if not image_id or not image_dir:
        raise ValueError("Select --image-path-column or both --image-id-column and --image-dir-column.")
    return _portable_relative_path(str(PurePosixPath(image_dir) / image_id))


def _selected_columns(args: argparse.Namespace, metadata: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    """Use only explicit names, except for the verified PadChest field names."""

    label = args.label_column or ("Labels" if "Labels" in metadata else None)
    projection = args.projection_column or ("Projection" if "Projection" in metadata else None)
    patient = args.patient_id_column or ("PatientID" if "PatientID" in metadata else None)
    return label, projection, patient


def _require_columns(metadata: pd.DataFrame, columns: list[str | None]) -> None:
    missing = [column for column in columns if column and column not in metadata]
    if missing:
        raise ValueError(f"Metadata missing selected column(s): {', '.join(missing)}")


def _labelled_rows(
    metadata: pd.DataFrame, *, label_column: str,
    projection_column: str | None = None, accepted_views: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    accepted = {normalize_concept(item) for item in accepted_views or []}
    rows: list[dict[str, Any]] = []
    exclusions = {"missing_or_unparseable_label": 0, "excluded_projection": 0}
    for index, record in metadata.iterrows():
        labels = parse_label_value(record[label_column])
        if labels is None:
            exclusions["missing_or_unparseable_label"] += 1
            continue
        projection = _value(record, projection_column)
        if accepted and normalize_concept(projection) not in accepted:
            exclusions["excluded_projection"] += 1
            continue
        rows.append({"source_row": int(index), "labels": labels, "projection": projection,
                     "pneumonia_target": int(PNEUMONIA_TOKEN in set(labels))})
    return rows, exclusions


def cohort_summary(rows: list[dict[str, Any]], metadata: pd.DataFrame, patient_id_column: str | None) -> dict[str, float | int | None]:
    targets = [row["pneumonia_target"] for row in rows]
    patients = {_value(metadata.iloc[row["source_row"]], patient_id_column) for row in rows}
    patients.discard("")
    positive = int(sum(targets))
    return {"total_images": len(rows), "unique_patients": len(patients), "pneumonia_positives": positive,
            "pneumonia_negatives": len(rows) - positive,
            "positive_prevalence": float(positive / len(rows)) if rows else None}


def build_cohort_request(
    metadata: pd.DataFrame, *, image_id_column: str, image_dir_column: str,
    patient_id_column: str, projection_column: str, label_column: str,
    method_label_column: str | None = None, accepted_views: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a pre-image-acquisition request CSV from full PadChest metadata."""

    _require_columns(metadata, [image_id_column, image_dir_column, patient_id_column, projection_column, label_column, method_label_column])
    labelled, exclusions = _labelled_rows(metadata, label_column=label_column, projection_column=projection_column, accepted_views=accepted_views)
    rows = []
    for item in labelled:
        record = metadata.iloc[item["source_row"]]
        rows.append({"ImageID": _value(record, image_id_column), "ImageDir": _value(record, image_dir_column),
                     "PatientID": _value(record, patient_id_column), "Projection": item["projection"],
                     "pneumonia_target": item["pneumonia_target"],
                     "MethodLabel": _value(record, method_label_column)})
    cohort = pd.DataFrame(rows)
    if cohort.ImageID.duplicated().any():
        raise ValueError("Cohort request has duplicate ImageID values.")
    return cohort, {"label_policy": "exact normalized Labels token == pneumonia", "exclusions": exclusions,
                    "cohort": cohort_summary(labelled, metadata, patient_id_column), "accepted_projections": accepted_views or []}


def build_manifest(
    metadata: pd.DataFrame, image_root: Path, *, image_path_column: str | None,
    image_id_column: str | None, image_dir_column: str | None, label_column: str,
    pneumonia_concepts: list[str], patient_id_column: str | None = None,
    case_id_column: str | None = None, projection_column: str | None = None,
    method_label_column: str | None = None, accepted_views: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build the predictor manifest after image acquisition; preserves extra provenance."""

    _require_columns(metadata, [image_path_column, image_id_column, image_dir_column, label_column,
                                patient_id_column, case_id_column, projection_column, method_label_column])
    concepts = {normalize_concept(value) for value in pneumonia_concepts if value.strip()}
    if not concepts:
        raise ValueError("At least one explicit --pneumonia-concept is required; substring matching is not used.")
    labelled, exclusions = _labelled_rows(metadata, label_column=label_column, projection_column=projection_column, accepted_views=accepted_views)
    rows, missing_rows = [], []
    exclusions["missing_image"] = 0
    for item in labelled:
        record = metadata.iloc[item["source_row"]]
        path = image_relative_path(record, image_path_column=image_path_column, image_id_column=image_id_column, image_dir_column=image_dir_column)
        case_id = _value(record, case_id_column) or path
        patient_id = _value(record, patient_id_column) or case_id
        if not case_id or not patient_id:
            raise ValueError(f"Missing case or patient identifier at metadata row {item['source_row']}.")
        provenance = {"patient_id": patient_id, "case_id": case_id, "image_path": path,
                      "binary_target": int(bool(set(item["labels"]) & concepts)),
                      "projection": item["projection"], "method_label": _value(record, method_label_column),
                      "source_row": item["source_row"], "labels": "|".join(item["labels"]),
                      "label_cuis": _value(record, "labelCUIS") if "labelCUIS" in metadata else ""}
        if not image_root.joinpath(*PurePosixPath(path).parts).is_file():
            exclusions["missing_image"] += 1
            missing_rows.append({**provenance, "exclusion_reason": "image_unavailable_in_image_root"})
            continue
        rows.append({"patient_id": patient_id, "case_id": case_id, "image_path": path,
                     "binary_target": provenance["binary_target"],
                     "has_bounding_box": False, "bounding_box_count": 0, "split": "external_test",
                     "dataset": "padchest", "projection": item["projection"], "source_row": item["source_row"],
                     "method_label": provenance["method_label"], "label_cuis": provenance["label_cuis"]})
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("No eligible PadChest rows remain after explicit label/projection/image checks.")
    if manifest.case_id.duplicated().any() or manifest.image_path.duplicated().any():
        raise ValueError("PadChest case/image identifiers must be unique.")
    if manifest.binary_target.nunique() != 2:
        raise ValueError("Eligible PadChest manifest must contain both pneumonia target classes.")
    missing = pd.DataFrame(missing_rows)
    intended = cohort_summary(labelled, metadata, patient_id_column)
    evaluable = cohort_summary(
        [{"source_row": int(row.source_row), "pneumonia_target": int(row.binary_target)} for row in manifest.itertuples()],
        metadata, patient_id_column,
    )
    return manifest, missing, {"row_count_input": len(metadata), "intended_cohort": intended,
                      "evaluable_cohort": evaluable, "intended_cohort_count": len(labelled),
                      "available_evaluable_count": len(manifest), "missing_image_count": len(missing),
                      "row_count_eligible": len(manifest),
                      "positive_count": int(manifest.binary_target.sum()), "negative_count": int((manifest.binary_target == 0).sum()),
                      "exclusions": exclusions, "pneumonia_concepts": sorted(concepts),
                      "label_column": label_column, "projection_column": projection_column,
                      "accepted_views": accepted_views or [], "label_cuis_retained_for_provenance": "labelCUIS" in metadata}


def schema_audit(metadata: pd.DataFrame, image_root: Path | None, args: argparse.Namespace) -> dict[str, Any]:
    """Produce a full-metadata audit; candidate tokens are never label mappings."""

    label, projection, patient = _selected_columns(args, metadata)
    method = args.method_label_column or ("MethodLabel" if "MethodLabel" in metadata else None)
    cuis = args.label_cuis_column or ("labelCUIS" if "labelCUIS" in metadata else None)
    image_id = args.image_id_column or ("ImageID" if "ImageID" in metadata else None)
    image_dir = args.image_dir_column or ("ImageDir" if "ImageDir" in metadata else None)
    valid, exclusions = (
        _labelled_rows(metadata, label_column=label, projection_column=projection)
        if label
        else ([], {"missing_or_unparseable_label": len(metadata)})
    )
    pneumonia_rows = [item for item in valid if item["pneumonia_target"]]
    all_tokens = Counter(token for item in valid for token in item["labels"])
    by_projection = Counter(item["projection"] for item in pneumonia_rows) if projection else Counter()
    by_method = Counter(_value(metadata.iloc[item["source_row"]], method) for item in pneumonia_rows) if method else Counter()
    cui_values = sorted({_value(metadata.iloc[item["source_row"]], cuis) for item in pneumonia_rows if _value(metadata.iloc[item["source_row"]], cuis)}) if cuis else []
    duplicate_images = int(metadata[image_id].duplicated().sum()) if image_id else None
    duplicate_pairs = int(metadata.duplicated([patient, image_id]).sum()) if patient and image_id else None
    candidates = {}
    for name, views in CANDIDATE_PROJECTION_SETS.items():
        selected, _ = _labelled_rows(metadata, label_column=label, projection_column=projection, accepted_views=list(views)) if label and projection else ([], {})
        candidates[name] = cohort_summary(selected, metadata, patient)
    missing_paths = None
    if image_root and image_root.is_dir() and (args.image_path_column or (image_id and image_dir)):
        missing_paths = 0
        for _, record in metadata.iterrows():
            try:
                path = image_relative_path(record, image_path_column=args.image_path_column, image_id_column=image_id, image_dir_column=image_dir)
                missing_paths += int(not image_root.joinpath(*PurePosixPath(path).parts).is_file())
            except ValueError:
                missing_paths += 1
    return {"columns": list(metadata.columns), "total_rows": len(metadata), "unique_images": int(metadata[image_id].nunique()) if image_id else None,
            "unique_patients": int(metadata[patient].nunique()) if patient else None, "duplicate_image_id_count": duplicate_images,
            "duplicate_patient_image_count": duplicate_pairs, "primary_label_column": label, "primary_projection_column": projection,
            "method_label_column": method, "label_cuis_column": cuis, "image_root": str(image_root) if image_root else None,
            "missing_image_path_count": missing_paths, "projection_distribution": metadata[projection].value_counts(dropna=False).to_dict() if projection else {},
            "method_label_distribution": metadata[method].value_counts(dropna=False).to_dict() if method else {},
            "valid_parsed_labels": len(valid), "excluded_missing_or_unparseable_labels": exclusions["missing_or_unparseable_label"],
            "exact_pneumonia_positive_images_before_view_filtering": len(pneumonia_rows),
            "exact_pneumonia_positive_patients": len({_value(metadata.iloc[x["source_row"]], patient) for x in pneumonia_rows} - {""}),
            "exact_pneumonia_positive_by_projection": dict(by_projection), "exact_pneumonia_positive_by_method_label": dict(by_method),
            "normalized_labels_containing_pneum_audit_only": sorted(token for token in all_tokens if "pneum" in token),
            "label_cuis_on_exact_pneumonia_rows": cui_values, "candidate_frontal_cohorts": candidates}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True); parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output-manifest", type=Path); parser.add_argument("--cohort-request-csv", type=Path)
    parser.add_argument("--exclusion-output", type=Path, help="CSV recording eligible images unavailable under --image-root.")
    parser.add_argument("--image-path-column"); parser.add_argument("--image-id-column"); parser.add_argument("--image-dir-column")
    parser.add_argument("--patient-id-column"); parser.add_argument("--case-id-column"); parser.add_argument("--projection-column")
    parser.add_argument("--method-label-column"); parser.add_argument("--label-cuis-column"); parser.add_argument("--accepted-view", action="append", default=[])
    parser.add_argument("--label-column"); parser.add_argument("--pneumonia-concept", action="append", default=[])
    parser.add_argument("--audit-only", action="store_true"); parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.metadata_csv.is_file():
        raise FileNotFoundError("--metadata-csv must exist.")
    if args.image_root is not None and not args.image_root.is_dir():
        raise FileNotFoundError("--image-root must be a directory when supplied.")
    metadata = pd.read_csv(args.metadata_csv)
    print(json.dumps(schema_audit(metadata, args.image_root, args), indent=2, allow_nan=False))
    if args.audit_only:
        return
    label, projection, patient = _selected_columns(args, metadata)
    image_id = args.image_id_column or ("ImageID" if "ImageID" in metadata else None)
    image_dir = args.image_dir_column or ("ImageDir" if "ImageDir" in metadata else None)
    if args.cohort_request_csv:
        if args.cohort_request_csv.exists() and not args.overwrite: raise FileExistsError(args.cohort_request_csv)
        if not all((image_id, image_dir, patient, projection, label)): raise ValueError("Cohort request requires ImageID, ImageDir, PatientID, Projection, and Labels columns.")
        cohort, summary = build_cohort_request(metadata, image_id_column=image_id, image_dir_column=image_dir, patient_id_column=patient, projection_column=projection, label_column=label, method_label_column=args.method_label_column or ("MethodLabel" if "MethodLabel" in metadata else None), accepted_views=args.accepted_view)
        args.cohort_request_csv.parent.mkdir(parents=True, exist_ok=True); cohort.to_csv(args.cohort_request_csv, index=False)
        args.cohort_request_csv.with_name(args.cohort_request_csv.stem + "_metadata.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    if args.output_manifest:
        if args.image_root is None: raise ValueError("--output-manifest requires --image-root.")
        if args.output_manifest.exists() and not args.overwrite: raise FileExistsError(args.output_manifest)
        manifest, missing, summary = build_manifest(metadata, args.image_root, image_path_column=args.image_path_column, image_id_column=image_id, image_dir_column=image_dir, label_column=label or "", pneumonia_concepts=args.pneumonia_concept, patient_id_column=patient, case_id_column=args.case_id_column or image_id, projection_column=projection, method_label_column=args.method_label_column or ("MethodLabel" if "MethodLabel" in metadata else None), accepted_views=args.accepted_view)
        exclusion_output = args.exclusion_output or args.output_manifest.with_name(args.output_manifest.stem + "_image_availability_exclusions.csv")
        if exclusion_output.exists() and not args.overwrite: raise FileExistsError(exclusion_output)
        args.output_manifest.parent.mkdir(parents=True, exist_ok=True); exclusion_output.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(args.output_manifest, index=False)
        missing.to_csv(exclusion_output, index=False)
        summary["image_availability_exclusion_output"] = str(exclusion_output)
        args.output_manifest.with_name(args.output_manifest.stem + "_metadata.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    if not args.cohort_request_csv and not args.output_manifest:
        raise ValueError("Use --audit-only, --cohort-request-csv, or --output-manifest.")


if __name__ == "__main__":
    main()
