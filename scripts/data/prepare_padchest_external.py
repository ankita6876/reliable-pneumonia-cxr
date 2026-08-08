"""Schema-driven PadChest external-manifest preparation; never runs inference."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

REQUIRED_MANIFEST_COLUMNS = (
    "patient_id", "image_path", "binary_target", "has_bounding_box",
    "bounding_box_count", "split", "dataset",
)


def parse_label_value(value: object, delimiter: str) -> list[str] | None:
    """Parse an explicitly selected report-level annotation value, never guessing missing labels."""

    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return None
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            return None
        return [item.strip() for item in parsed if item.strip()]
    return [item.strip() for item in text.split(delimiter) if item.strip()]


def normalize_concept(value: str) -> str:
    return " ".join(value.casefold().split())


def build_manifest(
    metadata: pd.DataFrame,
    image_root: Path,
    *,
    image_path_column: str,
    label_column: str,
    pneumonia_concepts: list[str],
    patient_id_column: str | None = None,
    case_id_column: str | None = None,
    projection_column: str | None = None,
    accepted_views: list[str] | None = None,
    label_delimiter: str = "|",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a portable, image-level pneumonia manifest from reviewed schema arguments."""

    required = {image_path_column, label_column}
    optional = {column for column in (patient_id_column, case_id_column, projection_column) if column}
    missing = sorted((required | optional) - set(metadata.columns))
    if missing:
        raise ValueError(f"Metadata missing selected column(s): {', '.join(missing)}")
    concepts = {normalize_concept(value) for value in pneumonia_concepts if value.strip()}
    if not concepts:
        raise ValueError("At least one explicit --pneumonia-concept is required; substring matching is not used.")
    accepted = {normalize_concept(value) for value in accepted_views or []}
    rows: list[dict[str, Any]] = []
    exclusions: dict[str, int] = {"missing_or_unparseable_label": 0, "excluded_view": 0, "missing_image": 0}
    for index, record in metadata.iterrows():
        labels = parse_label_value(record[label_column], label_delimiter)
        if labels is None:
            exclusions["missing_or_unparseable_label"] += 1
            continue
        view = "" if projection_column is None or pd.isna(record[projection_column]) else str(record[projection_column]).strip()
        if accepted and normalize_concept(view) not in accepted:
            exclusions["excluded_view"] += 1
            continue
        image_path = _portable_relative_path(record[image_path_column])
        source = image_root.joinpath(*PurePosixPath(image_path).parts)
        if not source.is_file():
            exclusions["missing_image"] += 1
            continue
        case_id = str(record[case_id_column]).strip() if case_id_column else image_path
        patient_id = str(record[patient_id_column]).strip() if patient_id_column else case_id
        if not case_id or not patient_id:
            raise ValueError(f"Missing case or patient identifier at metadata row {index}.")
        normalized_labels = {normalize_concept(value) for value in labels}
        rows.append({
            "patient_id": patient_id, "case_id": case_id, "image_path": image_path,
            "binary_target": int(bool(normalized_labels & concepts)), "has_bounding_box": False,
            "bounding_box_count": 0, "split": "external_test", "dataset": "padchest",
            "projection": view, "source_row": int(index),
        })
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("No eligible PadChest rows remain after explicit label/view/image checks.")
    if manifest.case_id.duplicated().any():
        raise ValueError("PadChest case/image identifiers must be unique.")
    if manifest.image_path.duplicated().any():
        raise ValueError("PadChest image paths must be unique.")
    if manifest.binary_target.nunique() != 2:
        raise ValueError("Eligible PadChest manifest must contain both pneumonia target classes.")
    summary = {
        "row_count_input": len(metadata), "row_count_eligible": len(manifest),
        "positive_count": int(manifest.binary_target.sum()), "negative_count": int((manifest.binary_target == 0).sum()),
        "exclusions": exclusions, "pneumonia_concepts": sorted(concepts),
        "image_path_column": image_path_column, "label_column": label_column,
        "patient_id_column": patient_id_column, "case_id_column": case_id_column,
        "projection_column": projection_column, "accepted_views": sorted(accepted),
        "label_delimiter": label_delimiter,
    }
    return manifest, summary


def schema_audit(metadata: pd.DataFrame, image_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Return inspection-only schema evidence; likely labels are never used as a mapping."""

    path_candidates = [column for column in metadata if re.search(r"(image|file|path)", column, re.IGNORECASE)]
    projection_candidates = [column for column in metadata if re.search(r"(projection|view|position)", column, re.IGNORECASE)]
    patient_candidates = [column for column in metadata if re.search(r"patient", column, re.IGNORECASE)]
    annotation_candidates = [column for column in metadata if re.search(r"(label|annotation|method|cui|report)", column, re.IGNORECASE)]
    path_column = args.image_path_column if args.image_path_column else (path_candidates[0] if len(path_candidates) == 1 else None)
    label_column = args.label_column
    examples = metadata[label_column].dropna().astype(str).head(10).tolist() if label_column in metadata else []
    tokens: list[str] = []
    if label_column in metadata:
        for value in metadata[label_column].dropna().head(5000):
            parsed = parse_label_value(value, args.label_delimiter)
            if parsed:
                tokens.extend(parsed)
    likely = sorted({token for token in tokens if "pneumonia" in token.casefold()})[:100]
    missing_paths: int | None = None
    if path_column:
        values = metadata[path_column]
        missing_paths = int(sum(not image_root.joinpath(*PurePosixPath(_portable_relative_path(value)).parts).is_file() for value in values.dropna()))
    projection = args.projection_column if args.projection_column else (projection_candidates[0] if len(projection_candidates) == 1 else None)
    return {
        "columns": list(metadata.columns), "row_count": len(metadata), "image_root": str(image_root),
        "candidate_image_path_columns": path_candidates, "selected_image_path_column": path_column,
        "missing_image_path_count": missing_paths, "candidate_projection_columns": projection_candidates,
        "selected_projection_column": projection,
        "unique_projection_values": sorted(metadata[projection].dropna().astype(str).unique().tolist()) if projection else [],
        "candidate_patient_id_columns": patient_candidates, "patient_identifier_available": bool(patient_candidates),
        "annotation_method_fields": annotation_candidates, "label_field_examples": examples,
        "candidate_pneumonia_related_labels": likely,
    }


def _portable_relative_path(value: object) -> str:
    if value is None or pd.isna(value):
        raise ValueError("Image path is missing.")
    raw = str(value).replace("\\", "/").strip()
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Image path must be safe and root-relative: {raw!r}")
    return str(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--image-path-column")
    parser.add_argument("--patient-id-column")
    parser.add_argument("--case-id-column")
    parser.add_argument("--projection-column")
    parser.add_argument("--accepted-view", action="append", default=[])
    parser.add_argument("--label-column")
    parser.add_argument("--label-delimiter", default="|")
    parser.add_argument("--pneumonia-concept", action="append", default=[])
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.metadata_csv.is_file() or not args.image_root.is_dir():
        raise FileNotFoundError("--metadata-csv and --image-root must exist.")
    metadata = pd.read_csv(args.metadata_csv)
    audit = schema_audit(metadata, args.image_root, args)
    print(json.dumps(audit, indent=2))
    if args.audit_only:
        return
    if args.output_manifest is None or not args.image_path_column or not args.label_column:
        raise ValueError("Full manifest creation requires --output-manifest, --image-path-column, and --label-column.")
    if args.output_manifest.exists() and not args.overwrite:
        raise FileExistsError(f"Output manifest exists: {args.output_manifest}")
    manifest, summary = build_manifest(metadata, args.image_root, image_path_column=args.image_path_column,
                                       label_column=args.label_column, pneumonia_concepts=args.pneumonia_concept,
                                       patient_id_column=args.patient_id_column, case_id_column=args.case_id_column,
                                       projection_column=args.projection_column, accepted_views=args.accepted_view,
                                       label_delimiter=args.label_delimiter)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_manifest, index=False)
    args.output_manifest.with_name(args.output_manifest.stem + "_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
