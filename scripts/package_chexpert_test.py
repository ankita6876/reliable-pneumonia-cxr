from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import pandas as pd


SPLIT_CSV = Path(
    r"C:\Research\datasets\chexpert\processed"
    r"\chexpert_pneumonia_splits.csv"
)

EXTRACTED_ROOT = Path(
    r"C:\Research\datasets\chexpert\extracted"
)

OUTPUT_ROOT = Path(
    r"C:\Research\datasets\chexpert"
    r"\test_package"
)

ZIP_PATH = Path(
    r"C:\Research\datasets\chexpert"
    r"\chexpert_test_package.zip"
)


def resolve_image_path(
    relative_path: str,
) -> Path:
    normalized = Path(
        str(relative_path)
        .replace("\\", "/")
    )

    candidates = [
        EXTRACTED_ROOT / normalized,
        EXTRACTED_ROOT
        / "CheXpert-v1.0-small"
        / normalized,
        EXTRACTED_ROOT
        / "CheXpert-v1.0"
        / normalized,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(
        EXTRACTED_ROOT.rglob(normalized.name)
    )

    matching_suffixes = [
        path
        for path in matches
        if str(path)
        .replace("\\", "/")
        .endswith(str(normalized).replace("\\", "/"))
    ]

    if len(matching_suffixes) == 1:
        return matching_suffixes[0]

    raise FileNotFoundError(
        f"Could not resolve image: {relative_path}"
    )


def main() -> None:
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"Split CSV not found: {SPLIT_CSV}"
        )

    if not EXTRACTED_ROOT.exists():
        raise FileNotFoundError(
            f"Extracted root not found: "
            f"{EXTRACTED_ROOT}"
        )

    frame = pd.read_csv(SPLIT_CSV)

    required_columns = {
        "image_path",
        "pneumonia_label",
        "split",
    }

    missing_columns = (
        required_columns - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    test = frame.loc[
        frame["split"]
        .astype(str)
        .str.lower()
        .isin(["test"])
    ].copy()

    # The binary baseline used only definite labels.
    test = test.loc[
        test["pneumonia_label"].isin([0, 1])
    ].reset_index(drop=True)

    print(
        "Test rows:",
        len(test),
    )

    print(
        "\nTest labels:"
    )
    print(
        test[
            "pneumonia_label"
        ].value_counts()
    )

    if test.empty:
        raise RuntimeError(
            "No test rows were found."
        )

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    images_root = OUTPUT_ROOT / "images"
    images_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    packaged_paths: list[str] = []
    missing_paths: list[str] = []

    for index, row in test.iterrows():
        relative_path = str(
            row["image_path"]
        ).replace("\\", "/")

        try:
            source = resolve_image_path(
                relative_path
            )
        except FileNotFoundError:
            missing_paths.append(
                relative_path
            )
            continue

        destination = (
            images_root
            / Path(relative_path)
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source,
            destination,
        )

        packaged_paths.append(
            str(
                Path("images")
                / Path(relative_path)
            ).replace("\\", "/")
        )

        if (
            (index + 1) % 250 == 0
            or index + 1 == len(test)
        ):
            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(test)}"
            )

    if missing_paths:
        missing_file = (
            OUTPUT_ROOT
            / "missing_images.txt"
        )

        missing_file.write_text(
            "\n".join(missing_paths),
            encoding="utf-8",
        )

        raise RuntimeError(
            f"{len(missing_paths)} images "
            f"were missing. See "
            f"{missing_file}"
        )

    test[
        "packaged_image_path"
    ] = packaged_paths

    manifest_output = (
        OUTPUT_ROOT
        / "test_manifest.csv"
    )

    test.to_csv(
        manifest_output,
        index=False,
    )

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(
        ZIP_PATH,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in OUTPUT_ROOT.rglob("*"):
            if path.is_file():
                archive.write(
                    path,
                    arcname=path.relative_to(
                        OUTPUT_ROOT
                    ),
                )

    total_size_mb = (
        ZIP_PATH.stat().st_size
        / 1024**2
    )

    print("\nPackaging complete")
    print(
        "Images packaged:",
        len(packaged_paths),
    )
    print(
        "Manifest:",
        manifest_output,
    )
    print(
        "ZIP:",
        ZIP_PATH,
    )
    print(
        "ZIP size:",
        round(total_size_mb, 2),
        "MB",
    )


if __name__ == "__main__":
    main()
