"""Inspect up to five CheXpert dataset samples with a deterministic basic transform."""

import argparse
import sys
from pathlib import Path

from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumonia_ai.data.chexpert_dataset import (  # noqa: E402
    CheXpertPneumoniaDataset,
    DatasetValidationError,
)


def parse_args() -> argparse.Namespace:
    """Parse smoke-test arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Path to the CheXpert root directory.")
    parser.add_argument("--manifest", required=True, help="Path to the split manifest CSV.")
    parser.add_argument(
        "--split", required=True, choices=("train", "validation", "test"), help="Split to inspect."
    )
    return parser.parse_args()


def main() -> int:
    """Inspect dataset samples without constructing a DataLoader."""
    args = parse_args()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    try:
        dataset = CheXpertPneumoniaDataset(
            args.root, args.manifest, args.split, transform=transform
        )
    except (FileNotFoundError, NotADirectoryError, DatasetValidationError) as error:
        print(f"Dataset smoke-test error: {error}", file=sys.stderr)
        return 1

    print(f"Dataset length: {len(dataset)}")
    for index in range(min(5, len(dataset))):
        sample = dataset[index]
        image = sample["image"]
        if not hasattr(image, "shape") or tuple(image.shape) != (3, 224, 224):
            print(f"Invalid image tensor shape for sample {index}: {getattr(image, 'shape', None)}")
            return 1
        print(f"Sample {index}:")
        print(f"  Image tensor shape: {tuple(image.shape)}")
        print(f"  Label: {sample['label'].item()}")
        print(f"  Patient ID: {sample['patient_id']}")
        print(f"  Study ID: {sample['study_id']}")
        print(f"  Relative image path: {sample['image_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
