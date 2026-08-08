# Phase 1.2 segmentation validation runbook

This run validates the frozen historical lung-segmentation pipeline on the audited JSRT Kaggle mirror. It is inference-only: no model is trained, fine-tuned, or modified. The mirror contains resized grayscale *combined* lung masks; it is decoded explicitly with `uint8 >= 128` because its anti-aliased boundary pixels are not strictly binary. This is a mirror-specific decoding rule, not a claim about an official SCR threshold.

## A. Required Kaggle inputs

1. This repository, including `src/pneumonia_ai/segmentation/` and `scripts/analysis/evaluate_external_lung_segmentation.py`.
2. Kaggle dataset `ankitaroy123567/pneumonia-thesis-assets`, containing `/kaggle/input/datasets/ankitaroy123567/pneumonia-thesis-assets/best_checkpoint.pt`. The evaluator verifies SHA-256 `bdcbb77d4872292721a5bce8328401274d1b7d917e5e413f173873c8cf886a1d` before inference.
3. Kaggle dataset `abduzzami/jsrt-247-image-lung-segmentation-mask-dataset`, with 247 images and 247 matching masks under `/kaggle/input/datasets/abduzzami/jsrt-247-image-lung-segmentation-mask-dataset/content/jsrt`. The audited paths are `cxr/` and `masks/`; IDs include `JPCLN###` and `JPCNN###`.

Internet is unnecessary after attaching inputs. CPU is sufficient for 247 images because the frozen model uses 128 x 128 inference; GPU is optional.

## B. Dataset inspection commands

Run these before selecting final paths or invoking inference. They are inspection only.

```bash
find /kaggle/input/datasets/abduzzami/jsrt-247-image-lung-segmentation-mask-dataset/content/jsrt -maxdepth 3 -type f | sort | head -200
find /kaggle/input/datasets/abduzzami/jsrt-247-image-lung-segmentation-mask-dataset/content/jsrt -type f | sed -n '1,60p'
python - <<'PY'
from pathlib import Path
import numpy as np
from PIL import Image
root = Path('/kaggle/input/datasets/abduzzami/jsrt-247-image-lung-segmentation-mask-dataset/content/jsrt')
for path in sorted(root.rglob('*')):
    if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}:
        with Image.open(path) as image:
            print(path, image.mode, image.size, np.unique(np.asarray(image))[:20])
PY
```

Confirm the 247 image/mask pairs retain matching geometry (audited as 224 x 224), and inspect the grayscale values. The evaluator requires one canonical `JPCLN###` or `JPCNN###` image and one matching combined mask per case. It does not resize ground truth. The audited mirror is decoded as `uint8 >= 128`.

## C. Smoke-test command

Audit-only command (does not load the checkpoint or run inference):

```bash
python scripts/analysis/evaluate_external_lung_segmentation.py \
  --images-root /kaggle/input/datasets/abduzzami/jsrt-247-image-lung-segmentation-mask-dataset/content/jsrt/cxr \
  --combined-masks-root /kaggle/input/datasets/abduzzami/jsrt-247-image-lung-segmentation-mask-dataset/content/jsrt/masks \
  --checkpoint /kaggle/input/datasets/ankitaroy123567/pneumonia-thesis-assets/best_checkpoint.pt \
  --output-dir /kaggle/working/phase1_2_jsrt_audit \
  --audit-only
```

Smoke-test command:

```bash
python scripts/analysis/evaluate_external_lung_segmentation.py \
  --images-root /kaggle/input/datasets/abduzzami/jsrt-247-image-lung-segmentation-mask-dataset/content/jsrt/cxr \
  --combined-masks-root /kaggle/input/datasets/abduzzami/jsrt-247-image-lung-segmentation-mask-dataset/content/jsrt/masks \
  --checkpoint /kaggle/input/datasets/ankitaroy123567/pneumonia-thesis-assets/best_checkpoint.pt \
  --output-dir /kaggle/working/phase1_2_jsrt_smoke \
  --device cpu --bootstrap-iterations 2000 --seed 42 --max-samples 12
```

## D. Smoke-test validation checklist

- Confirm `segmentation_mapping_audit.csv` contains only `mapped` rows for selected cases.
- Confirm checkpoint SHA matches exactly before any predictions are produced.
- Confirm image and mask dimensions match and `segmentation_failures.csv` is empty.
- Inspect the deterministic qualitative panel; do not alter masks or add post-processing based on it.
- Verify metadata records `ground_truth_mask_threshold: 128`, `prediction_mask_threshold: 0.5`, and `postprocessing: none`.

## E. Full-evaluation command

Use all eligible mapped cases (omit `--max-samples`):

```bash
python scripts/analysis/evaluate_external_lung_segmentation.py \
  --images-root /kaggle/input/datasets/abduzzami/jsrt-247-image-lung-segmentation-mask-dataset/content/jsrt/cxr \
  --combined-masks-root /kaggle/input/datasets/abduzzami/jsrt-247-image-lung-segmentation-mask-dataset/content/jsrt/masks \
  --checkpoint /kaggle/input/datasets/ankitaroy123567/pneumonia-thesis-assets/best_checkpoint.pt \
  --output-dir /kaggle/working/phase1_2_jsrt_full \
  --device cpu --bootstrap-iterations 2000 --seed 42
```

## F. Output inspection commands

```bash
column -s, -t < /kaggle/working/phase1_2_jsrt_full/segmentation_summary.csv
head -10 /kaggle/working/phase1_2_jsrt_full/segmentation_case_metrics.csv
cat /kaggle/working/phase1_2_jsrt_full/segmentation_metadata.json
cat /kaggle/working/phase1_2_jsrt_full/segmentation_failures.csv
```

Expected output files: `segmentation_case_metrics.csv`, `segmentation_summary.csv`, `segmentation_summary.json`, `segmentation_failures.csv`, `segmentation_mapping_audit.csv`, `segmentation_metadata.json`, and PNG/PDF qualitative panels.

## G. Archive command

```bash
cd /kaggle/working && zip -r phase1_2_jsrt_full_results.zip phase1_2_jsrt_full
```

Download the ZIP as a new artifact; never overwrite the thesis experiment archives.

## H. Expected success criteria

The full run is acceptable only if mapping is complete and unambiguous, the checkpoint hash matches, all evaluated image/mask pairs retain their original matching geometry, and all output provenance fields are present. Report Dice and IoU mean, sample SD, median, and case-level 95% bootstrap confidence intervals; do not use the JSRT/SCR results to change segmentation weights or classification thresholds.
