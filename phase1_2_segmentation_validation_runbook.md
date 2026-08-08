# Phase 1.2 segmentation validation runbook

This run validates the frozen historical lung-segmentation pipeline on JSRT images with SCR-derived, human-annotated left and right lung masks. It is inference-only: no model is trained, fine-tuned, or modified.

## A. Required Kaggle inputs

1. This repository, including `src/pneumonia_ai/segmentation/` and `scripts/analysis/evaluate_external_lung_segmentation.py`.
2. Kaggle dataset `pneumonia-thesis-assets`, containing `best_checkpoint.pt` at the historical input location. The evaluator verifies SHA-256 `bdcbb77d4872292721a5bce8328401274d1b7d917e5e413f173873c8cf886a1d` before inference.
3. A provenance-verified JSRT/SCR dataset attachment. It must contain converted JSRT radiographs plus distinct SCR left- and right-lung masks, each traceable to canonical `JPCLN###` IDs. Do not substitute a color-labelled mask bundle until its case mapping and label semantics are documented.

Internet is unnecessary after attaching inputs. CPU is sufficient for 247 images because the frozen model uses 128 x 128 inference; GPU is optional.

## B. Dataset inspection commands

Run these before selecting final paths or invoking inference. They are inspection only.

```bash
find /kaggle/input/jsrt-scr -maxdepth 4 -type f | sort | head -200
find /kaggle/input/jsrt-scr -type f | sed -n '1,60p'
python - <<'PY'
from pathlib import Path
import numpy as np
from PIL import Image
root = Path('/kaggle/input/jsrt-scr')
for path in sorted(root.rglob('*')):
    if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}:
        with Image.open(path) as image:
            print(path, image.mode, image.size, np.unique(np.asarray(image))[:20])
PY
```

Confirm the converted image and mask geometry match, that each side has one mask per `JPCLN###`, and that left/right directories are unambiguous. If the mirror has a nonstandard layout, create a reviewed mapping CSV with `case_id,image_path,left_mask_path,right_mask_path`; paths are relative to the corresponding supplied roots unless absolute.

## C. Smoke-test command

After inspection, using illustrative paths adjusted to the verified attachment:

```bash
python scripts/analysis/evaluate_external_lung_segmentation.py \
  --images-root /kaggle/input/jsrt-scr/images \
  --masks-root /kaggle/input/jsrt-scr/masks \
  --checkpoint /kaggle/input/pneumonia-thesis-assets/best_checkpoint.pt \
  --output-dir /kaggle/working/phase1_2_jsrt_smoke \
  --device cpu --bootstrap-iterations 2000 --seed 42 --max-samples 12
```

If mask directories are not named `left/right`, `leftMask/rightMask`, or `left_lung/right_lung`, add both `--left-masks-root` and `--right-masks-root`. To inspect mapping only, add `--audit-only`; this intentionally stops before checkpoint verification and inference.

## D. Smoke-test validation checklist

- Confirm `segmentation_mapping_audit.csv` contains only `mapped` rows for selected cases.
- Confirm checkpoint SHA matches exactly before any predictions are produced.
- Confirm image and mask dimensions match and `segmentation_failures.csv` is empty.
- Inspect the deterministic qualitative panel; do not alter masks or add post-processing based on it.
- Verify metadata records `mask_threshold: 0.5` and `postprocessing: none`.

## E. Full-evaluation command

Use all eligible mapped cases (omit `--max-samples`):

```bash
python scripts/analysis/evaluate_external_lung_segmentation.py \
  --images-root /kaggle/input/jsrt-scr/images \
  --masks-root /kaggle/input/jsrt-scr/masks \
  --checkpoint /kaggle/input/pneumonia-thesis-assets/best_checkpoint.pt \
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
