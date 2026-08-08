# Phase 2 PadChest external-validation runbook

PadChest is the second external **classification and calibration** cohort. Do not run Grad-CAM localization. No threshold, preprocessing, training, or checkpoint choice may use PadChest labels.

## 1. Schema inspection is the first Kaggle action

Attach the PadChest image/metadata dataset, the repository, six frozen classifier checkpoint assets, the frozen segmentation checkpoint asset, and `configs/operating_thresholds/`.

```bash
PADROOT=/kaggle/input/<verified-padchest-attachment>
find "$PADROOT" -maxdepth 3 -type f | sort | sed -n '1,200p'
find "$PADROOT" -type f \( -iname '*.csv' -o -iname '*.csv.gz' \) -print
python - <<'PY'
from pathlib import Path
import pandas as pd
root = Path('/kaggle/input/<verified-padchest-attachment>')
for path in sorted(root.rglob('*.csv')):
    frame = pd.read_csv(path, nrows=20)
    print('\nCSV:', path, '\ncolumns:', list(frame.columns), '\nhead:\n', frame.head())
PY
```

After identifying one metadata CSV and candidate image root, run **only** this audit command, substituting reviewed column names when available:

```bash
python scripts/data/prepare_padchest_external.py \
  --metadata-csv /kaggle/input/<verified-padchest-attachment>/<metadata>.csv \
  --image-root /kaggle/input/<verified-padchest-attachment>/<images> \
  --label-column <reviewed-report-level-label-column> \
  --image-path-column <reviewed-image-path-column> \
  --projection-column <reviewed-projection-column> \
  --audit-only
```

Review the printed columns, row count, label examples, likely pneumonia concepts, projection values, missing image paths, annotation-method fields, and patient-ID availability. Do not create a manifest or run prediction until this output is reviewed and the exact label/CUI and frontal-view values are frozen.

## 2. Manifest audit, then smoke test

After review, create the manifest by supplying exact report-level pneumonia concepts with repeated `--pneumonia-concept` flags and reviewed frontal values with repeated `--accepted-view` flags. The primary negatives are all eligible images without the exact pneumonia concept, including other abnormalities; missing or unparseable labels are excluded and counted.

The smoke test is a deterministic, stratified 8–16-case inference run per input mode using `scripts/predict_external.py --max-samples 12 --seed 42`. For hard-masked inference, pass `--expected-segmentation-sha256 bdcbb77d4872292721a5bce8328401274d1b7d917e5e413f173873c8cf886a1d`. Confirm raster paths load, case identities align, both classes are present, the hard-masked prediction metadata records the historical segmentation SHA, `mask_threshold: 0.5`, and `postprocessing: none`.

## 3. Frozen assets and thresholds

Use matched seeds 42, 123, and 2026 for original and hard-masked A4 classifiers. The hard-masked path always receives the segmentation checkpoint with SHA-256 `bdcbb77d4872292721a5bce8328401274d1b7d917e5e413f173873c8cf886a1d`; it is the U-Net (`1/1/16/depth 4`, input size 256) validated in Phase 1.2.

The checkpoint attachment must preserve these archive-member identities exactly before the first run:

| Condition | Seed | Frozen source / expected member |
| --- | ---: | --- |
| Original | 42 | Historical Kaggle asset: `/kaggle/input/datasets/ankitaroy123567/a4-original-control-results/A4_original_control/best_checkpoint.pt`; archive member: `A4_original_control_results.zip::A4_original_control/best_checkpoint.pt` |
| Original | 123 | `A4_original_seed_123_results.zip::A4_original_seed_123/best_checkpoint.pt` |
| Original | 2026 | `A4_original_seed_2026_results.zip::A4_original_seed_2026/best_checkpoint.pt` |
| Hard-masked | 42 | Historical Kaggle asset: `/kaggle/input/datasets/ankitaroy123567/a4-bestcheckpoint/best_checkpoint.pt`; reconstructed-run provenance must be recorded explicitly |
| Hard-masked | 123 | `A4_hard_masked_seed_123_results.zip::A4_hard_masked_seed_123/best_checkpoint.pt` |
| Hard-masked | 2026 | `A4_hard_masked_seed_2026_results.zip::A4_hard_masked_seed_2026/best_checkpoint.pt` |

Before smoke testing, print each mounted checkpoint path and SHA-256 and record the mapping in run metadata. The local archive evidence does not establish future Kaggle mount paths or hashes for seeds 123/2026; do not substitute a different checkpoint.

For each model/seed report fixed 0.5 and the corresponding frozen CheXpert Youden-J value in `configs/operating_thresholds/`. Never pass PadChest labels to threshold selection or calibration fitting.

## 4. Full run and archive

After schema and smoke-test review, run each of the six frozen checkpoints independently. Preserve a separate seed-42 paired result for RSNA comparability, then a descriptive all-three-seed summary. Use GPU for throughput; no training is involved. Archive only generated CSV, JSON, PNG, and PDF outputs from `/kaggle/working/results/phase2_padchest_external`; never overwrite historical RSNA archives.
