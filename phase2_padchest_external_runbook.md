# Phase 2 PadChest external-validation runbook

PadChest is the second external **classification and calibration** cohort. Do not run Grad-CAM localization. No threshold, preprocessing, training, or checkpoint choice may use PadChest labels.

## 1. Frozen cohort and image availability

The full metadata and image-availability audit is complete. The Kaggle metadata is `/kaggle/input/datasets/seoyunje/padchest-small-dataset/PC/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv`; the image root is `/kaggle/input/datasets/seoyunje/padchest-small-dataset/PC/images-224/images-224`. The full CSV has 160,861 rows and the mirror has 160,845 image files. The 24-image sample remains schema-only and must not be used for Phase 2 inference, smoke testing, threshold selection, calibration, or final cohort definition.

The primary cohort is frozen: valid report-level `Labels` lists, exact normalized `pneumonia` token, and `Projection` in `PA` or `AP`. This gives 96,219 intended images from 63,914 patients: 3,773 positives and 92,446 negatives. It includes all eligible `Physician` and `RNN_model` labels (Physician 24,536; RNN_model 71,676). The sensitivity analysis is the Physician-labelled subset only; do not redefine the primary cohort around MethodLabel.

At the real image root, 96,212 intended images are available and evaluable from 63,911 patients: 3,773 positives and 92,439 negatives (PA 91,658; AP 4,554). Seven intended images are absent from the mirror. All seven are `pneumonia_target=0` and `RNN_model`-labelled. They are image-availability exclusions, not a cohort redefinition, and must be retained in the generated exclusion CSV.

The recorded counts predate a parser correction that excludes empty `Labels` lists as unusable rather than negative. Regenerate the full Kaggle metadata audit and final cohort manifest before using these counts for execution; do not infer replacement counts locally.

Attach the PadChest dataset, repository, six frozen classifier checkpoint assets, frozen segmentation checkpoint asset, and `configs/operating_thresholds/`.

```bash
PADROOT=/kaggle/input/datasets/seoyunje/padchest-small-dataset/PC
find "$PADROOT" -maxdepth 3 -type f | sort | sed -n '1,200p'
```

The following reproducibility audit may be rerun, but does not change the frozen policy:

```bash
python scripts/data/prepare_padchest_external.py \
  --metadata-csv "$PADROOT/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv" \
  --image-root "$PADROOT/images-224/images-224" \
  --image-id-column ImageID --image-dir-column ImageDir \
  --patient-id-column PatientID --projection-column Projection \
  --method-label-column MethodLabel --label-column Labels \
  --label-cuis-column labelCUIS --audit-only
```

## 2. Generate the final inference manifest, then checkpoint verification

Generate the manifest against the real root. Although the metadata retains `ImageDir`, the seoyunje 224×224 mirror is flat: `<image_root>/<ImageID>.png`. With no `--image-path-column`, the adapter first checks canonical `ImageDir/ImageID`, then the exact flat `ImageID` path; it never recursively searches or matches another filename. `MethodLabel`, target, projection, source row, and `labelCUIS` are retained for provenance. The companion exclusion CSV explicitly records every eligible missing image and its target/projection/MethodLabel provenance.

```bash
python scripts/data/prepare_padchest_external.py \
  --metadata-csv "$PADROOT/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv" \
  --image-root "$PADROOT/images-224/images-224" \
  --output-manifest /kaggle/working/results/phase2_padchest_external/padchest_pa_ap_manifest.csv \
  --exclusion-output /kaggle/working/results/phase2_padchest_external/padchest_pa_ap_image_availability_exclusions.csv \
  --image-id-column ImageID --image-dir-column ImageDir \
  --patient-id-column PatientID --case-id-column ImageID \
  --projection-column Projection --method-label-column MethodLabel \
  --label-column Labels --label-cuis-column labelCUIS \
  --pneumonia-concept pneumonia --accepted-view PA --accepted-view AP
```

Verify the manifest metadata reports intended `96,219`, available/evaluable `96,212`, and missing `7`; verify the exclusion CSV has seven RNN-model negative rows. Do not downsample negatives.

The mirror files are 224×224, `I;16` (16-bit grayscale) PNGs. The external loader detects this mode and maps the native full `[0, 65535]` range to 8-bit RGB before the unchanged frozen `ToTensor()`/model normalization path; it does not use per-image contrast normalization. DICOM decoding remains the historical RSNA windowing and MONOCHROME path.

The next checkpoint is threshold/provenance verification: confirm the frozen CheXpert-selected threshold and calibration assets before any execution. Do not optimize a threshold on PadChest. After that verification only, the smoke test is a deterministic, stratified 8–16-case inference run per input mode using `scripts/predict_external.py --max-samples 12 --seed 42`. For hard-masked inference, pass `--expected-segmentation-sha256 bdcbb77d4872292721a5bce8328401274d1b7d917e5e413f173873c8cf886a1d`. Confirm raster paths load, case identities align, both classes are present, the hard-masked prediction metadata records the historical segmentation SHA, `mask_threshold: 0.5`, and `postprocessing: none`.

## 3. Frozen assets and thresholds

Use matched seeds 42, 123, and 2026 for original and hard-masked A4 classifiers. The hard-masked path always receives the segmentation checkpoint with SHA-256 `bdcbb77d4872292721a5bce8328401274d1b7d917e5e413f173873c8cf886a1d`; it is the U-Net (`1/1/16/depth 4`, input size 256) validated in Phase 1.2.

| Condition | Seed | Frozen source / expected member |
| --- | ---: | --- |
| Original | 42 | `/kaggle/input/datasets/ankitaroy123567/a4-original-control-results/A4_original_control/best_checkpoint.pt`; `A4_original_control_results.zip::A4_original_control/best_checkpoint.pt` |
| Original | 123 | `A4_original_seed_123_results.zip::A4_original_seed_123/best_checkpoint.pt` |
| Original | 2026 | `A4_original_seed_2026_results.zip::A4_original_seed_2026/best_checkpoint.pt` |
| Hard-masked | 42 | `/kaggle/input/datasets/ankitaroy123567/a4-bestcheckpoint/best_checkpoint.pt`; reconstructed-run provenance must be recorded explicitly |
| Hard-masked | 123 | `A4_hard_masked_seed_123_results.zip::A4_hard_masked_seed_123/best_checkpoint.pt` |
| Hard-masked | 2026 | `A4_hard_masked_seed_2026_results.zip::A4_hard_masked_seed_2026/best_checkpoint.pt` |

Before smoke testing, print each mounted checkpoint path and SHA-256 and record the mapping in run metadata. Do not substitute a different checkpoint.

For each model/seed report fixed 0.5 and the corresponding frozen CheXpert Youden-J value in `configs/operating_thresholds/`. Never pass PadChest labels to threshold selection or calibration fitting.

## 4. Full run and archive

After threshold/provenance and smoke-test review, run each of the six frozen checkpoints independently. Preserve a separate seed-42 paired result for RSNA comparability, then a descriptive all-three-seed summary. Use GPU for throughput; no training is involved. Archive only generated CSV, JSON, PNG, and PDF outputs from `/kaggle/working/results/phase2_padchest_external`; never overwrite historical RSNA archives.
