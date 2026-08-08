# Phase 1.2 segmentation validation report

## Purpose

Phase 1.2 quantitatively validates the frozen lung-segmentation inference pipeline that generated the hard-masked thesis inputs. It evaluates the exact historical checkpoint through `FrozenLungSegmenter`; no model selection, training, fine-tuning, or weight modification occurred.

## Dataset and ground-truth provenance

Evaluation used the JSRT Kaggle resized combined-mask mirror at the audited Kaggle paths recorded in the run metadata. This is not untouched original-resolution SCR data: the chest radiographs and corresponding combined lung masks are 224 x 224 PNGs in a Kaggle mirror. The 247 image/mask pairs were complete, uniquely mapped by canonical JSRT identifiers, and had matching geometry.

Ground truth was decoded specifically for this resized grayscale mirror as `uint8 >= 128`. This explicit interpolation-aware decoding rule is not represented as an official SCR annotation threshold.

## Frozen inference configuration

- Checkpoint SHA-256: `bdcbb77d4872292721a5bce8328401274d1b7d917e5e413f173873c8cf886a1d` (verified match).
- Architecture: `UNet` with `in_channels=1`, `out_channels=1`, `base_channels=16`, and `depth=4`.
- Segmentation input size: 256.
- Prediction threshold: 0.5.
- Backward-compatible metadata field `mask_threshold`: 0.5.
- Ground-truth combined-mask decoding threshold: 128.
- Post-processing: none. No morphology, hole filling, or connected-component cleanup was applied.

## Cohort completion

| Quantity | Value |
| --- | ---: |
| Cases discovered | 247 |
| Cases mapped | 247 |
| Cases evaluated | 247 |
| Failures | 0 |

## Segmentation accuracy

Case-level bootstrap used 2,000 resamples with seed 42.

| Metric | Mean | Sample SD | Median | 95% bootstrap CI |
| --- | ---: | ---: | ---: | --- |
| Dice | 0.9528014769453654 | 0.02951369022054901 | 0.9625101874490628 | [0.9489780044811696, 0.9562882865651664] |
| IoU | 0.9112545271462479 | 0.04983494082139124 | 0.9277297721916732 | [0.904925346815033, 0.9172241078177316] |

## Ten lowest-Dice cases

| Case ID | Dice | IoU |
| --- | ---: | ---: |
| JPCNN071 | 0.7756118111140898 | 0.6334688770722552 |
| JPCNN017 | 0.8023966028735966 | 0.6700019428793472 |
| JPCLN128 | 0.8104909213180901 | 0.6813658977838082 |
| JPCLN077 | 0.8340600438803566 | 0.7153541994187985 |
| JPCLN088 | 0.8519905341041848 | 0.7421459137877048 |
| JPCLN021 | 0.8596607182198722 | 0.7538639876352395 |
| JPCLN034 | 0.8608375845296002 | 0.7556759008539887 |
| JPCLN114 | 0.8617699652484901 | 0.757114062129476 |
| JPCLN151 | 0.8653754994243922 | 0.7626977021784542 |
| JPCLN136 | 0.8836225137537029 | 0.7915087187263078 |

## Deterministic qualitative panel

The panel selection rule was distribution-based (low, near-median, and high Dice), not hand-picked. Selected IDs: `JPCNN071`, `JPCNN017`, `JPCNN011`, `JPCLN146`, `JPCLN019`, and `JPCNN001`.

## Scientific interpretation

The frozen segmentation pipeline demonstrates high agreement with the decoded lung masks in this external JSRT mirror, including strong mean Dice and IoU across all 247 cases. This supports the claim that the hard-masking inputs used a competent anatomical lung-field localizer. It does not establish that improved segmentation or explainability necessarily improves pneumonia prediction; the frozen thesis finding remains unchanged: hard masking improved anatomical localization and sensitivity while reducing external AUROC, specificity, precision, and calibration.

## Limitations and provenance caveat

This validation uses a resized 224 x 224 Kaggle mirror with anti-aliased grayscale combined masks, rather than original-resolution JSRT images and untouched SCR left/right annotations. The `>=128` mask rule is a transparent decoding choice for that mirror and could affect boundary pixels. Results therefore validate the exact frozen inference pipeline against this reproducible representation, not every possible original SCR encoding. The checkpoint training metadata identifies a Montgomery training split, reinforcing the rationale for using JSRT rather than Montgomery as the evaluation cohort.

## Authoritative sources

- `experiments/kaggle_gpu/extracted/phase1_2_jsrt_full_results/segmentation_summary.json`
- `experiments/kaggle_gpu/extracted/phase1_2_jsrt_full_results/segmentation_summary.csv`
- `experiments/kaggle_gpu/extracted/phase1_2_jsrt_full_results/segmentation_metadata.json`
- `experiments/kaggle_gpu/extracted/phase1_2_jsrt_full_results/segmentation_case_metrics.csv`
- `experiments/kaggle_gpu/extracted/phase1_2_jsrt_full_results/segmentation_mapping_audit.csv`
- `experiments/kaggle_gpu/extracted/phase1_2_jsrt_full_results/segmentation_failures.csv`
