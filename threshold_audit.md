# Threshold forensic audit

This audit describes historical behavior only. No threshold code or result has been changed.

## Implemented methods

`scripts/classification/optimisation_thresholds.py:10-61` implements validation-only candidate search over 0, every observed probability, and 1. It supports `fixed_0.5`, `max_f1`, `youden`, `balanced_accuracy`, and `target_sensitivity`; its `selected` column identifies the chosen row. `src/pneumonia_ai/evaluation/core.py:83-100` independently implements `fixed_0.5`, Youden’s J, and max-F1, and rejects non-validation input for selection (`:85-87`).

## Reported result families

| Result family | Historical threshold used | Evidence |
|---|---|---|
| Internal CheXpert fixed metrics | Fixed 0.5, for every model and seed | `scripts/classification/run_optimisation_experiment.py:645-657` computes `fixed_metrics = _metrics(validation, 0.5)` and writes it as `fixed_threshold: 0.5`; every inspected A4 archive `validation_metrics.json` records `fixed_threshold: 0.5`. |
| Internal CheXpert max-F1 descriptive metrics | Per-run/per-seed max-F1 threshold selected from that run’s CheXpert validation predictions | `run_optimisation_experiment.py:638-646` calls `threshold_analysis(..., "max_f1")`, saves `threshold_analysis.csv`, then computes `max_f1_metrics`. The A4 archives show distinct thresholds (for example original 42=0.170069, 123=0.257877, 2026=0.184916; hard-masked 42=0.214306, 123=0.158976, 2026=0.269064). |
| Multi-seed thesis summary | Fixed-0.5 metrics are primary; max-F1 metrics are secondary descriptive metrics | `scripts/analysis/analyze_multiseed_ablation.py:24-29` defines `PRIMARY_METRICS` as the fixed fields and `SECONDARY_METRICS` as max-F1 fields. |
| RSNA external statistical analysis frozen in this baseline | Fixed 0.5 for both models | `external_validation_report.json` records `threshold: 0.5`; the classification counts and paired predictions are consistent with that report. It was not a per-model or per-seed optimized operating point. |
| Generic RSNA external evaluator | Validation-derived threshold, defaulting to Youden’s J; can instead be max-F1 or fixed 0.5 | `scripts/evaluate_external.py:24-27` defines the default `youden`; `:46-52` validates a CheXpert validation input, selects the threshold from it, and applies it to external data. |
| Grad-CAM correctness fields | Fixed 0.5, model probabilities thresholded independently | `scripts/analysis/evaluate_rsna_gradcam_localization.py:366-370` sets `predicted = int(probability >= .5)` and records `correct`; the frozen Grad-CAM model summary’s `correct` mean equals the RSNA fixed-0.5 sensitivity for the positive-only cohort. Localization metrics themselves are not classification-threshold dependent. |

## Was RSNA threshold frozen from CheXpert validation?

The reusable `evaluate_external.py` workflow is explicitly designed to freeze a threshold chosen on CheXpert validation (`scripts/evaluate_external.py:1, 46-52`) and defaults to Youden’s J. However, the specific frozen RSNA comparison documented in `results_baseline.md` records threshold 0.5 in `external_validation_report.json`. Its preserved report does not identify a CheXpert validation-derived threshold or method. Therefore the accurate conclusion is: **the historical frozen RSNA comparison used fixed 0.5, not a documented validation-selected threshold**. The generic evaluator’s capability must not be conflated with the archived analysis.

## Model/seed variation and ambiguities

The fixed reporting threshold did not differ by model or seed: it was 0.5. The descriptive max-F1 threshold did differ by model and seed, because it was separately selected on each run’s validation predictions. The separately identifiable hard-masked seed-42 A4 archive is absent; its metrics/threshold-analysis provenance is the preserved `A4_regularised_optimisation` extracted result rather than a dedicated archive. No command line or run manifest was found for the archived external statistical comparison, so its invocation-level threshold method cannot be established beyond the artifact’s explicit `threshold: 0.5`.

## Recommended Phase 1.1 policy (not implemented)

Pre-specify one clinical operating-point objective and select its threshold on CheXpert validation only, separately for each frozen trained model/seed. Store the selected value, selection method, validation cohort identity, and calibration state in a machine-readable manifest; apply it unchanged to all external and explainability correctness reports. Keep fixed 0.5 results as a clearly labeled secondary reference, and never select or tune a threshold on RSNA test data.
