# Phase 1.1: threshold justification and ROC analysis

## Historical policy verified

The Phase-0 conclusion is confirmed against the archived A4 metrics and source. Internal primary operating-point metrics used fixed threshold 0.5; per-run max-F1 fields were descriptive validation analyses. The archived paired RSNA comparison records threshold 0.5. `scripts/evaluate_external.py` already selected Youden, max-F1, or fixed 0.5 using CheXpert validation input, not external input.

## New publication threshold policy

For each frozen model/seed, select a single operating threshold on its CheXpert validation predictions only by maximizing Youden's J (TPR - FPR). Ties are resolved by selecting the highest finite threshold among the tied maxima, a deterministic choice that favors specificity at equal J. The derived threshold is then frozen and may be applied unchanged to external data. AUROC and PR-AUC remain separate, threshold-independent measures.

The reusable implementation is `pneumonia_ai.evaluation.core.roc_threshold_analysis`. It fails for empty arrays, NaN/infinite/out-of-range probabilities, non-binary targets, single-class targets, or an ROC curve without a finite threshold. It never substitutes 0.5 after an invalid estimation.

## Why external labels are excluded

An external threshold optimized using RSNA labels would tune the reported operating point to the external test set and bias external performance. The Phase 1.1 external application therefore only reads the frozen CheXpert-derived threshold artifacts. `external_threshold_metadata` rejects supplied provenance whose selection dataset contains `external` or `RSNA`.

## Frozen CheXpert-derived thresholds

| Condition | Seed | Threshold | Youden J | Validation AUROC | Source artifact |
|---|---:|---:|---:|---:|---|
| Original | 42 | 0.421927 | 0.413645 | 0.754214 | `A4_original_control_results.zip::validation_predictions.csv` |
| Original | 123 | 0.454737 | 0.410503 | 0.758894 | `A4_original_seed_123_results.zip::validation_predictions.csv` |
| Original | 2026 | 0.534504 | 0.421920 | 0.758843 | `A4_original_seed_2026_results.zip::validation_predictions.csv` |
| Hard-masked | 42 | 0.560857 | 0.396495 | 0.754826 | `A4_regularised_optimisation/validation_predictions.csv` |
| Hard-masked | 123 | 0.626494 | 0.380748 | 0.735912 | `A4_hard_masked_seed_123_results.zip::validation_predictions.csv` |
| Hard-masked | 2026 | 0.485797 | 0.418582 | 0.754358 | `A4_hard_masked_seed_2026_results.zip::validation_predictions.csv` |

Machine-readable files are under `configs/operating_thresholds/`; the seed-42 hard-masked artifact retains the existing reconstructed-run provenance limitation.

Threshold summaries (sample SD): original mean 0.470389, SD 0.057898, minimum 0.421927, maximum 0.534504; hard-masked mean 0.557716, SD 0.070401, minimum 0.485797, maximum 0.626494. These summaries are descriptive only and are not applied as averaged external thresholds.

## Threshold sensitivity analysis

The complete table is `results/phase1_1_threshold_analysis/threshold_sensitivity_table.csv`. CheXpert rows are explicitly in-sample operating-point analyses because selection and measurement use the same saved validation predictions. RSNA rows apply only the corresponding frozen seed-42 CheXpert threshold; no RSNA labels were used for selection. AUROC and PR-AUC are repeated only to demonstrate that they do not vary with threshold.

| Dataset | Condition / seed | Policy | Threshold | AUROC | PR-AUC | Accuracy | Balanced accuracy | Sensitivity | Specificity | Precision | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CheXpert | Original / 42 | Fixed 0.5 | 0.500000 | 0.754214 | 0.864012 | 0.661900 | 0.691027 | 0.623932 | 0.758123 | 0.867327 | 0.725766 |
| CheXpert | Original / 42 | Youden-J | 0.421927 | 0.754214 | 0.864012 | 0.722165 | 0.706823 | 0.742165 | 0.671480 | 0.851307 | 0.792998 |
| CheXpert | Original / 123 | Fixed 0.5 | 0.500000 | 0.758894 | 0.869032 | 0.697651 | 0.702842 | 0.690883 | 0.714801 | 0.859929 | 0.766193 |
| CheXpert | Original / 123 | Youden-J | 0.454737 | 0.758894 | 0.869032 | 0.729316 | 0.705252 | 0.760684 | 0.649819 | 0.846276 | 0.801200 |
| CheXpert | Original / 2026 | Fixed 0.5 | 0.500000 | 0.758843 | 0.874820 | 0.727273 | 0.702734 | 0.759259 | 0.646209 | 0.844691 | 0.799700 |
| CheXpert | Original / 2026 | Youden-J | 0.534504 | 0.758843 | 0.874820 | 0.713994 | 0.710960 | 0.717949 | 0.703971 | 0.860068 | 0.782609 |
| CheXpert | Hard-masked / 42 | Fixed 0.5 | 0.500000 | 0.754826 | 0.883049 | 0.699694 | 0.684596 | 0.719373 | 0.649819 | 0.838870 | 0.774540 |
| CheXpert | Hard-masked / 42 | Youden-J | 0.560857 | 0.754826 | 0.883049 | 0.665986 | 0.698247 | 0.623932 | 0.772563 | 0.874251 | 0.728180 |
| CheXpert | Hard-masked / 123 | Fixed 0.5 | 0.500000 | 0.735912 | 0.853507 | 0.693565 | 0.680323 | 0.710826 | 0.649819 | 0.837248 | 0.768875 |
| CheXpert | Hard-masked / 123 | Youden-J | 0.626494 | 0.735912 | 0.853507 | 0.634321 | 0.690374 | 0.561254 | 0.819495 | 0.887387 | 0.687609 |
| CheXpert | Hard-masked / 2026 | Fixed 0.5 | 0.500000 | 0.754358 | 0.873656 | 0.710930 | 0.702266 | 0.722222 | 0.682310 | 0.852101 | 0.781804 |
| CheXpert | Hard-masked / 2026 | Youden-J | 0.485797 | 0.754358 | 0.873656 | 0.727273 | 0.709291 | 0.750712 | 0.667870 | 0.851373 | 0.797880 |
| RSNA | Original / 42 | Fixed 0.5 | 0.500000 | 0.790984 | 0.466485 | 0.765590 | 0.690525 | 0.553892 | 0.827158 | 0.482399 | 0.515679 |
| RSNA | Original / 42 | CheXpert Youden-J | 0.421927 | 0.790984 | 0.466485 | 0.725941 | 0.721968 | 0.714737 | 0.729199 | 0.434260 | 0.540265 |
| RSNA | Hard-masked / 42 | Fixed 0.5 | 0.500000 | 0.748844 | 0.425004 | 0.622395 | 0.690231 | 0.813706 | 0.566757 | 0.353264 | 0.492649 |
| RSNA | Hard-masked / 42 | CheXpert Youden-J | 0.560857 | 0.748844 | 0.425004 | 0.685317 | 0.683835 | 0.681138 | 0.686533 | 0.387234 | 0.493760 |

## ROC methodology and figure status

The generator uses case-level predictions only. For internal CheXpert it plots each condition's mean interpolated ROC over the three matched seeds, with a sample-SD band; it does not concatenate independently trained-model predictions. For RSNA it plots the paired frozen seed-42 original and hard-masked prediction curves. It also creates a two-panel combined figure, keeping datasets separate.

Generated output paths are:

- `results/phase1_1_threshold_analysis/chexpert_internal_roc_comparison.{png,pdf}`
- `results/phase1_1_threshold_analysis/rsna_external_roc_comparison.{png,pdf}`
- `results/phase1_1_threshold_analysis/chexpert_rsna_roc_comparison.{png,pdf}`

All six files were generated from saved case-level predictions. No ROC was made from summary metrics.

## Environment and tests

The local environment was recreated with Python 3.11.9 at `.venv\\Scripts\\python.exe`, after retaining the prior environment as `.venv_broken_phase1_1`. Dependencies were installed strictly from `requirements.txt`. Verified imports: numpy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, matplotlib 3.11.1, torch 2.13.0+cpu, and `pneumonia_ai.evaluation.core`.

`python -m pytest -q tests/test_evaluation.py` passed: 19 passed, 0 failed (five pre-existing sklearn warnings). This includes the Phase 1.1 Youden-J, deterministic tie, invalid/single-class, NaN, supplied-threshold, provenance, and external-leakage-prevention tests. No implementation correction was required after testing.

Additional relevant tests (`test_evaluation.py`, `test_multiseed_ablation_analysis.py`, and `test_freeze_thesis_baseline.py`) passed: 31 passed, 0 failed. `py_compile`, targeted Ruff `--select E,F`, and `git diff --check` passed. The full latest-Ruff default rule set still reports legacy style-only rules outside this Phase 1.1 change; no unrelated mass reformat was performed.

## Reproduction

From the repository root, run:

```text
.\.venv\Scripts\python.exe -m pytest -q tests/test_evaluation.py
.\.venv\Scripts\python.exe scripts/analysis/generate_phase1_1_threshold_analysis.py --overwrite
.\.venv\Scripts\ruff.exe check src/pneumonia_ai/evaluation scripts/evaluate_external.py scripts/analysis/generate_phase1_1_threshold_analysis.py tests/test_evaluation.py
```

The generator refuses to overwrite derived artifacts unless `--overwrite` is supplied. It reads only archived/extracted predictions, opens no checkpoint, and performs no training.

## Limitations and preservation statement

All six CheXpert validation-prediction files were recovered. The hard-masked seed-42 provenance remains the preserved `A4_regularised_optimisation` extract rather than a separately identifiable A4 archive. No models were retrained, no historical baseline artifact was overwritten, and `results_baseline.md` was not changed.
