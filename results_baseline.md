# Frozen thesis baseline results

Frozen reference: Git tag `v1-thesis-baseline` at commit `37966623bb490431308394c1e51f46dd5f870d0d`. This is a forensic snapshot; no model was retrained, no threshold was changed, and no existing artifact was modified.

## Internal CheXpert validation: matched three-seed summary

All operating-point values use the archived fixed threshold of 0.5. Values are mean ± sample SD across seeds 42, 123, and 2026.

| Metric | Original | Hard-masked |
|---|---:|---:|
| AUROC | 0.757 ± 0.003 | 0.748 ± 0.011 |
| PR-AUC | 0.869 ± 0.005 | 0.870 ± 0.015 |
| Accuracy | 0.696 ± 0.033 | 0.701 ± 0.009 |
| Balanced accuracy | 0.699 ± 0.007 | 0.689 ± 0.012 |
| F1 | 0.764 ± 0.037 | 0.775 ± 0.006 |
| Precision | 0.857 ± 0.012 | 0.843 ± 0.008 |
| Sensitivity | 0.691 ± 0.068 | 0.717 ± 0.006 |
| Specificity | 0.706 ± 0.056 | 0.661 ± 0.019 |

Seed-level best epochs: original 42/123/2026 = 9/8/9; hard-masked 42/123/2026 = 7/9/7.

### Internal paired seed statistics

Hard-masked minus original. These tests are exploratory because n=3 paired seeds. The 95% intervals are t intervals (df=2).

| Metric | Mean difference | 95% CI | paired t p | Wilcoxon p | Cohen’s dz |
|---|---:|---:|---:|---:|---:|
| AUROC | -0.009 | [-0.040, 0.022] | 0.338 | 0.250 | -0.721 |
| PR-AUC | 0.001 | [-0.042, 0.044] | 0.945 | 1.000 | 0.045 |
| Accuracy | 0.006 | [-0.065, 0.076] | 0.758 | 1.000 | 0.204 |
| Balanced accuracy | -0.010 | [-0.038, 0.019] | 0.275 | 0.250 | -0.860 |
| F1 | 0.011 | [-0.074, 0.096] | 0.628 | 0.750 | 0.328 |
| Precision | -0.015 | [-0.062, 0.033] | 0.320 | 0.500 | -0.757 |
| Sensitivity | 0.026 | [-0.139, 0.191] | 0.566 | 0.750 | 0.393 |
| Specificity | -0.046 | [-0.230, 0.138] | 0.397 | 0.250 | -0.617 |

## RSNA external classification

Paired RSNA cohort: 26,684 cases (6,012 positive; 20,672 negative), evaluated at threshold 0.5.

| Metric | Original | Hard-masked | Difference (masked − original), 95% paired-bootstrap CI |
|---|---:|---:|---:|
| AUROC | 0.791 | 0.749 | -0.042 [-0.048, -0.036]; p < 0.001 |
| PR-AUC | 0.466 | 0.425 | -0.041 [-0.053, -0.030]; p < 0.001 |
| Accuracy | 0.766 | 0.622 | -0.143 [-0.149, -0.137]; p < 0.001 |
| Balanced accuracy | 0.691 | 0.690 | -0.000 [-0.008, 0.008]; p = 0.936 |
| F1 | 0.516 | 0.493 | -0.023 [-0.033, -0.014]; p < 0.001 |
| Precision | 0.482 | 0.353 | -0.129 [-0.138, -0.120]; p < 0.001 |
| Sensitivity | 0.554 | 0.814 | 0.260 [0.246, 0.273]; p < 0.001 |
| Specificity | 0.827 | 0.567 | -0.260 [-0.267, -0.254]; p < 0.001 |
| NPV | 0.864 | 0.913 | 0.048 [0.044, 0.053]; p < 0.001 |
| Brier | 0.163 | 0.229 | 0.066 [0.064, 0.068]; p < 0.001 |
| NLL | 0.497 | 0.649 | 0.152 [0.148, 0.156]; p < 0.001 |
| ECE | 0.139 | 0.275 | 0.136 [0.134, 0.138]; p < 0.001 |

McNemar: original-correct/masked-wrong = 6,233; original-wrong/masked-correct = 2,412; discordant pairs = 8,645; exact p < 0.001 (the stored value is 0.0). Masking helped 2,412 cases and hurt 6,233 cases.

## RSNA Grad-CAM localization

Paired localization cohort: 6,011 of 6,012 RSNA pneumonia-positive patients with boxes. One hard-masked case (`3d895c52-0627-42b3-ba67-41a08f387da0`) was excluded after deterministic retry because its Grad-CAM activation map was constant or unavailable; no heatmap was imputed.

| Metric | Original | Hard-masked | Difference (masked − original), 95% paired-bootstrap CI |
|---|---:|---:|---:|
| Pointing Game | 0.072 | 0.408 | 0.336 [0.322, 0.349]; p < 0.001 |
| Energy inside boxes | 0.101 | 0.217 | 0.116 [0.113, 0.120]; p < 0.001 |
| Heatmap IoU (0.5) | 0.037 | 0.150 | 0.113 [0.110, 0.117]; p < 0.001 |
| Top-10% energy inside | 0.088 | 0.318 | 0.230 [0.224, 0.237]; p < 0.001 |
| Top-20% energy inside | 0.099 | 0.254 | 0.155 [0.151, 0.159]; p < 0.001 |
| Lesion coverage (0.5) | 0.072 | 0.235 | 0.163 [0.157, 0.170]; p < 0.001 |
| Activation-area ratio (0.5) | 0.085 | 0.070 | -0.015 [-0.016, -0.013]; p < 0.001 |

Pointing Game McNemar: original-only hit = 149; hard-masked-only hit = 2,169; exact p < 0.001 (stored value 0.0).

## Artifact provenance

- Internal seed archives: `experiments/kaggle_gpu/archives/A4_original_control_results.zip`, `A4_original_seed_123_results.zip`, `A4_original_seed_2026_results.zip`, `A4_hard_masked_seed_123_results.zip`, and `A4_hard_masked_seed_2026_results.zip` (`validation_metrics.json`, `run_summary.json`).
- The hard-masked seed-42 values, including best epoch 7, AUROC 0.7548263342487169, and PR-AUC 0.8830489836730908, are preserved in `experiments/kaggle_gpu/extracted/thesis_optimisation_results/classification_optimisation/A4_regularised_optimisation/{validation_metrics.json,run_summary.json}`. No separately identifiable local `A4_hard_masked_seed_42` archive exists; that provenance limitation is retained here.
- External classification: `experiments/kaggle_gpu/extracted/RSNA_external_statistical_analysis/RSNA_external_statistical_analysis/{external_validation_report.json,paired_bootstrap_comparison.csv,masking_outcome_summary.csv,mcnemar_test.json,domain_shift_summary.json,paired_predictions.csv}`.
- Localization: `experiments/kaggle_gpu/extracted/rsna_gradcam_full_results/rsna_gradcam_full/{rsna_gradcam_model_summary.csv,rsna_gradcam_bootstrap_comparison.csv,rsna_gradcam_paired_comparison.csv,rsna_gradcam_completion_summary.json,rsna_gradcam_report.json,rsna_gradcam_failures.csv,rsna_gradcam_case_metrics_paired_complete.csv}`.

## Frozen interpretation

Hard masking improves anatomical localization and increases external sensitivity. It reduces external discrimination, specificity, precision, and calibration (higher Brier, NLL, and ECE). Explainability improvement therefore does not imply predictive improvement.
