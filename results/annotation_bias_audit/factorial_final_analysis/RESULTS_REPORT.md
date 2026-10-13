# Controlled Anatomical Shortcut Bias — Factorial Results

## Experiment design

- Dataset: matched CheXpert development cohort.
- Training images: 3,220.
- Validation images: 679, from 594 patients.
- Backbone: ResNet50.
- Marker: contrast-controlled striped patch, 24 × 24 pixels.
- Anatomical placements: lung interior, lung boundary, lung exterior.
- Training marker-label correlations: 50%, 75%, 100%.
- Random seeds: 42, 123, 2026.
- Total models: 27.
- Evaluation conditions: clean, matching, reversed.
- Total predictions: 54,999.
- Model selection: clean CheXpert validation AUROC.
- Statistical analysis: 2,000 patient-cluster bootstrap samples.

## Seed-averaged shortcut effects

|   correlation | region   |   AUROC_gap_mean |   AUROC_gap_sd |   probability_change_mean |   probability_change_sd |   flip_rate_mean |   flip_rate_sd |
|--------------:|:---------|-----------------:|---------------:|--------------------------:|------------------------:|-----------------:|---------------:|
|          0.5  | boundary |          -0.0061 |         0.0105 |                    0.026  |                  0.0023 |           0.0368 |         0.0078 |
|          0.5  | exterior |          -0.0079 |         0.0072 |                    0.0141 |                  0.0011 |           0.0275 |         0.0045 |
|          0.5  | interior |          -0.0056 |         0.0255 |                    0.0316 |                  0.0062 |           0.054  |         0.0112 |
|          0.75 | boundary |           0.7636 |         0.0257 |                    0.3616 |                  0.0018 |           0.6303 |         0.0449 |
|          0.75 | exterior |           0.6348 |         0.11   |                    0.2791 |                  0.0442 |           0.4811 |         0.1141 |
|          0.75 | interior |           0.8461 |         0.0277 |                    0.4069 |                  0.0476 |           0.728  |         0.0561 |
|          1    | boundary |           0.9994 |         0.001  |                    0.6165 |                  0.0459 |           0.9961 |         0.0009 |
|          1    | exterior |           0.9989 |         0.0019 |                    0.8045 |                  0.3213 |           0.9882 |         0.0204 |
|          1    | interior |           1      |         0      |                    0.905  |                  0.0452 |           1      |         0      |

## Clean-image performance

|   correlation | region   |   clean_AUROC_mean |   clean_AUROC_sd |   clean_AUPRC_mean |   clean_F1_mean |   clean_F1_sd |
|--------------:|:---------|-------------------:|-----------------:|-------------------:|----------------:|--------------:|
|          0.5  | boundary |             0.7433 |           0.01   |             0.8509 |          0.7371 |        0.0161 |
|          0.5  | exterior |             0.7391 |           0.0049 |             0.85   |          0.7487 |        0.0147 |
|          0.5  | interior |             0.7379 |           0.0045 |             0.8486 |          0.707  |        0.0158 |
|          0.75 | boundary |             0.7418 |           0.0065 |             0.851  |          0.383  |        0.0818 |
|          0.75 | exterior |             0.7431 |           0.0061 |             0.8467 |          0.5509 |        0.098  |
|          0.75 | interior |             0.7312 |           0.0064 |             0.8461 |          0.2987 |        0.1134 |
|          1    | boundary |             0.694  |           0.0065 |             0.8255 |          0      |        0      |
|          1    | exterior |             0.7009 |           0.0194 |             0.8214 |          0.0519 |        0.0899 |
|          1    | interior |             0.6651 |           0.022  |             0.8051 |          0.1002 |        0.1698 |

## Paired anatomical comparisons at 75% correlation

|   correlation | comparison        | metric       |   estimate |   CI_2.5 |   CI_97.5 |
|--------------:|:------------------|:-------------|-----------:|---------:|----------:|
|          0.75 | interior-boundary | regional_gap |     0.0824 |   0.0517 |    0.112  |
|          0.75 | interior-exterior | regional_gap |     0.2113 |   0.1844 |    0.2414 |
|          0.75 | boundary-exterior | regional_gap |     0.1289 |   0.0985 |    0.1585 |

## Interpretation

At 50% training correlation, matching-minus-reversed
AUROC gaps are approximately zero, consistent with the
marker orientation providing no systematic label signal.

At 75%, the seed-averaged gap follows:
interior > boundary > exterior.

At 100%, gaps approach one for all three regions,
indicating near-saturated response to marker reversal.

Clean-image performance generally declines at perfect
marker-label correlation.

## Statistical limitations

The bootstrap intervals quantify validation-patient
sampling uncertainty conditional on the fitted models.
They do not capture the full uncertainty associated with
training seeds. Only three training seeds were evaluated.

Matching and reversed markers encode the ground-truth
label by construction. They are synthetic interventions,
not naturally occurring clinical test conditions.

All results are from the matched CheXpert cohort.
External validation and clinical relevance must not be
claimed from this experiment alone.

The effect of anatomical placement may also depend on
local texture, visibility, segmentation quality and
marker-placement geometry.

## Generated artifacts

- performance_all_models.csv
- clean_performance_seed_summary.csv
- shortcut_effects_all_models.csv
- shortcut_effects_seed_summary.csv
- patient_cluster_bootstrap_all_contrasts.csv
- regional_differences_by_seed.csv
- Six figures, each saved as PNG and PDF.
