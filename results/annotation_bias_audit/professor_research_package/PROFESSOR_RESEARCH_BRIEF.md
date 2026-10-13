# Professor Research Brief — Anatomical Shortcut Bias

**Prepared:** 2026-10-13  
**Status:** Completed factorial training, evaluation and statistical analysis  
**Scope:** Controlled synthetic-marker experiments on matched CheXpert

## Research question

How do anatomical marker placement and marker-label correlation
jointly influence shortcut susceptibility and clean-image
generalization in pneumonia chest X-ray classification?

## Experimental setup

- Dataset: CheXpert matched development cohort
- Training images: 3,220
- Validation images: 679 from 594 patients
- Backbone: ImageNet-pretrained ResNet50
- Training: three epochs per run
- Marker: 24 × 24 contrast-controlled striped pattern
- Marker locations: lung interior, lung boundary, lung exterior
- Training correlations: 50%, 75%, 100%
- Seeds: 42, 123, 2026
- Total models: 27
- Evaluation: clean, label-matching and label-reversed marker
- Total evaluation predictions: 54,999
- Statistical analysis: 2,000 patient-cluster bootstrap repetitions

## Finding 1: Correlation determines shortcut susceptibility

|   correlation | region   |   AUROC_gap_mean |   AUROC_gap_sd |   flip_rate_mean |
|--------------:|:---------|-----------------:|---------------:|-----------------:|
|          0.5  | boundary |          -0.0061 |         0.0105 |           0.0368 |
|          0.5  | exterior |          -0.0079 |         0.0072 |           0.0275 |
|          0.5  | interior |          -0.0056 |         0.0255 |           0.054  |
|          0.75 | boundary |           0.7636 |         0.0257 |           0.6303 |
|          0.75 | exterior |           0.6348 |         0.11   |           0.4811 |
|          0.75 | interior |           0.8461 |         0.0277 |           0.728  |
|          1    | boundary |           0.9994 |         0.001  |           0.9961 |
|          1    | exterior |           0.9989 |         0.0019 |           0.9882 |
|          1    | interior |           1      |         0      |           1      |

The 50% condition has near-zero mean matching-minus-reversed
AUROC gaps. At 75%, the models exhibit strong marker dependence
with an anatomical ordering. At 100%, marker dependence nearly
saturates across regions.

## Finding 2: Anatomical placement matters at 75%

- Interior AUROC gap: 0.8461
- Boundary AUROC gap: 0.7636
- Exterior AUROC gap: 0.6348

### Patient-cluster bootstrap comparisons

| comparison        |   estimate |   CI_2.5 |   CI_97.5 |
|:------------------|-----------:|---------:|----------:|
| interior-boundary |     0.0824 |   0.0517 |    0.112  |
| interior-exterior |     0.2113 |   0.1844 |    0.2414 |
| boundary-exterior |     0.1289 |   0.0985 |    0.1585 |

All three pairwise differences are positive in each of the
three training seeds.

The intervals describe validation-patient sampling uncertainty
conditional on these trained models. They do not fully capture
training-seed uncertainty.

## Finding 3: Strong shortcuts reduce clean performance

|   correlation | region   |   clean_AUROC_mean |   clean_AUROC_sd |   clean_F1_mean |
|--------------:|:---------|-------------------:|-----------------:|----------------:|
|          0.5  | boundary |             0.7433 |           0.01   |          0.7371 |
|          0.5  | exterior |             0.7391 |           0.0049 |          0.7487 |
|          0.5  | interior |             0.7379 |           0.0045 |          0.707  |
|          0.75 | boundary |             0.7418 |           0.0065 |          0.383  |
|          0.75 | exterior |             0.7431 |           0.0061 |          0.5509 |
|          0.75 | interior |             0.7312 |           0.0064 |          0.2987 |
|          1    | boundary |             0.694  |           0.0065 |          0      |
|          1    | exterior |             0.7009 |           0.0194 |          0.0519 |
|          1    | interior |             0.6651 |           0.022  |          0.1002 |

At 100% correlation, clean-image AUROC is lower than at 50%
for all three anatomical regions. F1 at a fixed 0.5 threshold
also deteriorates, although it is sensitive to calibration.

## Potential research contribution

The study provides a controlled, three-seed factorial
investigation of the interaction between anatomical marker
placement and label-correlation strength.

A notable empirical observation is that interior markers
produce the largest reversal susceptibility at intermediate
correlation, while perfect correlation leads to near-saturation
across all regions.

Novelty relative to existing literature still requires a
careful comparison with the selected base papers and related
shortcut-learning studies.

## Important limitations

1. Synthetic label-matching and label-reversed markers use
   the true label to construct the intervention. They are
   stress tests, not natural clinical test distributions.
2. The current factorial evidence is internal to CheXpert.
   No external generalization claim is established here.
3. Marker placement may affect local contrast, texture,
   visibility and geometric context; these mechanisms are
   not separately identified.
4. Only three training seeds were evaluated.
5. The bootstrap intervals do not capture full training
   randomness.
6. F1 is reported at threshold 0.5 and should not be
   interpreted independently of calibration and prevalence.
7. The study demonstrates controlled shortcut susceptibility,
   not clinical deployment readiness.

## Recommended next steps

- Review the experiment design and findings with supervisor.
- Confirm whether RSNA/PadChest external testing is required.
- Finalize comparison with base papers and related work.
- Draft thesis methodology, results, discussion and conclusion.
- Preserve checkpoints separately from compact Git artifacts.

## Reproducibility

- Git commit: 4105093db7ba3ef60de5ac165d4326bf6786e6b3
- Frozen segmenter SHA-256: bdcbb77d4872292721a5bce8328401274d1b7d917e5e413f173873c8cf886a1d
- Full model checkpoint hashes: checkpoint_inventory.csv
- Source hashes: source_code_inventory.csv
- Training configurations: training_configs/
- Main figures: figures/
- Main tables: tables/

This is a preliminary professor-facing research summary.
Final manuscript claims remain subject to protocol review
and literature comparison.
