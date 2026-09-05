# ResNet50 Complementary Logit Fusion

## Purpose

This experiment evaluates a context-preserving alternative to
destructive lung masking for pneumonia classification under
cross-dataset distribution shift.

The final method combines two full-radiograph ResNet50 branches:

1. **Primary branch:** the frozen ResNet50 Original seed-42 model.
2. **Complementary branch:** a ResNet50 trained with mild
   appearance/style augmentation.

No lung pixels or surrounding anatomical context are removed at
inference time.

## Fusion rule

Predictions are combined in logit space:

`z_fusion = 0.65 * z_original + 0.35 * z_complementary`

followed by the sigmoid function.

The 65/35 weight was selected using **CheXpert validation only**,
with validation AUROC as the selection criterion. The weight was
frozen before evaluation on the untouched CheXpert test set,
RSNA, and PadChest.

No external labels were used for fusion-weight selection.

## Operating threshold

The recorded operating threshold is:

`0.5414520502090454`

This threshold was derived on CheXpert validation for the
Original42 model as the lowest threshold achieving at least
80% specificity. It is retained in the frozen analysis to avoid
target-domain threshold tuning.

Threshold-dependent fusion results should therefore be interpreted
as performance under a pre-specified source-derived operating
point rather than as an operating point optimized specifically for
the fusion.

## Main discrimination results

| Dataset | Original AUROC | Fusion AUROC | Delta |
|---|---:|---:|---:|
| CheXpert test | 0.732115 | 0.749810 | +0.017695 |
| RSNA | 0.779455 | 0.806087 | +0.026633 |
| PadChest full | 0.681157 | 0.728623 | +0.047466 |
| PadChest Physician-only | 0.700669 | 0.739014 | +0.038345 |

## Paired bootstrap results

Patient-cluster bootstrap resampling was used where repeated
patient images were present.

### AUROC

- CheXpert test:
  delta +0.017695,
  95% CI [+0.005473, +0.032117],
  p=0.007992.

- RSNA:
  delta +0.026633,
  95% CI [+0.024173, +0.029348],
  p=0.001998.

- PadChest full:
  delta +0.047466,
  95% CI [+0.044334, +0.050184],
  p=0.003992.

- PadChest Physician-only:
  delta +0.038345,
  95% CI [+0.032417, +0.044121],
  p=0.003992.

### AUPRC

AUPRC improved significantly on RSNA and both PadChest analyses.
The CheXpert-test AUPRC increased from 0.859909 to 0.867484, but
its paired bootstrap interval included zero
(delta +0.007575, 95% CI [-0.005045, +0.020811], p=0.223776).

## PadChest operating performance

On the full PadChest cohort, the frozen fusion improved AUROC,
AUPRC, Brier score, NLL, sensitivity, specificity, balanced
accuracy, precision, accuracy, NPV, and F1 relative to the
Original42 branch at the recorded source-derived threshold.

The Physician-only sensitivity analysis showed the same direction
of improvement.

## RSNA operating performance

On RSNA, fusion improved discrimination, Brier score, NLL,
sensitivity, balanced accuracy, and F1. Specificity decreased
relative to Original42, demonstrating the expected operating-point
trade-off.

## Interpretation

Earlier segmentation-guided masking experiments showed that
suppressing extra-pulmonary context could improve localization
while degrading classification discrimination. The present
complementary fusion does not claim that segmentation itself was
improved. Instead, it provides evidence that preserving the full
radiograph while combining predictors with complementary error
structure can mitigate the classification degradation associated
with destructive context suppression.

The complementary branch alone performed particularly strongly on
PadChest. It was not substituted for the frozen fusion after
observing external results, because doing so would constitute
target-set model selection.

## Reproducibility

Compact frozen results are stored under:

`results/resnet50_complementary_fusion/`

Generic scripts are provided for:

- frozen weighted-logit evaluation:
  `scripts/evaluate_complementary_fusion.py`
- paired patient-cluster bootstrap:
  `scripts/paired_bootstrap_fusion.py`

Large prediction files, datasets, and model checkpoints are
intentionally excluded from Git.
