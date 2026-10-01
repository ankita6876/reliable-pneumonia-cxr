# PadChest primary external-validation results

Frozen cohort: **95,757 images from 63,706 patients**
(**3,773 pneumonia-positive; 91,984 negative**).

| Model | AUROC (95% CI) | AUPRC (95% CI) |
|---|---:|---:|
| P7 ResNet50 | 0.6812 (0.6727–0.6900) | 0.0693 (0.0655–0.0736) |
| LCDR | 0.7499 (0.7418–0.7582) | 0.1096 (0.1035–0.1174) |

Paired LCDR − P7 difference:

- AUROC: **+0.0687**
  (95% CI +0.0612 to +0.0761)
- AUPRC: **+0.0404**
  (95% CI +0.0355 to +0.0460)

The pre-specified AUROC non-inferiority margin was **−0.01**.
The lower confidence bound for the paired AUROC difference was
**+0.0612**.

Confidence intervals were obtained using **1,000 paired
patient-cluster bootstrap replicates (seed 42)**.

PadChest labels were not used for training, model selection,
threshold selection, or calibration fitting.

This result uses the **recovered/corrected LCDR checkpoint**
(SHA-256:
`302728a6997d6ce5c1f30dd1098c05390fd29aad4f31f79e7cf68936652cf65f`)
and must not be combined with RSNA results from the earlier
lost LCDR checkpoint.
