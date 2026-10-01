# LCDR — PadChest external validation

This directory contains compact publication and provenance
artifacts for the corrected PadChest evaluation of the
recovered LCDR seed-42 model against the frozen P7 ResNet50
control.

## Cohort

- 95,757 PA/AP images
- 63,706 patients
- 3,773 pneumonia-positive images
- 91,984 negative images

## Evaluation

Primary discrimination metrics are AUROC and AUPRC.

Confidence intervals and paired model differences use 1,000
paired patient-cluster bootstrap replicates with seed 42.

The pre-specified AUROC non-inferiority margin is -0.01.

## Leakage controls

PadChest labels were not used for:

- training
- model selection
- alpha selection
- threshold selection
- calibration fitting

Segmentation is not required during LCDR inference.

## PadChest preprocessing

Native 16-bit grayscale PNG values are explicitly mapped from
[0, 65535] to [0, 255], rounded to uint8, converted to RGB,
resized to 224 x 224, converted to tensor, and normalized using
ImageNet statistics.

This preprocessing was verified to be tensor-identical to the
historical external-validation pipeline.

## Important provenance note

These results belong to the recovered/corrected LCDR checkpoint
trained with paired image/lung-mask rotation.

They must not be combined with RSNA results produced by the
earlier lost LCDR checkpoint. The recovered checkpoint requires
its own RSNA evaluation.
