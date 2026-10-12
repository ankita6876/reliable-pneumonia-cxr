# Contrast-Controlled Anatomical Shortcut Experiment

## Experimental configuration

- Dataset: CheXpert matched cohort
- Training images: 3,220
- Validation images: 679
- Validation patients: 594
- Backbone: ResNet50
- Seed: 42
- Training epochs: 3
- Batch size: 8
- BPEC lambda: 0
- Marker size: 24 x 24
- Marker amplitude: 45
- Marker patterns: vertical and horizontal stripes
- Training marker-label correlation: 1.0
- Checkpoint selection: clean validation AUROC
- Bootstrap: 2,000 patient-clustered replicates

## Shortcut sensitivity

Matching AUROC minus reversed AUROC:

| Region | Original solid marker | Contrast-controlled stripe marker |
|---|---:|---:|
| Interior | 0.8876 | 1.0000 |
| Boundary | 0.0306 | 0.9994 |
| Exterior | 0.4923 | 0.9965 |

## Interpretation

The contrast-controlled stripe protocol produced near-perfect
marker reliance in all three anatomical regions.

The original region-dependent shortcut effect did not retain
its large magnitude under the alternative marker encoding.

This suggests that anatomical placement, marker appearance,
and their interaction must be considered when interpreting
shortcut learning.

The experiment changes both contrast behavior and marker
encoding. It therefore does not isolate contrast as a single
causal variable.

All reported results come from one training seed and the
matched CheXpert development cohort.

External RSNA and PadChest datasets were not used.
