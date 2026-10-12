# Anatomical Boundary Sensitivity: Exploratory Findings

## Experimental setup
- CheXpert validation: 64 of 979 explicitly labeled cases
- Frozen ResNet50, seed 42
- Frozen U-Net lung segmenter
- Resolution: 224 x 224
- Fixed threshold: 0.252866
- No retraining or external validation

## Initial pilot
- Original AUROC: 0.7614
- Hard-mask AUROC: 0.6886
- Highest prediction flip rate: 12.5% (6-pixel dilation)

## Matched-area controls

| Intervention | Boundary change | Control change | Valid pairs |
|---|---:|---:|---:|
| Erode 3 px | 0.0343 | 0.0841 | 64 |
| Dilate 3 px | 0.0345 | 0.0421 | 64 |
| Erode 6 px | 0.0539 | 0.0413 | 2 |
| Dilate 6 px | 0.0461 | 0.0358 | 64 |

## Interpretation
No consistent evidence of disproportionate sensitivity to lung
boundaries relative to the tested equal-area controls.

## Limitations
- Exploratory sample of 64 images
- Scattered-pixel controls differ from continuous boundary bands
- Only two valid comparisons for 6-pixel erosion
- Original classifier preprocessing requires verification
- Morphological mask perturbation is not calibrated uncertainty

## Next experiment
Spatially coherent, area-matched perturbations across anatomical regions.
