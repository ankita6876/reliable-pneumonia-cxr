# Reliable Pneumonia Detection from Chest X-Rays

This project develops and evaluates reliable methods for detecting pneumonia from chest X-ray images, with an emphasis on sound data practices, robust evaluation, and clinically responsible experimentation.

Datasets are not stored in GitHub.

## Setup

_Setup instructions will be added here._

## Classification optimisation

`scripts/classification/run_optimisation_experiment.py` applies hard masking to
the source CheXpert images using either a frozen segmentation model or a
complete indexed probability-mask cache. It refuses to fall back to unmasked
images. The current A0 cache is incomplete, so A0 uses the trained Montgomery
U-Net checkpoint:

```powershell
python scripts\classification\run_optimisation_experiment.py --config C:\Research\reliable-pneumonia-cxr\configs\classification_optimisation\A0_reproduce_current.yaml --splits-csv C:\Research\datasets\chexpert\processed\chexpert_splits.csv --image-root C:\Research\datasets\chexpert\extracted --segmentation-checkpoint C:\Research\outputs\montgomery_unet_v1\best_checkpoint.pt --output-root C:\Research\reliable-pneumonia-cxr\outputs\classification_optimisation --device cpu
```
