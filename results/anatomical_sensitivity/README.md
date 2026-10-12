# Anatomical Region Sensitivity Experiments

## MSc Thesis
Reliability of pneumonia classification from chest X-rays
under cross-dataset shift.

## Datasets
- CheXpert: development and internal validation
- RSNA: external evaluation
- PadChest: external evaluation
- JSRT: lung segmentation development

## Experimental Models
- Frozen ResNet50 pneumonia classifier
- Frozen U-Net lung segmenter

## Experiments
1. Initial anatomical boundary perturbation pilot
2. Matched-area perturbation controls
3. Repeated spatial-patch sensitivity experiment
4. Blackout versus Gaussian blur comparison
5. Full CheXpert validation sensitivity analysis

## Patch Protocol
- Input size: 224 x 224
- Patch size: 8 x 8
- Regions: lung interior, inner boundary, exterior
- 10 patch placements per region
- Blackout and Gaussian blur
- Frozen models; no additional training

## Statistical Analysis
- Within-image repeated patch aggregation
- Paired anatomical region comparisons
- Bootstrap confidence intervals
- Patient-aware analysis where applicable

## Important Interpretation
The experiments measure prediction sensitivity under
localized image perturbations.

They do not establish causal anatomical reliance,
prove clinical correctness, or eliminate shortcut
learning explanations.

## Storage Policy
Store:
- Publication-ready figures
- Statistical tables
- Experiment metadata
- Compact or compressed predictions
- Experiment reports

Do not store:
- Raw datasets
- Large lung-mask caches
- Model checkpoints
- Temporary inference files
