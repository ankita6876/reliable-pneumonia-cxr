# Anatomical Region Sensitivity Experiments

MSc thesis: pneumonia classification reliability under dataset shift.

## Experiments
- Initial lung-boundary perturbation pilot
- Repeated spatial-patch sensitivity pilot
- Blackout versus Gaussian-blur control
- Full CheXpert validation analysis

## Protocol
- Frozen ResNet50 pneumonia classifier
- Frozen U-Net lung segmenter
- Lung interior, inner boundary, and exterior regions
- 8 x 8 pixel patches
- 10 sampled patches per region
- Blackout and Gaussian-blur perturbations

## Interpretation
These experiments investigate prediction sensitivity under
localized interventions. They do not establish causal anatomical
reliance or eliminate shortcut-learning explanations.

## Storage
Compact tables, figures, metadata, and compressed prediction
files are retained here. Datasets and model checkpoints are excluded.

## Reproducibility
Refer to the experiment metadata and the repository's source code.
