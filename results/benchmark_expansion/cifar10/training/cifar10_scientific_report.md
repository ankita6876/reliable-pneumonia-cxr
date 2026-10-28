# CIFAR-10 Spatial Shortcut Size-Ablation Study

## Dataset
Official CIFAR-10, 10 classes.

## Training
Nested class-stratified subsets:
[5000, 15000, 45000]

## Validation
Fixed 5,000-image stratified subset from the official
training split.

## Test
Official 10,000-image test split reserved and not accessed.

## Architecture
ImageNet-pretrained ResNet18, frozen backbone,
linear ten-class classification head.

## Conditions
Three seeds: [42, 123, 2026]
Reliability: ['independent', 0.5, 0.75, 1.0]
Position: ['center', 'bottom_right', 'random']

## Completed models
117

## Evaluation
Macro-AUROC, macro-F1, balanced accuracy, accuracy,
log loss, total variation, counterfactual decision flips.

## Limitations
- Synthetic shortcut markers.
- Frozen backbone.
- Validation split used for model selection and analysis.
- Marker pattern recognition may vary between codes.
- This experiment isolates sample size within CIFAR-10,
  but does not establish universal effects across domains.
- Held-out test evaluation is still required.
