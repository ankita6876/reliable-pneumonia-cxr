# BreastMNIST Spatial Shortcut Study

## Experimental status

- Dataset: BreastMNIST+ (64 × 64)
- Training images: 546
- Validation images: 78
- Test images: 156
- Models completed: 30
- Training seeds: 42, 123, 2026
- Shortcut reliability: 50%, 75%, 100%
- Training locations: center, bottom-right, random
- Architecture: ImageNet-pretrained ResNet-18 with frozen backbone
- Evaluation: matching, reversed, neutral, clean

## Main descriptive findings

### Reliability-dependent shortcut reliance

At the center training/evaluation location, the mean
matching-minus-reversed AUROC gaps were:

- 50% reliability: 0.0272
- 75% reliability: 0.4211
- 100% reliability: 0.8338

This demonstrates a strong descriptive increase in
shortcut reliance as the marker becomes more predictive.

### Spatial dependence

At 100% reliability, the matched-location AUROC gaps were:

- Center: 0.8338
- Bottom-right: 0.3400
- Random: 0.5018

The magnitude of shortcut reliance differs by location.

### Random-location shortcuts

The random-location model retained a substantial
matching-minus-reversed AUROC gap at 100% reliability.

Thus, randomizing marker position alone did not eliminate
shortcut dependence in this experiment.

## Important limitations

1. Only three training seeds were used.
2. The test set contains 156 images.
3. BreastMNIST has substantial class imbalance.
4. The model uses a frozen ImageNet-pretrained backbone.
5. Synthetic marker sensitivity is not proof of
   naturally occurring clinical shortcut learning.
6. Probability changes can also reflect visual
   perturbation sensitivity.
7. Neutral-marker and clean-model comparisons must be
   considered alongside matching/reversed interventions.
8. Results should not be treated as an isolated causal
   effect of dataset size.
9. Hyperparameters and model selection use validation
   performance; the official test split should not be
   used to tune the protocol.
10. The code used a frozen backbone, so the reported
    results characterize the trained classifier head
    operating on pretrained image representations.

## Scientific interpretation

The experiment supports the hypothesis that shortcut
reliance depends on marker reliability and spatial
placement.

It does not establish a universal ranking of locations,
nor does it establish that larger datasets necessarily
reduce shortcut learning.

## Next experiment

Repeat the standardized protocol on DermaMNIST,
while preserving the multiclass labels and defining
a scientifically valid multiclass shortcut encoding.
