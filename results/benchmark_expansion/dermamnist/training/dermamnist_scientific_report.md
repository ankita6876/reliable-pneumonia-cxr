# DermaMNIST Multiclass Spatial Shortcut Study

## Experimental design

- Dataset: Official DermaMNIST+ 64x64
- Dataset MD5: b70a2f5635c6199aeaa28c31d7202e1f
- Training images: 7007
- Validation images: 1003
- Classes: 7
- Completed models: 39
- Seeds: 42, 123, 2026
- Architecture: ImageNet-pretrained ResNet18
- Backbone: frozen
- Classification head: linear, seven outputs
- Loss: class-weighted cross-entropy
- Model selection: validation macro-F1
- Reliability conditions: independent, 50%, 75%, 100%
- Locations: center, bottom-right, random

## Counterfactual evaluation

Each validation image is evaluated with:

1. No marker
2. Neutral marker
3. Marker matching its true class
4. Each of the seven possible marker codes

For each image, the six incorrect-code predictions are
averaged to construct a descriptive alternative-code
probability distribution.

The primary reliance summaries include:

- Matching-minus-alternative macro AUROC
- Matching-minus-alternative macro-F1
- Mean total variation probability change
- Counterfactual decision flip rate

The averaged alternative-code distribution is a
summary of multiple interventions, not a single
realizable marker assignment.

## Dataset imbalance

Training class counts:
[228, 359, 769, 80, 779, 4693, 99]

Class-weighted training and macro metrics are used
because the class distribution is highly imbalanced.

## Interpretation limitations

- This is a controlled synthetic-shortcut study.
- Results are based on validation data.
- The official test split is reserved for final evaluation.
- Only three training seeds are used.
- The backbone is frozen.
- Spatial rankings may depend on the dataset and marker.
- Multiclass AUROC gaps should not be pooled directly
  with binary BreastMNIST AUROC gaps.
- A difference between datasets is not evidence that
  training-set size alone caused that difference.
- Independent marker controls can still show visual
  perturbation sensitivity.

## Next steps

1. Review validation results.
2. Audit the experimental implementation.
3. Preserve exact executable code.
4. Compare with BreastMNIST using compatible metrics.
5. Extend to CIFAR-10 and PathMNIST.
6. Perform within-dataset training-size ablations.
