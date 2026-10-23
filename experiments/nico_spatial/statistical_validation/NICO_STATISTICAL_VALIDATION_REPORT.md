
# NICO Spatial–Context Shortcut Study
## Paired Statistical Validation Report

### Status

Exploratory NICO pilot with image-level paired statistical analysis.

This report does not claim that the experiment is a
corrected confirmatory training run.

### Dataset

- Source: NICO dataset, Kaggle alexandraneagu101/nico-dataset
- Classes: airplane and car
- Contexts: autumn, dim, grass, outdoor, rock, water
- Original files: 3318
- Exact duplicate copies removed: 56
- Conflicting duplicate groups excluded: 2
- Final unique cohort: 3260
- Train: 2282
- Validation: 489
- Locked test: 489
- Test image access during this analysis: none

### Model and training provenance

- Backbone: ImageNet-pretrained ResNet50
- Training conditions: clean, center, bottom_right
- Seeds: 42, 123, 2026
- Marker: 24x24 pixels
- Input: 224x224 RGB
- Counterfactual positions: five
- Checkpoints: nine existing pilot checkpoints
- Training during this analysis: none

### Known pilot limitation

The original training implementation executed an
unused mixed-precision forward pass before its
full-precision optimization pass.

Because ResNet50 uses BatchNorm, this could affect
running statistics.

All findings are therefore classified as exploratory
until reproduced using corrected training code.

### Bootstrap methodology

- Resampling unit: validation image
- Pairing: all model/marker predictions for an image
  share the same bootstrap indices
- Strata: class x context
- Bootstrap repetitions: 2000
- Random seed: 240126
- Reported intervals: percentile 95% confidence intervals
- Training-seed variability: not fully represented
- Three seeds are averaged before image resampling

### Primary matched-location findings

- Clean-trained, center-tested:
  0.0132 [0.0107, 0.0161]

- Center-trained, center-tested:
  0.8805 [0.8689, 0.8919]

- Bottom-right-trained, bottom-right-tested:
  0.6908 [0.6791, 0.7028]

### Full spatial sensitivity matrix

| condition    |   top_left |   top_right |   bottom_left |   bottom_right |   center |
|:-------------|-----------:|------------:|--------------:|---------------:|---------:|
| clean        |    0.00184 |     0.00193 |       0.00229 |        0.00153 |  0.01321 |
| center       |    0.49842 |     0.2777  |       0.22628 |        0.13724 |  0.88053 |
| bottom_right |    0.20689 |     0.34089 |       0.33099 |        0.69079 |  0.47809 |

### Clean-image validation performance

| model                 | condition    |   seed |   auroc |   auprc |   accuracy |   precision |   recall |      f1 |
|:----------------------|:-------------|-------:|--------:|--------:|-----------:|------------:|---------:|--------:|
| clean_seed42          | clean        |     42 | 0.99745 | 0.99813 |    0.9816  |     0.98155 |  0.98519 | 0.98336 |
| clean_seed123         | clean        |    123 | 0.99711 | 0.99782 |    0.9816  |     0.98155 |  0.98519 | 0.98336 |
| clean_seed2026        | clean        |   2026 | 0.99782 | 0.99835 |    0.97955 |     0.99618 |  0.96667 | 0.9812  |
| center_seed42         | center       |     42 | 0.8728  | 0.841   |    0.60327 |     0.5819  |  1       | 0.73569 |
| center_seed123        | center       |    123 | 0.97122 | 0.97081 |    0.78119 |     0.71618 |  1       | 0.83462 |
| center_seed2026       | center       |   2026 | 0.95523 | 0.95514 |    0.67689 |     0.63084 |  1       | 0.77364 |
| bottom_right_seed42   | bottom_right |     42 | 0.99481 | 0.99533 |    0.93456 |     0.89667 |  0.9963  | 0.94386 |
| bottom_right_seed123  | bottom_right |    123 | 0.99719 | 0.99772 |    0.92229 |     0.87662 |  1       | 0.93426 |
| bottom_right_seed2026 | bottom_right |   2026 | 0.99767 | 0.99782 |    0.97955 |     0.97101 |  0.99259 | 0.98168 |

### Paired contrasts

| contrast                                      | metric   |   estimate |   ci_lower |   ci_upper |
|:----------------------------------------------|:---------|-----------:|-----------:|-----------:|
| center_trained_vs_clean_at_center             | switch   |    0.86731 |    0.8562  |    0.87864 |
| bottom_right_trained_vs_clean_at_bottom_right | switch   |    0.68926 |    0.67766 |    0.70109 |
| center_matched_minus_bottom_right             | switch   |    0.74329 |    0.72892 |    0.75716 |
| bottom_right_matched_minus_center             | switch   |    0.21271 |    0.19302 |    0.23159 |

### Context-specific matched-location results

| condition    | position     | context   |   n |   mean_abs_switch |   ci_lower |   ci_upper |   decision_flip_rate |   flip_ci_lower |   flip_ci_upper |
|:-------------|:-------------|:----------|----:|------------------:|-----------:|-----------:|---------------------:|----------------:|----------------:|
| clean        | center       | autumn    |  47 |           0.01722 |    0.00584 |    0.03481 |              0.00709 |         0       |         0.02128 |
| clean        | center       | dim       |  79 |           0.0104  |    0.00577 |    0.01623 |              0.01266 |         0       |         0.02954 |
| clean        | center       | grass     |  87 |           0.01968 |    0.01321 |    0.02705 |              0.01533 |         0.00383 |         0.03065 |
| clean        | center       | outdoor   | 111 |           0.01426 |    0.00898 |    0.02109 |              0.00601 |         0       |         0.01502 |
| clean        | center       | rock      |  81 |           0.00677 |    0.00473 |    0.00925 |              0.00412 |         0       |         0.01235 |
| clean        | center       | water     |  84 |           0.01176 |    0.00775 |    0.01668 |              0.00397 |         0       |         0.0119  |
| center       | center       | autumn    |  47 |           0.89717 |    0.86875 |    0.92756 |              0.91489 |         0.87234 |         0.95035 |
| center       | center       | dim       |  79 |           0.88948 |    0.86262 |    0.91533 |              0.91983 |         0.88186 |         0.95781 |
| center       | center       | grass     |  87 |           0.91178 |    0.88836 |    0.93617 |              0.9387  |         0.90805 |         0.96552 |
| center       | center       | outdoor   | 111 |           0.85559 |    0.82595 |    0.88585 |              0.86787 |         0.82282 |         0.90991 |
| center       | center       | rock      |  81 |           0.87312 |    0.84595 |    0.89742 |              0.91358 |         0.87243 |         0.95062 |
| center       | center       | water     |  84 |           0.87052 |    0.84278 |    0.89712 |              0.89286 |         0.84921 |         0.93254 |
| bottom_right | bottom_right | autumn    |  47 |           0.70055 |    0.65325 |    0.74701 |              0.73759 |         0.65957 |         0.8156  |
| bottom_right | bottom_right | dim       |  79 |           0.6344  |    0.60778 |    0.66293 |              0.65401 |         0.61603 |         0.69198 |
| bottom_right | bottom_right | grass     |  87 |           0.72421 |    0.69879 |    0.74774 |              0.7433  |         0.69732 |         0.78544 |
| bottom_right | bottom_right | outdoor   | 111 |           0.69617 |    0.67028 |    0.72323 |              0.69369 |         0.65465 |         0.73574 |
| bottom_right | bottom_right | rock      |  81 |           0.6839  |    0.65985 |    0.70772 |              0.69547 |         0.65844 |         0.73251 |
| bottom_right | bottom_right | water     |  84 |           0.7033  |    0.67888 |    0.72917 |              0.69841 |         0.65476 |         0.74206 |

### Context heterogeneity

| condition    |   observed_context_range |   bootstrap_ci_lower |   bootstrap_ci_upper | interpretation                                        |
|:-------------|-------------------------:|---------------------:|---------------------:|:------------------------------------------------------|
| center       |                  0.05619 |              0.03417 |              0.09856 | descriptive_heterogeneity_not_formal_interaction_test |
| bottom_right |                  0.08981 |              0.05833 |              0.12977 | descriptive_heterogeneity_not_formal_interaction_test |

### Neutral-patch controls

| condition    | position     |   orientation_switch |   neutral_vs_clean |
|:-------------|:-------------|---------------------:|-------------------:|
| clean        | center       |              0.01321 |            0.0107  |
| center       | center       |              0.88053 |            0.01615 |
| bottom_right | bottom_right |              0.69079 |            0.00168 |

### Interpretation

The principal measurement is the absolute change in
predicted probability of the car class when marker
orientation is changed while the source image and
marker coordinates are held constant.

Decision-flip rate is the proportion of images for
which the predicted binary class changes under
that paired intervention.

Matched-location contrasts examine whether a model
is more sensitive at the position where its marker
was placed during training.

The neutral-patch control compares a uniform gray
patch with the clean image. This measures a distinct
intervention from changing stripe orientation.

The difference between these magnitudes must not be
interpreted as an additive causal decomposition.

NICO context labels do not provide segmentation
masks or direct evidence of foreground overlap.

Differences between contexts are descriptive and
do not by themselves demonstrate causal
environmental interactions.

Exact-image deduplication does not eliminate the
possibility of visually near-identical images.

### Main limitations

1. Original pilot training contained a redundant
   BatchNorm-updating forward pass.
2. Model selection used validation AUROC under the
   corresponding training-marker condition.
3. Image-bootstrap intervals are conditional on
   the three trained seeds.
4. No segmentation-based object-overlap control.
5. No near-duplicate perceptual-hash audit yet.
6. Locked test set has not been evaluated.
7. Novelty relative to closest papers remains to
   be established through literature comparison.

### Recommended next steps

1. Preserve all pilot checkpoints and artifacts.
2. Correct training implementation in a versioned
   confirmatory protocol.
3. Audit near-duplicate image leakage.
4. Define confirmatory contrasts and multiplicity
   handling before test evaluation.
5. Compare results against Oxford-IIIT Pet.
6. Review base papers and define the precise
   contribution without unsupported novelty claims.

### Reproducibility

- Manifest SHA256: 37804e81504d0184e171ca1691e7a1ac4d0d66f98edcb98dd8ddee2c043660c8
- Validation images: 489
- Models: 9
- Model-position combinations: 45
- Image-level counterfactual rows: 22005
- Test evaluation: none
