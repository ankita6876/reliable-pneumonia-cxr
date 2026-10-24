# NICO Confirmatory V1 — Statistical Validation Report

## Study design

- Dataset: NICO airplane versus car
- Training: 2,133 images
- Validation: 557 images
- Reserved test: 489 images, not evaluated
- Excluded: 81 images
- Backbone: ImageNet-pretrained ResNet50
- Training conditions: clean, center, bottom_right
- Seeds: 42, 123, 2026
- Selected checkpoints: matched-condition validation AUROC
- Marker: 24x24 orientation-coded striped patch
- Analysis: validation predictions only

## Primary outcome

Mean absolute change in predicted probability of car
when marker orientation changes from 0 to 1 at a
fixed image and marker position.

## Spatial sensitivity

| condition    | position     |   estimate |   ci_low |   ci_high |
|:-------------|:-------------|-----------:|---------:|----------:|
| clean        | top_left     |    0.00545 |  0.0043  |   0.00673 |
| clean        | top_right    |    0.00491 |  0.00395 |   0.00591 |
| clean        | bottom_left  |    0.00445 |  0.00336 |   0.00568 |
| clean        | bottom_right |    0.00664 |  0.00537 |   0.00806 |
| clean        | center       |    0.02308 |  0.01984 |   0.02666 |
| center       | top_left     |    0.82504 |  0.81832 |   0.83175 |
| center       | top_right    |    0.75204 |  0.74328 |   0.76089 |
| center       | bottom_left  |    0.75556 |  0.7468  |   0.76398 |
| center       | bottom_right |    0.67672 |  0.66695 |   0.6864  |
| center       | center       |    0.99692 |  0.99666 |   0.99712 |
| bottom_right | top_left     |    0.18367 |  0.1765  |   0.19094 |
| bottom_right | top_right    |    0.48109 |  0.4674  |   0.49435 |
| bottom_right | bottom_left  |    0.59434 |  0.5799  |   0.60855 |
| bottom_right | bottom_right |    0.94137 |  0.93445 |   0.94771 |
| bottom_right | center       |    0.55643 |  0.53801 |   0.57409 |

## Paired comparisons

| comparison                                      | condition_a   | position_a   | condition_b   | position_b   |   estimate |   ci_low |   ci_high |
|:------------------------------------------------|:--------------|:-------------|:--------------|:-------------|-----------:|---------:|----------:|
| center vs clean at center                       | center        | center       | clean         | center       |    0.97383 |  0.9703  |   0.97702 |
| bottom-right vs clean at bottom-right           | bottom_right  | bottom_right | clean         | bottom_right |    0.93472 |  0.92791 |   0.9409  |
| center-trained: center vs bottom-right          | center        | center       | center        | bottom_right |    0.3202  |  0.31059 |   0.32995 |
| bottom-right-trained: bottom-right vs center    | bottom_right  | bottom_right | bottom_right  | center       |    0.38494 |  0.37036 |   0.40013 |
| center vs bottom-right training at center       | center        | center       | bottom_right  | center       |    0.44049 |  0.42286 |   0.45898 |
| center vs bottom-right training at bottom-right | center        | bottom_right | bottom_right  | bottom_right |   -0.26465 | -0.27412 |  -0.25499 |

## Neutral patch control

| condition    | position     |   estimate |   ci_low |   ci_high |
|:-------------|:-------------|-----------:|---------:|----------:|
| clean        | top_left     |    0.00479 |  0.0038  |   0.00578 |
| clean        | top_right    |    0.00547 |  0.00439 |   0.00669 |
| clean        | bottom_left  |    0.00401 |  0.00322 |   0.00483 |
| clean        | bottom_right |    0.00512 |  0.00406 |   0.00621 |
| clean        | center       |    0.02402 |  0.01932 |   0.02923 |
| center       | top_left     |    0.00687 |  0.0065  |   0.00728 |
| center       | top_right    |    0.00668 |  0.00627 |   0.00711 |
| center       | bottom_left  |    0.00864 |  0.00811 |   0.00921 |
| center       | bottom_right |    0.01115 |  0.01037 |   0.01193 |
| center       | center       |    0.03565 |  0.03335 |   0.03793 |
| bottom_right | top_left     |    0.00847 |  0.00775 |   0.00919 |
| bottom_right | top_right    |    0.00973 |  0.00899 |   0.01048 |
| bottom_right | bottom_left  |    0.00784 |  0.00724 |   0.00843 |
| bottom_right | bottom_right |    0.01709 |  0.01573 |   0.01846 |
| bottom_right | center       |    0.02611 |  0.02371 |   0.02878 |

## Seed robustness

| condition    | position     |    mean |   sd_across_seeds |   min_seed |   max_seed |
|:-------------|:-------------|--------:|------------------:|-----------:|-----------:|
| clean        | center       | 0.02308 |           0.01086 |    0.014   |    0.03511 |
| center       | center       | 0.99692 |           0.00037 |    0.99653 |    0.99728 |
| bottom_right | bottom_right | 0.94137 |           0.01175 |    0.933   |    0.9548  |

## Context heterogeneity

| condition    | position     | minimum_context   | maximum_context   |   minimum_effect |   maximum_effect |   range |
|:-------------|:-------------|:------------------|:------------------|-----------------:|-----------------:|--------:|
| clean        | center       | autumn            | grass             |          0.0157  |          0.03178 | 0.01608 |
| center       | center       | rock              | grass             |          0.99607 |          0.99761 | 0.00155 |
| bottom_right | bottom_right | autumn            | grass             |          0.92175 |          0.968   | 0.04624 |

## Statistical procedure

For each image, outcome values were averaged over
the three trained seeds within each condition.

Two thousand bootstrap samples were drawn with
replacement within the original label-by-context
strata, preserving stratum sample sizes.

Paired differences were calculated on the same
validation images.

The resulting percentile intervals describe
image-sampling uncertainty conditional on the
three fitted seeds. They do not capture full
random-seed training uncertainty.

These intervals are descriptive and unadjusted
for multiple comparisons.

## Important limitations

1. Validation data were used for checkpoint selection.
2. The same validation images were used for
   counterfactual analysis.
3. The historical reserved test cohort was exposed
   to previous exploratory v2 development.
4. Perceptual hashing cannot detect every possible
   near-duplicate.
5. Context differences are descriptive, not proof
   of causal context effects.
6. Neutral patches measure a different intervention
   from switching marker orientation. Their effects
   should not be interpreted as a direct decomposition
   of the marker effect.
7. The original v2 training source was unavailable,
   so this is a newly documented implementation.

## Interpretation guide

Large marker-orientation effects indicate sensitivity
to the encoded marker under the tested intervention.

A high matched-condition AUROC does not establish
robust recognition of the underlying object.

Differences across positions provide evidence about
spatial sensitivity, conditional on this dataset,
marker design, and training recipe.

## Test integrity

No test predictions were generated by this analysis.
