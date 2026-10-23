# Oxford-IIIT Pet Spatial Shortcut Study
## Consolidated Experimental Report

Generated: 2026-10-23 17:03 UTC

### 1. Research objective

Investigate how synthetic label-associated markers influence
image-classifier predictions, with particular attention to:

- Absolute marker location
- Animal foreground versus background placement
- Training-time marker location
- Counterfactual changes in marker orientation
- Image-content occlusion

This study is a nonmedical extension of the broader thesis
investigating shortcut sensitivity and model reliability.

### 2. Experimental setup

Dataset: Oxford-IIIT Pet.

Classifier: ImageNet-pretrained ResNet50, two-class cat/dog head.

Input resolution: 224 × 224 pixels.

Synthetic marker: 24 × 24 striped patch.

Locked development validation cohort: 722 images.

Training conditions represented in the replicated analysis:
clean, interior, background, center, bottom-right.

Seeds: 42, 123, 2026.

Total evaluated checkpoints: 15.

Counterfactual positions:
top-left, top-right, bottom-left, bottom-right,
center, animal interior, and background.

The official Oxford-IIIT Pet test set has not been used
in the reported analyses.

### 3. Coordinate-controlled counterfactual results

The same trained model and image were evaluated using two
marker orientations at identical coordinates.

The primary measure was:

mean absolute change in predicted dog probability
between the two marker orientations.

| tested_position | mean_abs_change | mean_flip_pct | mean_foreground_overlap |
| --- | --- | --- | --- |
| center | 0.01356 | 0.85873 | 0.85546 |
| interior | 0.01194 | 0.88643 | 1.0 |
| bottom_right | 0.0094 | 0.70175 | 0.1094 |
| background | 0.00865 | 0.62789 | 0.0 |
| bottom_left | 0.00696 | 0.52632 | 0.09422 |
| top_right | 0.00638 | 0.45245 | 0.01611 |
| top_left | 0.0048 | 0.44321 | 0.0247 |

#### Training-condition-by-test-position sensitivity

| trained_region | top_left | top_right | bottom_left | bottom_right | center | interior | background |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.00467 | 0.00356 | 0.00387 | 0.00337 | 0.01274 | 0.01127 | 0.00554 |
| interior | 0.00421 | 0.00465 | 0.00541 | 0.0047 | 0.01501 | 0.01418 | 0.00792 |
| background | 0.0051 | 0.00719 | 0.00729 | 0.00786 | 0.01133 | 0.01056 | 0.01025 |
| center | 0.00382 | 0.0039 | 0.00426 | 0.00334 | 0.01619 | 0.01185 | 0.00565 |
| bottom_right | 0.00622 | 0.01259 | 0.01397 | 0.02775 | 0.01256 | 0.01183 | 0.01387 |

**Interpretation**

Marker sensitivity varies by both the location used for
testing and the training condition.

The center and animal interior show relatively high
sensitivity when averaged across trained models.

The bottom-right-trained models exhibit especially high
sensitivity to bottom-right marker changes.

These observations support spatially heterogeneous
responses to synthetic shortcut cues.

They do not establish that training position is the sole
cause of the observed differences, because separately
trained models may differ in other learned properties.

### 4. Neutral-patch control

A fixed RGB (128, 128, 128) patch was applied at the same
coordinates as the striped markers.

Three descriptive comparisons were used:

1. Stripe encoding effect:
   absolute prediction difference between marker types.

2. Neutral occlusion:
   absolute prediction difference between neutral patch
   and clean image.

3. Encoding-minus-neutral:
   difference between the two preceding measures.

| tested_position | encoding_effect | neutral_occlusion | encoding_minus_neutral |
| --- | --- | --- | --- |
| center | 0.01356 | 0.01125 | 0.00231 |
| interior | 0.01194 | 0.00985 | 0.00209 |
| bottom_right | 0.0094 | 0.00246 | 0.00694 |
| background | 0.00865 | 0.00371 | 0.00494 |
| bottom_left | 0.00696 | 0.00347 | 0.00349 |
| top_right | 0.00638 | 0.00342 | 0.00296 |
| top_left | 0.0048 | 0.00457 | 0.00023 |

#### Model-specific encoding-minus-neutral matrix

| trained_region | top_left | top_right | bottom_left | bottom_right | center | interior | background |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clean | -6e-05 | 0.00024 | 0.00035 | 0.00085 | 0.00167 | 0.00142 | 0.0018 |
| interior | -0.00036 | 0.00121 | 0.002 | 0.00231 | 0.00352 | 0.0042 | 0.00421 |
| background | 0.00051 | 0.00357 | 0.0038 | 0.00541 | 0.00014 | 0.00083 | 0.00654 |
| center | -0.00061 | 0.00056 | 0.00089 | 0.00093 | 0.00489 | 0.00194 | 0.00201 |
| bottom_right | 0.0017 | 0.0092 | 0.01041 | 0.02522 | 0.00134 | 0.00204 | 0.01013 |

**Interpretation**

Stripe-orientation sensitivity is not equivalent to the
effect of introducing a neutral patch.

At several positions, especially bottom-right, the
orientation-change measure exceeds the neutral-occlusion
measure.

This is consistent with sensitivity to marker appearance
beyond simple image occlusion.

However, these measures are not an additive causal
decomposition. A single gray patch does not control for
every possible local texture, contrast, or replacement
effect.

### 5. Exploratory bootstrap analysis

Paired bootstrap comparisons were computed across
images and training seeds.

| trained_region | tested_position | n_seeds | mean_encoding_minus_neutral | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| clean | top_left | 3 | -6e-05 | -0.00062 | 0.00049 |
| clean | top_right | 3 | 0.00024 | -0.00036 | 0.00086 |
| clean | bottom_left | 3 | 0.00035 | -8e-05 | 0.00079 |
| clean | bottom_right | 3 | 0.00085 | 0.0004 | 0.00139 |
| clean | center | 3 | 0.00167 | 0.00055 | 0.00277 |
| clean | interior | 3 | 0.00142 | 0.00029 | 0.00246 |
| clean | background | 3 | 0.0018 | 0.00101 | 0.00265 |
| interior | top_left | 3 | -0.00036 | -0.00127 | 0.00084 |
| interior | top_right | 3 | 0.00121 | -0.00032 | 0.00361 |
| interior | bottom_left | 3 | 0.002 | 0.00019 | 0.005 |
| interior | bottom_right | 3 | 0.00231 | 0.0001 | 0.00554 |
| interior | center | 3 | 0.00352 | -0.00132 | 0.00852 |
| interior | interior | 3 | 0.0042 | -0.00076 | 0.00959 |
| interior | background | 3 | 0.00421 | 0.00072 | 0.00861 |
| background | top_left | 3 | 0.00051 | -0.00098 | 0.00179 |
| background | top_right | 3 | 0.00357 | 0.0006 | 0.0064 |
| background | bottom_left | 3 | 0.0038 | 0.00098 | 0.00751 |
| background | bottom_right | 3 | 0.00541 | 0.00179 | 0.01008 |
| background | center | 3 | 0.00014 | -0.00146 | 0.00182 |
| background | interior | 3 | 0.00083 | -0.00113 | 0.00334 |
| background | background | 3 | 0.00654 | 0.00274 | 0.01211 |
| center | top_left | 3 | -0.00061 | -0.00103 | -0.00022 |
| center | top_right | 3 | 0.00056 | -0.00011 | 0.00118 |
| center | bottom_left | 3 | 0.00089 | 0.00032 | 0.00157 |
| center | bottom_right | 3 | 0.00093 | 0.00027 | 0.00178 |
| center | center | 3 | 0.00489 | 0.00086 | 0.00781 |
| center | interior | 3 | 0.00194 | -0.0001 | 0.00391 |
| center | background | 3 | 0.00201 | 0.00075 | 0.00331 |
| bottom_right | top_left | 3 | 0.0017 | -0.00042 | 0.00338 |
| bottom_right | top_right | 3 | 0.0092 | 0.00218 | 0.01346 |
| bottom_right | bottom_left | 3 | 0.01041 | 0.00351 | 0.0154 |
| bottom_right | bottom_right | 3 | 0.02522 | 0.01101 | 0.0338 |
| bottom_right | center | 3 | 0.00134 | -8e-05 | 0.00282 |
| bottom_right | interior | 3 | 0.00204 | 0.00033 | 0.00372 |
| bottom_right | background | 3 | 0.01013 | 0.00422 | 0.01476 |

**Statistical caution**

Only three training seeds are available. Bootstrap
intervals should be interpreted as exploratory and
should not be presented as definitive evidence of
generalization across arbitrary training runs.

The analysis also involves multiple comparisons;
unadjusted intervals do not provide family-wise
error control.

### 6. Main findings

**Finding A — Position sensitivity**

The response to marker orientation differs substantially
across spatial locations.

**Finding B — Training-position interaction**

Models trained with markers at different positions show
different spatial sensitivity profiles.

**Finding C — Neutral-patch comparison**

For some model-position combinations, switching the
marker orientation has a greater measured effect than
introducing a fixed neutral patch.

**Finding D — High predictive performance does not
guarantee marker invariance**

A classifier may achieve strong validation performance
while remaining sensitive to synthetic marker changes.

### 7. Limitations

- The development training subset is small.
- Only three training seeds were used for the replicated
  conditions.
- The seven test positions do not form a complete
  factorial training design.
- Marker position and object overlap are not fully
  disentangled.
- Synthetic markers are controlled perturbations,
  not necessarily realistic dataset shortcuts.
- A neutral gray patch is an incomplete occlusion control.
- Validation results are exploratory.
- No final held-out test results are included.
- The Oxford-IIIT Pet experiments do not themselves
  demonstrate improvement in CheXpert-to-RSNA/PadChest
  pneumonia classification.

### 8. Final evaluation protocol — proposed, not yet executed

Before accessing the official test set:

1. Freeze the primary research questions and hypotheses.
2. Freeze the exact model checkpoints and preprocessing.
3. Verify that seed-specific training recipes are equivalent.
4. Predefine the primary counterfactual effect and
   comparisons, with secondary outcomes clearly separated.
5. Specify uncertainty estimation and multiple-comparison
   handling.
6. Audit for train/validation/test overlap.
7. Record checkpoint hashes and the source-code commit.
8. Confirm that no official test labels influenced
   experiment selection or model development.
9. Obtain supervisor approval of the evaluation protocol.
10. Execute the official test evaluation once, report all
    predefined comparisons, and disclose any deviations.

### 9. Remaining thesis integration

The Oxford-IIIT Pet study provides a nonmedical
generalization experiment for the spatial-shortcut
research question.

The clinical thesis must still separately establish:

- The final CheXpert-trained proposed model.
- Its distinction from relevant base papers.
- Comparison with locked clinical baselines.
- External RSNA and PadChest performance.
- AUROC, AUPRC, F1, sensitivity, specificity,
  calibration, and appropriate uncertainty estimates.
- Whether the predefined clinical claims are supported.

The nonmedical results should not substitute for
these clinical evaluations.

### 10. Current conclusion

The Oxford-IIIT Pet experiments demonstrate that
ResNet50 classifiers can exhibit position-dependent
sensitivity to synthetic label-associated visual markers,
and that this sensitivity cannot be described solely by
the effect of replacing image content with a neutral
gray patch.

The strongest defensible contribution at this stage is
a controlled characterization of spatial shortcut
sensitivity, not a claim of universal robustness
improvement.

