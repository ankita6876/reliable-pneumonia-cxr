# NICO v2 Statistical Validation

## Dataset and experimental design

- Validation images: 489
- Models: 9
- Training conditions: clean, center, bottom_right
- Seeds: 42, 123, 2026
- Bootstrap replicates: 2000
- Bootstrap stratification: class × context
- Bootstrap unit: image
- Three seeds averaged before resampling

## Main sensitivity estimates

| condition    | position     |   estimate |   ci_lower |   ci_upper |
|:-------------|:-------------|-----------:|-----------:|-----------:|
| clean        | top_left     |    0.00201 |    0.00145 |    0.00267 |
| clean        | top_right    |    0.00171 |    0.00116 |    0.0024  |
| clean        | bottom_left  |    0.00197 |    0.00134 |    0.00283 |
| clean        | bottom_right |    0.00138 |    0.00103 |    0.00177 |
| clean        | center       |    0.01124 |    0.00885 |    0.01396 |
| center       | top_left     |    0.49726 |    0.48464 |    0.51102 |
| center       | top_right    |    0.32282 |    0.3099  |    0.33667 |
| center       | bottom_left  |    0.27729 |    0.26437 |    0.29127 |
| center       | bottom_right |    0.16972 |    0.16055 |    0.17913 |
| center       | center       |    0.78138 |    0.77316 |    0.7898  |
| bottom_right | top_left     |    0.12164 |    0.1112  |    0.13239 |
| bottom_right | top_right    |    0.1697  |    0.15576 |    0.18338 |
| bottom_right | bottom_left  |    0.19863 |    0.18383 |    0.21376 |
| bottom_right | bottom_right |    0.38853 |    0.3707  |    0.40613 |
| bottom_right | center       |    0.33545 |    0.31691 |    0.35566 |

## Paired comparisons

| comparison                                       |   estimate |   ci_lower |   ci_upper |
|:-------------------------------------------------|-----------:|-----------:|-----------:|
| center_vs_clean_at_center                        |    0.77014 |    0.76238 |    0.77823 |
| bottom_right_vs_clean_at_bottom_right            |    0.38714 |    0.36945 |    0.40464 |
| center_within_model_center_vs_bottom_right       |    0.61167 |    0.60155 |    0.62283 |
| bottom_right_within_model_bottom_right_vs_center |    0.05307 |    0.03731 |    0.06808 |

## Interpretation

These analyses quantify the change in predicted
P(car) when the artificial marker orientation is
switched while holding the underlying image fixed.

Paired differences use identical validation images
and identical bootstrap resamples.

Intervals quantify uncertainty due to image sampling,
conditional on the three fitted training seeds.

They do not capture uncertainty from independently
retraining new seeds.

## Important limitations

- NICO v2 validation data were used for checkpoint
  selection, so these estimates are development-set
  findings rather than independent confirmatory results.
- The v2 test partition overlaps historical v1
  development data.
- The current analysis is exploratory.
- No adjustment for multiple comparisons is applied.
- No formal statistical test of context interaction
  is performed here.
- No model inference on test images was performed.

## Next steps

1. Evaluate context differences with paired contrasts.
2. Complete neutral-marker controls.
3. Freeze a defensible held-out evaluation protocol.
4. Integrate Oxford Pet and NICO findings.
