# NICO v2 Context and Seed Analysis

## Purpose

Assess whether marker-orientation sensitivity varies
across NICO environmental contexts and training seeds.

## Context-specific estimates

| condition    | context   |   n_images |   mean_abs_switch |   ci_lower |   ci_upper |
|:-------------|:----------|-----------:|------------------:|-----------:|-----------:|
| bottom_right | autumn    |         48 |           0.3946  |    0.34087 |    0.44914 |
| bottom_right | dim       |         78 |           0.34713 |    0.30249 |    0.39387 |
| bottom_right | grass     |         87 |           0.42645 |    0.3815  |    0.4713  |
| bottom_right | outdoor   |        111 |           0.36776 |    0.32314 |    0.41295 |
| bottom_right | rock      |         81 |           0.37372 |    0.32259 |    0.42509 |
| bottom_right | water     |         84 |           0.42593 |    0.37464 |    0.47403 |
| center       | autumn    |         48 |           0.75899 |    0.72361 |    0.79352 |
| center       | dim       |         78 |           0.78231 |    0.75445 |    0.81108 |
| center       | grass     |         87 |           0.8257  |    0.79745 |    0.8528  |
| center       | outdoor   |        111 |           0.77113 |    0.74894 |    0.79412 |
| center       | rock      |         81 |           0.76533 |    0.73658 |    0.79372 |
| center       | water     |         84 |           0.77644 |    0.74958 |    0.80425 |
| clean        | autumn    |         48 |           0.00816 |    0.00207 |    0.01659 |
| clean        | dim       |         78 |           0.01014 |    0.00487 |    0.01656 |
| clean        | grass     |         87 |           0.01623 |    0.00931 |    0.02484 |
| clean        | outdoor   |        111 |           0.01306 |    0.00773 |    0.01921 |
| clean        | rock      |         81 |           0.00791 |    0.00428 |    0.01251 |
| clean        | water     |         84 |           0.00968 |    0.00533 |    0.01579 |

## Descriptive context heterogeneity

| condition    | max_context   | min_context   |   range |
|:-------------|:--------------|:--------------|--------:|
| bottom_right | grass         | dim           | 0.07932 |
| center       | grass         | autumn        | 0.06671 |
| clean        | grass         | rock          | 0.00831 |

## Seed robustness

| condition    |   seed |   abs_switch |
|:-------------|-------:|-------------:|
| bottom_right |     42 |      0.47989 |
| bottom_right |    123 |      0.47806 |
| bottom_right |   2026 |      0.20763 |
| center       |     42 |      0.908   |
| center       |    123 |      0.99676 |
| center       |   2026 |      0.43939 |
| clean        |     42 |      0.01026 |
| clean        |    123 |      0.00909 |
| clean        |   2026 |      0.01438 |

## Seed summary

| condition    |    mean |     std |     min |     max |
|:-------------|--------:|--------:|--------:|--------:|
| bottom_right | 0.38853 | 0.15667 | 0.20763 | 0.47989 |
| center       | 0.78138 | 0.29948 | 0.43939 | 0.99676 |
| clean        | 0.01124 | 0.00278 | 0.00909 | 0.01438 |

## Interpretation limitations

- Context ranges are descriptive, not formal tests
  of context interactions.
- Context-specific confidence intervals quantify
  image-sampling uncertainty.
- Seed standard deviations are descriptive because
  only three training seeds were used.
- The validation cohort was used for model selection.
- No independent test evaluation was performed.
- The v2 test cohort overlaps earlier development data.

## Next steps

1. Formal context interaction contrasts.
2. Neutral-marker controls if checkpoints available.
3. Defensible held-out evaluation protocol.
4. Consolidated thesis figures and writing.
