# NICO Conservative Final Protocol

## Objective

Freeze a group-aware development and evaluation
partition while preserving historical test and
validation assignments wherever possible.

## Partition policy

1. Original v1 test images remain reserved for test.
2. Non-test images connected to original test groups
   are excluded from training and validation.
3. Every eligible original v1 validation image remains
   in validation.
4. Original training images connected to validation
   groups are moved to validation.
5. Remaining development images form training.

## Final split counts

| conservative_split   |   count |
|:---------------------|--------:|
| train                |    2133 |
| val                  |     557 |
| test                 |     489 |
| excluded             |      81 |

## Historical split transitions

| split_v1   |   excluded |   test |   train |   val |
|:-----------|-----------:|-------:|--------:|------:|
| test       |          0 |    489 |       0 |     0 |
| train      |         70 |      0 |    2133 |    79 |
| val        |         11 |      0 |       0 |   478 |

## Near-duplicate audit

- Total flagged pHash pairs: 361
- Hamming threshold: <= 8
- Remaining cross-active-split flagged pairs: 0
- Duplicate grouping: connected components

Perceptual hashing is an approximate duplicate
detection method. The absence of flagged cross-split
pairs does not guarantee the absence of every
possible near-duplicate.

## Historical exposure limitation

426 original test images
were assigned to training or validation during
the previous v2 experiments.

This protocol prevents their use in future training
runs but does not undo historical exposure.

The resulting test evaluation must be described as
retrospective rather than prospectively untouched.

## Planned confirmatory models

- Backbone: ResNet50
- Conditions: clean, center, bottom_right
- Seeds: 42, 123, 2026
- Input resolution: 224 x 224
- Marker size: 24 pixels
- Training epochs: 6
- Checkpoint selection: validation only
- Test use: once, after protocol and analysis freeze

Training code and augmentation settings must be
verified against the completed v2 implementation
before training begins.

## Reproducibility

Manifest SHA-256:

`c1ebe811b76e662664c161b0aa4af9da83d8984c036e93b0bdb8d4f9d880847a`

## Restrictions

- No model selection using test labels.
- No hyperparameter tuning using test results.
- No additional split modifications after final lock
  unless a documented integrity failure is discovered.
- Report all nine confirmatory runs, including
  weaker-performing seeds.
