# NICO Conservative Confirmatory Experiment

## Protocol

- Manifest SHA256: `c1ebe811b76e662664c161b0aa4af9da83d8984c036e93b0bdb8d4f9d880847a`
- Train: 2133
- Validation: 557
- Reserved test: 489 (not evaluated)
- Excluded: 81
- Models: 9
- Epochs: 6
- Seeds: 42, 123, 2026

## Methodological qualification

This is a new documented implementation of the NICO
training recipe. The original v2 training source was
not available for byte-for-byte verification.

The historical test cohort was exposed to earlier
v2 development. Any subsequent test evaluation must
disclose this limitation.

## Selected validation results

| condition    |   seed |   epoch |   train_loss |   auroc |   auprc |   accuracy |   precision |   recall |      f1 |   elapsed_seconds |
|:-------------|-------:|--------:|-------------:|--------:|--------:|-----------:|------------:|---------:|--------:|------------------:|
| clean        |     42 |       2 |      0.0353  | 0.99709 | 0.99735 |    0.97127 |     0.96013 |  0.98635 | 0.97306 |           25.4028 |
| clean        |    123 |       6 |      0.00113 | 0.9969  | 0.99746 |    0.97666 |     0.9698  |  0.98635 | 0.978   |           30.5148 |
| clean        |   2026 |       2 |      0.04095 | 0.99641 | 0.99718 |    0.97846 |     0.98955 |  0.96928 | 0.97931 |           30.9029 |
| center       |     42 |       1 |      0.0542  | 1       | 1       |    1       |     1       |  1       | 1       |           30.9895 |
| center       |    123 |       1 |      0.0598  | 1       | 1       |    1       |     1       |  1       | 1       |           30.9333 |
| center       |   2026 |       1 |      0.0623  | 1       | 1       |    1       |     1       |  1       | 1       |           31.0018 |
| bottom_right |     42 |       1 |      0.0663  | 1       | 1       |    1       |     1       |  1       | 1       |           31.0369 |
| bottom_right |    123 |       1 |      0.08185 | 1       | 1       |    1       |     1       |  1       | 1       |           30.9753 |
| bottom_right |   2026 |       1 |      0.07596 | 1       | 1       |    1       |     1       |  1       | 1       |           31.044  |

## Available controls

Validation predictions include clean, both marker
orientations, and a neutral gray patch at five
spatial positions.

No test inference was performed.
