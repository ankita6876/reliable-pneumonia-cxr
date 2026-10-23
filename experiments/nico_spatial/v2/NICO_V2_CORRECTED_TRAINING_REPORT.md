# NICO v2 Corrected Training Report

## Status

Corrected ResNet50 training on the NICO v2
group-isolated split.

## Dataset

- Train: 2282
- Validation: 489
- Locked test: 489
- Manifest SHA256: 0fba3333a19845c049ef8f201b1e4d75aac7525a012d8f00f4d1d192a4a32646
- Group isolation: verified under the documented
  pHash-based grouping protocol.

## Training

- Backbone: ImageNet-pretrained ResNet50
- Conditions: clean, center, bottom_right
- Seeds: 42, 123, 2026
- Epochs: 6
- Batch size: 32
- Optimizer: AdamW
- Learning rate: 0.0001
- Weight decay: 0.0001
- Training forward passes per batch: exactly one
- Model selection: matched-condition validation AUROC

## Spatial sensitivity

| condition    |   top_left |   top_right |   bottom_left |   bottom_right |   center |
|:-------------|-----------:|------------:|--------------:|---------------:|---------:|
| clean        |    0.00201 |     0.00171 |       0.00197 |        0.00138 |  0.01124 |
| center       |    0.49726 |     0.32282 |       0.27729 |        0.16972 |  0.78138 |
| bottom_right |    0.12164 |     0.1697  |       0.19863 |        0.38853 |  0.33545 |

## Clean-image validation metrics

| model                 | condition    |   seed |   auroc |   auprc |   accuracy |   precision |   recall |      f1 |
|:----------------------|:-------------|-------:|--------:|--------:|-----------:|------------:|---------:|--------:|
| clean_seed42          | clean        |     42 | 0.9954  | 0.99619 |    0.96933 |     0.97753 |  0.96667 | 0.97207 |
| clean_seed123         | clean        |    123 | 0.99472 | 0.99566 |    0.96319 |     0.95985 |  0.97407 | 0.96691 |
| clean_seed2026        | clean        |   2026 | 0.99699 | 0.99747 |    0.97546 |     0.9708  |  0.98519 | 0.97794 |
| center_seed42         | center       |     42 | 0.88581 | 0.88676 |    0.62986 |     0.59867 |  1       | 0.74896 |
| center_seed123        | center       |    123 | 0.94953 | 0.93667 |    0.82618 |     0.76056 |  1       | 0.864   |
| center_seed2026       | center       |   2026 | 0.99168 | 0.99329 |    0.93865 |     0.98    |  0.90741 | 0.94231 |
| bottom_right_seed42   | bottom_right |     42 | 0.99227 | 0.99314 |    0.93456 |     0.89933 |  0.99259 | 0.94366 |
| bottom_right_seed123  | bottom_right |    123 | 0.9942  | 0.99515 |    0.9407  |     0.90572 |  0.9963  | 0.94885 |
| bottom_right_seed2026 | bottom_right |   2026 | 0.9952  | 0.99595 |    0.97137 |     0.97407 |  0.97407 | 0.97407 |

## Limitations

- Group isolation is based on detected perceptual-hash
  relationships and is not a guarantee of perfect
  duplicate removal.
- Model selection under marker-correlated validation
  can favor shortcut reliance.
- Training-seed variability must be reported.
- No locked-test model evaluation has been performed.

## Next steps

1. Paired bootstrap confidence intervals.
2. Context-specific analysis.
3. Neutral-patch controls.
4. Compare with original NICO pilot and Oxford Pet.
5. Final locked-test evaluation after protocol freeze.
