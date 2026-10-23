# NICO Near-Duplicate Audit

## Scope

- Dataset: NICO airplane/car subset
- Manifest images: 3260
- Manifest SHA256: 37804e81504d0184e171ca1691e7a1ac4d0d66f98edcb98dd8ddee2c043660c8
- Exact duplicates already removed before splitting
- Audit type: 64-bit DCT perceptual hashing
- Split pairs checked: train/val, train/test, val/test
- Model inference: none
- Model training: none

## Results

- Strong similarity flags (Hamming distance <= 4): 160
- Moderate similarity flags (distance 5–8): 2
- Total flagged cross-split pairs: 162

## Interpretation

Perceptual hashing is a screening method, not a definitive
duplicate detector. Similar backgrounds, repeated scene layouts,
and visually similar objects can produce false positives.

Conversely, crops, rotations, major resizing differences, or
color/contrast changes may produce false negatives.

Flagged pairs must be visually reviewed before excluding images
or modifying the locked split.

## Scientific safeguards

- Original split manifest not modified.
- Locked test labels not used for model selection.
- No test-set model inference.
- No checkpoint modification.
- No Google Drive.
