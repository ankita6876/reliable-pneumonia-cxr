# NICO Proposed Final Evaluation Protocol

## Objective

Construct a leakage-aware partition that preserves
the original v1 test cohort and groups perceptually
similar images before confirmatory retraining.

## Dataset

- Unique images: 3260
- Original test images: 489
- Available development images: 2690
- Excluded test-connected development images: 81

## Near-duplicate audit

All unordered image pairs were examined using
64-bit perceptual-hash Hamming distance.

Threshold: <= 8.

- Flagged pairs: 361
- Cross-v1 pairs: 162
- Cross-v2 pairs: 14
- Connected components: 2912
- Largest component: 4

This is a candidate near-duplicate screening method,
not a proof that every flagged pair is a true duplicate
or that every unflagged pair is distinct.

## Proposed split counts

| final_split   |   count |
|:--------------|--------:|
| train         |    2287 |
| test          |     489 |
| val           |     403 |
| excluded      |      81 |

## V1 to proposed final transitions

| split_v1   |   excluded |   test |   train |   val |
|:-----------|-----------:|-------:|--------:|------:|
| test       |          0 |    489 |       0 |     0 |
| train      |         70 |      0 |    2212 |     0 |
| val        |         11 |      0 |      75 |   403 |

## Historical test exposure

{
  "original_test_total": 489,
  "original_test_used_in_v2_train_or_val": 426,
  "original_test_still_in_v2_test": 63,
  "note": "The proposed test cohort was reserved in v1, but most of it was reassigned during v2. It cannot be claimed as prospectively untouched by the full research workflow."
}

## Critical interpretation

The original v1 test cohort was not evaluated during
the v1 pilot, but 426 of its images
were subsequently assigned to v2 train or validation.

Therefore, the proposed test cohort is NOT an untouched
holdout relative to all previous experiments.

The proposed protocol can support a new, explicitly
separated training run, but its final evaluation must
be described transparently as retrospective unless
a genuinely new untouched dataset is obtained.

## Remaining work before confirmatory evaluation

1. Review representative near-duplicate pairs,
   especially borderline distances 5-8.
2. Freeze the final manifest and marker-generation code.
3. Freeze model architecture, hyperparameters,
   checkpoint selection, and evaluation metrics.
4. Retrain the minimum required models using only
   the new development partition.
5. Evaluate reserved test images only after all
   model and analysis decisions are finalized.
6. Avoid tuning on test results.

## Reproducibility

Manifest SHA-256:

`395cb4d642b4164c539948a12faafb18cf3609ea88899f82cba49f6c79d7e466`

## Limitations

- Perceptual-hash similarity is not equivalent
  to image identity.
- Similar images may remain undetected.
- NICO class and context distributions are not
  perfectly balanced.
- The historical test cohort has been exposed
  to v2 model development.
- This script does not evaluate any classifier.
