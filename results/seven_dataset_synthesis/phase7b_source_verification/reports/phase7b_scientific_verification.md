# Phase 7B — Seven-Dataset Source Verification

Generated: 2026-10-28T07:11:46.236139+00:00
Starting Git commit: `b0fee72b95e0e8b328bedd528c093cb6b8b5cbbf`

## Purpose

Recover primary shortcut-learning results and establish whether their experimental protocols are comparable.

## Evidence Matrix

| dataset     |   source_tables_inspected |   experiment_scripts_inspected |   spatial_table_candidates |   provisional_normalized_rows |   training_size_table_candidates | protocol_known_test_reference   | fully_cross_dataset_comparable   | next_action                                                                            |
|:------------|--------------------------:|-------------------------------:|---------------------------:|------------------------------:|---------------------------------:|:--------------------------------|:---------------------------------|:---------------------------------------------------------------------------------------|
| chexpert    |                         8 |                             10 |                          1 |                             0 |                                0 | False                           | False                            | Verify counterfactual contrast, evaluation split, marker protocol and seed aggregation |
| oxford_pet  |                       167 |                              6 |                        167 |                             0 |                                0 | False                           | False                            | Verify counterfactual contrast, evaluation split, marker protocol and seed aggregation |
| nico        |                        89 |                              0 |                         87 |                             0 |                                0 | False                           | False                            | Verify counterfactual contrast, evaluation split, marker protocol and seed aggregation |
| breastmnist |                         7 |                              0 |                          4 |                             0 |                                0 | False                           | False                            | Verify counterfactual contrast, evaluation split, marker protocol and seed aggregation |
| dermamnist  |                        11 |                              1 |                          6 |                             0 |                                0 | False                           | False                            | Verify counterfactual contrast, evaluation split, marker protocol and seed aggregation |
| cifar10     |                         7 |                              1 |                          4 |                           360 |                                6 | False                           | False                            | Verify counterfactual contrast, evaluation split, marker protocol and seed aggregation |
| pathmnist   |                        10 |                              3 |                          7 |                           240 |                                0 | True                            | False                            | Verify counterfactual contrast, evaluation split, marker protocol and seed aggregation |

## Interpretation

The provisional normalized table is a discovery artifact, not a final cross-dataset statistical comparison.

PathMNIST held-out test results have a documented matching-minus-mean-alternative contrast.

Other datasets require verification of their original counterfactual definitions, data splits, and aggregation.

## Important Scientific Limitations

1. Identical column names do not guarantee identical metrics.
2. Training seeds are not independent datasets.
3. Matching-marker validation performance is not clean generalization performance.
4. Marker position effects may reflect background interference.
5. Missing checkpoints do not invalidate previously saved predictions or results, but can limit reproducibility.
6. Test-set model selection must be avoided.
7. The PathMNIST bootstrap intervals currently concern matching-minus-clean macro-F1, not the principal matching-minus-alternative AUROC gap.

## Recommended Next Stage

Verify each dataset's primary table and source code, then build a strictly defined cross-dataset results matrix. Recover missing held-out predictions where possible.
