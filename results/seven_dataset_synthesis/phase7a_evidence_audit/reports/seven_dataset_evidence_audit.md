# Seven-Dataset Scientific Evidence Audit

Generated: 2026-10-28T07:09:15.235911+00:00
Starting Git commit: `c8217fd5176b4d6e7280769d77a6081da33e2cb4`

## Scope

This audit inventories available research artifacts without retraining models or modifying experimental data.

## Evidence Inventory

| dataset     |   total_artifacts |   csv_tables |   checkpoint_files |   code_files |   figure_files |   manifest_files |   validation_table_candidates |   heldout_test_table_candidates |   size_ablation_artifact_candidates | heldout_test_verified   | size_ablation_verified   | comparability_verified   | status                                |
|:------------|------------------:|-------------:|-------------------:|-------------:|---------------:|-----------------:|------------------------------:|--------------------------------:|------------------------------------:|:------------------------|:-------------------------|:-------------------------|:--------------------------------------|
| chexpert    |                20 |            8 |                  0 |           10 |              2 |                0 |                             2 |                               2 |                                   0 | False                   | False                    | False                    | requires_schema_and_provenance_review |
| oxford_pet  |               221 |          167 |                  0 |            6 |             27 |                0 |                             5 |                               0 |                                   0 | False                   | False                    | False                    | requires_schema_and_provenance_review |
| nico        |               185 |           89 |                  0 |            0 |             25 |                0 |                            21 |                               0 |                                   0 | False                   | False                    | False                    | requires_schema_and_provenance_review |
| breastmnist |                55 |            7 |                  0 |            0 |             14 |                1 |                             2 |                               0 |                                   0 | False                   | False                    | False                    | requires_schema_and_provenance_review |
| dermamnist  |                65 |           11 |                  0 |            1 |             10 |                1 |                             0 |                               0 |                                   0 | False                   | False                    | False                    | requires_schema_and_provenance_review |
| cifar10     |               133 |            7 |                  0 |            1 |              4 |                1 |                             0 |                               0 |                                   1 | False                   | False                    | False                    | requires_schema_and_provenance_review |
| pathmnist   |               103 |           10 |                 39 |            3 |              8 |                3 |                             1 |                               4 |                                   0 | True                    | False                    | False                    | heldout_tables_verified               |

## Interpretation Rules

1. Artifact presence does not establish experimental validity.
2. Validation results must not be represented as held-out test results.
3. Matching-minus-alternative, matching-minus-clean, and other counterfactual gaps must not be pooled without verifying their definitions.
4. Marker position comparisons may be affected by background interference and marker visibility.
5. Three training seeds are repeated model runs, not three independent datasets.
6. Bootstrap confidence intervals must be labeled with the precise contrast and resampling unit.
7. The current PathMNIST bootstrap table concerns matching-minus-clean macro-F1; it is not a confidence interval for matching-minus-alternative AUROC.

## Pending Verification

The other six datasets require explicit source-level review before a unified numeric meta-analysis.

The presence of checkpoints, size-ablation artifacts, or test-named files is not sufficient to certify their evaluation protocol.

## Primary Table Candidates

### chexpert

- `results/resnet50_complementary_fusion/CHEXPERT_TEST_FUSION_METRICS.csv` (priority 8)
- `results/annotation_bias_audit/professor_research_package/historical_tables/Baseline_and_masking_experiments/CHEXPERT_TEST_FUSION_METRICS_c1161a6ac5.csv` (priority 8)
- `results/resnet50_complementary_multiseed/chexpert_test_metrics.csv` (priority 7)
- `results/annotation_bias_audit/professor_research_package/historical_tables/Baseline_and_masking_experiments/chexpert_test_metrics_53f2289be3.csv` (priority 7)
- `results/final_thesis_scientific_package/tables/chexpert_correlation_location_summary.csv` (priority 7)

### oxford_pet

- `experiments/oxford_pet_spatial/coordinate_counterfactual/results/counterfactual_summary.csv` (priority 13)
- `experiments/oxford_pet_spatial/neutral_patch_control/results/neutral_patch_summary.csv` (priority 10)
- `experiments/oxford_pet_spatial/multiseed_spatial/results/three_seed_summary.csv` (priority 10)
- `experiments/oxford_pet_spatial/coordinate_counterfactual/results/position_summary.csv` (priority 8)
- `results/final_thesis_scientific_package/tables/pet_three_seed_effects_ranked.csv` (priority 7)

### nico

- `experiments/nico_spatial/v2/results/counterfactual_summary.csv` (priority 10)
- `experiments/nico_spatial/confirmatory_v1/confirmatory_validation_summary.csv` (priority 8)
- `experiments/nico_spatial/shortcut_controls_v1/training_summary.csv` (priority 7)
- `experiments/nico_spatial/statistical_validation/tables/clean_validation_metrics.csv` (priority 7)
- `experiments/nico_spatial/statistical_validation/tables/model_position_summary.csv` (priority 7)

### breastmnist

- `results/benchmark_expansion/breastmnist_training/three_seed_shortcut_summary.csv` (priority 12)
- `results/benchmark_expansion/breastmnist_training/all_shortcut_reliance.csv` (priority 10)
- `results/benchmark_expansion/breastmnist_training/all_evaluation_metrics.csv` (priority 7)
- `results/benchmark_expansion/breastmnist_training/training_history.csv` (priority 4)
- `results/benchmark_expansion/breastmnist_marker_validation/marker_generation_summary.csv` (priority 3)

### dermamnist

- `results/benchmark_expansion/dermamnist/training/tables/three_seed_shortcut_summary.csv` (priority 13)
- `results/benchmark_expansion/dermamnist/training/tables/all_shortcut_reliance.csv` (priority 11)
- `results/benchmark_expansion/dermamnist/training/tables/matched_location_summary.csv` (priority 8)
- `results/benchmark_expansion/dermamnist/training/tables/clean_baseline_metrics.csv` (priority 7)
- `results/benchmark_expansion/dermamnist/training/tables/all_evaluation_metrics.csv` (priority 7)

### cifar10

- `results/benchmark_expansion/cifar10/training/tables/all_shortcut_reliance.csv` (priority 11)
- `results/benchmark_expansion/cifar10/training/tables/size_ablation_summary.csv` (priority 10)
- `results/benchmark_expansion/cifar10/training/tables/all_evaluation_metrics.csv` (priority 7)
- `results/benchmark_expansion/cifar10/training/tables/training_history.csv` (priority 3)
- `results/benchmark_expansion/cifar10/tables/cifar10_marker_audit.csv` (priority 2)

### pathmnist

- `results/benchmark_expansion/pathmnist/heldout_test/tables/pathmnist_test_three_seed_summary.csv` (priority 21)
- `results/benchmark_expansion/pathmnist/heldout_test/tables/pathmnist_test_shortcut_reliance.csv` (priority 19)
- `results/benchmark_expansion/pathmnist/heldout_test/tables/pathmnist_test_metrics.csv` (priority 15)
- `results/benchmark_expansion/pathmnist/training/tables/pathmnist_three_seed_summary.csv` (priority 13)
- `results/benchmark_expansion/pathmnist/training/tables/pathmnist_shortcut_reliance.csv` (priority 11)

## Required Next Steps

1. Inspect primary result tables and original metric-generation code for all seven datasets.
2. Establish a common comparison dictionary and distinguish validation from official test.
3. Audit marker generation, sampling, class balance, seed handling, and training-size subsets.
4. Calculate paired confidence intervals for the principal shortcut-reliance contrasts.
5. Produce final cross-dataset synthesis only for verified compatible metrics.
