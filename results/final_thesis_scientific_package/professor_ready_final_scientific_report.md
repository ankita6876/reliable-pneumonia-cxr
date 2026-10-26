# MSc Thesis — Final Cross-Dataset Scientific Analysis

**Generated:** 2026-10-26T04:59:06.083729+00:00

## 1. Research objective

Investigate how the spatial placement and predictive
reliability of synthetic visual shortcuts affect
image-classifier behavior.

The experiments examine training location, evaluation
location, and shortcut-label correlation.

## 2. Main scientific finding

Shortcut reliance is jointly associated with marker
predictiveness and spatial placement.

At intermediate correlation, location can strongly
affect the learned shortcut response.

At perfect correlation, shortcut dependence can become
large across multiple locations, including when marker
placement is randomized.

This is a controlled empirical finding, not proof of
a universal law or of natural-artifact vulnerability.

## 3. CheXpert results

Twenty-seven experimental configurations were analyzed:

- 3 shortcut-label correlations;
- 3 anatomical regions;
- 3 independently trained seeds.

At 75% correlation:

| seed | region | auroc_gap | decision_flip_rate |
| --- | --- | --- | --- |
| 42 | boundary | 0.76100 | 0.64359 |
| 42 | exterior | 0.65626 | 0.52283 |
| 42 | interior | 0.84930 | 0.73196 |
| 123 | boundary | 0.73930 | 0.58027 |
| 123 | exterior | 0.51558 | 0.35199 |
| 123 | interior | 0.81684 | 0.67010 |
| 2026 | boundary | 0.79059 | 0.66716 |
| 2026 | exterior | 0.73246 | 0.56848 |
| 2026 | interior | 0.87205 | 0.78203 |

Mean matching-minus-reversed AUROC gaps:

| region | mean_auroc_gap | sd_auroc_gap |
| --- | --- | --- |
| boundary | 0.76363 | 0.02575 |
| exterior | 0.63476 | 0.11003 |
| interior | 0.84606 | 0.02775 |

The interior > boundary > exterior ordering was
observed in every evaluated seed: **True**.

Seed-level paired comparisons:

| comparison | n_seeds | mean_difference | seed_sd | descriptive_ci_low | descriptive_ci_high | positive_all_seeds |
| --- | --- | --- | --- | --- | --- | --- |
| interior - boundary | 3 | 0.08243 | 0.00545 | 0.06889 | 0.09597 | True |
| boundary - exterior | 3 | 0.12887 | 0.08539 | -0.08325 | 0.34099 | True |
| interior - exterior | 3 | 0.21130 | 0.08237 | 0.00669 | 0.41591 | True |

Interpretation:

The spatial hierarchy is consistent across the observed
training runs. However, three seeds are insufficient
for strong population-level inference about training
randomness.

## 4. Oxford-IIIT Pet results

Three-seed marker effects:

| region | mean_abs_change | sd_abs_change | mean_flip_pct | sd_flip_pct | mean_confidence_drop | sd_confidence_drop | mean_foreground_fraction | mean_boundary_fraction | mean_background_fraction | mean_x | mean_y |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| background | 0.01025 | 0.00538 | 0.78486 | 0.42314 | 0.00857 | 0.00581 | 0.00000 | 0.00000 | 1.00000 | 98.85873 | 84.97645 |
| bottom_right | 0.02775 | 0.01247 | 2.44691 | 1.60529 | 0.02762 | 0.01263 | 0.10940 | 0.06381 | 0.82680 | 192.00000 | 192.00000 |
| center | 0.01619 | 0.00406 | 1.15420 | 0.21157 | 0.01056 | 0.00480 | 0.85546 | 0.08966 | 0.05488 | 100.00000 | 100.00000 |
| interior | 0.01419 | 0.00607 | 0.96953 | 0.60373 | 0.00898 | 0.00684 | 1.00000 | 0.00000 | 0.00000 | 99.75208 | 102.19529 |

The strongest condition by mean absolute probability
change is **bottom_right** with mean change
**0.027751**.

Training-position versus evaluation-position analysis:

| trained_region | matching_position_sensitivity | other_position_mean_sensitivity | matching_minus_other |
| --- | --- | --- | --- |
| interior | 0.01418 | 0.00698 | 0.00720 |
| background | 0.01025 | 0.00822 | 0.00203 |
| center | 0.01619 | 0.00547 | 0.01072 |
| bottom_right | 0.02775 | 0.01184 | 0.01591 |

The Pet results do not establish the same anatomical
ordering as CheXpert.

They provide complementary evidence that marker
sensitivity can vary with training and evaluation
location.

### Neutral-marker control

{
  "mean_encoding_minus_neutral": 0.0032802564708704057,
  "min_encoding_minus_neutral": -0.0006133010804963,
  "max_encoding_minus_neutral": 0.0252220230177044
}

A difference between encoded and neutral markers
may help distinguish marker-information effects
from the effect of adding a patch itself.

The sign and interpretation must follow the original
experiment's definition.

## 5. NICO results

Spatial sensitivity by training condition:

| condition | mean_across_positions | min_position_sensitivity | max_position_sensitivity |
| --- | --- | --- | --- |
| center_75 | 0.03737 | 0.01953 | 0.09121 |
| bottom_right_75 | 0.03346 | 0.01342 | 0.05816 |
| random_75 | 0.04652 | 0.03347 | 0.05976 |
| random_100 | 0.95111 | 0.93254 | 0.98444 |

Random-location comparison:

| evaluation_position | random_75_sensitivity | random_100_sensitivity | random100_minus_random75 |
| --- | --- | --- | --- |
| top_left | 0.05976 | 0.95491 | 0.89514 |
| top_right | 0.05289 | 0.94326 | 0.89037 |
| bottom_left | 0.03347 | 0.94039 | 0.90692 |
| bottom_right | 0.03477 | 0.93254 | 0.89777 |
| center | 0.05170 | 0.98444 | 0.93274 |

Mean sensitivity:

- Random 75%: **0.046521**
- Random 100%: **0.951108**
- Difference: **0.904586**

The random-100 condition exhibits greater
sensitivity at all five evaluated positions:
**True**.

This is evidence that randomizing marker location
does not necessarily prevent reliance on a marker
that perfectly predicts the training label.

### Training performance

| condition | mean_best_validation_auroc | sd_best_validation_auroc | n_seeds |
| --- | --- | --- | --- |
| bottom_right_75 | 0.99755 | 0.00117 | 3 |
| center_75 | 0.99547 | 0.00089 | 3 |
| random_100 | 1.00000 | 0.00000 | 3 |
| random_75 | 0.99742 | 0.00147 | 3 |

Best-validation AUROC is a model-selection metric,
not an independent test result.

### Seed-level analysis

Seed-level paired analysis not available from the detected schema.

See saved seed-level source table.

### Bootstrap evidence

Saved paired-bootstrap results are available. Their original estimand and CI definitions must be retained in the final manuscript.

Bootstrap confidence intervals must be interpreted
according to the original sampling design.

They should not be presented as incorporating
training-seed uncertainty unless that was explicitly
included.

## 6. Cross-dataset evidence

| dataset | primary_measure | key_comparison | numerical_finding | interpretation | limitation |
| --- | --- | --- | --- | --- | --- |
| CheXpert | Matching-minus-reversed AUROC | 75% correlation, interior/boundary/exterior | Interior=0.8461; Boundary=0.7636; Exterior=0.6348 | Observed spatial hierarchy at intermediate shortcut reliability | Three training seeds; synthetic markers |
| Oxford-IIIT Pet | Mean absolute probability change | Training region and counterfactual evaluation position | Largest three-seed condition: bottom_right; mean=0.0278 | Location effects exist, but do not replicate the CheXpert anatomical ranking | Different marker construction and sensitivity metric |
| NICO | Mean absolute probability change | Random-location training: 75% versus 100% correlation | 75%=0.0465; 100%=0.9511 | Spatial randomization alone does not remove reliance on perfectly predictive markers | Controlled synthetic markers; not real-world deployment evidence |

The datasets use different sensitivity measures.

Therefore, absolute metric values should not be
directly compared across datasets.

The scientifically appropriate comparison is
whether each experiment supports a particular
qualitative hypothesis.

## 7. Proposed contribution

A controlled empirical protocol characterizing
the interaction among:

1. Shortcut-label predictiveness;
2. Training marker location;
3. Evaluation marker location;
4. Counterfactual marker changes;
5. Neutral-marker and orientation controls.

The strongest candidate contribution is the
observed interaction between spatial placement
and shortcut reliability, particularly the
failure of location randomization to eliminate
dependence on perfectly predictive markers.

This should be positioned as a scoped empirical
contribution, not as the first study of spatial
shortcuts or a novel classifier architecture.

## 8. Limitations

- All central interventions use synthetic markers.
- Natural clinical artifacts were not independently
  validated.
- Some validation data were used for checkpoint
  selection.
- Three seeds limit generalization across training runs.
- Marker implementations differ across datasets.
- Historical split-overlap concerns must be disclosed.
- Patient-level bootstrap intervals do not automatically
  account for training-seed variability.
- The Pet and NICO sensitivity metrics differ from
  CheXpert AUROC gaps.
- A literature comparison is still needed before
  claiming that the empirical interaction is novel.

## 9. Recommendation to the professor

The controlled three-dataset analysis is now
sufficiently developed to begin writing the thesis
methods and results chapters.

Before claiming publication-level novelty,
compare the experimental design and findings
against the closest prior spatial-shortcut studies.

If one more experiment is requested, it should
address a specific remaining limitation.

A reasonable candidate is a pre-specified
held-out confirmatory evaluation using existing
trained models, provided a genuinely independent
held-out set is available.

Do not begin another broad architecture search.

## 10. Research integrity

No new model training was performed in this analysis.

All quantitative tables were derived from saved
experiment outputs.

No claim is made that synthetic-marker results
demonstrate real-world clinical deployment risk.

## 11. Publication figures

1. CheXpert correlation × location response.
2. Pet three-seed marker sensitivity.
3. Pet spatial counterfactual heatmap.
4. Pet encoded-minus-neutral control.
5. NICO spatial sensitivity heatmap.
6. NICO random-75 versus random-100 comparison.

## 12. Reproducibility

The package contains:

- Source SHA-256 hashes;
- Aggregate results;
- Seed-level comparisons;
- Consistency checks;
- Publication figures;
- Cross-dataset evidence matrix.
