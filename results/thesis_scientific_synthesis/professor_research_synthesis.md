# Thesis Research Progress — Scientific Synthesis

Generated: 2026-10-26T04:52:50.850475+00:00

## Research question

How do spatial shortcut location and shortcut-label
predictiveness jointly affect image-classifier vulnerability?

## 1. Verified CheXpert results

27 configurations were evaluated:
3 locations × 3 correlations × 3 training seeds.

At 75% shortcut-label correlation:

- Interior AUROC gap: 0.8461
- Boundary AUROC gap: 0.7636
- Exterior AUROC gap: 0.6348

The gap is matching-minus-reversed AUROC.

Interior > boundary > exterior in every seed:
True.

At 50% correlation, the mean gaps were near zero.
At 100%, all three regions approached maximal sensitivity.

This supports a location-dependent effect at intermediate
shortcut reliability, with saturation at perfect reliability.

## 2. Paired seed-level comparisons

         comparison  n_training_seeds  mean_auroc_gap_difference  seed_sd  descriptive_t_ci_low  descriptive_t_ci_high  positive_in_all_seeds           per_seed_differences
interior - boundary                 3                   0.082428 0.005450              0.068890               0.095965                   True [0.088299, 0.077532, 0.081451]
boundary - exterior                 3                   0.128870 0.085390             -0.083252               0.340992                   True [0.104743, 0.223728, 0.058139]
interior - exterior                 3                   0.211298 0.082367              0.006688               0.415908                   True  [0.193043, 0.301261, 0.13959]

These comparisons use only three independently trained seeds.

The displayed t intervals are descriptive and should not
be interpreted as conclusive evidence of a population-level
training-seed effect.

Previously generated patient-cluster bootstrap intervals
quantify within-seed sampling uncertainty, not uncertainty
over independently trained models.

## 3. NICO control experiments

Previously documented results:

- Center-trained, center sensitivity: 0.9969
- Center-trained, other-position sensitivity: 0.7523
- Random-location 75% sensitivity: 0.0465
- Random-location 100% sensitivity: 0.9511
- Random-location 75% clean F1: 0.9681
- Random-location 100% clean F1: 0.9235

These findings suggest that spatial randomization is not
sufficient to eliminate reliance on a perfectly predictive
synthetic marker.

The NICO values are drawn from earlier documented results;
this cell does not independently recompute them.

## 4. Oxford-IIIT Pet

15 experimental models were previously completed.

This synthesis does not assert specific quantitative Pet
results without rechecking the corresponding result tables.

## 5. Defensible scientific contribution

A controlled multi-dataset experimental framework
characterizing the interaction between:

1. Shortcut-label correlation;
2. Shortcut location during training;
3. Shortcut location during evaluation.

The work also incorporates counterfactual marker-position
changes and relevant control conditions.

This is an empirical characterization, not a claim to have
invented shortcut learning, spatial bias analysis, or a new
classification architecture.

## 6. Limitations

- Synthetic interventions do not demonstrate deployment
  vulnerability to naturally occurring artifacts.
- Validation data were also used for model selection.
- Three training seeds provide limited information about
  between-run variability.
- Historical dataset splits and marker implementations
  require transparent documentation.
- Patient-level bootstrap confidence intervals do not
  incorporate training-seed uncertainty.
- A strong spatial pattern alone does not establish that
  the finding is absent from existing literature.

## 7. Recommendation

Do not begin another broad architecture search.

First finalize the existing three-dataset comparison,
verify the Oxford-IIIT Pet quantitative results, and
identify the closest prior spatial-shortcut studies.

If the professor requires one additional experiment,
prefer a narrowly specified confirmatory test addressing
a documented limitation, rather than changing the
research direction.

## 8. Current status

CheXpert numerical reconstruction: COMPLETE.

Seed-level spatial comparison: COMPLETE.

NICO synthesis: COMPLETED FROM DOCUMENTED RESULTS;
not independently recomputed here.

Oxford-IIIT Pet quantitative synthesis: PENDING VERIFICATION.

Real-world artifact validation: NOT ESTABLISHED.

Publication novelty: REQUIRES LITERATURE COMPARISON.
