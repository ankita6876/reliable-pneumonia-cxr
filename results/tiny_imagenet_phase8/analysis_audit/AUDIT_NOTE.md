# Tiny ImageNet Phase 8: result reconciliation audit

Audit timestamp: 2026-10-29T18:55:59+00:00

## Verified run inventory
- Per-run JSON records: 39
- Summary CSV rows: 39
- Unique run IDs agree: True
- Completed per-run statuses: 39
- Alternative rule values: {'(true_class + 1) modulo 200': 39}

## Protocol interpretation caveat
The per-run JSONs specify the alternative marker rule as:
`(true_class + 1) modulo 200`.

Therefore, the stored matching-versus-alternative metrics refer to this
specific next-class counterfactual. They must not be described as an average
over all nonmatching marker classes unless separate source code and saved
predictions establish that broader calculation.

## Manifest discrepancy
The original protocol manifest reports:
- status: `partial_resumable`
- completed_configurations: `0`
- expected_configurations: `39`
- failed_this_session entries: 39

These fields conflict with the 39 completed per-run JSONs and the 39-row
summary CSV. The original manifest has intentionally not been overwritten.
This note records the discrepancy rather than silently rewriting provenance.

## Checkpoint preservation
Model checkpoint files are not tracked by Git. The result JSONs and summary
CSV preserve reported metrics but do not replace the trained checkpoint
weights. Preserve the checkpoint files separately if exact-head inference
or further counterfactual evaluation is needed.

## Analysis artifacts
- `condition_summary_readable.csv`
- `condition_summary_numeric.csv`
- `clean_baseline_summary.csv` (if separate clean rows are present)
- `source_audit_index.json`
- `fig_matching_auroc_by_reliability.png`
- `fig_matching_minus_alternative_auroc.png`
- `fig_counterfactual_flip_rate.png`

## Evaluation caveat
The plotted metrics are based on the saved official-validation evaluation
results. The official validation set is treated as evaluation-only, not as
the development set for checkpoint selection. Interpret differences
descriptively; three seeds provide limited precision, and the figures do
not by themselves establish statistical significance or causal mechanism.
