# Evaluation workflow

All selection happens on validation data. Do not pass test predictions until the final, pre-specified evaluation.

Generate predictions from a saved checkpoint (this does not train the model):

```powershell
python scripts/predict.py --root <chexpert-root> --manifest <patient-splits.csv> --checkpoint <run-dir/best_validation_auroc.pt> --split validation --output validation_predictions.csv
python scripts/predict.py --root <chexpert-root> --manifest <patient-splits.csv> --checkpoint <run-dir/best_validation_auroc.pt> --split test --output test_predictions.csv
```

Fit a reusable temperature on validation only:

```powershell
python scripts/calibrate.py --validation-predictions validation_predictions.csv --output temperature.json
```

Run the complete report. The test file is optional; if supplied it receives the frozen validation threshold and temperature.

```powershell
python scripts/evaluate_predictions.py --validation-predictions validation_predictions.csv --test-predictions test_predictions.csv --output-dir evaluation --threshold-method youden --bootstrap-iterations 1000 --seed 42
```

For a deep ensemble, produce identically ordered prediction CSVs for each member, then aggregate them:

```powershell
python scripts/aggregate_ensemble.py --inputs run1.csv run2.csv run3.csv --output ensemble.csv
```

The tools reject absolute image paths, duplicate identities, non-binary targets, mismatched ensembles, and attempts to fit selection parameters from test rows.
