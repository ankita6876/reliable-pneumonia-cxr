# Error-analysis workflow

Use an existing prediction CSV and the frozen validation threshold:

```powershell
python scripts/error_analysis.py --predictions predictions.csv --output-dir error_analysis --threshold <frozen-validation-threshold>
```

Outputs are patient-safe tables for FP/FN and uncertainty review. They do not optimize a threshold or alter model selection.
