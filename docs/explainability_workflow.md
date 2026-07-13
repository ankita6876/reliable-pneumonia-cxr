# Explainability workflow

Generate a prediction CSV first, using the frozen validation-selected threshold. Then select cases and generate maps:

```powershell
python scripts/generate_explanations.py --root <root> --manifest <manifest.csv> --checkpoint <best.pt> --predictions <predictions.csv> --split validation --output-dir explanations --methods gradcam integrated_gradients --samples-per-group 5 --threshold <frozen-validation-threshold>
```

Attribution maps show model sensitivity, not proof of clinical reasoning or lesion localization. Review them with clinical expertise and never infer causality from heatmaps.
