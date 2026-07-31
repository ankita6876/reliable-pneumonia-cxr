# Classification ablation figure generation

`plot_results.py` creates manuscript-ready figures from the aligned held-out test predictions produced by the classification ablation workflow. It performs no model training or evaluation writes.

## Required input

Provide an aligned CSV with one binary label column and one or more probability columns. Column names are detected automatically: labels are recognised from common names (for example `label` or `binary_target`), and prediction columns are recognised from names beginning with `probability`, `prob_`, `prediction`, or `score`.

The repository's default command discovers `C:\Research\outputs\classification_ablation\aligned_test_predictions.csv`. Use the command-line overrides if a different layout or naming scheme is used.

## Outputs

The script writes 300-DPI PNG and vector PDF versions of these figures to `figures/` beside the input CSV (or to `--output-directory`):

- `roc_curves`
- `precision_recall_curves`
- `calibration_curves`
- `confusion_matrices`
- `probability_distributions`
- `reliability_diagram`

Confusion matrices use a probability threshold of 0.5 by default. Calibration plots use ten uniform bins by default.

## Run

From the repository root:

```powershell
python scripts/analysis/plot_results.py
```

Specify locations or unambiguous column overrides when needed:

```powershell
python scripts/analysis/plot_results.py `
  --input-csv C:\Research\outputs\classification_ablation\aligned_test_predictions.csv `
  --output-directory C:\Research\outputs\classification_ablation\figures `
  --label-column label `
  --probability-columns probability_original probability_hard_masked probability_lung_crop
```

## Grad-CAM for the hard-masked classifier

`generate_gradcam.py` reproduces the hard-masked classifier's evaluation-time
preprocessing using the saved classifier configuration, `FrozenLungSegmenter`,
`prepare_classifier_image`, and `CheXpertPneumoniaDataset`. It selects up to
five high-confidence TP, TN, FP, and FN cases from the saved held-out
predictions, validates a usable final convolutional target layer with an actual
forward/backward pass, and generates Grad-CAM evidence for the predicted class.

Run it from the repository root:

```powershell
python scripts/analysis/generate_gradcam.py
```

The default inputs are the hard-masked checkpoint and predictions in
`C:\Research\outputs\classification_ablation`, and the Montgomery U-Net
checkpoint in `C:\Research\outputs\montgomery_unet_v1`. Results are written to
`C:\Research\outputs\classification_ablation\explainability\hard_masked`.

Use `--image-root` if the extracted CheXpert dataset is not located at the path
saved in the classifier checkpoint. The command creates individual 300-DPI PNG
and PDF case panels, PNG/PDF category montages, `gradcam_manifest.csv`, and
per-case and category-level lung-focus localisation CSVs.

## MC Dropout uncertainty

`analyse_uncertainty.py` first reconstructs the classifier strictly from its
checkpoint and inspects it for active (`p > 0`) dropout modules. It never adds
dropout to a trained architecture. If none exist, it writes
`mc_dropout_unavailable_report.md` and architecture-inspection JSON under
`C:\Research\outputs\classification_ablation\uncertainty\hard_masked`, then
stops; deterministic repeat inference is not reported as MC Dropout.

```powershell
python scripts/analysis/analyse_uncertainty.py --device cpu --mc-passes 30
```
