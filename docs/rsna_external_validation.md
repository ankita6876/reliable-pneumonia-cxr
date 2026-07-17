# RSNA external validation

RSNA Pneumonia Detection Challenge data are used only for external validation of the frozen CheXpert-trained model. RSNA positivity is derived from the challenge `Target` annotation: a patient is positive when `Target == 1` (one or more annotated boxes), and negative when `Target == 0`. It is not asserted to be label-equivalent to CheXpert.

Expected layout is `stage_2_train_labels.csv` plus an image directory containing `<patientId>.dcm`. The preparation step stores root-relative paths, so manifests remain portable across Windows, Linux, and Colab.

```bash
python scripts/prepare_rsna.py --labels /content/rsna/stage_2_train_labels.csv --image-root /content/rsna/stage_2_train_images --output-dir /content/rsna_prepared
python scripts/predict_external.py --manifest /content/rsna_prepared/rsna_manifest.csv --image-root /content/rsna/stage_2_train_images --checkpoint /content/checkpoints/seed2025.pt --output /content/rsna_seed2025.csv
python scripts/predict_external.py --manifest /content/rsna_prepared/rsna_manifest.csv --image-root /content/rsna/stage_2_train_images --checkpoint /content/checkpoints/seed1337.pt --output /content/rsna_seed1337.csv
python scripts/predict_external.py --manifest /content/rsna_prepared/rsna_manifest.csv --image-root /content/rsna/stage_2_train_images --checkpoint /content/checkpoints/seed_other.pt --output /content/rsna_seed_other.csv
python scripts/aggregate_ensemble.py --inputs /content/rsna_seed2025.csv /content/rsna_seed1337.csv /content/rsna_seed_other.csv --output /content/rsna_ensemble_predictions.csv
python scripts/evaluate_external.py --external-predictions /content/rsna_ensemble_predictions.csv --validation-predictions /content/chexpert_validation_ensemble.csv --output-dir /content/rsna_evaluation --threshold-method youden --bootstrap-iterations 1000 --seed 42
```

Outputs include the manifest, integrity summary, ensemble predictions, frozen-parameter metrics and patient-level bootstrap CIs, selective-prediction and failure-detection tables, and ROC, PR, reliability, confusion-matrix, risk-coverage, and uncertainty figures. The workflow never tunes a threshold or temperature on RSNA.

Important limitations are dataset shift, different labeling procedures, box-derived RSNA classification labels, possible external calibration degradation, and the deliberate absence of external-test tuning.
