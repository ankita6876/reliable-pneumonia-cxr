# Results workflow

Store downloaded Colab run ZIPs outside the repository, extract them under a local `experiment-results` folder, then build a metadata-only registry:

```powershell
python scripts/build_experiment_registry.py --inputs experiment-results --output registry.csv
python scripts/compare_experiments.py --registry registry.csv --output-dir comparisons
python scripts/generate_publication_artifacts.py --registry registry.csv --output-dir publication
```

The registry never copies checkpoints or predictions. Rankings use validation AUPRC then validation AUROC only; test and external results remain descriptive.
