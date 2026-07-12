# Google Colab GPU Training

Use Google Colab only as a disposable execution environment. GitHub remains the source of
truth for the canonical research code: do not edit the canonical code directly in Colab.
Do not commit datasets, split manifests, outputs, checkpoints, secrets, or runtime-specific
paths.

## 1. Enable a GPU runtime

1. Open a new Colab notebook.
2. Choose **Runtime** > **Change runtime type**.
3. Set **Hardware accelerator** to **GPU** and select **Save**.
4. In a code cell, confirm that Colab assigned a GPU:

```bash
!nvidia-smi
```

If no GPU is shown, reconnect to a GPU runtime before continuing.

## 2. Clone the canonical repository

Replace the placeholder with the repository's public GitHub URL, then run:

```bash
!git clone https://github.com/<organization>/<repository>.git
%cd <repository>
```

Use the checked-out revision for the experiment. Do not make edits to the canonical research
code in the Colab session; prepare and review code changes outside Colab, then push them to
GitHub before running them here.

## 3. Install dependencies and verify CUDA

```bash
!python -m pip install --upgrade pip
!python -m pip install -r requirements.txt
!python scripts/check_runtime.py --require-cuda
```

The final command prints the Python and PyTorch versions, CUDA availability, the device count,
and the GPU name. It exits with an error if Colab did not provide CUDA.

## 4. Make CheXpert available without adding it to Git

Obtain CheXpert through its approved access process. Keep the dataset outside the cloned
repository, for example by mounting approved storage or uploading it to the temporary Colab
runtime. Never add image files or dataset metadata to Git.

Set `DATA_ROOT` to the directory containing the approved CheXpert files:

```python
DATA_ROOT = "/content/chexpert"
```

The directory must contain the images referenced by the split manifest. This path is passed only
at runtime and is not written to the run manifest or saved configuration.

## 5. Supply a portable patient-level split manifest

Place the already-created patient-level split manifest outside the repository or in an ignored
local staging location. Set its runtime location in the notebook:

```python
SPLIT_MANIFEST = "/content/patient_splits.csv"
```

The manifest must contain the required columns and use root-relative POSIX values in
`image_path`; do not put absolute local paths in the manifest. The training command records only
the manifest filename, never its full path. Do not regenerate or alter split assignments in
Colab, and do not use the test split during training.

## 6. Run a dry run

Start with a one-batch dry run. It disables pretrained-weight downloads and writes a new run
directory with a manifest, resolved config, history, and checkpoint.

```python
!python scripts/train_baseline.py --root "$DATA_ROOT" --manifest "$SPLIT_MANIFEST" --config configs/baseline_densenet121.yaml --dry-run
```

## 7. Run one full experiment

After the dry run succeeds, run the selected predefined configuration. This example runs the
reference DenseNet121 configuration; select another tracked configuration only when it is part
of the predefined protocol.

```python
!python scripts/train_baseline.py --root "$DATA_ROOT" --manifest "$SPLIT_MANIFEST" --config configs/baseline_densenet121.yaml
```

The configured output location contains a uniquely named run directory. It includes
`run_manifest.json`, `config.yaml`, `training_history.csv`, and the checkpoint selected by
validation AUROC. Training uses only train and validation data; the test split is not requested.

## 8. Preserve the run directory

Before the Colab runtime disconnects, archive the run directory and download it or copy it to
approved storage. Replace the placeholder with the run directory printed by the training script.

```python
!tar -czf experiment_run.tar.gz <run-directory>
from google.colab import files
files.download("experiment_run.tar.gz")
```

Keep the archived output out of Git. Preserve its run manifest together with the output so the
experiment can be reproduced from the tracked code revision and configuration.
