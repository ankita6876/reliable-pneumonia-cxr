# Study Protocol

## 1. Research Objective

**Objective:** [State the primary objective of this study.]

**Target population and clinical setting:** [Specify the intended population and setting.]

**Primary outcome:** [Specify the pneumonia detection outcome and reference standard.]

## 2. Research Questions

1. [What is the primary research question regarding pneumonia detection from chest X-rays?]
2. [How will model performance vary across the selected datasets and evaluation settings?]
3. [How well calibrated are the planned models' predicted probabilities?]
4. [How reliably do the planned models quantify predictive uncertainty?]
5. [What are the planned analyses of performance across relevant subgroups?]

## 3. Hypotheses

**Primary null hypothesis (H0):** [State the null hypothesis for the primary comparison or outcome.]

**Primary alternative hypothesis (H1):** [State the corresponding alternative hypothesis.]

**Secondary null hypotheses (H0):**

- [State secondary null hypothesis 1.]
- [State secondary null hypothesis 2.]

**Secondary alternative hypotheses (H1):**

- [State secondary alternative hypothesis 1.]
- [State secondary alternative hypothesis 2.]

## 4. Datasets

### CheXpert

- Version: [Specify version.]
- Local path: [Specify local path; do not commit data to GitHub.]
- Label definition and uncertainty handling: Frontal images with Pneumonia labels `1`, `0`,
  or `-1` are eligible. Missing Pneumonia labels are excluded. Uncertain (`-1`) labels are
  retained without transformation in the source cohort. Four predefined uncertainty-label
  experiments will be compared: `ignore` excludes `-1` records; `u_zero` maps `-1` to `0`;
  `u_one` maps `-1` to `1`; and `soft_uncertain` assigns uncertain rows a configurable soft
  target (initially 0.5) and lower loss weight (initially 0.5). Raw labels remain available for
  analysis. Training may include uncertain rows according to strategy, but validation model
  selection uses only definite `0`/`1` labels for directly comparable AUROC and AUPRC. No
  strategy is designated as best before evaluation.
- Intended role: CheXpert `train.csv` supplies the eligible development cohort; the official
  `valid.csv` is reserved as a secondary evaluation set.

### MIMIC-CXR

- Version: [Specify version.]
- Local path: [Specify local path; do not commit data to GitHub.]
- Label definition and uncertainty handling: MIMIC-CXR-JPG metadata is converted to the shared
  portable manifest, retaining only definite Pneumonia labels `0` or `1` for external evaluation.
- Intended role: external validation only; it is never used for threshold selection, calibration,
  architecture selection, or model training.

### NIH ChestX-ray14

- Version: [Specify version.]
- Local path: [Specify local path; do not commit data to GitHub.]
- Label definition and uncertainty handling: `Pneumonia` in `Finding Labels` is positive and other
  findings are negative in the prespecified weak-label external analysis.
- Intended role: external validation only under the shared preprocessing, manifest, inference, and
  fixed-threshold evaluation interfaces.

## 5. Inclusion and Exclusion Criteria

**Inclusion criteria:**

- CheXpert frontal images only.
- CheXpert records with Pneumonia labels `1`, `0`, or `-1`.
- [Define any population or metadata requirements.]

**Exclusion criteria:**

- [Define image-quality exclusions.]
- [Define duplicate, missing, or incomplete-record handling.]
- Lateral CheXpert images and rows with missing Pneumonia labels.

## 6. Data Splitting Strategy

All splitting must occur at the **patient level** so that no patient contributes images to more than one split.

- Training split: 70% of CheXpert cohort patients, assigned deterministically with seed 42.
- Validation split: 15% of CheXpert cohort patients, assigned deterministically with seed 42.
- Test split: 15% of CheXpert cohort patients, assigned deterministically with seed 42.
- External evaluation: NIH ChestX-ray14 and MIMIC-CXR are converted to a shared root-relative
  manifest format. The CheXpert validation-selected threshold and calibration parameters are frozen
  before external inference; cross-dataset reports are descriptive and do not alter development.
- Stratification: patient-level allocation stratified as closely as practical by original
  Pneumonia label (`1`, `0`, or `-1`); uncertain labels remain untransformed.

Test data must not be used for model selection, threshold tuning, or other development decisions.

## 7. Planned Models

### DenseNet121

- Role: reference baseline.
- Architecture and initialization: timm `densenet121`, configured for one binary output logit;
  pretrained ImageNet weights are a configurable initialization option.
- Training configuration: the shared initial protocol uses 224x224 inputs, ImageNet
  normalization, weighted BCEWithLogitsLoss, the predefined uncertainty-label experiments, and
  train/validation splits only; validation selection is restricted to definite labels.

### ConvNeXt-Tiny

- Role: predefined comparator backbone.
- Architecture and initialization: timm `convnext_tiny`, configured for one binary output logit;
  pretrained ImageNet weights are a configurable initialization option.
- Training configuration: the same shared initial protocol as DenseNet121.

### EfficientNetV2-S

- Role: predefined comparator backbone.
- Architecture and initialization: timm `tf_efficientnetv2_s`, configured for one binary output
  logit; pretrained ImageNet weights are a configurable initialization option.
- Training configuration: the same shared initial protocol as DenseNet121.

### TorchXRayVision DenseNet121 (all datasets)

- Role: predefined domain-pretraining comparison, not a proposed final model.
- Architecture and initialization: TorchXRayVision DenseNet121 with the
  `densenet121-res224-all` chest-X-ray weights when enabled; its multi-pathology classifier is
  replaced with one binary pneumonia logit.
- Training configuration: uses one-channel TorchXRayVision preprocessing and the same
  patient-level `ignore`, train/validation-only protocol, seed, optimizer, scheduler, AMP,
  early stopping, class weighting, and definite-label validation selection as ConvNeXt-Tiny.

No architecture is designated the winner before evaluation.

## 8. Planned Experiments

1. [Establish a baseline training and validation protocol.]
2. [Evaluate each planned architecture under the same predefined protocol.]
3. [Assess performance on held-out internal test data.]
4. [Assess external generalization across applicable datasets.]
5. [Evaluate calibration and uncertainty-estimation approaches.]
6. [Conduct predefined subgroup and robustness analyses.]
7. [Perform explainability review using the planned methods.]

## 9. Evaluation Metrics

### Primary Metrics

- AUROC and AUPRC with patient-level 95% bootstrap confidence intervals.
- [Specify primary operating-point metric(s), if applicable.]

### Secondary Metrics

- A validation-only frozen operating threshold (Youden's J, maximum validation F1, or prespecified 0.5) will be applied unchanged to the final test set.
- [Specify sensitivity, specificity, precision, recall, and F1-score reporting plan.]
- [Specify subgroup and external-evaluation reporting plan.]

### Calibration Metrics

- Negative log-likelihood, Brier score, ECE, adaptive ECE, calibration intercept/slope, and reliability diagrams will be reported. Temperature scaling is fitted only on validation logits and frozen before test evaluation.
- [Specify calibration-curve reporting plan.]
- [Specify probability-calibration method(s), if planned.]

### Uncertainty Metrics

- Deep ensembles are the primary uncertainty method. Predictive entropy, expected entropy, mutual information, probability variance, and logit variance will be recorded. Uncertainty-based failure detection compares correct and incorrect predictions using AUROC/AUPRC, Mann–Whitney U, and rank-biserial effect size.
- Selective prediction will retain the least-uncertain cases at prespecified coverage levels and report risk–coverage curves. Test data remains untouched until the single final evaluation using frozen validation choices.
- [Specify uncertainty-quality metric(s) and referral/abstention analysis plan.]

## 10. Statistical Analysis

- Confidence intervals: [Specify confidence level, resampling method, and number of resamples.]
- Statistical tests: [Specify planned tests for model or metric comparisons.]
- Multiple comparisons: [Specify correction method or rationale.]
- Significance threshold: [Specify threshold and interpretation plan.]
- Missing data: [Specify handling strategy.]

## 11. Explainability Plan

### Grad-CAM

- [Specify target layers, image selection procedure, and review process.]

### Integrated Gradients

- [Specify baseline, attribution settings, image selection procedure, and review process.]

## 12. Reproducibility Checklist

- [ ] Record dataset versions, access dates, and preprocessing decisions.
- [ ] Maintain patient-level split assignments and random seeds.
- [ ] Version configuration files and dependency specifications.
- [ ] Document model architectures, initialization, and training settings.
- [ ] Log evaluation settings, thresholds, and metric implementations.
- [ ] Preserve analysis scripts and generated figures where appropriate.
- [ ] Add tests for important data-processing and evaluation code.
- [ ] Keep medical images and private credentials out of version control.
- [ ] Record deviations from this protocol with justification.

## 13. Risks and Limitations

- [Placeholder: dataset representativeness and potential selection bias.]
- [Placeholder: label quality, uncertainty, and reference-standard limitations.]
- [Placeholder: domain shift across institutions, devices, and patient populations.]
- [Placeholder: confounding, shortcuts, and spurious correlations.]
- [Placeholder: limitations of explainability and uncertainty methods.]
- [Placeholder: ethical, privacy, and clinical-deployment considerations.]
