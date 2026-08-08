# Phase 2 external dataset decision

## A. Existing external-evaluation pipeline

The current RSNA path is reusable in components, but is not yet a generic external-dataset workflow.

- Manifest construction: `scripts/prepare_rsna.py` calls `pneumonia_ai.data.rsna.build_rsna_manifest`; the generic historical adapter is `scripts/build_external_manifest.py` plus `src/pneumonia_ai/data/multidataset.py`.
- External prediction: `scripts/predict_external.py:predict_external` loads the saved original or hard-masked checkpoint, resolves its saved input mode, and uses `prepare_classifier_image`. Hard-masked inference requires `FrozenLungSegmenter` and its mask cache.
- The current external predictor unconditionally calls `load_dicom_as_pil`, so it supports the RSNA DICOM representation, not a PNG/JPEG dataset without a narrow loader dispatch extension.
- Per-model evaluation: `scripts/evaluate_external.py` validates a CheXpert validation prediction table and an `external_test` prediction table; it applies frozen threshold provenance, CheXpert-fitted temperature scaling, discrimination and calibration metrics, confidence intervals, and ROC/PR/reliability/confusion-matrix figures.
- Statistics: `src/pneumonia_ai/evaluation/core.py` provides discrimination, calibration, and bootstrap utilities. `scripts/analysis/statistical_analysis.py` implements an older, hard-coded paired bootstrap analysis and should not be copied unchanged; Phase 2 should parameterize its aligned-prediction logic. The frozen RSNA paired results are authoritative artifacts rather than a fully reusable current CLI.
- Grad-CAM: `scripts/analysis/evaluate_rsna_gradcam_localization.py` reads a manifest and a separate box file, filters positive boxed cases, uses the original/hard-masked checkpoints and the frozen segmenter, then produces paired bootstrap and Pointing Game McNemar outputs. It is RSNA-named and DICOM-specific, but its localization metric functions are reusable if a scientifically valid boxed cohort exists.
- Curves: `scripts/analysis/plot_results.py` is the reusable two-model ROC/PR/calibration/confusion-matrix figure layer. Phase 1.1's ROC generator documents the appropriate no-pooling approach for three seeds.

## B. NIH label suitability

Local code documents NIH metadata fields `Image Index`, `Finding Labels`, and optionally `Patient ID`. `src/pneumonia_ai/data/multidataset.py:_parse_nih` splits the pipe-delimited multi-label `Finding Labels` field and sets a binary target to one when the exact label `Pneumonia` is present. The repository protocol calls this a prespecified **weak-label** external analysis: images with other findings are negative for the pneumonia target.

NIH therefore offers image-level multi-label labels, image IDs, patient IDs when supplied, and a `No Finding` label as one possible label value. Projection/view metadata is not consumed by the current adapter and was not documented locally. The `Pneumonia` label is not semantically equivalent to either CheXpert's pneumonia label or RSNA's box-derived opacity target; all cross-dataset comparisons must remain descriptive.

For an NIH-only sensitivity analysis, the defensible prespecified policy would be: positive iff exact `Pneumonia` is present; negative iff it is absent, including images with other findings; retain co-findings in positives; exclude missing/malformed labels. This is a weak-label target, not a radiographic reference standard.

## C. NIH bounding-box suitability

The repository has no NIH bounding-box parser, annotation file, class inventory, coordinate documentation, or locally documented pneumonia-box count. Consequently, this audit cannot establish from local evidence which NIH pathologies have boxes, whether pneumonia boxes exist, their prevalence, coordinate convention, or whether multiple boxes occur.

The RSNA evaluator could support Pointing Game, energy inside boxes, heatmap IoU, top-10%/top-20% energy inside, lesion coverage, and activation-area ratio only after a verified NIH box schema is mapped to its required `patient_id,x,y,width,height` format and demonstrated to contain an adequate pneumonia cohort. That evidence is absent locally. NIH localization is therefore **not approved** for Phase 2 planning.

## D. NIH/RSNA overlap concern

Local RSNA artifacts establish only that RSNA is an external DICOM cohort with box-derived `Target` labels; they do not themselves contain an image-lineage audit. However, the RSNA Pneumonia Detection Challenge is known to have been assembled from NIH ChestX-ray14 source images. Because browser research is prohibited for this audit and there is no local image-hash crosswalk, the exact overlap count cannot be established here.

For publication interpretation, NIH must be treated as source-overlapping with RSNA, not as a genuinely independent second external dataset. A separate NIH evaluation could be descriptive or a lineage/split sensitivity analysis, but it would not substantiate a second-independent-cohort claim.

## E. NIH versus PadChest

| Criterion | NIH ChestX-ray14 | PadChest |
| --- | --- | --- |
| Independence from RSNA | No for the second-independent-cohort claim because of RSNA/NIH source lineage | Expected yes; confirm no image-level overlap in dataset documentation before execution |
| Pneumonia target | Exact `Pneumonia` weak label is already implemented | Requires a one-time schema audit and explicit mapping of the PadChest pneumonia concept |
| Localization | Potentially possible, but not locally verified and not approved | Classification-only in the current scope |
| Existing code fit | Generic NIH manifest adapter exists; raster-image loading still needs a narrow extension | No adapter exists; needs manifest adapter plus the same raster-image loading extension |
| Publication value | Adds a related-source sensitivity cohort, not independent replication | Stronger independent external classification replication |

## F. Recommended dataset

**Recommend PadChest for Phase 2 classification.** Its expected independence from RSNA outweighs NIH's convenient but unverified localization possibility. This recommendation is conditional on a pre-run PadChest provenance and schema inspection that confirms a distinct source cohort and an unambiguous pneumonia label mapping. Do not claim NIH as the second independent external dataset.

## G. Exact target-label definition

For PadChest, the pre-registered target is: **positive iff the reviewed PadChest study-level annotation contains the canonical concept mapped to pneumonia; negative iff it lacks that concept; retain other thoracic findings in both groups; exclude only records with missing, malformed, or unmappable labels.**

The exact source column, literal label spelling, uncertainty convention, and multi-study/image aggregation rule are unresolved because no PadChest schema exists locally. They must be frozen after a dataset inspection cell and before any model inference. This is a data-definition step, not model selection or preprocessing tuning.

## H. Threshold policy

Report two prespecified operating-point analyses per matched model/seed:

1. Historical fixed threshold 0.5, clearly labelled as the RSNA-comparable reference.
2. The matching frozen CheXpert-validation Youden-J threshold from `configs/operating_thresholds/`, applied unchanged: original/hard-masked seed 42 = 0.4219265282154083 / 0.5608569383621216; use the corresponding frozen artifacts for seeds 123 and 2026.

No PadChest or NIH labels may select, optimize, recalibrate, or otherwise influence either threshold. AUROC and PR-AUC remain threshold-independent. `external_threshold_metadata` already rejects external threshold-selection provenance.

## I. Evaluation design

Primary design: run all three matched seeds (42, 123, 2026) for original and hard-masked classifiers, with each condition/seed using its own frozen CheXpert operating threshold. This is scientifically stronger than a seed-42-only analysis because it quantifies seed variability.

For comparability with the historical RSNA result, include a clearly labelled seed-42 paired table and ROC/PR overlay as a prespecified supplementary replication. Do not pool or concatenate predictions across independently trained seeds. Plot seed-specific curves or a mean interpolated ROC with seed variability as in Phase 1.1.

Outputs should mirror RSNA where applicable: predictions, metric and confusion-matrix JSON, calibration metrics and reliability plot, ROC/PR plots, paired predictions, masking outcome summary, paired bootstrap comparisons, domain-shift summary, McNemar results, and the fixed-0.5/validation-Youden threshold sensitivity table. Report AUROC, PR-AUC, accuracy, balanced accuracy, sensitivity, specificity, precision, F1, NPV, Brier, NLL, and ECE.

## J. Localization feasibility

**Not planned for PadChest in Phase 2.** The approved recommendation is classification-only. NIH Grad-CAM localization remains scientifically unresolved: it requires an independently verified pneumonia-box cohort and coordinate schema, and would not resolve NIH/RSNA image-lineage dependence.

## K. Minimal code changes

Do not create a parallel evaluation architecture.

1. Parameterize `scripts/build_external_manifest.py`/the shared manifest adapters with a PadChest adapter after its schema is verified.
2. Add a conservative image-loader dispatch to `scripts/predict_external.py`: retain existing DICOM decoding for `.dcm`; use PIL raster loading for verified PNG/JPEG PadChest paths. Preserve the classifier's existing transform and hard-mask path exactly.
3. Parameterize RSNA-labelled titles/output names in `scripts/evaluate_external.py`; retain its frozen CheXpert threshold/temperature safeguards.
4. Generalize the aligned paired-bootstrap/McNemar reporting around existing `evaluation.core` utilities rather than copying the older absolute-path `statistical_analysis.py`.
5. Reuse Phase 1.1 threshold artifacts directly; do not create or estimate any threshold on the new dataset.

## L. Kaggle execution plan

Attach: (1) a provenance-verified PadChest image and metadata dataset; (2) the frozen A4 original and hard-masked checkpoints for seeds 42, 123, and 2026; (3) the frozen segmentation checkpoint required by hard-masked inference; and (4) the repository code plus frozen threshold JSONs.

Required inspection cell before implementation: print the metadata columns and unique candidate labels; inspect image extensions, root-relative paths, image count, patient/study IDs, duplicate image IDs, missing files, and any view/projection fields. Freeze the manifest mapping before inference.

Run a stratified, deterministic 8–16-case smoke test for both target classes and both input modes. Then run the full paired all-seed evaluation. GPU is useful for six full inference passes; CPU is functionally sufficient but likely slow. No GPU-heavy work is needed for the inspection or manifest audit.

## M. Risks and unresolved questions

- PadChest metadata schema, licence/access route, exact pneumonia concept, and image layout are not present locally and must be inspected before implementation.
- Expected PadChest independence from RSNA must be documented during its provenance audit; exact image overlap cannot be established from the present local repository.
- The current `predict_external.py` only decodes DICOM; raster dispatch needs focused tests before execution.
- The historical RSNA paired classification artifact identifies only seed-42 checkpoint paths: original `/kaggle/input/datasets/ankitaroy123567/a4-original-control-results/A4_original_control/best_checkpoint.pt` and hard-masked `/kaggle/input/datasets/ankitaroy123567/a4-bestcheckpoint/best_checkpoint.pt`. Checkpoints for seeds 123 and 2026 exist in preserved A4 ZIP archives, but their future Kaggle asset paths and hashes need freezing before Phase 2 inference.
- No NIH box cohort decision should be made without authoritative annotation-schema verification.
