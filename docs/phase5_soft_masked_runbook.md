# Phase 5: soft-masking mitigation

The classifier uses the frozen lung-segmentation checkpoint and the unmodified binary mask `M = (probability >= 0.5)`.  For each grayscale input pixel `x`, the Phase 5 input is `x` inside `M` and `0.20 * x` outside `M`, converted deterministically to `uint8` before the established resize and classifier transforms.

`soft_mask_outside_factor` is frozen at `0.20`; it is recorded in the resolved configuration, checkpoint configuration, and external-prediction metadata.  The probability mask is never used as a continuous alpha mask, and no blur, feathering, morphology, crop, or other mask postprocessing is applied.

The factor must not be tuned on RSNA, PadChest, or any other external dataset.  The controlled comparison is Original versus Hard-Masked versus Soft-Masked, with the same A4 training settings, patient-level split, frozen segmentation checkpoint, and mask threshold `0.5`.
