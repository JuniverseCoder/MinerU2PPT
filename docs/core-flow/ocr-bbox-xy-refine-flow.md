# Core Flow: OCR Text BBox XY Refinement (Y Flow + X Extension)

## Flow Scope
- Flow name: OCR text bbox refinement in `refine_ocr_text_elements`
- Trigger: OCR text elements are extracted and line fragments are merged
- Entry/exit points:
  - Entry: `converter/ocr_merge.py::refine_ocr_text_elements`
  - Exit: refined OCR text elements with updated `bbox`, `lines[].bbox`, `spans[].bbox`
- OCR detection parameters:
  - Optional DB detection settings (det_db_thresh / det_db_box_thresh / det_db_unclip_ratio) are passed when the OCR engine is initialized and influence upstream detection before refinement.

## Happy Path
1. Convert element/line bbox from JSON coordinates to pixel coordinates.
2. Expand bbox to a padded window (`_expand_bbox_with_pad`).
3. Estimate `bg_color` and `font_color` from original bbox.
4. Build row and column hit flags in the padded window:
   - rows: `_build_row_font_flags`
   - cols: `_build_col_font_flags`
5. Refine Y boundaries (top/bottom) with two-stage logic:
   - first-wave trim inside original box,
   - second-wave extend on untouched side,
   - boundary alignment by row flags.
6. Refine X boundaries (left/right) with the same two-stage logic:
   - first-wave trim inside original box,
   - second-wave extend on untouched side,
   - boundary alignment by column flags.
7. Clamp refined bbox to image boundaries; if invalid, fallback to original bbox.
8. Convert refined pixel bboxes back to JSON coordinates.
9. Synchronize output geometry:
   - `line.bbox` uses refined line bbox,
   - `span.bbox` is aligned to refined line bbox in both X and Y,
   - final element bbox is union of refined lines.
10. Preserve debug fields: `ocr_bbox_original`, `ocr_bbox_pad`, `ocr_bbox_refined`.

## Exception/Alternative Paths
- Condition: No valid bbox input.
  - Handling: keep original element unchanged.
  - Outcome: no refinement on that element.
- Condition: Row/column hit flags are empty or no hit in original window.
  - Handling: keep original side(s) unchanged.
  - Outcome: partial or no refinement.
- Condition: Refined bbox becomes invalid (`x2 <= x1` or `y2 <= y1`).
  - Handling: fallback to original bbox.
  - Outcome: conversion remains stable.

## Ownership & Handoffs
- Component/role ownership by step:
  - OCR refinement implementation: `converter/ocr_merge.py`
  - Rendering consumption: `converter/generator.py`
- Input/output contracts:
  - Input: OCR text element list, page image, JSON/image geometry scale.
  - Output: OCR text element list with synchronized refined bboxes.

## State/Status Model (if applicable)
- States:
  - original bbox
  - padded bbox
  - refined bbox
  - fallback-to-original (if invalid)
- Transitions:
  - original -> padded -> refined OR original -> fallback
- Terminal conditions:
  - refined bbox valid and emitted
  - original bbox emitted after fallback

## Operational Notes
- Timeouts/retries:
  - No retry loop in refine stage; deterministic per element.
- Idempotency/concurrency:
  - Pure function behavior for same inputs; no shared mutable state.
- Observability hooks:
  - Debug fields retained in refined output for diagnostics.

## Validation
- Integration/e2e scenarios for this flow:
  - `tests/integration/test_generator_ocr_merge.py`
  - CLI smoke conversion with demo assets (OCR-to-PPT path)

## References
- PRD/Plan refs:
  - `docs/interview/spec-ocr-bbox-xy-refine.md`
  - `docs/plan/ocr-bbox-xy-refine-plan.md`
- Related code paths:
  - `converter/ocr_merge.py`
  - `tests/unit/test_ocr_bbox_refine.py`
