# OCR Engine Configuration

## Scope
- Affected modules/layers:
  - GUI input wiring (`gui.py`)
  - CLI flags (`main.py`)
  - OCR engine initialization (`converter/ocr_merge.py`, `converter/generator.py`)
- In-scope decisions:
  - Which OCR detection parameters are exposed to users
  - Parameter precedence across GUI/CLI/engine defaults
- Out-of-scope:
  - OCR bbox refinement algorithms (documented in core-flow docs)

## Rules & Constraints
- Layer/import boundaries:
  - GUI/CLI only pass config values; OCR logic remains inside `PaddleOCREngine`.
- Dependency direction:
  - UI/CLI -> `convert_mineru_to_ppt` -> `PaddleOCREngine`.
- DI/composition constraints:
  - GUI reuses a shared OCR engine instance and must reset `_ocr` when settings change.

## Design Decisions
- Decision: expose OCR detection DB parameters in GUI advanced options.
- Rationale: allow fine-tuning detection strictness and bbox tightness without CLI use.
- Alternatives considered:
  - CLI-only exposure (rejected due to GUI usage focus).
- Trade-offs:
  - More UI complexity; advanced panel is collapsible by default.

## Implementation Impact
- Files/components impacted:
  - `gui.py`: advanced OCR section (det_db_thresh / det_db_box_thresh / det_db_unclip_ratio), batch and single wiring.
  - `main.py`: CLI flags already pass these parameters.
  - `converter/generator.py`: passes OCR params into `PaddleOCREngine` when engine is built.
  - `converter/ocr_merge.py`: uses det_db_* overrides when initializing PaddleOCR.
- Migration/cutover notes:
  - No migration required; defaults remain unchanged when fields are empty.
- Backward-compatibility notes:
  - Empty fields preserve model variant defaults.

## Validation
- Architecture rule checks/tests:
  - Unit tests for OCR engine init and merge logic.
- Expected pass criteria:
  - GUI starts with advanced options collapsed.
  - OCR detection params only affect engine when provided.

## References
- PRD/Plan refs:
  - `docs/plan/ocr-model-variant-selection-plan.md`
- Related docs:
  - `docs/core-flow/ocr-bbox-xy-refine-flow.md`
