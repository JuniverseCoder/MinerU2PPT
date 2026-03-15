# OCR Configuration Testing

## Scope
- Feature/workflow under test:
  - OCR engine configuration from GUI/CLI into `PaddleOCREngine`
- Test layers covered: unit / integration

## Shared Testing Rules
- Mandatory coverage expectations:
  - GUI/CLI parameter wiring must not change defaults when fields are empty.
  - OCR engine should accept det_db_* overrides without initialization errors.
- Test data/fixtures conventions:
  - Use small synthetic bboxes; avoid model downloads where possible.
- Mock/adapter expectations:
  - Prefer unit tests to assert parameter values on engine instances.

## How to Run
- Unit:
  - `python -m pytest tests/unit`
- Integration:
  - `python -m pytest tests/integration/test_generator_ocr_merge.py`

## Workflow Validation Requirements
- When workflow changes, required test updates:
  - Update/extend unit tests if OCR config parameters are added or renamed.
  - Ensure GUI advanced options remain collapsible and do not crash on start.
- Required evidence for sign-off:
  - Unit tests green.
  - GUI launch smoke test (instantiate `App`).

## Failure Handling
- Common failure patterns:
  - GUI startup errors due to missing widget attributes.
  - OCR engine reuse without resetting `_ocr` after config changes.
- Debug/triage checklist:
  - Confirm advanced options are hidden by default.
  - Verify det_db_* are `None` when fields are empty.

## References
- PRD/Plan refs:
  - `docs/plan/ocr-model-variant-selection-plan.md`
- Related test files:
  - `tests/unit/test_ocr_engine_init.py`
  - `tests/integration/test_generator_ocr_merge.py`
