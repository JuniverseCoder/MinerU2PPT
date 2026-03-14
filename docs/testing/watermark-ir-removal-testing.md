# Testing Guide: Watermark Semantic in Unified IR and Render Removal

## Scope
- Feature/workflow under test: watermark semantic propagation from MinerU source blocks to final render filtering
- Test layers covered: unit / integration-smoke

## Shared Testing Rules
- Mandatory coverage expectations:
  - MinerU adapter maps watermark semantic into IR field `is_watermark`.
  - `discarded_blocks` default to watermark `True` when no explicit field exists.
  - Explicit watermark field on normal blocks is propagated.
  - OCR overlay replacement does not drop watermark semantic from base elements.
  - Final render filtering is driven by `is_watermark` and `remove_watermark`.
- Test data/fixtures conventions:
  - Keep fixtures minimal: one non-watermark element and one watermark element in same page.
  - Include one overlap-replace scenario to verify merge inheritance behavior.
- Mock/adapter expectations:
  - Generator filtering tests should mock rendering side effects and assert filtered element set.

## How to Run
- Unit:
  - `python -m pytest "tests/unit/test_mineru_adapter.py" "tests/unit/test_ir.py" "tests/unit/test_ir_merge.py" "tests/unit/test_generator_cleanup_order.py"`
- Integration:
  - `python -m pytest "tests/integration/test_generator_ocr_merge.py"`
- E2E/Smoke:
  - `python main.py --json "demo/case1/MinerU_PixPin_2026-03-05_21-52-43__20260305135318.json" --input "demo/case1/PixPin_2026-03-05_21-52-43.png" --output "tmp/watermark-ir-smoke-output.pptx" --no-watermark`

## Workflow Validation Requirements
- When workflow changes, required test updates:
  - Any change in adapter watermark fallback rules must update adapter unit assertions.
  - Any change in merge replacement logic must keep watermark inheritance checks.
  - Any change in render filtering must keep `is_watermark`-based filtering assertions.
- Required evidence for sign-off:
  - Unit command above passes.
  - At least one integration test run confirms OCR pipeline compatibility.

## Failure Handling
- Common failure patterns:
  - Watermark flag lost after OCR replacement due to missing inheritance in merge stage.
  - Render filtering mistakenly tied to `is_discarded` only.
  - Source JSON uses alternate watermark key and adapter parse path is missing.
- Debug/triage checklist:
  1. Confirm adapter output includes `is_watermark` on expected elements.
  2. Confirm merged IR retains `is_watermark` on replaced text elements.
  3. Confirm generator filter condition references `is_watermark`.

## References
- PRD/Plan refs:
  - `docs/interview/spec-unified-ir-decouple-mineru.md`
  - `docs/plan/unified-ir-decouple-mineru-plan.md`
- Related test files:
  - `tests/unit/test_mineru_adapter.py`
  - `tests/unit/test_ir.py`
  - `tests/unit/test_ir_merge.py`
  - `tests/unit/test_generator_cleanup_order.py`
