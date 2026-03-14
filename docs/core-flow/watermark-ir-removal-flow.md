# Core Flow: MinerU Watermark Flag -> IR -> Final Removal

## Flow Scope
- Flow name: Watermark semantic propagation and final render removal
- Trigger: `convert_mineru_to_ppt` builds page IR and runs OCR merge before `process_page`
- Entry/exit points:
  - Entry: `converter/adapters/mineru_adapter.py::MinerUAdapter.extract_page_elements`
  - Exit: `converter/generator.py::PPTGenerator.process_page` render filtering

## Happy Path
1. MinerU adapter reads page blocks (`para_blocks/images/tables/discarded_blocks`).
2. Adapter resolves watermark semantic per block:
   - explicit `is_watermark` / `watermark` if present,
   - for `discarded_blocks`, fallback watermark is `True`.
3. Adapter writes `is_watermark` into unified IR (`TextIR` / `ImageIR`).
4. OCR adapter generates OCR text IR with `is_watermark=False` by default.
5. IR merge replaces MinerU text with OCR text where needed, but preserves base semantic flags:
   - `is_discarded = base OR overlay`
   - `is_watermark = base OR overlay`
6. Generator builds cleanup/render sets using watermark semantic:
   - keep watermark elements only when `remove_watermark=False`
   - drop watermark elements when `remove_watermark=True`.
7. Slide rendering uses filtered text/image sets and outputs final PPT.

## Exception/Alternative Paths
- Condition: block has no explicit watermark field and is not in `discarded_blocks`.
  - Handling: watermark defaults to `False`.
  - Outcome: block is treated as normal content.
- Condition: OCR replacement happens on a watermark block.
  - Handling: merge stage inherits watermark from base element.
  - Outcome: watermark semantic is not lost after replacement.

## Ownership & Handoffs
- Component/role ownership by step:
  - Watermark source extraction: `converter/adapters/mineru_adapter.py`
  - IR schema/normalization: `converter/ir.py`
  - Merge semantic retention: `converter/ir_merge.py`
  - Final render filtering: `converter/generator.py`
- Input/output contracts:
  - Input: MinerU JSON + OCR text elements
  - Output: merged typed IR where watermark semantics survive to render stage

## State/Status Model
- States:
  - source semantic (`discarded_blocks` or explicit watermark field)
  - IR semantic (`is_watermark`)
  - merged semantic (`is_watermark` retained through overlay replacement)
  - render decision (`removed` or `kept`)
- Transitions:
  - source -> adapter mapping -> IR normalize -> merge inherit -> render filter
- Terminal conditions:
  - watermark element removed when `remove_watermark=True`
  - watermark element retained when `remove_watermark=False`

## Operational Notes
- Timeouts/retries:
  - No retry loop specific to watermark logic.
- Idempotency/concurrency:
  - Semantic mapping is deterministic for same JSON + OCR inputs.
- Observability hooks:
  - Existing `[OCR]` merge stats remain available in conversion logs.

## Validation
- Integration/e2e scenarios for this flow:
  - `tests/unit/test_mineru_adapter.py`
  - `tests/unit/test_ir.py`
  - `tests/unit/test_ir_merge.py`
  - `tests/unit/test_generator_cleanup_order.py`

## References
- PRD/Plan refs:
  - `docs/interview/spec-unified-ir-decouple-mineru.md`
  - `docs/plan/unified-ir-decouple-mineru-plan.md`
- Related code paths:
  - `converter/adapters/mineru_adapter.py`
  - `converter/adapters/ocr_adapter.py`
  - `converter/ir.py`
  - `converter/ir_merge.py`
  - `converter/generator.py`
