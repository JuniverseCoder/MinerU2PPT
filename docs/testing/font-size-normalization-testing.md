# Testing Guide: Font Size Normalization (Page-Level Pre-Render)

## Purpose
Define reusable testing guidance for page-level font-size normalization behavior.

## Test Layers

### Unit Tests
Target: normalization algorithm behavior in isolation.

Required coverage:
- Bold-bucket separation and cluster normalization outcomes.
- Threshold behavior at ratio `1.3` (merge vs split expectations).
- Text-element eligibility gate (ineligible elements remain unchanged).
- Two-stage clustering correctness:
  - seed grouping by center threshold,
  - K-means optimization preserving threshold constraints,
  - fallback-safe outcomes.
- Writeback behavior:
  - target size applied to `elem.style.font_size`,
  - target size applied to all run styles when runs exist.

Primary file:
- `tests/unit/test_generator_font_normalization.py`

### Integration Tests
Target: pipeline insertion point and non-regression.

Required coverage:
- Normalization is invoked once in `process_page` before final render.
- Non-text rendering behavior is preserved (e.g., image bbox/path unaffected).
- Existing OCR merge/render behavior remains intact.

Primary file:
- `tests/integration/test_generator_ocr_merge.py`

## Recommended Commands
```bash
python -m pytest "tests/unit/test_generator_font_normalization.py" "tests/integration/test_generator_ocr_merge.py"
```

Optional broader check:
```bash
python -m pytest "tests/unit/test_generator_page_range.py" "tests/unit/test_generator_cleanup_order.py" "tests/unit/test_generator_text_cleanup_margin.py" "tests/unit/test_generator_font_normalization.py" "tests/integration/test_generator_ocr_merge.py"
```

## Acceptance Expectations
- All unit and integration checks above pass.
- No regression in existing OCR merge integration behavior.
- Normalization constraints remain enforceable after optimization stage.
