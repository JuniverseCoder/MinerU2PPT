# Core Flow: Page-Level Font Size Normalization (Pre-Render)

## Scope
This document describes the page-level text font-size normalization step in `PPTGenerator.process_page`.

## Runtime Position
The normalization step runs **after** image/text cleanup and **before** element rendering to slide.

High-level order in `process_page`:
1. Validate/materialize page IR.
2. Build cleanup/render element sets.
3. Run `_cleanup_text_and_images(...)`.
4. Run `_normalize_page_text_font_sizes(...)`.
5. Process elements into `PageContext` and render (`render_to_slide`).

## Eligibility Rules (TextElement Gate)
Normalization is applied at `TextIR` (text_element) level.

A text element is considered eligible only if it can be internally unified:
- Bold status is consistent across runs (or can be inferred for no-run element).
- Effective run/element font sizes are available.
- Internal spread satisfies threshold (`max_size / min_size <= 1.3`).

If eligibility checks fail, the text element is excluded from normalization and kept unchanged.

## Grouping and Optimization Strategy
Within each page:
1. Split eligible elements by bold bucket (`bold=True` and `bold=False`).
2. **Stage A (seed grouping):** cluster by center-distance threshold (`<= 1.3`).
3. **Stage B (optimization):** use Stage A cluster count as `K`, run 1D K-means in log-size space.
4. Validate optimized clusters with the same center-distance threshold.
   - If validation fails, fallback to Stage A clusters.

## Writeback Rule
For each final cluster with size >= 2:
- Compute median font size, round to integer pt.
- Apply this target size to the whole text element:
  - `elem.style.font_size`
  - every `run.style.font_size` when runs exist

Single-element clusters are not normalized.

## Design Constraints
- Page-local only (no cross-page accumulation).
- Bold bucket separation is mandatory.
- Threshold semantics are ratio-based (`1.3`), not absolute point difference.
- Non-text elements are unaffected.
