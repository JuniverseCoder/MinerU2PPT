# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Development Commands

### Environment Setup
- **Install Dependencies**: `pip install -r requirements.txt`
- **Python Version**: Python 3.10+ recommended

### Execution
- **Run GUI**: `python gui.py` (Recommended for most uses)
- **Run CLI**: `python main.py --json <path_to_json> --input <path_to_input> --output <path_to_ppt> [OPTIONS]`
  - `--no-watermark`: Erase elements marked as `discarded_blocks`.
  - `--debug-images`: Generate diagnostic images in the `tmp/` folder.

### Packaging
- **Create Executable**: `pyinstaller --windowed --onefile --name MinerU2PPT gui.py`

## Code Architecture

### High-Level Structure
- **`gui.py`**: The main entry point for end-users. A `tkinter`-based GUI that provides a user-friendly interface for the conversion process.
- **`main.py`**: The entry point for the command-line interface (CLI).
- **`converter/`**: Core conversion logic.
  - **`generator.py`**: Contains `PPTGenerator` and `PageContext`. It orchestrates the entire conversion from a source (PDF/image) and a MinerU JSON file to a PPTX presentation.
  - **`ocr_merge.py`**: PaddleOCR adapter and OCR-first merge/refine policy. It normalizes OCR outputs, performs two-stage bbox refinement, and groups OCR boxes with MinerU overlap as reference while keeping OCR as the final text source.
  - **`utils.py`**: Low-level helpers for image processing, color analysis, and segmentation.

### Key Implementation Details
- **Unified Input**: The core logic in `convert_mineru_to_ppt` handles both PDF and single-image files as input, using the same sophisticated MinerU JSON-driven pipeline for both.
- **Stateful Page Processing**: A `PageContext` class holds the state for each page, including the original image, a progressively cleaned background, and a list of all detected elements.
- **Two-Phase Conversion**: Page processing is split into two phases:
  1.  **Analysis**: Elements from the JSON are processed to extract their data and populate the `PageContext`. A clean background is created by inpainting the area under each element.
  2.  **Rendering**: The cleaned background is rendered, followed by images, and finally text. This ensures correct Z-order layering.
- **Watermark/Footer Handling**: Elements marked as `discarded_blocks` in the JSON are handled based on the `remove_watermark` option.
- **Advanced Text Processing**:
  - **Bullet Point Correction**: A heuristic prepends a bullet character (`•`) when source text starts with markdown-style list marker.
  - **Single-Line Textbox Widening**: Single-line textboxes are widened by 20% during rendering to prevent unwanted wrapping.
- **Forced OCR-First Pipeline (PaddleOCR)**:
  - OCR is mandatory in CLI and GUI paths.
  - In `PPTGenerator.process_page`, OCR extraction runs before per-element analysis. OCR failures raise exceptions and abort conversion.
  - OCR bbox refinement uses a two-stage vertical strategy (pad window, inner trim, then conditional extend).
  - Overlap handling uses grouping + union (instead of overlap-drop), with OCR kept as the final text source.
  - Merge statistics are emitted as `[OCR]` logs (`candidates`, `groups`, `merged`, `added`).
  - GUI caches a shared `PaddleOCREngine` instance (`self.shared_ocr_engine`) and reuses it across single/batch tasks to avoid repeated initialization overhead.
  - Debug mode additionally exports page-level text bounding box overlays (`tmp/page_<index>_text_boxes.png`) alongside existing char-level debug images.
- **GUI Logic**:
  - **Single and Batch Modes**: The GUI supports two modes of operation. Users can switch between converting a single file and managing a list of files for batch processing.
  - **Modal Task Dialog**: In batch mode, a modal `AddTaskDialog` is used to add new conversion tasks. This dialog includes its own file browsers and drag-and-drop functionality.
  - **Dynamic UI**: The main window's UI changes dynamically based on the selected mode. Options that are not relevant for batch mode (like debugging) are hidden to simplify the interface.
  - **Per-Task Options**: In batch mode, options like "Remove Watermark" are configured individually for each task within the `AddTaskDialog`.
  - **Internationalization (i18n)**: Auto-detects OS language for English or Chinese UI.
  - **Drag and Drop**: `tkinterdnd2` is used for file inputs in both the main window and the task dialog.
  - **Asynchronous Processing**: The conversion process runs in a separate thread to keep the GUI responsive, for both single and batch conversions.

## Testing Notes
- OCR merge/group logic is covered by focused unit tests (`tests/unit/test_ocr_merge.py`) for grouping and union behavior.
- OCR bbox refine behavior is covered by `tests/unit/test_ocr_bbox_refine.py`.
- Pipeline and wiring coverage is provided by integration tests for forced-OCR behavior and removed CLI OCR flag handling (`tests/integration/test_generator_ocr_merge.py`, `tests/integration/test_cli_ocr_option.py`).
- Real-sample OCR structure and bbox alignment checks use demo assets under `demo/` (`tests/integration/test_case1_ocr*.py`, `tests/integration/test_case3_ocr_bbox_bottom.py`).

## Docs Index
- Core flow:
  - `docs/core-flow/font-size-normalization-pre-render.md`
- Testing:
  - `docs/testing/font-size-normalization-testing.md`
