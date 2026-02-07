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
  - **`utils.py`**: Low-level helpers for image processing, color analysis, and segmentation.

### Key Implementation Details
- **Unified Input**: The core logic in `convert_mineru_to_ppt` handles both PDF and single-image files as input, using the same sophisticated MinerU JSON-driven pipeline for both.
- **Stateful Page Processing**: A `PageContext` class holds the state for each page, including the original image, a progressively cleaned background, and a list of all detected elements.
- **Two-Phase Conversion**: Page processing is split into two phases:
  1.  **Analysis**: Elements from the JSON are processed to extract their data and populate the `PageContext`. A clean background is created by inpainting the area under each element.
  2.  **Rendering**: The cleaned background is rendered, followed by images, and finally text. This ensures correct Z-order layering.
- **Watermark/Footer Handling**: Elements marked as `discarded_blocks` in the JSON are handled based on the `remove_watermark` option.
- **Advanced Text Processing**:
  - **Bullet Point Correction**: A heuristic prepends a bullet character (`•`) if the first two detected `raw_chars` in a text block have different colors.
  - **Single-Line Textbox Widening**: Single-line textboxes are widened by 20% during rendering to prevent unwanted wrapping.
- **GUI Logic**:
  - **Single and Batch Modes**: The GUI supports two modes of operation. Users can switch between converting a single file and managing a list of files for batch processing.
  - **Modal Task Dialog**: In batch mode, a modal `AddTaskDialog` is used to add new conversion tasks. This dialog includes its own file browsers and drag-and-drop functionality.
  - **Dynamic UI**: The main window's UI changes dynamically based on the selected mode. Options that are not relevant for batch mode (like debugging) are hidden to simplify the interface.
  - **Per-Task Options**: In batch mode, options like "Remove Watermark" are configured individually for each task within the `AddTaskDialog`.
  - **Internationalization (i18n)**: Auto-detects OS language for English or Chinese UI.
  - **Drag and Drop**: `tkinterdnd2` is used for file inputs in both the main window and the task dialog.
  - **Asynchronous Processing**: The conversion process runs in a separate thread to keep the GUI responsive, for both single and batch conversions.
