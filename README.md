[简体中文](README_zh.md)

# MinerU to PPT Converter

This tool converts PDF files and images into editable PowerPoint presentations (`.pptx`) by leveraging structured data from the [MinerU PDF Extractor](https://mineru.net/OpenSourceTools/Extractor). It accurately reconstructs text, images, and layout, providing a high-fidelity, editable version of the original document.

The application features a user-friendly graphical interface (GUI) and is designed for easy use.

![GUI Screenshot](img/gui.png)

## For Users: How to Use

As a user, you only need the packaged Windows release (CPU or GPU variant). You do not need to install Python or any libraries.

1.  **Download the Application**: Get the latest package from the project's [Releases page](https://github.com/YOUR_USERNAME/YOUR_REPO/releases).
    -   `MinerU2PPT-win64-cpu-setup.exe`: CPU-only package (recommended default).
    -   `MinerU2PPT-win64-gpu-cu118-setup.exe`: CUDA 11.8 GPU package.
    -   `MinerU2PPT-win64-gpu-cu126-setup.exe`: CUDA 12.6 GPU package.
    -   `MinerU2PPT-win64-gpu-cu129-setup.exe`: CUDA 12.9 GPU package.

2.  **Get the MinerU JSON File**:
    -   Go to the [MinerU PDF/Image Extractor](https://mineru.net/OpenSourceTools/Extractor).
    -   Upload your PDF or image file and let it process.
    -   Download the resulting JSON file. This file contains the structural information that our tool needs for the conversion.
    ![Download JSON](img/download_json.png)

3.  **Run the Converter**:
    -   Double-click the executable to start the application.
    -   **Select Input File**: Drag and drop your PDF or image file onto the first input field, or use the "Browse..." button.
    -   **Select JSON File**: Drag and drop the JSON file you downloaded from MinerU onto the second input field.
    -   **Output Path**: The output path for your new PowerPoint file will be automatically filled in. You can change it by typing directly or using the "Save As..." button.
    -   **Options**:
        -   **Remove Watermark**: Check this box to automatically erase elements like page numbers or footers.
        -   **Generate Debug Images**: Keep this unchecked unless you are troubleshooting.
    -   Click **Start Conversion**.

4.  **Open Your File**: Once the conversion is complete, click the "Open Output Folder" button to find your new `.pptx` file.

### Using Batch Mode

The application also supports converting multiple files at once in Batch Mode.

1.  **Switch to Batch Mode**: Click the "Batch Mode" button in the top-right corner of the application. The interface will switch to the batch processing view.
2.  **Add Tasks**:
    -   Click the "Add Task" button. A new window will pop up.
    -   In the popup, select the **Input File**, the corresponding **MinerU JSON File**, and specify the **Output Path**.
    -   Set the **Remove Watermark** option for this specific task.
    -   Click "OK" to add the task to the list.
3.  **Manage Tasks**: You can add multiple tasks to the list. If you need to remove a task, select it from the list and click "Delete Task".
4.  **Start Batch Conversion**: Once all your tasks are added, click "Start Batch Conversion". The application will process each task sequentially. A log will show the progress for each file.

## For Developers

This section provides instructions for running the application from source and packaging it for distribution.

### Environment Setup

1.  Clone the repository.
2.  It is recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install dependencies based on your development target.
    -   **Default (GPU CUDA 11.8 full dev)**:
        ```bash
        pip install -r requirements.txt
        ```
    -   **GPU CUDA 12.6 full dev**:
        ```bash
        pip install -r requirements-gpu-cu126.txt -r requirements-build.txt
        ```
    -   **GPU CUDA 12.9 full dev**:
        ```bash
        pip install -r requirements-gpu-cu129.txt -r requirements-build.txt
        ```
    -   **CPU full dev (for CI/other contributors)**:
        ```bash
        pip install -r requirements-dev-cpu.txt
        ```

### Running from Source

-   **To run the GUI application**:
    ```bash
    python gui.py
    ```
-   **To use the CLI**:
    ```bash
    python main.py --json <path_to_json> --input <path_to_pdf_or_image> --output <path_to_ppt> [OPTIONS]
    ```

#### OCR CLI Options

-   `--ocr-device {auto,gpu,cpu}`: OCR device policy. Default is `auto` (`gpu -> cpu` fallback).
-   `--ocr-model-root <path>`: Optional local PaddleOCR model root.

Example:
```bash
python main.py --json "demo/case1/MinerU_xxx.json" --input "demo/case1/PixPin_xxx.png" --output "out.pptx" --ocr-device auto --ocr-model-root "models/paddleocr"
```

### Regression: Generate PPT for All Demo Cases

If you want regression to also produce PPT files for direct visual review, run:

```bash
python -m pytest "tests/integration/test_case1_ocr.py" -k all_demo_cases_generate_ppt_outputs_for_manual_review
```

Generated PPT files will be saved to:

- `tmp/regression_ppt_outputs/case1.pptx`
- `tmp/regression_ppt_outputs/case2.pptx`
- `tmp/regression_ppt_outputs/case3.pptx`
- `tmp/regression_ppt_outputs/case4.pptx`
- `tmp/regression_ppt_outputs/case5.pptx`

### Packaging as a Standalone Executable (Windows)

This project now recommends **onedir/installer-style packaging** over onefile for better runtime stability and easier model deployment.

1.  **Install PyInstaller**:
    ```bash
    pip install pyinstaller
    ```

2.  **Prepare Local OCR Models (offline-first)**:
    Put local models under one of the supported roots (priority order):
    1. CLI/engine `model_root`
    2. environment variable `MINERU_OCR_MODEL_ROOT`
    3. executable directory `models/paddleocr`
    4. source repository `models/paddleocr`

    Directory layout:
    ```
    models/paddleocr/<lang>/det
    models/paddleocr/<lang>/rec
    models/paddleocr/<lang>/cls
    ```

3.  **Build the onedir package**:
    ```bash
    pyinstaller --clean gui.spec
    ```

4.  **Find build output**:
    The packaged app directory will be generated under `dist/MinerU2PPT/`.

## Documentation

- Documentation domains:
  - `docs/architecture/`
  - `docs/testing/`
  - `docs/core-flow/`
  - `docs/api/`
- Core flow docs:
  - `docs/core-flow/font-size-normalization-pre-render.md`
  - `docs/core-flow/ocr-bbox-xy-refine-flow.md`
  - `docs/core-flow/watermark-ir-removal-flow.md`
- Testing docs:
  - `docs/testing/font-size-normalization-testing.md`
  - `docs/testing/ocr-bbox-refine-testing.md`
  - `docs/testing/watermark-ir-removal-testing.md`
