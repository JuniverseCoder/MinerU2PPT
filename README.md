[简体中文](README_zh.md)

# MinerU to PPT Converter

This tool converts PDF files and images into editable PowerPoint presentations (`.pptx`) by leveraging structured data from the [MinerU PDF Extractor](https://mineru.net/OpenSourceTools/Extractor). It accurately reconstructs text, images, and layout, providing a high-fidelity, editable version of the original document.

The application features a user-friendly graphical interface (GUI) and is designed for easy use.

![GUI Screenshot](img/gui.png)

## For Users: How to Use

As a user, you only need the standalone executable file (e.g., `MinerU2PPT.exe`). You do not need to install Python or any libraries.

1.  **Download the Application**: Get the latest executable from the project's [Releases page](https://github.com/YOUR_USERNAME/YOUR_REPO/releases).

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

## For Developers

This section provides instructions for running the application from source and packaging it for distribution.

### Environment Setup

1.  Clone the repository.
2.  It is recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install the required dependencies from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

### Running from Source

-   **To run the GUI application**:
    ```bash
    python gui.py
    ```
-   **To use the CLI**:
    ```bash
    python main.py --json <path_to_json> --pdf <path_to_pdf> --output <path_to_ppt> [OPTIONS]
    ```

### Packaging as a Standalone Executable (.exe)

You can package the GUI application into a single `.exe` file for easy distribution.

1.  **Install PyInstaller**:
    ```bash
    pip install pyinstaller
    ```

2.  **Build the Executable**:
    Run the `pyinstaller` command from the project's root directory. Use the `--name` flag to specify a professional name for your application.
    -   `--windowed`: Prevents a console window from appearing in the background.
    -   `--onefile`: Bundles everything into a single executable file.
    -   `--name`: Sets the name of the final executable.

    ```bash
    pyinstaller --windowed --onefile --name MinerU2PPT gui.py
    ```

3.  **Find the Executable**:
    Your standalone application, `MinerU2PPT.exe`, will be located in the `dist` folder.
