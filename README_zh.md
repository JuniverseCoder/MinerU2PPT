# MinerU 转 PPT 转换器

本工具利用 [MinerU PDF 提取器](https://mineru.net/OpenSourceTools/Extractor) 生成的结构化数据，将 PDF 文件和图片文件转换为可编辑的 PowerPoint 演示文稿（`.pptx`）。它能准确地重建文本、图片和布局，提供一个高保真的、可编辑的原始文档副本。

本应用带有一个为用户设计的图形界面（GUI），简单易用。

![GUI Screenshot](img/gui.png)

## 用户指南：如何使用

作为普通用户，您只需要下载发布包（CPU 或 GPU 版本）。您无需安装 Python 或任何编程库。

1.  **下载应用程序**: 从本项目的 [Releases 页面](https://github.com/YOUR_USERNAME/YOUR_REPO/releases) 下载最新发布包。
    -   `MinerU2PPT-win64-cpu-setup.exe`：CPU 版本（默认推荐）。
    -   `MinerU2PPT-win64-gpu-cu118-setup.exe`：CUDA 11.8 GPU 版本。
    -   `MinerU2PPT-win64-gpu-cu126-setup.exe`：CUDA 12.6 GPU 版本。
    -   `MinerU2PPT-win64-gpu-cu129-setup.exe`：CUDA 12.9 GPU 版本。

2.  **获取 MinerU JSON 文件**:
    -   访问 [MinerU PDF/图片提取器](https://mineru.net/OpenSourceTools/Extractor)。
    -   上传您的 PDF 或图片文件，等待其处理完成。
    -   下载生成的 JSON 文件。该文件包含了您的文档结构信息，是本工具进行转换所必需的。
    ![下载 JSON](img/download_json.png)

3.  **运行转换器**:
    -   双击 `.exe` 文件以启动应用程序。
    -   **选择输入文件**: 将您的 PDF 或图片文件拖拽到第一个输入框中，或使用“浏览...”按钮选择。
    -   **选择 JSON 文件**: 将您从 MinerU 下载的 JSON 文件拖拽到第二个输入框中。
    -   **输出路径**: 您的新 PowerPoint 文件的输出路径将被自动填充。您也可以直接输入路径，或使用“另存为...”按钮来更改。
    -   **选项**:
        -   **移除水印**: 勾选此项可自动擦除被标记为水印语义的元素（例如页脚、页码等）。
        -   **生成调试图片**: 除非您需要排查问题，否则请勿勾选此项。
    -   点击 **开始转换**。

4.  **打开您的文件**: 转换完成后，点击“打开输出文件夹”按钮，即可找到您新生成的 `.pptx` 文件。

### 使用批量模式

该应用还支持批量模式，可以一次性转换多个文件。

1.  **切换到批量模式**: 点击应用右上角的“批量模式”按钮。界面将切换到批量处理视图。
2.  **添加任务**:
    -   点击“添加任务”按钮，会弹出一个新窗口。
    -   在弹出窗口中，通过拖拽或浏览选择**输入文件**、对应的 **MinerU JSON 文件**，并指定**输出路径**。
    -   为此特定任务设置**移除水印**选项。
    -   点击“确定”将任务添加到列表中。
3.  **管理任务**: 您可以向列表中添加多个任务。如果需要删除任务，请从列表中选中它，然后点击“删除任务”。
4.  **开始批量转换**: 添加完所有任务后，点击“开始批量转换”。应用将按顺序处理每个任务。日志区域会显示每个文件的处理进度。

## 开发者指南

本部分为需要从源代码运行或打包分发本应用的开发者提供说明。

### 环境配置

1.  克隆本仓库。
2.  建议使用虚拟环境：
    ```bash
    python -m venv venv
    source venv/bin/activate  # 在 Windows 上: venv\Scripts\activate
    ```
3.  按开发目标安装依赖。
    -   **默认（GPU CUDA 11.8 全量开发）**：
        ```bash
        pip install -r requirements.txt
        ```
    -   **GPU CUDA 12.6 全量开发**：
        ```bash
        pip install -r requirements-gpu-cu126.txt -r requirements-build.txt
        ```
    -   **GPU CUDA 12.9 全量开发**：
        ```bash
        pip install -r requirements-gpu-cu129.txt -r requirements-build.txt
        ```
    -   **CPU 全量开发（CI/其他协作者）**：
        ```bash
        pip install -r requirements-dev-cpu.txt
        ```

### 从源代码运行

-   **运行 GUI 应用程序**:
    ```bash
    python gui.py
    ```
-   **使用命令行界面 (CLI)**:
    ```bash
    python main.py --json <json文件路径> --input <pdf或图片路径> --output <ppt输出路径> [OPTIONS]
    ```

#### OCR 相关 CLI 参数

-   `--ocr-device {auto,gpu,cpu}`：OCR 设备策略，默认 `auto`（`gpu -> cpu` 回退）。
-   `--ocr-model-root <path>`：可选，本地 PaddleOCR 模型根目录。未指定时会自动下载模型。
-   `--ocr-model-variant {auto,lite,server}`：OCR 模型版本，默认 `auto`（有 GPU 用 server，无 GPU 用 lite/mobile）。
-   `--ocr-font-distance-threshold <float>`：字体敏感度，用于 OCR bbox 细化（默认 `60.0`）。数值越大，文本框越大。

示例：
```bash
python main.py --json "demo/case1/MinerU_xxx.json" --input "demo/case1/PixPin_xxx.png" --output "out.pptx" --ocr-device auto --ocr-font-distance-threshold 60
```

### 回归：为所有 Demo Case 生成 PPT

如果希望回归流程同时产出 PPT 便于直接人工查看，可执行：

```bash
python -m pytest "tests/integration/test_case1_ocr.py" -k all_demo_cases_generate_ppt_outputs_for_manual_review
```

生成的 PPT 位于：

- `tmp/regression_ppt_outputs/case1.pptx`
- `tmp/regression_ppt_outputs/case2.pptx`
- `tmp/regression_ppt_outputs/case3.pptx`
- `tmp/regression_ppt_outputs/case4.pptx`
- `tmp/regression_ppt_outputs/case5.pptx`

### 打包为 Windows 独立应用

本项目现推荐使用 **onedir/安装包风格**，而不是 onefile，稳定性更好，也更适合离线模型分发。

1.  **安装 PyInstaller**:
    ```bash
    pip install pyinstaller
    ```

2.  **OCR 模型（默认自动下载）**:
    默认会在首次运行时自动下载模型。如需使用本地模型，可传 `--ocr-model-root` 或设置 `MINERU_OCR_MODEL_ROOT`。

    可选本地目录结构（仅在提供时需要）：
    ```
    models/paddleocr/<variant>/<lang>/det
    models/paddleocr/<variant>/<lang>/rec
    models/paddleocr/<variant>/<lang>/cls  # 仅在启用方向分类时需要
    ```
    其中 `<variant>` 为 `lite` 或 `server`。

3.  **构建 onedir 包**:
    ```bash
    pyinstaller --clean gui.spec
    ```

4.  **查看构建产物**:
    产物目录位于 `dist/MinerU2PPT/`。

## 文档导航

- 文档域：
  - `docs/architecture/`
  - `docs/testing/`
  - `docs/core-flow/`
  - `docs/api/`
- 核心流程文档：
  - `docs/core-flow/font-size-normalization-pre-render.md`
  - `docs/core-flow/ocr-bbox-xy-refine-flow.md`
  - `docs/core-flow/watermark-ir-removal-flow.md`
- 架构文档：
  - `docs/architecture/ocr-engine-configuration.md`
- 测试文档：
  - `docs/testing/font-size-normalization-testing.md`
  - `docs/testing/ocr-bbox-refine-testing.md`
  - `docs/testing/ocr-configuration-testing.md`
  - `docs/testing/watermark-ir-removal-testing.md`
