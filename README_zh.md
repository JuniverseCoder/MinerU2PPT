# MinerU 转 PPT 转换器

本工具利用 [MinerU PDF 提取器](https://mineru.net/OpenSourceTools/Extractor) 生成的结构化数据，将 PDF 文件和图片文件转换为可编辑的 PowerPoint 演示文稿（`.pptx`）。它能准确地重建文本、图片和布局，提供一个高保真的、可编辑的原始文档副本。

本应用带有一个为用户设计的图形界面（GUI），简单易用。

![GUI Screenshot](img/gui.png)

## 用户指南：如何使用

作为普通用户，您只需要下载发布包（CPU 或 GPU 版本）。您无需安装 Python 或任何编程库。

1.  **下载应用程序**: 从本项目的 [Releases 页面](https://github.com/YOUR_USERNAME/YOUR_REPO/releases) 下载最新发布包。
    -   `MinerU2PPT-win64-cpu.zip`：CPU 版本（默认推荐）。
    -   `MinerU2PPT-win64-gpu-cu118.zip`：CUDA 11.8 GPU 版本。

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
        -   **移除水印**: 勾选此项可自动擦除 MinerU JSON 中标记为“丢弃”的元素，例如页脚或页码。
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
3.  从 `requirements.txt` 文件安装所需依赖。
    ```bash
    pip install -r requirements.txt
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
-   `--ocr-model-root <path>`：可选，本地 PaddleOCR 模型根目录。

示例：
```bash
python main.py --json "demo/case1/MinerU_xxx.json" --input "demo/case1/PixPin_xxx.png" --output "out.pptx" --ocr-device auto --ocr-model-root "models/paddleocr"
```

### 打包为 Windows 独立应用

本项目现推荐使用 **onedir/安装包风格**，而不是 onefile，稳定性更好，也更适合离线模型分发。

1.  **安装 PyInstaller**:
    ```bash
    pip install pyinstaller
    ```

2.  **准备本地 OCR 模型（离线优先）**:
    模型根目录解析优先级为：
    1. CLI/引擎参数 `model_root`
    2. 环境变量 `MINERU_OCR_MODEL_ROOT`
    3. 可执行目录 `models/paddleocr`
    4. 源码目录 `models/paddleocr`

    目录结构要求：
    ```
    models/paddleocr/<lang>/det
    models/paddleocr/<lang>/rec
    models/paddleocr/<lang>/cls
    ```

3.  **构建 onedir 包**:
    ```bash
    pyinstaller --clean gui.spec
    ```

4.  **查看构建产物**:
    产物目录位于 `dist/MinerU2PPT/`。
