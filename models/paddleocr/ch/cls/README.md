---
license: Apache License 2.0
library_name: PaddleOCR
language:
- English
- Chinese
pipeline_tag: image-to-text
tags:
- OCR
- PaddlePaddle
- PaddleOCR
---

# PP-LCNet_x1_0_textline_ori

## Introduction

The text line orientation classification module primarily distinguishes the orientation of text lines and corrects them using post-processing. In processes such as document scanning and license/certificate photography, to capture clearer images, the capture device may be rotated, resulting in text lines in various orientations. Standard OCR pipelines cannot handle such data well. By utilizing image classification technology, the orientation of text lines can be predetermined and adjusted, thereby enhancing the accuracy of OCR processing. The key accuracy metrics are as follow:

<table>
<tr>
<th>Model</th>
<th>Recognition Avg Accuracy(%)</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<tr>
<td>PP-LCNet_x1_0_textline_ori</td>
<td>98.85</td>
<td>0.96</td>
<td>Text line classification model based on PP-LCNet_x0_25, with two classes: 0 degrees and 180 degrees</td>
</tr>
</table>

## Quick Start

### Installation

1. PaddlePaddle

Please refer to the following commands to install PaddlePaddle using pip:

```bash
# for CUDA11.8
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# for CUDA12.6
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# for CPU
python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

For details about PaddlePaddle installation, please refer to the [PaddlePaddle official website](https://www.paddlepaddle.org.cn/en/install/quick).

2. PaddleOCR

Install the latest version of the PaddleOCR inference package from PyPI:

```bash
python -m pip install paddleocr
```

### Model Usage

You can quickly experience the functionality with a single command:

```bash
paddleocr text_line_orientation_classification \
    --model_name PP-LCNet_x1_0_textline_ori \
    -i https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/m3ZmUPAnst1f9xXvTVLKS.png
```

You can also integrate the model inference of the text recognition module into your project. Before running the following code, please download the sample image to your local machine.

```python
from paddleocr import TextLineOrientationClassificatio
model = TextLineOrientationClassification(model_name="PP-LCNet_x1_0_textline_ori")
output = model.predict(input="m3ZmUPAnst1f9xXvTVLKS.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

After running, the obtained result is as follows:

```json
{'res': {'input_path': '/root/.paddlex/predict_input/m3ZmUPAnst1f9xXvTVLKS.png', 'page_index': None, 'class_ids': array([1], dtype=int32), 'scores': array([0.99829], dtype=float32), 'label_names': ['180_degree']}}
```

The visualized image is as follows:

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/0y5rEbMTzgsqP6Ptnj-Er.png)

For details about usage command and descriptions of parameters, please refer to the [Document](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/module_usage/text_recognition.html#iii-quick-start).


## Links

[PaddleOCR Repo](https://github.com/paddlepaddle/paddleocr)

[PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html)
