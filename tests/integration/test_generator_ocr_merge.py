import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from converter.generator import PPTGenerator, convert_mineru_to_ppt
from converter.ir_merge import merge_ir_elements
from converter.ocr_merge import PaddleOCREngine


class FakeOCREngine:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail

    def extract_text_elements(self, page_image, json_w, json_h):
        if self.should_fail:
            raise RuntimeError("ocr init failed")
        return [
            {
                "type": "text",
                "bbox": [60, 60, 100, 100],
                "index": 1,
                "lines": [
                    {
                        "bbox": [60, 60, 100, 100],
                        "spans": [{"bbox": [60, 60, 100, 100], "content": "ocr-text", "type": "text"}],
                    }
                ],
                "is_discarded": False,
            }
        ]


class TestGeneratorOCRMerge(unittest.TestCase):
    def test_process_page_with_forced_ocr_does_not_raise(self):
        generator = PPTGenerator("out.pptx", ocr_engine=FakeOCREngine())
        slide = generator.add_slide()
        page_image = np.zeros((200, 200, 3), dtype=np.uint8)

        elements = [
            {
                "type": "text",
                "bbox": [10, 10, 40, 40],
                "text": "mineru",
                "is_discarded": False,
            }
        ]

        generator.process_page(slide, elements, page_image, page_size=(200, 200), page_index=0, debug_images=False)

    def test_convert_function_ocr_failure_raises(self):
        fake_json_data = [{"para_blocks": [], "images": [], "tables": [], "discarded_blocks": []}]

        with (
            mock.patch("converter.generator.open", mock.mock_open(read_data="[]")),
            mock.patch("converter.generator.json.load", return_value=fake_json_data),
            mock.patch("converter.generator.Image.open") as mocked_image_open,
            mock.patch("converter.generator.np.array", return_value=np.zeros((600, 600, 3), dtype=np.uint8)),
            mock.patch("converter.generator.PPTGenerator") as mocked_generator_cls,
        ):
            rgb_image = mock.Mock()
            rgb_image.mode = "RGB"
            rgb_image.convert.return_value = rgb_image
            mocked_image_open.return_value = rgb_image

            generator_instance = mocked_generator_cls.return_value
            generator_instance.add_slide.return_value = object()

            with self.assertRaises(RuntimeError):
                convert_mineru_to_ppt(
                    "in.json",
                    "in.png",
                    "out.pptx",
                    ocr_engine=FakeOCREngine(should_fail=True),
                )

    def test_convert_constructs_ocr_engine_with_forwarded_config(self):
        fake_json_data = [{"para_blocks": [], "images": [], "tables": [], "discarded_blocks": []}]

        class _Engine:
            def extract_text_elements(self, *_args, **_kwargs):
                return []

        with (
            mock.patch("converter.generator.open", mock.mock_open(read_data="[]")),
            mock.patch("converter.generator.json.load", return_value=fake_json_data),
            mock.patch("converter.generator.Image.open") as mocked_image_open,
            mock.patch("converter.generator.np.array", return_value=np.zeros((600, 600, 3), dtype=np.uint8)),
            mock.patch("converter.generator.PaddleOCREngine", return_value=_Engine()) as mocked_engine,
            mock.patch("converter.generator.PPTGenerator") as mocked_generator_cls,
        ):
            rgb_image = mock.Mock()
            rgb_image.mode = "RGB"
            rgb_image.convert.return_value = rgb_image
            mocked_image_open.return_value = rgb_image

            generator_instance = mocked_generator_cls.return_value
            generator_instance.add_slide.return_value = object()

            convert_mineru_to_ppt(
                "in.json",
                "in.png",
                "out.pptx",
                ocr_device_policy="cpu",
                ocr_model_root="models/paddleocr",
                ocr_offline_only=True,
            )

        mocked_engine.assert_called_once_with(
            device_policy="cpu",
            model_root="models/paddleocr",
            offline_only=True,
        )

    def test_convert_function_forwards_ocr_config_to_generator(self):
        fake_json_data = [{"para_blocks": [], "images": [], "tables": [], "discarded_blocks": []}]

        class _Engine:
            def extract_text_elements(self, *_args, **_kwargs):
                return []

        with (
            mock.patch("converter.generator.open", mock.mock_open(read_data="[]")),
            mock.patch("converter.generator.json.load", return_value=fake_json_data),
            mock.patch("converter.generator.Image.open") as mocked_image_open,
            mock.patch("converter.generator.PPTGenerator") as mocked_generator_cls,
            mock.patch("converter.generator.np.array", return_value=np.zeros((600, 600, 3), dtype=np.uint8)),
            mock.patch("converter.generator.PaddleOCREngine", return_value=_Engine()),
        ):
            rgb_image = mock.Mock()
            rgb_image.mode = "RGB"
            rgb_image.convert.return_value = rgb_image
            mocked_image_open.return_value = rgb_image

            generator_instance = mocked_generator_cls.return_value
            generator_instance.add_slide.return_value = object()

            convert_mineru_to_ppt(
                "in.json",
                "in.png",
                "out.pptx",
                ocr_device_policy="gpu",
                ocr_model_root="x/models/paddleocr",
                ocr_offline_only=True,
            )

        self.assertEqual(mocked_generator_cls.call_count, 1)
        kwargs = mocked_generator_cls.call_args.kwargs
        self.assertEqual(kwargs["remove_watermark"], True)
        self.assertEqual(kwargs["ocr_device_policy"], "gpu")
        self.assertEqual(kwargs["ocr_model_root"], "x/models/paddleocr")
        self.assertEqual(kwargs["ocr_offline_only"], True)
        self.assertIsNotNone(kwargs["ocr_engine"])


    def test_case2_ir_merge_outputs_text_elements(self):
        repo_root = Path(__file__).resolve().parents[2]
        input_png = repo_root / "demo" / "case2" / "PixPin_2026-03-05_22-01-24.png"
        input_json = repo_root / "demo" / "case2" / "MinerU_PixPin_2026-03-05_22-01-24__20260305140239.json"

        self.assertTrue(input_png.exists(), f"Missing demo image: {input_png}")
        self.assertTrue(input_json.exists(), f"Missing demo json: {input_json}")

        data = json.loads(input_json.read_text(encoding="utf-8"))
        page_data = data["pdf_info"][0]

        elements = []
        for item in page_data.get("para_blocks", []):
            if item:
                elements.append(item)
        for item in page_data.get("discarded_blocks", []):
            if item:
                item["is_discarded"] = True
                elements.append(item)

        page_image = np.array(Image.open(input_png).convert("RGB"))
        ocr_engine = PaddleOCREngine()
        ocr_elements = ocr_engine.extract_text_elements(
            page_image,
            page_image.shape[1],
            page_image.shape[0],
        )

        def _has_overlap(a, b):
            x1 = max(a[0], b[0])
            y1 = max(a[1], b[1])
            x2 = min(a[2], b[2])
            y2 = min(a[3], b[3])
            return x1 < x2 and y1 < y2

        # simulate adapter output contract for this integration case
        mineru_ir = []
        for elem in elements:
            if not elem.get("bbox"):
                continue
            lines = elem.get("lines", [])
            text_from_lines = "\n".join(
                "".join(span.get("content", "") for span in line.get("spans", []))
                for line in lines
                if line.get("spans")
            )
            text_value = text_from_lines or elem.get("text", "")
            if not text_value and not lines:
                continue
            mineru_ir.append(
                {
                    "type": "text",
                    "bbox": elem.get("bbox"),
                    "lines": lines,
                    "text": text_value,
                    "group_id": elem.get("group_id"),
                    "order": [elem.get("index", 0), 0],
                    "style": {"bold": False, "font_size": None, "align": "left"},
                    "is_discarded": elem.get("is_discarded", False),
                    "source": "mineru",
                }
            )

        ocr_ir = [
            {
                "type": "text",
                "bbox": elem.get("bbox"),
                "lines": elem.get("lines", []),
                "text": "\n".join(
                    "".join(span.get("content", "") for span in line.get("spans", []))
                    for line in elem.get("lines", [])
                    if line.get("spans")
                ),
                "text_runs": [
                    {
                        "text": span.get("content", ""),
                        "bbox": span.get("bbox") or line.get("bbox") or elem.get("bbox"),
                        "line_index": line_index,
                        "style": {},
                    }
                    for line_index, line in enumerate(elem.get("lines", []))
                    for span in line.get("spans", [])
                    if span.get("content", "")
                ],
                "group_id": elem.get("group_id"),
                "order": [elem.get("index", 0), 0],
                "style": {"bold": False, "font_size": None, "align": "left"},
                "is_discarded": False,
                "source": "ocr",
            }
            for elem in ocr_elements
            if elem.get("bbox")
        ]

        merged, _stats = merge_ir_elements(mineru_ir, ocr_ir, _has_overlap)

        merged_texts = [
            str(elem.get("text", ""))
            for elem in merged
            if elem.get("type") == "text"
        ]

        self.assertTrue(any("构建" in text and "流水线" in text for text in merged_texts))
        self.assertTrue(any("形成统一规范的数据开发标准" in text for text in merged_texts))

        target_phrase_elements = [
            elem
            for elem in merged
            if elem.get("type") == "text"
            and "设计即开发" in str(elem.get("text", ""))
            and "十倍提效" in str(elem.get("text", ""))
        ]
        self.assertEqual(
            len(target_phrase_elements),
            1,
            "Expected sentence '以“设计即开发”实现开发侧十倍提效。' to stay in a single TextIR",
        )

if __name__ == "__main__":
    unittest.main()
