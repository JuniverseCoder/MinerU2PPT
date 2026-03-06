import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from converter.generator import PPTGenerator, convert_mineru_to_ppt
from converter.ocr_merge import PaddleOCREngine, TEXT_ELEMENT_TYPES, merge_ocr_text_elements


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
                "text": "ocr-text",
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

    def test_process_page_ocr_failure_raises(self):
        generator = PPTGenerator("out.pptx", ocr_engine=FakeOCREngine(should_fail=True))
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

        with self.assertRaises(RuntimeError):
            generator.process_page(slide, elements, page_image, page_size=(200, 200), page_index=0, debug_images=False)

    def test_generator_constructs_ocr_engine_with_forwarded_config(self):
        page_image = np.zeros((120, 120, 3), dtype=np.uint8)

        class _Engine:
            def extract_text_elements(self, *_args, **_kwargs):
                return []

        with mock.patch("converter.generator.PaddleOCREngine", return_value=_Engine()) as mocked_engine:
            generator = PPTGenerator(
                "out.pptx",
                ocr_device_policy="cpu",
                ocr_model_root="models/paddleocr",
                ocr_offline_only=True,
            )
            slide = generator.add_slide()
            generator.process_page(slide, [], page_image, page_size=(120, 120), page_index=0, debug_images=False)

        mocked_engine.assert_called_once_with(
            device_policy="cpu",
            model_root="models/paddleocr",
            offline_only=True,
        )

    def test_convert_function_forwards_ocr_config_to_generator(self):
        fake_json_data = [{"para_blocks": [], "images": [], "tables": [], "discarded_blocks": []}]

        with (
            mock.patch("builtins.open", mock.mock_open(read_data="[]")),
            mock.patch("converter.generator.json.load", return_value=fake_json_data),
            mock.patch("converter.generator.Image.open") as mocked_image_open,
            mock.patch("converter.generator.PPTGenerator") as mocked_generator_cls,
            mock.patch("converter.generator.np.array", return_value=np.zeros((60, 60, 3), dtype=np.uint8)),
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

        mocked_generator_cls.assert_called_once_with(
            "out.pptx",
            remove_watermark=True,
            ocr_engine=None,
            ocr_device_policy="gpu",
            ocr_model_root="x/models/paddleocr",
            ocr_offline_only=True,
        )


    def test_case2_list_structure_kept_after_ocr_merge(self):
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

        merged, _stats = merge_ocr_text_elements(
            elements,
            ocr_elements,
            _has_overlap,
            TEXT_ELEMENT_TYPES,
        )

        target_list = None
        for elem in merged:
            if elem.get("type") == "list" and len(elem.get("blocks") or []) == 2:
                first_text = "".join(
                    span.get("content", "")
                    for line in (elem.get("blocks") or [])[0].get("lines", [])
                    for span in line.get("spans", [])
                )
                if "构建" in first_text and "流水线" in first_text:
                    target_list = elem
                    break

        self.assertIsNotNone(target_list, "Expected case2 list block to exist after merge")
        self.assertEqual(target_list.get("type"), "list")
        self.assertEqual(len(target_list.get("blocks", [])), 2)

        block_texts = [
            "".join(
                span.get("content", "")
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            )
            for block in target_list.get("blocks", [])
        ]

        self.assertTrue(any("构建" in text and "流水线" in text for text in block_texts))
        self.assertTrue(any("形成统一规范的数据开发标准" in text for text in block_texts))

if __name__ == "__main__":
    unittest.main()
