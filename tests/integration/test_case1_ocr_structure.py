import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from converter.ocr_merge import PaddleOCREngine


class TestCase1OCRStructure(unittest.TestCase):
    def test_case1_ocr_elements_match_mineru_text_shape(self):
        repo_root = Path(__file__).resolve().parents[2]
        input_png = repo_root / "demo" / "case1" / "PixPin_2026-03-05_21-52-43.png"

        self.assertTrue(input_png.exists(), f"Missing demo image: {input_png}")

        page_image = np.array(Image.open(input_png).convert("RGB"))
        ocr_engine = PaddleOCREngine()
        ocr_elements = ocr_engine.extract_text_elements(
            page_image,
            page_image.shape[1],
            page_image.shape[0],
        )

        self.assertGreater(len(ocr_elements), 0, "OCR returned no text elements")

        required_elem_keys = {"angle", "bbox", "index", "is_discarded", "lines", "type"}
        for elem in ocr_elements:
            self.assertTrue(required_elem_keys.issubset(elem.keys()))
            self.assertEqual(elem["type"], "text")
            self.assertIsInstance(elem["bbox"], list)
            self.assertEqual(len(elem["bbox"]), 4)
            self.assertIsInstance(elem["lines"], list)
            self.assertGreater(len(elem["lines"]), 0)

            for line in elem["lines"]:
                self.assertIn("bbox", line)
                self.assertIn("spans", line)
                self.assertIsInstance(line["bbox"], list)
                self.assertEqual(len(line["bbox"]), 4)
                self.assertIsInstance(line["spans"], list)
                self.assertGreater(len(line["spans"]), 0)

                for span in line["spans"]:
                    self.assertIn("bbox", span)
                    self.assertIn("content", span)
                    self.assertIn("type", span)
                    self.assertIsInstance(span["bbox"], list)
                    self.assertEqual(len(span["bbox"]), 4)
                    self.assertEqual(span["type"], "text")
                    self.assertTrue(str(span["content"]).strip())

        extracted_texts = [
            span["content"]
            for elem in ocr_elements
            for line in elem["lines"]
            for span in line["spans"]
        ]
        self.assertIn("基于AI的声明式数据开发新范式", extracted_texts)


if __name__ == "__main__":
    unittest.main()
