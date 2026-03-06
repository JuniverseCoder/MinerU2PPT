import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from converter.ocr_merge import PaddleOCREngine


class TestCase1OCRBBoxAlignment(unittest.TestCase):
    def _iou(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        if x1 >= x2 or y1 >= y2:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter)

    def test_case1_title_bbox_is_aligned_with_json(self):
        repo_root = Path(__file__).resolve().parents[2]
        input_png = repo_root / "demo" / "case1" / "PixPin_2026-03-05_21-52-43.png"
        input_json = repo_root / "demo" / "case1" / "MinerU_PixPin_2026-03-05_21-52-43__20260305135318.json"

        self.assertTrue(input_png.exists(), f"Missing demo image: {input_png}")
        self.assertTrue(input_json.exists(), f"Missing demo json: {input_json}")

        data = json.loads(input_json.read_text(encoding="utf-8"))
        para_blocks = data["pdf_info"][0]["para_blocks"]
        target = next(
            block
            for block in para_blocks
            if block.get("lines")
            and block["lines"][0].get("spans")
            and block["lines"][0]["spans"][0].get("content") == "基于AI的声明式数据开发新范式"
        )
        target_bbox = target["bbox"]

        page_image = np.array(Image.open(input_png).convert("RGB"))
        ocr_engine = PaddleOCREngine()
        ocr_elements = ocr_engine.extract_text_elements(
            page_image,
            page_image.shape[1],
            page_image.shape[0],
        )

        matched = []
        for elem in ocr_elements:
            text = "".join(
                span.get("content", "")
                for line in elem.get("lines", [])
                for span in line.get("spans", [])
            )
            if text == "基于AI的声明式数据开发新范式":
                matched.append(elem["bbox"])

        self.assertTrue(matched, "OCR did not extract target title text")

        best_iou = max(self._iou(target_bbox, bbox) for bbox in matched)
        self.assertGreaterEqual(
            best_iou,
            0.87,
            f"OCR bbox not aligned enough. target={target_bbox}, matched={matched}, best_iou={best_iou}",
        )


if __name__ == "__main__":
    unittest.main()
