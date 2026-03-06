import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from converter.ocr_merge import PaddleOCREngine


class TestCase3OCRBBoxBottom(unittest.TestCase):
    def test_case3_target_text_y2_row_is_non_font_color(self):
        repo_root = Path(__file__).resolve().parents[2]
        input_png = repo_root / "demo" / "case3" / "PixPin_2026-03-05_22-01-36.png"

        self.assertTrue(input_png.exists(), f"Missing demo image: {input_png}")

        page_image = np.array(Image.open(input_png).convert("RGB"))
        ocr_engine = PaddleOCREngine()
        ocr_elements = ocr_engine.extract_text_elements(
            page_image,
            page_image.shape[1],
            page_image.shape[0],
        )

        target_text = "AI辅助、效率大幅提升且质量高度稳定"
        matched = None
        for elem in ocr_elements:
            text = "".join(
                span.get("content", "")
                for line in elem.get("lines", [])
                for span in line.get("spans", [])
            )
            if text == target_text:
                matched = elem
                break

        self.assertIsNotNone(matched, "Target text not found by OCR on case3")

        bbox = matched.get("bbox")
        y2 = int(round(bbox[3]))
        y_row = min(max(y2, 0), page_image.shape[0] - 1)

        row = page_image[y_row, int(round(bbox[0])):int(round(bbox[2]))]
        self.assertGreater(row.shape[0], 0, f"Invalid row slice for bbox={bbox}")

        bg = np.median(row, axis=0)
        diff = np.linalg.norm(row.astype(np.float32) - bg.astype(np.float32), axis=1)

        # boundary row can keep small non-background ratio due to anti-aliasing and subpixel rendering
        non_bg_ratio = float(np.sum(diff > 55.0)) / float(len(diff))
        self.assertLess(
            non_bg_ratio,
            0.35,
            f"Expected boundary row to be mostly non-font-like, got non_bg_ratio={non_bg_ratio}, bbox={bbox}",
        )


if __name__ == "__main__":
    unittest.main()
