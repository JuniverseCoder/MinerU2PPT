import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from converter.ocr_merge import PaddleOCREngine
from converter.utils import extract_background_color, extract_font_color


def _font_ratio_for_row(page_image, bbox, y):
    x1 = max(0, int(round(bbox[0])))
    x2 = min(page_image.shape[1], int(round(bbox[2])))
    y = min(max(0, int(y)), page_image.shape[0] - 1)

    row = page_image[y, x1:x2]
    if row.shape[0] == 0:
        return 0.0

    px_bbox = [
        max(0, int(round(bbox[0]))),
        max(0, int(round(bbox[1]))),
        min(page_image.shape[1], int(round(bbox[2]))),
        min(page_image.shape[0], int(round(bbox[3]))),
    ]
    bg = extract_background_color(page_image, px_bbox)
    font_color, _, _ = extract_font_color(page_image, px_bbox, bg)

    diff = np.linalg.norm(row.astype(np.float32) - np.array(font_color, dtype=np.float32), axis=1)
    return float(np.sum(diff < 60.0)) / float(len(diff))


class TestCase3OCRBBoxDevTrackPadding(unittest.TestCase):
    def test_case3_devtrack_vertical_padding_rows_are_tight(self):
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

        # In current OCR output this footer line is recognized as "开发资产NotebookLM"
        # and corresponds to the user's "开发侧（DevTrack）" target area.
        target_text = "开发资产NotebookLM"
        matched_bbox = None
        for elem in ocr_elements:
            text = "".join(
                span.get("content", "")
                for line in elem.get("lines", [])
                for span in line.get("spans", [])
            )
            if text == target_text:
                matched_bbox = elem.get("bbox")
                break

        self.assertIsNotNone(matched_bbox, f"Target text not found by OCR on case3: {target_text}")

        y1 = int(round(matched_bbox[1]))
        y2 = int(round(matched_bbox[3]))

        top_first_ratio = _font_ratio_for_row(page_image, matched_bbox, y1)
        top_second_ratio = _font_ratio_for_row(page_image, matched_bbox, y1 + 1)
        top_third_ratio = _font_ratio_for_row(page_image, matched_bbox, y1 + 2)
        top_fourth_ratio = _font_ratio_for_row(page_image, matched_bbox, y1 + 3)
        bottom_second_last_ratio = _font_ratio_for_row(page_image, matched_bbox, y2 - 1)
        bottom_last_ratio = _font_ratio_for_row(page_image, matched_bbox, y2)

        # Expected boundary pattern with anti-alias tolerance:
        # - first row should be non-font
        # - nearby inner rows should contain font pixels (within 3 rows)
        # - second-last row should contain font pixels
        # - last row should be non-font
        self.assertLess(
            top_first_ratio,
            0.08,
            f"top first row should be sparse, first={top_first_ratio}, bbox={matched_bbox}",
        )
        self.assertGreater(
            max(top_second_ratio, top_third_ratio, top_fourth_ratio),
            0.02,
            f"top boundary should quickly enter font rows, second={top_second_ratio}, third={top_third_ratio}, fourth={top_fourth_ratio}, bbox={matched_bbox}",
        )
        self.assertGreater(bottom_second_last_ratio, 0.02, f"bottom second-last row should contain font, ratio={bottom_second_last_ratio}, bbox={matched_bbox}")
        self.assertLess(bottom_last_ratio, 0.01, f"bottom last row should be non-font, ratio={bottom_last_ratio}, bbox={matched_bbox}")


if __name__ == "__main__":
    unittest.main()
