import unittest

import numpy as np

from converter.ocr_merge import refine_ocr_text_elements


def _make_test_image(height=120, width=220):
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    # draw a dark text-like horizontal band (with thickness) around y=52..64
    image[52:64, 40:180] = (20, 20, 20)
    return image


class TestOCRBBoxRefine(unittest.TestCase):
    def test_first_wave_inner_trim_removes_extra_top_bottom(self):
        image = _make_test_image()
        # bbox intentionally too tall (contains lots of blank space)
        ocr_elements = [
            {
                "type": "text",
                "bbox": [40, 30, 180, 90],
                "lines": [{"bbox": [40, 30, 180, 90], "spans": [{"bbox": [40, 30, 180, 90], "content": "hello", "type": "text"}]}],
                "source": "ocr",
            }
        ]

        refined = refine_ocr_text_elements(ocr_elements, image, json_w=220, json_h=120)
        refined_bbox = refined[0]["bbox"]

        # Should trim toward the actual dark band region
        self.assertLess(refined_bbox[1], 56)
        self.assertGreater(refined_bbox[1], 40)
        self.assertGreater(refined_bbox[3], 60)
        self.assertLess(refined_bbox[3], 80)

    def test_second_wave_extend_can_recover_bottom_when_bbox_too_small(self):
        image = _make_test_image()
        # bbox misses lower part of band (bottom too high)
        ocr_elements = [
            {
                "type": "text",
                "bbox": [40, 52, 180, 58],
                "lines": [{"bbox": [40, 52, 180, 58], "spans": [{"bbox": [40, 52, 180, 58], "content": "hello", "type": "text"}]}],
                "source": "ocr",
            }
        ]

        refined = refine_ocr_text_elements(ocr_elements, image, json_w=220, json_h=120)
        refined_bbox = refined[0]["bbox"]

        self.assertGreater(refined_bbox[3], 58)

    def test_refine_keeps_debug_bboxes(self):
        image = _make_test_image()
        ocr_elements = [
            {
                "type": "text",
                "bbox": [40, 30, 180, 90],
                "lines": [{"bbox": [40, 30, 180, 90], "spans": [{"bbox": [40, 30, 180, 90], "content": "hello", "type": "text"}]}],
                "source": "ocr",
            }
        ]

        refined = refine_ocr_text_elements(ocr_elements, image, json_w=220, json_h=120)
        elem = refined[0]

        self.assertIn("ocr_bbox_original", elem)
        self.assertIn("ocr_bbox_pad", elem)
        self.assertIn("ocr_bbox_refined", elem)


if __name__ == "__main__":
    unittest.main()
