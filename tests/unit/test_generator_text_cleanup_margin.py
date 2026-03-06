import unittest

import numpy as np

from converter.generator import PPTGenerator


class TestGeneratorTextCleanupMargin(unittest.TestCase):
    def test_cleanup_margin_uses_single_line_height_when_lines_present(self):
        generator = PPTGenerator("out.pptx")
        context = type("Ctx", (), {})()
        context.coords = {"img_h": 2000, "json_h": 1000}

        elem = {
            "bbox": [0, 0, 100, 200],  # multi-line large bbox
            "lines": [
                {"bbox": [0, 0, 100, 20], "spans": [{"content": "a", "bbox": [0, 0, 100, 20], "type": "text"}]},
                {"bbox": [0, 30, 100, 50], "spans": [{"content": "b", "bbox": [0, 30, 100, 50], "type": "text"}]},
            ],
        }

        margin_px = generator._compute_text_cleanup_margin_px(context, elem)
        # single line json height=20, scale=2 => 40px, 5% => 2px
        self.assertEqual(margin_px, 2)

    def test_cleanup_margin_falls_back_to_element_bbox_height(self):
        generator = PPTGenerator("out.pptx")
        context = type("Ctx", (), {})()
        context.coords = {"img_h": 1000, "json_h": 1000}

        elem = {
            "bbox": [0, 0, 100, 40],
        }

        margin_px = generator._compute_text_cleanup_margin_px(context, elem)
        # 40 * 5% = 2
        self.assertEqual(margin_px, 2)

    def test_process_text_registers_margin_px_not_ratio(self):
        generator = PPTGenerator("out.pptx")

        calls = []

        class _Context:
            def __init__(self):
                self.coords = {"img_h": 1000, "json_h": 1000, "scale_y": 1.0, "img_w": 1000, "json_w": 1000}
                self.original_image = np.zeros((100, 100, 3), dtype=np.uint8)

            def add_element_bbox_for_cleanup(self, bbox, margin_px=0, margin_ratio=0.0, min_margin_px=1):
                calls.append({
                    "bbox": bbox,
                    "margin_px": margin_px,
                    "margin_ratio": margin_ratio,
                    "min_margin_px": min_margin_px,
                })

            def add_processed_element(self, _elem_type, _data):
                pass

        elem = {
            "type": "text",
            "bbox": [0, 0, 100, 40],
            "lines": [
                {"bbox": [0, 0, 100, 20], "spans": [{"content": "a", "bbox": [0, 0, 100, 20], "type": "text"}]}
            ],
            "source": "mineru",
        }

        generator._process_text(_Context(), elem)

        self.assertEqual(len(calls), 1)
        self.assertGreater(calls[0]["margin_px"], 0)
        self.assertEqual(calls[0]["margin_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
