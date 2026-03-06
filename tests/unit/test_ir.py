import unittest

from converter.ir import (
    normalize_element_ir,
    normalize_style,
    rebuild_text_from_runs,
    sort_elements,
    validate_ir_elements,
)


class TestIR(unittest.TestCase):
    def test_normalize_text_element_with_order_fallback(self):
        elem = {
            "type": "text",
            "bbox": [20, 10, 60, 40],
            "text": "hello",
            "source": "mineru",
        }

        normalized = normalize_element_ir(elem)
        self.assertEqual(normalized["type"], "text")
        self.assertEqual(normalized["bbox"], [20.0, 10.0, 60.0, 40.0])
        self.assertEqual(normalized["order"], [10.0, 20.0])
        self.assertEqual(normalized["style"]["align"], "left")

    def test_normalize_style_minimum_fields(self):
        style = normalize_style({"bold": 1, "font_size": "18", "align": "CENTER"})
        self.assertTrue(style["bold"])
        self.assertEqual(style["font_size"], 18.0)
        self.assertEqual(style["align"], "center")

    def test_sort_elements_uses_y_then_x_fallback(self):
        elems = [
            {"type": "text", "bbox": [100, 50, 120, 60], "text": "c", "style": {"bold": False, "font_size": None, "align": "left"}, "order": [50, 100], "is_discarded": False, "source": "x", "group_id": None},
            {"type": "text", "bbox": [10, 20, 30, 30], "text": "a", "style": {"bold": False, "font_size": None, "align": "left"}, "order": [20, 10], "is_discarded": False, "source": "x", "group_id": None},
            {"type": "text", "bbox": [40, 20, 55, 30], "text": "b", "style": {"bold": False, "font_size": None, "align": "left"}, "order": [20, 40], "is_discarded": False, "source": "x", "group_id": None},
        ]

        sorted_elems = sort_elements(elems)
        texts = [elem["text"] for elem in sorted_elems]
        self.assertEqual(texts, ["a", "b", "c"])

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            normalize_element_ir({"type": "list", "bbox": [1, 1, 2, 2], "text": "x"})

    def test_text_run_bbox_is_required(self):
        with self.assertRaises(ValueError):
            normalize_element_ir(
                {
                    "type": "text",
                    "bbox": [0, 0, 10, 10],
                    "text": "a",
                    "text_runs": [{"text": "a", "line_index": 0}],
                    "source": "ocr",
                }
            )

    def test_text_run_bbox_geometry_is_validated(self):
        with self.assertRaises(ValueError):
            normalize_element_ir(
                {
                    "type": "text",
                    "bbox": [0, 0, 10, 10],
                    "text": "a",
                    "text_runs": [{"text": "a", "bbox": [5, 5, 1, 1], "line_index": 0}],
                    "source": "ocr",
                }
            )

    def test_rebuild_text_from_runs_uses_line_index_and_geometry(self):
        runs = [
            {"text": "B", "bbox": [20, 0, 25, 10], "line_index": 0},
            {"text": "A", "bbox": [0, 0, 10, 10], "line_index": 0},
            {"text": "C", "bbox": [0, 20, 10, 30], "line_index": 1},
        ]
        rebuilt = rebuild_text_from_runs(runs)
        self.assertEqual(rebuilt, "AB\nC")

    def test_validate_ir_elements_checks_text_runs_consistency(self):
        elements = [
            {
                "type": "text",
                "bbox": [0, 0, 100, 20],
                "text": "wrong",
                "text_runs": [{"text": "ok", "bbox": [0, 0, 100, 20], "line_index": 0, "style": {}}],
                "order": [0, 0],
                "style": {"bold": False, "font_size": None, "align": "left"},
                "is_discarded": False,
                "source": "ocr",
                "group_id": None,
            }
        ]
        with self.assertRaises(ValueError):
            validate_ir_elements(elements, require_text_runs_consistency=True)


if __name__ == "__main__":
    unittest.main()
