import unittest

from converter.ir_merge import merge_ir_elements


class TestIRMerge(unittest.TestCase):
    def test_group_id_replaces_text_first(self):
        base = [
            {
                "type": "text",
                "bbox": [10, 10, 40, 30],
                "text": "mineru",
                "text_runs": [{"text": "mineru", "bbox": [10, 10, 40, 30], "line_index": 0, "style": {}}],
                "group_id": "g1",
                "order": [10, 10],
                "style": {"bold": False, "font_size": None, "align": "left"},
                "is_discarded": False,
                "source": "mineru",
            }
        ]
        overlay = [
            {
                "type": "text",
                "bbox": [10, 10, 41, 31],
                "text": "ocr",
                "text_runs": [{"text": "ocr", "bbox": [10, 10, 41, 31], "line_index": 0, "style": {}}],
                "group_id": "g1",
                "order": [10, 10],
                "style": {"bold": False, "font_size": None, "align": "left"},
                "is_discarded": False,
                "source": "ocr",
            }
        ]

        merged, stats = merge_ir_elements(base, overlay)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "ocr")
        self.assertEqual(stats["group_replaced"], 1)

    def test_overlap_replaces_text_when_group_missing(self):
        base = [
            {
                "type": "text",
                "bbox": [10, 10, 40, 30],
                "text": "old",
                "text_runs": [{"text": "old", "bbox": [10, 10, 40, 30], "line_index": 0, "style": {}}],
                "group_id": None,
                "order": [10, 10],
                "style": {"bold": False, "font_size": None, "align": "left"},
                "is_discarded": False,
                "source": "mineru",
            }
        ]
        overlay = [
            {
                "type": "text",
                "bbox": [20, 20, 50, 35],
                "text": "new",
                "text_runs": [{"text": "new", "bbox": [20, 20, 50, 35], "line_index": 0, "style": {}}],
                "group_id": None,
                "order": [20, 20],
                "style": {"bold": False, "font_size": None, "align": "left"},
                "is_discarded": False,
                "source": "ocr",
            }
        ]

        merged, stats = merge_ir_elements(base, overlay)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "new")
        self.assertEqual(stats["overlap_replaced"], 1)

    def test_non_overlapping_overlay_is_appended(self):
        base = [
            {
                "type": "text",
                "bbox": [10, 10, 40, 30],
                "text": "old",
                "text_runs": [{"text": "old", "bbox": [10, 10, 40, 30], "line_index": 0, "style": {}}],
                "group_id": None,
                "order": [10, 10],
                "style": {"bold": False, "font_size": None, "align": "left"},
                "is_discarded": False,
                "source": "mineru",
            }
        ]
        overlay = [
            {
                "type": "text",
                "bbox": [100, 100, 150, 120],
                "text": "new",
                "text_runs": [{"text": "new", "bbox": [100, 100, 150, 120], "line_index": 0, "style": {}}],
                "group_id": None,
                "order": [100, 100],
                "style": {"bold": False, "font_size": None, "align": "left"},
                "is_discarded": False,
                "source": "ocr",
            }
        ]

        merged, stats = merge_ir_elements(base, overlay)
        self.assertEqual(len(merged), 2)
        self.assertEqual(stats["overlay_added"], 1)

    def test_splits_from_same_base_are_combined(self):
        base = [
            {
                "type": "text",
                "bbox": [74, 265, 731, 312],
                "text": "Design as Code, Asset as Service",
                "text_runs": [
                    {
                        "text": "Design as Code, Asset as Service",
                        "bbox": [74, 265, 731, 312],
                        "line_index": 0,
                        "style": {},
                    }
                ],
                "group_id": None,
                "order": [265, 74],
                "style": {"bold": False, "font_size": None, "align": "left"},
                "is_discarded": False,
                "source": "mineru",
            }
        ]
        overlay = [
            {
                "type": "text",
                "bbox": [74, 270, 400, 313],
                "lines": [{"bbox": [74, 270, 400, 313], "spans": [{"bbox": [74, 270, 400, 313], "content": "Design as Code,", "type": "text"}]}],
                "text": "Design as Code,",
                "text_runs": [
                    {"bbox": [74, 270, 400, 313], "text": "Design as Code,", "line_index": 0, "style": {}}
                ],
                "group_id": None,
                "order": [270, 74],
                "style": {"bold": False, "font_size": None, "align": "left"},
                "is_discarded": False,
                "source": "ocr",
            },
            {
                "type": "text",
                "bbox": [408, 271, 730, 304],
                "lines": [{"bbox": [408, 271, 730, 304], "spans": [{"bbox": [408, 271, 730, 304], "content": "Asset as Service", "type": "text"}]}],
                "text": "Asset as Service",
                "text_runs": [
                    {"bbox": [408, 271, 730, 304], "text": "Asset as Service", "line_index": 0, "style": {}}
                ],
                "group_id": None,
                "order": [271, 408],
                "style": {"bold": False, "font_size": None, "align": "left"},
                "is_discarded": False,
                "source": "ocr",
            },
        ]

        merged, stats = merge_ir_elements(base, overlay)
        self.assertEqual(len(merged), 1)
        self.assertIn("Design as Code,", merged[0]["text"])
        self.assertIn("Asset as Service", merged[0]["text"])
        self.assertIsInstance(merged[0]["text_runs"], list)
        self.assertEqual(
            "".join(run["text"] for run in merged[0]["text_runs"]),
            "Design as Code,Asset as Service",
        )
        self.assertGreaterEqual(stats["overlay_fragment_groups"], 1)


if __name__ == "__main__":
    unittest.main()
