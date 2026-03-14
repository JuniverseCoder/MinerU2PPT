import unittest

from converter.ir import TextIR, TextRunIR
from converter.ir_merge import merge_ir_elements


def _text(
    bbox,
    text,
    *,
    group_id=None,
    source="mineru",
    bold=False,
    is_watermark=False,
    order=None,
    lines=None,
):
    return TextIR(
        type="text",
        bbox=[float(v) for v in bbox],
        text=text,
        text_runs=[
            TextRunIR(
                text=text,
                bbox=[float(v) for v in bbox],
                line_index=0,
                style={"bold": bold},
            )
        ],
        group_id=group_id,
        order=[float(v) for v in (order or [bbox[1], bbox[0]])],
        style={"bold": bold, "font_size": None, "align": "left"},
        is_discarded=False,
        is_watermark=is_watermark,
        source=source,
        lines=lines,
    )


class TestIRMerge(unittest.TestCase):
    def test_group_id_replaces_text_first(self):
        base = [_text([10, 10, 40, 30], "mineru", group_id="g1", source="mineru")]
        overlay = [_text([10, 10, 41, 31], "ocr", group_id="g1", source="ocr")]

        merged, stats = merge_ir_elements(base, overlay)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "ocr")
        self.assertEqual(stats["group_replaced"], 1)

    def test_overlap_replaces_text_when_group_missing(self):
        base = [_text([10, 10, 40, 30], "old", source="mineru")]
        overlay = [_text([20, 20, 50, 35], "new", source="ocr")]

        merged, stats = merge_ir_elements(base, overlay)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "new")
        self.assertEqual(stats["overlap_replaced"], 1)

    def test_overlap_replacement_inherits_bold_from_base(self):
        base = [_text([10, 10, 40, 30], "old", source="mineru", bold=True)]
        overlay = [_text([20, 20, 50, 35], "new", source="ocr", bold=False)]

        merged, _ = merge_ir_elements(base, overlay)
        self.assertEqual(len(merged), 1)
        self.assertTrue(merged[0].style["bold"])
        self.assertTrue(merged[0].text_runs[0].style["bold"])

    def test_group_replacement_inherits_bold_from_base(self):
        base = [_text([10, 10, 40, 30], "mineru", group_id="g1", source="mineru", bold=True)]
        overlay = [_text([10, 10, 41, 31], "ocr", group_id="g1", source="ocr", bold=False)]

        merged, _ = merge_ir_elements(base, overlay)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "ocr")
        self.assertTrue(merged[0].style["bold"])
        self.assertTrue(merged[0].text_runs[0].style["bold"])

    def test_overlap_replacement_inherits_watermark_from_base(self):
        base = [_text([10, 10, 40, 30], "old", source="mineru", is_watermark=True)]
        overlay = [_text([20, 20, 50, 35], "new", source="ocr", is_watermark=False)]

        merged, _ = merge_ir_elements(base, overlay)
        self.assertEqual(len(merged), 1)
        self.assertTrue(merged[0].is_watermark)

    def test_non_overlapping_overlay_is_appended(self):
        base = [_text([10, 10, 40, 30], "old", source="mineru")]
        overlay = [_text([100, 100, 150, 120], "new", source="ocr")]

        merged, stats = merge_ir_elements(base, overlay)
        self.assertEqual(len(merged), 2)
        self.assertEqual(stats["overlay_added"], 1)

    def test_splits_from_same_base_are_combined(self):
        base = [
            _text(
                [74, 265, 731, 312],
                "Design as Code, Asset as Service",
                source="mineru",
            )
        ]
        overlay = [
            _text(
                [74, 270, 400, 313],
                "Design as Code,",
                source="ocr",
                lines=[
                    {
                        "bbox": [74, 270, 400, 313],
                        "spans": [{"bbox": [74, 270, 400, 313], "content": "Design as Code,", "type": "text"}],
                    }
                ],
            ),
            _text(
                [408, 271, 730, 304],
                "Asset as Service",
                source="ocr",
                lines=[
                    {
                        "bbox": [408, 271, 730, 304],
                        "spans": [{"bbox": [408, 271, 730, 304], "content": "Asset as Service", "type": "text"}],
                    }
                ],
            ),
        ]

        merged, stats = merge_ir_elements(base, overlay)
        self.assertEqual(len(merged), 1)
        self.assertIn("Design as Code,", merged[0].text)
        self.assertIn("Asset as Service", merged[0].text)
        self.assertIsInstance(merged[0].text_runs, list)
        self.assertEqual(
            "".join(run.text for run in merged[0].text_runs),
            "Design as Code,Asset as Service",
        )
        self.assertGreaterEqual(stats["overlay_fragment_groups"], 1)


if __name__ == "__main__":
    unittest.main()
