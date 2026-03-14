import unittest

from converter.generator import PPTGenerator
from converter.ir import ImageIR, TextIR, TextRunIR


class TestGeneratorFontNormalization(unittest.TestCase):
    def _text_with_run(self, text: str, size: float, bold: bool, order_x: float) -> TextIR:
        return TextIR(
            type="text",
            bbox=[order_x, 0, order_x + 40, 20],
            text=text,
            source="ocr",
            order=[0, order_x],
            style={"bold": bold, "font_size": None, "align": "left"},
            text_runs=[
                TextRunIR(
                    text=text,
                    bbox=[order_x, 0, order_x + 40, 20],
                    line_index=0,
                    style={"bold": bold, "font_size": float(size), "align": "left"},
                )
            ],
        )

    def test_normalize_groups_by_bold_and_replaces_with_group_median(self):
        generator = PPTGenerator("out.pptx")

        elements = [
            self._text_with_run("b1", 20, True, 0),
            self._text_with_run("b2", 22, True, 50),
            self._text_with_run("b3", 22, True, 100),
            self._text_with_run("n1", 10, False, 150),
            self._text_with_run("n2", 11, False, 200),
            self._text_with_run("n3", 11, False, 250),
        ]

        normalized = generator._normalize_page_text_font_sizes(elements)

        bold_sizes = []
        normal_sizes = []
        for elem in normalized:
            run_size = elem.text_runs[0].style.get("font_size")
            if elem.style.get("bold"):
                bold_sizes.append(run_size)
            else:
                normal_sizes.append(run_size)

        self.assertEqual(bold_sizes, [22.0, 22.0, 22.0])
        self.assertEqual(normal_sizes, [11.0, 11.0, 11.0])

    def test_normalize_merges_group_when_ratio_to_center_within_1_3_threshold(self):
        generator = PPTGenerator("out.pptx")

        elements = [
            self._text_with_run("a", 20.0, False, 0),
            self._text_with_run("b", 22.0, False, 50),
            self._text_with_run("c", 24.0, False, 100),
        ]

        normalized = generator._normalize_page_text_font_sizes(elements)
        sizes = [elem.text_runs[0].style.get("font_size") for elem in normalized]

        self.assertEqual(sizes[0], 22.0)
        self.assertEqual(sizes[1], 22.0)
        self.assertEqual(sizes[2], 22.0)

    def test_center_distance_clustering_keeps_far_sample_outside_cluster(self):
        generator = PPTGenerator("out.pptx")

        elements = [
            self._text_with_run("a", 20.0, False, 0),
            self._text_with_run("b", 24.0, False, 50),
            self._text_with_run("c", 32.0, False, 100),
        ]

        normalized = generator._normalize_page_text_font_sizes(elements)
        sizes = [elem.text_runs[0].style.get("font_size") for elem in normalized]

        self.assertEqual(sizes[0], 22.0)
        self.assertEqual(sizes[1], 22.0)
        self.assertEqual(sizes[2], 32.0)

    def test_kmeans_optimization_preserves_threshold_constraints(self):
        generator = PPTGenerator("out.pptx")

        samples = [(20.0, 0), (23.0, 1), (28.0, 2), (40.0, 3)]
        seed_groups = generator._assign_groups_by_center_distance(samples, 1.3)
        optimized_groups = generator._optimize_groups_with_kmeans(samples, seed_groups, 1.3)

        self.assertTrue(generator._groups_within_center_threshold(optimized_groups, 1.3))
        flattened = sorted([item for group in optimized_groups for item in group], key=lambda item: item[1])
        self.assertEqual(flattened, sorted(samples, key=lambda item: item[1]))

    def test_normalize_writes_run_first_and_elem_fallback(self):
        generator = PPTGenerator("out.pptx")

        elem_with_runs = TextIR(
            type="text",
            bbox=[0, 0, 40, 20],
            text="runs",
            source="ocr",
            order=[0, 0],
            style={"bold": False, "font_size": None, "align": "left"},
            text_runs=[
                TextRunIR(text="r1", bbox=[0, 0, 20, 20], line_index=0, style={"bold": False, "font_size": 20.0, "align": "left"}),
                TextRunIR(text="r2", bbox=[20, 0, 40, 20], line_index=0, style={"bold": False, "font_size": 22.0, "align": "left"}),
            ],
        )
        elem_without_runs_1 = TextIR(
            type="text",
            bbox=[50, 0, 90, 20],
            text="elem1",
            source="mineru",
            order=[0, 50],
            style={"bold": False, "font_size": 20.0, "align": "left"},
            text_runs=None,
        )
        elem_without_runs_2 = TextIR(
            type="text",
            bbox=[100, 0, 140, 20],
            text="elem2",
            source="mineru",
            order=[0, 100],
            style={"bold": False, "font_size": 22.0, "align": "left"},
            text_runs=None,
        )
        image_elem = ImageIR(
            type="image",
            bbox=[0, 30, 40, 60],
            source="mineru",
            order=[30, 0],
            style={"bold": False, "font_size": None, "align": "left"},
        )

        normalized = generator._normalize_page_text_font_sizes(
            [elem_with_runs, elem_without_runs_1, elem_without_runs_2, image_elem]
        )

        updated_with_runs = normalized[0]
        updated_elem_1 = normalized[1]
        updated_elem_2 = normalized[2]

        self.assertEqual(updated_with_runs.text_runs[0].style.get("font_size"), 21.0)
        self.assertEqual(updated_with_runs.text_runs[1].style.get("font_size"), 21.0)
        self.assertEqual(updated_with_runs.style.get("font_size"), 21.0)

        self.assertEqual(updated_elem_1.style.get("font_size"), 21.0)
        self.assertEqual(updated_elem_2.style.get("font_size"), 21.0)

        self.assertIs(normalized[3], image_elem)

    def test_skip_ineligible_text_element_when_internal_runs_cannot_unify(self):
        generator = PPTGenerator("out.pptx")

        eligible_1 = self._text_with_run("ok1", 20.0, False, 0)
        eligible_2 = self._text_with_run("ok2", 22.0, False, 50)
        ineligible = TextIR(
            type="text",
            bbox=[100, 0, 160, 20],
            text="bad",
            source="ocr",
            order=[0, 100],
            style={"bold": False, "font_size": None, "align": "left"},
            text_runs=[
                TextRunIR(text="b1", bbox=[100, 0, 130, 20], line_index=0, style={"bold": False, "font_size": 12.0, "align": "left"}),
                TextRunIR(text="b2", bbox=[130, 0, 160, 20], line_index=0, style={"bold": False, "font_size": 20.0, "align": "left"}),
            ],
        )

        normalized = generator._normalize_page_text_font_sizes([eligible_1, eligible_2, ineligible])

        self.assertEqual(normalized[0].text_runs[0].style.get("font_size"), 21.0)
        self.assertEqual(normalized[1].text_runs[0].style.get("font_size"), 21.0)

        self.assertEqual(normalized[2].text_runs[0].style.get("font_size"), 12.0)
        self.assertEqual(normalized[2].text_runs[1].style.get("font_size"), 20.0)
        self.assertIsNone(normalized[2].style.get("font_size"))


if __name__ == "__main__":
    unittest.main()
