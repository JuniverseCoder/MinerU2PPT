import unittest

from converter.ocr_merge import merge_ocr_text_elements


def _has_overlap(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return x1 < x2 and y1 < y2


class TestOCRMerge(unittest.TestCase):
    def test_merge_groups_overlapping_ocr_and_removes_overlapping_mineru_text(self):
        mineru = [{"type": "text", "bbox": [10, 10, 30, 30]}]
        ocr = [
            {"type": "text", "bbox": [20, 20, 30, 30], "lines": [{"bbox": [20, 20, 30, 30], "spans": [{"bbox": [20, 20, 30, 30], "content": "A", "type": "text"}]}]},
            {"type": "text", "bbox": [29, 20, 40, 30], "lines": [{"bbox": [29, 20, 40, 30], "spans": [{"bbox": [29, 20, 40, 30], "content": "B", "type": "text"}]}]},
        ]

        merged, stats = merge_ocr_text_elements(mineru, ocr, _has_overlap)

        self.assertEqual(len(merged), 1)
        ocr_merged = merged[0]
        text = "".join(span.get("content", "") for line in ocr_merged["lines"] for span in line.get("spans", []))
        self.assertEqual(text.replace("\n", ""), "AB")
        self.assertEqual(stats["ocr_candidates"], 2)
        self.assertEqual(stats["ocr_groups"], 1)
        self.assertEqual(stats["ocr_merged"], 1)
        self.assertEqual(stats["ocr_added"], 1)
        self.assertEqual(stats["mineru_removed_overlap"], 1)

    def test_merge_keeps_non_overlapping_ocr(self):
        mineru = [{"type": "text", "bbox": [10, 10, 30, 30]}]
        ocr = [{"type": "text", "bbox": [35, 35, 50, 50], "lines": [{"bbox": [35, 35, 50, 50], "spans": [{"bbox": [35, 35, 50, 50], "content": "new", "type": "text"}]}]}]

        merged, stats = merge_ocr_text_elements(mineru, ocr, _has_overlap)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[-1]["lines"][0]["spans"][0]["content"], "new")
        self.assertEqual(stats["ocr_candidates"], 1)
        self.assertEqual(stats["ocr_groups"], 1)
        self.assertEqual(stats["ocr_merged"], 0)
        self.assertEqual(stats["ocr_added"], 1)
        self.assertEqual(stats["mineru_removed_overlap"], 0)

    def test_merge_only_removes_overlapping_mineru_text_types(self):
        mineru = [{"type": "image", "bbox": [10, 10, 30, 30]}]
        ocr = [{"type": "text", "bbox": [15, 15, 20, 20], "lines": [{"bbox": [15, 15, 20, 20], "spans": [{"bbox": [15, 15, 20, 20], "content": "inside-image", "type": "text"}]}]}]

        merged, stats = merge_ocr_text_elements(mineru, ocr, _has_overlap)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["type"], "image")
        self.assertEqual(stats["ocr_added"], 1)
        self.assertEqual(stats["mineru_removed_overlap"], 0)


    def test_merge_replaces_list_blocks_at_text_level(self):
        mineru = [
            {
                "type": "list",
                "bbox": [492, 196, 1148, 277],
                "index": 8,
                "is_discarded": False,
                "sub_type": "text",
                "blocks": [
                    {
                        "type": "text",
                        "bbox": [492, 196, 1148, 230],
                        "index": 6,
                        "lines": [
                            {
                                "bbox": [492, 196, 1148, 230],
                                "spans": [
                                    {
                                        "bbox": [492, 196, 1148, 230],
                                        "content": "✓ 构建“指标定义 -> 声明式开发 -> 服务发布”的完整流水线。",
                                        "type": "text",
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "type": "text",
                        "bbox": [492, 244, 857, 277],
                        "index": 7,
                        "lines": [
                            {
                                "bbox": [492, 244, 857, 277],
                                "spans": [
                                    {
                                        "bbox": [492, 244, 857, 277],
                                        "content": "形成统一规范的数据开发标准。",
                                        "type": "text",
                                    }
                                ],
                            }
                        ],
                    },
                ],
            }
        ]

        ocr = [
            {
                "type": "text",
                "bbox": [500, 200, 1142, 229],
                "lines": [
                    {
                        "bbox": [500, 200, 1142, 229],
                        "spans": [
                            {
                                "bbox": [500, 200, 1142, 229],
                                "content": "✓ 构建“指标定义 -> 声明式开发 -> 服务发布”的完整流水线。",
                                "type": "text",
                            }
                        ],
                    }
                ],
            },
            {
                "type": "text",
                "bbox": [500, 246, 860, 276],
                "lines": [
                    {
                        "bbox": [500, 246, 860, 276],
                        "spans": [
                            {
                                "bbox": [500, 246, 860, 276],
                                "content": "形成统一规范的数据开发标准。",
                                "type": "text",
                            }
                        ],
                    }
                ],
            },
        ]

        merged, stats = merge_ocr_text_elements(mineru, ocr, _has_overlap)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].get("type"), "list")
        self.assertEqual(len(merged[0].get("blocks", [])), 2)
        self.assertEqual(
            merged[0]["blocks"][0]["lines"][0]["spans"][0]["content"],
            "✓ 构建“指标定义 -> 声明式开发 -> 服务发布”的完整流水线。",
        )
        self.assertEqual(
            merged[0]["blocks"][1]["lines"][0]["spans"][0]["content"],
            "形成统一规范的数据开发标准。",
        )
        self.assertEqual(stats["ocr_added"], 2)


if __name__ == "__main__":
    unittest.main()
