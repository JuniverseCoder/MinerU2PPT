import unittest

from converter.adapters.mineru_adapter import MinerUAdapter, MinerUPageData
from converter.ir import ImageIR, TextIR


class TestMinerUAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = MinerUAdapter()

    def test_maps_para_block_to_text_ir(self):
        page = {
            "para_blocks": [
                {
                    "type": "title",
                    "index": 1,
                    "bbox": [10, 10, 100, 30],
                    "lines": [{"bbox": [10, 10, 100, 30], "spans": [{"bbox": [10, 10, 100, 30], "content": "Title", "type": "text"}]}],
                }
            ],
            "images": [],
            "tables": [],
            "discarded_blocks": [],
        }

        elements = self.adapter.extract_page_elements(MinerUPageData.from_dict(page))
        self.assertEqual(len(elements), 1)
        self.assertIsInstance(elements[0], TextIR)
        self.assertEqual(elements[0].type, "text")
        self.assertEqual(elements[0].source, "mineru")
        self.assertFalse(elements[0].is_discarded)
        self.assertTrue(elements[0].style["bold"])
        self.assertIsNone(elements[0].text_runs)

    def test_maps_list_blocks_into_multiple_text_elements(self):
        page = {
            "para_blocks": [
                {
                    "type": "list",
                    "index": 8,
                    "bbox": [10, 10, 200, 80],
                    "blocks": [
                        {
                            "type": "text",
                            "index": 11,
                            "bbox": [10, 10, 200, 40],
                            "lines": [{"bbox": [10, 10, 200, 40], "spans": [{"bbox": [10, 10, 200, 40], "content": "A", "type": "text"}]}],
                        },
                        {
                            "type": "text",
                            "index": 12,
                            "bbox": [10, 45, 200, 80],
                            "lines": [{"bbox": [10, 45, 200, 80], "spans": [{"bbox": [10, 45, 200, 80], "content": "B", "type": "text"}]}],
                        },
                    ],
                }
            ],
            "images": [],
            "tables": [],
            "discarded_blocks": [],
        }

        elements = self.adapter.extract_page_elements(MinerUPageData.from_dict(page))
        self.assertEqual(len(elements), 2)
        self.assertTrue(all(isinstance(elem, TextIR) for elem in elements))
        self.assertEqual(elements[0].group_id, elements[1].group_id)

    def test_include_text_runs_generates_runs_for_mineru(self):
        page = {
            "para_blocks": [
                {
                    "type": "text",
                    "bbox": [0, 0, 100, 20],
                    "lines": [
                        {
                            "bbox": [0, 0, 100, 20],
                            "spans": [
                                {"bbox": [0, 0, 40, 20], "content": "A", "type": "text"},
                                {"bbox": [40, 0, 100, 20], "content": "B", "type": "text"},
                            ],
                        }
                    ],
                }
            ],
            "images": [],
            "tables": [],
            "discarded_blocks": [],
        }

        elements = self.adapter.extract_page_elements(MinerUPageData.from_dict(page), include_text_runs=True)
        self.assertEqual(len(elements), 1)
        self.assertIsInstance(elements[0].text_runs, list)
        self.assertEqual(len(elements[0].text_runs), 2)
        self.assertEqual(elements[0].text_runs[0].bbox, [0.0, 0.0, 40.0, 20.0])

    def test_maps_image_caption_and_discarded(self):
        page = {
            "para_blocks": [],
            "images": [
                {
                    "type": "image",
                    "index": 2,
                    "bbox": [50, 50, 160, 140],
                    "blocks": [
                        {"type": "image_body", "bbox": [52, 52, 150, 130]},
                        {
                            "type": "image_caption",
                            "bbox": [52, 132, 150, 142],
                            "lines": [{"bbox": [52, 132, 150, 142], "spans": [{"bbox": [52, 132, 150, 142], "content": "caption", "type": "text"}]}],
                        },
                    ],
                }
            ],
            "tables": [],
            "discarded_blocks": [{"type": "text", "bbox": [1, 1, 2, 2], "text": "wm"}],
        }

        elements = self.adapter.extract_page_elements(MinerUPageData.from_dict(page))
        image_elems = [e for e in elements if isinstance(e, ImageIR)]
        text_elems = [e for e in elements if isinstance(e, TextIR)]

        self.assertEqual(len(image_elems), 1)
        self.assertEqual(len(text_elems), 2)
        self.assertTrue(any(elem.is_discarded for elem in text_elems))
        self.assertTrue(any(elem.is_watermark for elem in text_elems if elem.is_discarded))

    def test_discarded_block_defaults_to_watermark_true(self):
        page = {
            "para_blocks": [],
            "images": [],
            "tables": [],
            "discarded_blocks": [{"type": "text", "bbox": [1, 1, 2, 2], "text": "wm"}],
        }

        elements = self.adapter.extract_page_elements(MinerUPageData.from_dict(page))
        self.assertEqual(len(elements), 1)
        self.assertTrue(elements[0].is_discarded)
        self.assertTrue(elements[0].is_watermark)

    def test_explicit_watermark_flag_on_normal_block_is_propagated(self):
        page = {
            "para_blocks": [{"type": "text", "bbox": [10, 10, 20, 20], "text": "tag", "is_watermark": True}],
            "images": [],
            "tables": [],
            "discarded_blocks": [],
        }

        elements = self.adapter.extract_page_elements(MinerUPageData.from_dict(page))
        self.assertEqual(len(elements), 1)
        self.assertFalse(elements[0].is_discarded)
        self.assertTrue(elements[0].is_watermark)


if __name__ == "__main__":
    unittest.main()
