import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from converter.adapters.mineru_adapter import MinerUAdapter, MinerUPageData
from converter.adapters.ocr_adapter import OCRAdapter
from converter.ir_merge import merge_ir_elements
from converter.ocr_merge import PaddleOCREngine


class TestCase1IRMergeLineMerge(unittest.TestCase):
    def test_case1_design_asset_should_merge_into_single_text_box(self):
        repo_root = Path(__file__).resolve().parents[2]
        input_png = repo_root / "demo" / "case1" / "PixPin_2026-03-05_21-52-43.png"
        input_json = repo_root / "demo" / "case1" / "MinerU_PixPin_2026-03-05_21-52-43__20260305135318.json"

        self.assertTrue(input_png.exists(), f"Missing demo image: {input_png}")
        self.assertTrue(input_json.exists(), f"Missing demo json: {input_json}")

        data = json.loads(input_json.read_text(encoding="utf-8"))
        page_data = data["pdf_info"][0]

        page_image = np.array(Image.open(input_png).convert("RGB"))
        mineru_elements = MinerUAdapter().extract_page_elements(MinerUPageData.from_dict(page_data))
        ocr_elements = OCRAdapter(PaddleOCREngine()).extract_page_elements(
            page_image,
            page_image.shape[1],
            page_image.shape[0],
        )

        merged, stats = merge_ir_elements(mineru_elements, ocr_elements)

        target = [
            elem
            for elem in merged
            if elem.get("type") == "text"
            and "Design as Code," in (elem.get("text") or "")
            and "Asset as Service" in (elem.get("text") or "")
        ]

        self.assertEqual(
            len(target),
            1,
            f"Expected merged single text element for Design/Asset phrase, got {len(target)}",
        )
        self.assertGreaterEqual(stats.get("overlay_fragment_groups", 0), 1)
        self.assertIsInstance(target[0].get("text_runs"), list)
        self.assertEqual(
            "".join(run.get("text", "") for run in target[0]["text_runs"]),
            "Design as Code,Asset as Service",
        )


if __name__ == "__main__":
    unittest.main()
