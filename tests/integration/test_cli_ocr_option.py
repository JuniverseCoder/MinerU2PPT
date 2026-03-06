import unittest
from unittest import mock

import main


class TestCliOCROption(unittest.TestCase):
    def test_cli_forwards_core_flags_and_default_ocr_options(self):
        argv = [
            "main.py",
            "--json",
            "a.json",
            "--input",
            "a.pdf",
            "--output",
            "a.pptx",
            "--debug-images",
            "--no-watermark",
        ]

        with mock.patch("sys.argv", argv), mock.patch("main.convert_mineru_to_ppt") as mocked_convert:
            main.main()

        self.assertEqual(mocked_convert.call_count, 1)
        kwargs = mocked_convert.call_args.kwargs
        self.assertTrue(kwargs["debug_images"])
        self.assertTrue(kwargs["remove_watermark"])
        self.assertEqual(kwargs["ocr_device_policy"], "auto")
        self.assertIsNone(kwargs["ocr_model_root"])
        self.assertTrue(kwargs["ocr_offline_only"])

    def test_cli_forwards_explicit_ocr_options(self):
        argv = [
            "main.py",
            "--json",
            "a.json",
            "--input",
            "a.pdf",
            "--output",
            "a.pptx",
            "--ocr-device",
            "gpu",
            "--ocr-model-root",
            "models/paddleocr",
        ]

        with mock.patch("sys.argv", argv), mock.patch("main.convert_mineru_to_ppt") as mocked_convert:
            main.main()

        kwargs = mocked_convert.call_args.kwargs
        self.assertEqual(kwargs["ocr_device_policy"], "gpu")
        self.assertEqual(kwargs["ocr_model_root"], "models/paddleocr")
        self.assertTrue(kwargs["ocr_offline_only"])

    def test_cli_rejects_removed_ocr_flag(self):
        argv = [
            "main.py",
            "--json",
            "a.json",
            "--input",
            "a.pdf",
            "--output",
            "a.pptx",
            "--ocr-merge",
        ]

        with mock.patch("sys.argv", argv), self.assertRaises(SystemExit):
            main.main()


if __name__ == "__main__":
    unittest.main()
