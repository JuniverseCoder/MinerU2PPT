import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from converter.ocr_merge import PaddleOCREngine


class TestPaddleOCREngineInit(unittest.TestCase):
    def _create_local_model_tree(self, root: Path, lang: str = "ch"):
        lang_root = root / lang
        for sub in ("det", "rec", "cls"):
            subdir = lang_root / sub
            subdir.mkdir(parents=True, exist_ok=True)
            (subdir / "inference.pdmodel").write_text("m", encoding="utf-8")
            (subdir / "inference.pdiparams").write_text("p", encoding="utf-8")

    def test_model_root_resolution_prefers_constructor_argument(self):
        with tempfile.TemporaryDirectory() as tempdir:
            explicit_root = Path(tempdir) / "explicit"
            explicit_root.mkdir(parents=True)
            os.environ["MINERU_OCR_MODEL_ROOT"] = str(Path(tempdir) / "env")
            try:
                engine = PaddleOCREngine(model_root=str(explicit_root))
                resolved, source = engine._resolve_model_root()
            finally:
                os.environ.pop("MINERU_OCR_MODEL_ROOT", None)

        self.assertEqual(resolved, explicit_root.resolve())
        self.assertEqual(source, "argument:model_root")

    def test_model_root_resolution_prefers_env_when_argument_missing(self):
        with tempfile.TemporaryDirectory() as tempdir:
            env_root = Path(tempdir) / "env"
            env_root.mkdir(parents=True)
            os.environ["MINERU_OCR_MODEL_ROOT"] = str(env_root)
            try:
                engine = PaddleOCREngine()
                resolved, source = engine._resolve_model_root()
            finally:
                os.environ.pop("MINERU_OCR_MODEL_ROOT", None)

        self.assertEqual(resolved, env_root.resolve())
        self.assertEqual(source, "env:MINERU_OCR_MODEL_ROOT")

    def test_model_root_resolution_prefers_executable_internal_dir_when_present(self):
        with tempfile.TemporaryDirectory() as tempdir:
            exe_path = Path(tempdir) / "app.exe"
            exe_path.write_text("", encoding="utf-8")
            exe_model_root = exe_path.parent / "_internal" / "models" / "paddleocr"
            exe_model_root.mkdir(parents=True)

            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MINERU_OCR_MODEL_ROOT", None)
                with mock.patch("converter.ocr_merge.sys.executable", str(exe_path)):
                    engine = PaddleOCREngine()
                    resolved, source = engine._resolve_model_root()

        self.assertEqual(resolved, exe_model_root.resolve())
        self.assertEqual(source, "executable:_internal/models/paddleocr")

    def test_model_root_resolution_prefers_executable_dir_when_present(self):
        with tempfile.TemporaryDirectory() as tempdir:
            exe_path = Path(tempdir) / "app.exe"
            exe_path.write_text("", encoding="utf-8")
            exe_model_root = exe_path.parent / "models" / "paddleocr"
            exe_model_root.mkdir(parents=True)

            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MINERU_OCR_MODEL_ROOT", None)
                with mock.patch("converter.ocr_merge.sys.executable", str(exe_path)):
                    engine = PaddleOCREngine()
                    resolved, source = engine._resolve_model_root()

        self.assertEqual(resolved, exe_model_root.resolve())
        self.assertEqual(source, "executable:models/paddleocr")

    def test_model_root_resolution_falls_back_to_source_dir(self):
        with tempfile.TemporaryDirectory() as tempdir:
            fake_module_file = Path(tempdir) / "repo" / "converter" / "ocr_merge.py"
            fake_module_file.parent.mkdir(parents=True, exist_ok=True)
            fake_module_file.write_text("# placeholder", encoding="utf-8")
            source_root = fake_module_file.parents[1] / "models" / "paddleocr"
            source_root.mkdir(parents=True)

            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MINERU_OCR_MODEL_ROOT", None)
                with (
                    mock.patch("converter.ocr_merge.__file__", str(fake_module_file)),
                    mock.patch("converter.ocr_merge.sys.executable", str(Path(tempdir) / "no_app.exe")),
                ):
                    engine = PaddleOCREngine()
                    resolved, source = engine._resolve_model_root()

        self.assertEqual(resolved, source_root.resolve())
        self.assertEqual(source, "source:models/paddleocr")

    def test_missing_model_path_raises_readable_error(self):
        with tempfile.TemporaryDirectory() as tempdir:
            missing_root = Path(tempdir) / "missing-models"
            fake_module = SimpleNamespace(PaddleOCR=lambda **_kwargs: object())
            engine = PaddleOCREngine(model_root=str(missing_root), offline_only=True)

            with mock.patch.dict("sys.modules", {"paddleocr": fake_module}):
                with self.assertRaises(RuntimeError) as ctx:
                    engine._ensure_initialized()

        self.assertIn("Local model root not found", str(ctx.exception))
        self.assertIn("models/paddleocr", str(ctx.exception))

    def test_auto_mode_falls_back_from_gpu_to_cpu(self):
        with tempfile.TemporaryDirectory() as tempdir:
            model_root = Path(tempdir) / "models" / "paddleocr"
            self._create_local_model_tree(model_root)

            calls = []

            class FakeOCR:
                def __init__(self, device):
                    self.device = device

            def fake_constructor(**kwargs):
                calls.append(kwargs.get("device"))
                if kwargs.get("device") == "gpu":
                    raise RuntimeError("gpu failed")
                return FakeOCR("cpu")

            fake_module = SimpleNamespace(PaddleOCR=fake_constructor)
            engine = PaddleOCREngine(device_policy="auto", model_root=str(model_root), offline_only=True)

            with (
                mock.patch.dict("sys.modules", {"paddleocr": fake_module}),
                mock.patch.object(PaddleOCREngine, "_is_gpu_available", return_value=True),
            ):
                engine._ensure_initialized()

        self.assertIsNotNone(engine._ocr)
        self.assertEqual(engine._ocr.device, "cpu")
        self.assertIn("gpu", calls)
        self.assertIn("cpu", calls)

    def test_gpu_mode_does_not_fallback_to_cpu(self):
        with tempfile.TemporaryDirectory() as tempdir:
            model_root = Path(tempdir) / "models" / "paddleocr"
            self._create_local_model_tree(model_root)

            calls = []

            def fake_constructor(**kwargs):
                calls.append(kwargs.get("device"))
                raise RuntimeError("gpu failed")

            fake_module = SimpleNamespace(PaddleOCR=fake_constructor)
            engine = PaddleOCREngine(device_policy="gpu", model_root=str(model_root), offline_only=True)

            with (
                mock.patch.dict("sys.modules", {"paddleocr": fake_module}),
                mock.patch.object(PaddleOCREngine, "_is_gpu_available", return_value=True),
            ):
                with self.assertRaises(RuntimeError):
                    engine._ensure_initialized()

        self.assertTrue(calls)
        self.assertTrue(all(device == "gpu" for device in calls))

    def test_cpu_mode_uses_cpu_only(self):
        with tempfile.TemporaryDirectory() as tempdir:
            model_root = Path(tempdir) / "models" / "paddleocr"
            self._create_local_model_tree(model_root)

            calls = []

            class FakeOCR:
                pass

            def fake_constructor(**kwargs):
                calls.append(kwargs.get("device"))
                return FakeOCR()

            fake_module = SimpleNamespace(PaddleOCR=fake_constructor)
            engine = PaddleOCREngine(device_policy="cpu", model_root=str(model_root), offline_only=True)

            with mock.patch.dict("sys.modules", {"paddleocr": fake_module}):
                engine._ensure_initialized()

        self.assertIsNotNone(engine._ocr)
        self.assertTrue(calls)
        self.assertTrue(all(device == "cpu" for device in calls))


if __name__ == "__main__":
    unittest.main()
