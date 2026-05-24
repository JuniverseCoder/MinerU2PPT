import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from .utils import extract_background_color, extract_font_color

TEXT_ELEMENT_TYPES = (
    "list",
    "text",
    "title",
    "caption",
    "footnote",
    "footer",
    "header",
    "page_number",
)

OCR_BBOX_PAD_RATIO = 0.2
OCR_BBOX_MIN_PAD_PIXELS = 1
OCR_FONT_DISTANCE_THRESHOLD = 60.0
OCR_FONT_DISTANCE_THRESHOLD_LITE = 50.0
OCR_REFINE_MIN_PIXEL_RATIO = 0.005
ROW_NON_BG_DISTANCE_THRESHOLD = 55.0


class PaddleOCREngine:
    _MODEL_ROOT_ENV = "MINERU_OCR_MODEL_ROOT"
    _MODEL_REQUIRED_DIRS = ("det", "rec")
    _MODEL_FILE_CANDIDATES = ("inference.pdmodel", "inference.json")
    _PARAM_FILE_CANDIDATES = ("inference.pdiparams",)
    _MODEL_VARIANTS = {"auto", "lite", "server"}
    SUPPORTED_LANGS = {"ch", "en"}
    _MODEL_NAME_MAPPING = {
        "lite": {
            "ch": {
                "text_detection_model_name": "PP-OCRv5_mobile_det",
                "text_recognition_model_name": "PP-OCRv5_mobile_rec",
            },
            "en": {
                "text_detection_model_name": "PP-OCRv5_mobile_det",
                "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
            },
        },
        "server": {
            "ch": {
                "text_detection_model_name": "PP-OCRv5_server_det",
                "text_recognition_model_name": "PP-OCRv5_server_rec",
            },
            "en": {
                "text_detection_model_name": "PP-OCRv5_server_det",
                "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
            },
        },
    }
    _DB_PARAM_DEFAULTS = {
        "lite": {
            "det_db_thresh": 0.35,
            "det_db_box_thresh": 0.8,
            "det_db_unclip_ratio": 0.9,
        },
        "server": {
            "det_db_thresh": 0.35,
            "det_db_box_thresh": 0.8,
            "det_db_unclip_ratio": 1.0,
        },
    }

    def __init__(
        self,
        lang: str = "ch",
        use_angle_cls: bool = False,
        device_policy: str = "auto",
        model_root: str | None = None,
        offline_only: bool = True,
        det_db_thresh: float | None = None,
        det_db_box_thresh: float | None = None,
        det_db_unclip_ratio: float | None = None,
        refine_font_distance_threshold: float | None = None,
        model_variant: str = "auto",
    ):
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.device_policy = device_policy
        self.model_root = model_root
        self.offline_only = offline_only
        self.det_db_thresh = det_db_thresh
        self.det_db_box_thresh = det_db_box_thresh
        self.det_db_unclip_ratio = det_db_unclip_ratio
        self.refine_font_distance_threshold = refine_font_distance_threshold
        self.model_variant = model_variant.lower() if model_variant else "auto"
        self._ocr = None
        self._active_device = None
        self._active_model_variant = None
        self._active_db_defaults = None

        if self.device_policy not in {"auto", "gpu", "cpu"}:
            raise ValueError("device_policy must be one of: auto, gpu, cpu")
        if self.model_variant not in self._MODEL_VARIANTS:
            raise ValueError("model_variant must be one of: auto, lite, server")
        if self.lang not in self.SUPPORTED_LANGS:
            raise ValueError("lang must be one of: ch, en")

    def _resolve_device_order(self):
        if self.device_policy == "gpu":
            return ["gpu"]
        if self.device_policy == "cpu":
            return ["cpu"]
        return ["gpu", "cpu"]

    def _resolve_cpu_threads(self) -> int:
        env_value = os.getenv("MINERU_OCR_CPU_THREADS")
        if env_value:
            try:
                parsed = int(env_value)
                if parsed > 0:
                    return parsed
            except ValueError:
                pass

        cpu_count = os.cpu_count() or 4
        # Keep some headroom for GUI responsiveness on CPU-only runs.
        return max(1, min(6, max(1, cpu_count // 2)))

    def _is_gpu_available(self) -> bool:
        try:
            import paddle

            if not paddle.device.is_compiled_with_cuda():
                return False

            try:
                return paddle.device.cuda.device_count() > 0
            except Exception:
                device_name = paddle.device.get_device()
                return isinstance(device_name, str) and device_name.startswith("gpu")
        except Exception:
            return False

    def _resolve_model_root(self):
        if self.model_root:
            root = Path(self.model_root).expanduser().resolve()
            return root, "argument:model_root"

        env_root = os.getenv(self._MODEL_ROOT_ENV)
        if env_root:
            root = Path(env_root).expanduser().resolve()
            return root, f"env:{self._MODEL_ROOT_ENV}"

        return None, "default:download"

    def _assert_local_models_integrity(self, model_root: Path, variant: str):
        if not model_root.exists() or not model_root.is_dir():
            raise RuntimeError(
                f"[OCR] Local model root not found: {model_root}. "
                "Expected a models/paddleocr directory with per-variant/per-language det/rec/cls subfolders."
            )

        variant_root = model_root / variant
        if not variant_root.exists() or not variant_root.is_dir():
            raise RuntimeError(
                f"[OCR] Missing model variant directory: {variant_root}. "
                f"Expected layout: {model_root}/<variant>/<lang>/det|rec|cls"
            )

        lang_root = variant_root / self.lang
        if not lang_root.exists() or not lang_root.is_dir():
            raise RuntimeError(
                f"[OCR] Missing language model directory: {lang_root}. "
                f"Expected layout: {model_root}/<variant>/<lang>/det|rec|cls"
            )

        required_dirs = list(self._MODEL_REQUIRED_DIRS)
        if self.use_angle_cls:
            required_dirs.append("cls")
        missing_dirs = [name for name in required_dirs if not (lang_root / name).is_dir()]
        if missing_dirs:
            raise RuntimeError(
                f"[OCR] Missing model subdirectories for lang='{self.lang}': {missing_dirs}. "
                f"Expected under: {lang_root}"
            )

        for subdir in required_dirs:
            model_dir = lang_root / subdir
            has_model_file = any((model_dir / name).exists() for name in self._MODEL_FILE_CANDIDATES)
            has_param_file = any((model_dir / name).exists() for name in self._PARAM_FILE_CANDIDATES)
            if not has_model_file or not has_param_file:
                raise RuntimeError(
                    f"[OCR] Incomplete local model files in {model_dir}. "
                    f"Required one of {self._MODEL_FILE_CANDIDATES} and one of {self._PARAM_FILE_CANDIDATES}."
                )

    def _build_model_dirs(self, model_root: Path, variant: str):
        lang_root = model_root / variant / self.lang
        model_dirs = {
            "text_detection_model_dir": str(lang_root / "det"),
            "text_recognition_model_dir": str(lang_root / "rec"),
        }
        if self.use_angle_cls:
            model_dirs["textline_orientation_model_dir"] = str(lang_root / "cls")
        return model_dirs

    def _constructor_attempts_for_device(
        self,
        device: str,
        model_dirs: dict[str, str] | None,
        model_names: dict[str, str] | None,
        db_defaults: dict[str, float] | None,
    ):
        base_kwargs = {
            "lang": self.lang,
            "enable_mkldnn": False,
            "enable_hpi": False,
            "use_tensorrt": False,
            "enable_cinn": False,
            "return_word_box": True,
        }
        if model_dirs:
            base_kwargs.update(model_dirs)
        if model_names:
            base_kwargs.update(model_names)

        defaults = db_defaults or {}
        det_db_thresh = self.det_db_thresh if self.det_db_thresh is not None else defaults.get("det_db_thresh")
        det_db_box_thresh = (
            self.det_db_box_thresh if self.det_db_box_thresh is not None else defaults.get("det_db_box_thresh")
        )
        det_db_unclip_ratio = (
            self.det_db_unclip_ratio if self.det_db_unclip_ratio is not None else defaults.get("det_db_unclip_ratio")
        )

        det_param_variants = [{}]
        if det_db_thresh is not None or det_db_box_thresh is not None or det_db_unclip_ratio is not None:
            legacy_params = {}
            text_det_params = {}

            if det_db_thresh is not None:
                legacy_params["det_db_thresh"] = float(det_db_thresh)
                text_det_params["text_det_thresh"] = float(det_db_thresh)

            if det_db_box_thresh is not None:
                legacy_params["det_db_box_thresh"] = float(det_db_box_thresh)
                text_det_params["text_det_box_thresh"] = float(det_db_box_thresh)

            if det_db_unclip_ratio is not None:
                legacy_params["det_db_unclip_ratio"] = float(det_db_unclip_ratio)
                text_det_params["text_det_unclip_ratio"] = float(det_db_unclip_ratio)

            # Try modern text_det_* names first, then legacy det_db_* for older versions.
            det_param_variants = [text_det_params, legacy_params]

        if self.offline_only:
            base_kwargs["download"] = False

        compatibility_attempts = [
            {
                "use_textline_orientation": self.use_angle_cls,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
            },
            {
                "use_angle_cls": self.use_angle_cls,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
            },
            {
                "use_textline_orientation": self.use_angle_cls,
            },
            {
                "use_angle_cls": self.use_angle_cls,
            },
            {
                "use_angle_cls": self.use_angle_cls,
            },
        ]

        if device == "cpu":
            cpu_threads = self._resolve_cpu_threads()
            device_variants = [
                {"device": device, "cpu_threads": cpu_threads},
                {"device": device},
            ]
        else:
            device_variants = [{"device": device}]

        for det_params in det_param_variants:
            for device_kwargs in device_variants:
                for attempt in compatibility_attempts:
                    yield {**base_kwargs, **det_params, **device_kwargs, **attempt}

        if self.offline_only and "download" in base_kwargs:
            fallback_base = dict(base_kwargs)
            fallback_base.pop("download", None)
            for det_params in det_param_variants:
                for device_kwargs in device_variants:
                    for attempt in compatibility_attempts:
                        yield {**fallback_base, **det_params, **device_kwargs, **attempt}

    def _ensure_initialized(self):
        if self._ocr is not None:
            return

        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        os.environ.setdefault("FLAGS_enable_pir_api", "0")
        os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
        os.environ.setdefault("FLAGS_enable_new_ir_in_executor", "0")

        from paddleocr import PaddleOCR

        model_root, model_root_source = self._resolve_model_root()

        requested_order = self._resolve_device_order()
        gpu_available = self._is_gpu_available()
        if not gpu_available and "gpu" in requested_order:
            if self.device_policy == "gpu":
                raise RuntimeError("[OCR] GPU device policy requested but CUDA GPU is not available.")
            print("[OCR] GPU not available; skipping GPU initialization and using CPU.")
        device_order = [d for d in requested_order if d != "gpu" or gpu_available]

        resolved_variant = self.model_variant
        resolved_source = "explicit"
        if resolved_variant == "auto":
            resolved_variant = "server" if gpu_available else "lite"
            resolved_source = "auto"
        self._active_model_variant = resolved_variant
        self._active_db_defaults = dict(self._DB_PARAM_DEFAULTS.get(resolved_variant, {}))
        print(f"[OCR] det_db_defaults={self._active_db_defaults}")

        model_dirs = None
        model_names = self._MODEL_NAME_MAPPING.get(resolved_variant, {}).get(self.lang)
        if model_root is not None:
            model_names = None
            self._assert_local_models_integrity(model_root, resolved_variant)
            model_dirs = self._build_model_dirs(model_root, resolved_variant)
            model_root_hint = f"{model_root}/{resolved_variant}"
        else:
            model_root_hint = "default:download"

        print(f"[OCR] model_root={model_root} ({model_root_source})")
        print(f"[OCR] model_variant={resolved_variant} ({resolved_source})")
        print(f"[OCR] lang={self.lang}")
        print(f"[OCR] device_order={device_order}")
        if model_root is not None:
            print(f"[OCR] model_root_layout={model_root_hint}")
            print(f"[OCR] model_variant_source=local")
        else:
            print(f"[OCR] model_variant_source=download")
        if "cpu" in device_order:
            print(
                f"[OCR] cpu_threads={self._resolve_cpu_threads()} "
                "(set MINERU_OCR_CPU_THREADS to override)"
            )

        last_error = None
        gpu_error = None
        gpu_failed = False
        for device in device_order:
            for kwargs in self._constructor_attempts_for_device(device, model_dirs, model_names, self._active_db_defaults):
                try:
                    self._ocr = PaddleOCR(**kwargs)
                    self._active_device = device
                    if device == "cpu" and gpu_failed:
                        print("[OCR] CPU fallback initialization succeeded.")
                    return
                except Exception as exc:
                    last_error = exc

            if device == "gpu":
                gpu_failed = True
                gpu_error = last_error
                if "cpu" in device_order:
                    print(f"[OCR] GPU initialization failed: {gpu_error}")

        if self.offline_only:
            raise RuntimeError(
                f"Failed to initialize PaddleOCR in offline-only mode: {last_error}. "
                "Local models are required and online download fallback is disabled."
            )

        raise RuntimeError(f"Failed to initialize PaddleOCR: {last_error}")

    def _run_ocr(self, rgb_image):
        if hasattr(self._ocr, "predict"):
            return self._ocr.predict(rgb_image)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        try:
            return self._ocr.ocr(bgr_image, cls=self.use_angle_cls)
        except TypeError as exc:
            message = str(exc)
            if "unexpected keyword argument 'cls'" in message:
                return self._ocr.ocr(bgr_image)
            raise

    def _prepare_image_for_ocr(self, rgb_image):
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        ocr_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        env_value = os.getenv("MINERU_OCR_MAX_SIDE") or os.getenv("MINERU_OCR_CPU_MAX_SIDE", "1920")
        try:
            max_side = int(env_value)
        except ValueError:
            max_side = 1920

        if max_side <= 0:
            return ocr_image, 1.0, 1.0

        img_h, img_w = ocr_image.shape[:2]
        longest_side = max(img_w, img_h)
        if longest_side <= max_side:
            return ocr_image, 1.0, 1.0

        scale = float(max_side) / float(longest_side)
        new_w = max(1, int(round(img_w * scale)))
        new_h = max(1, int(round(img_h * scale)))
        resized = cv2.resize(ocr_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        sx = float(img_w) / float(new_w)
        sy = float(img_h) / float(new_h)
        print(
            f"[OCR] resize before OCR: {img_w}x{img_h} -> {new_w}x{new_h} "
            f"(max_side={max_side}, grayscale=True)"
        )
        return resized, sx, sy

    def extract_text_elements(self, page_image, json_w: float, json_h: float, return_stage_elements: bool = False):
        self._ensure_initialized()

        if page_image is None or page_image.size == 0:
            if return_stage_elements:
                return {
                    "before_refined_elements": [],
                    "after_refined_elements": [],
                }
            return []

        img_h, img_w = page_image.shape[:2]
        if img_h == 0 or img_w == 0:
            if return_stage_elements:
                return {
                    "before_refined_elements": [],
                    "after_refined_elements": [],
                }
            return []

        ocr_image, sx, sy = self._prepare_image_for_ocr(page_image)
        ocr_h, ocr_w = ocr_image.shape[:2]

        raw_result = self._run_ocr(ocr_image)

        raw_elements = []
        for idx, (points, text, score) in enumerate(_iter_ocr_results(raw_result), start=1):
            if not text:
                continue

            bbox_px = _polygon_to_bbox(points, ocr_w, ocr_h)
            if not bbox_px:
                continue

            scaled_bbox_px = _scale_pixel_bbox(bbox_px, sx, sy, img_w, img_h)
            bbox_json = _pixel_bbox_to_json_bbox(scaled_bbox_px, json_w, json_h, img_w, img_h)
            raw_elements.append(
                {
                    "angle": 0,
                    "bbox": bbox_json,
                    "index": idx,
                    "is_discarded": False,
                    "lines": [
                        {
                            "bbox": bbox_json,
                            "spans": [
                                {
                                    "bbox": bbox_json,
                                    "content": text,
                                    "type": "text",
                                }
                            ],
                        }
                    ],
                    "type": "text",
                    "ocr_score": score,
                    "source": "ocr",
                }
            )

        merged_line_elements = _merge_ocr_line_fragments(raw_elements)
        font_distance_threshold = self.refine_font_distance_threshold
        if font_distance_threshold is None:
            font_distance_threshold = (
                OCR_FONT_DISTANCE_THRESHOLD_LITE
                if self._active_model_variant == "lite"
                else OCR_FONT_DISTANCE_THRESHOLD
            )

        refined_elements = refine_ocr_text_elements(
            merged_line_elements,
            page_image,
            json_w,
            json_h,
            font_distance_threshold=font_distance_threshold,
        )

        for idx, elem in enumerate(refined_elements, start=1):
            elem["index"] = idx

        if return_stage_elements:
            return {
                "before_refined_elements": merged_line_elements,
                "after_refined_elements": refined_elements,
            }

        return refined_elements


def merge_ocr_text_elements(
    elements: list[dict[str, Any]],
    ocr_elements: list[dict[str, Any]],
    has_overlap: Callable[[list[float], list[float]], bool],
    text_types: Iterable[str] = TEXT_ELEMENT_TYPES,
):
    text_type_set = set(text_types)

    mineru_text_units = _collect_mineru_text_units(elements, text_type_set)
    mineru_text_bboxes = [unit["bbox"] for unit in mineru_text_units]

    grouped_ocr_elements = _group_ocr_elements_by_overlap(ocr_elements, mineru_text_bboxes, has_overlap)

    assigned_by_unit, unassigned_ocr = _assign_ocr_groups_to_mineru_units(
        grouped_ocr_elements,
        mineru_text_units,
        has_overlap,
    )

    consumed_ocr_indices = set()
    merged_elements = []
    mineru_removed_overlap = 0

    for elem_idx, elem in enumerate(elements):
        elem_type = elem.get("type", "text")

        if elem_type == "list":
            blocks = elem.get("blocks") or []
            new_blocks = []

            for block_idx, block in enumerate(blocks):
                block_bbox = block.get("bbox")
                block_type = block.get("type", "text")
                block_unit_key = ("list_block", elem_idx, block_idx)
                assigned_indices = assigned_by_unit.get(block_unit_key, [])

                if block_bbox and block_type in text_type_set and assigned_indices:
                    mineru_removed_overlap += 1
                    for ocr_idx in assigned_indices:
                        consumed_ocr_indices.add(ocr_idx)
                        new_blocks.append(_to_list_text_block(grouped_ocr_elements[ocr_idx], block))
                else:
                    new_blocks.append(block)

            new_elem = {**elem, "blocks": new_blocks}
            new_elem["bbox"] = _compute_list_bbox_from_blocks(new_blocks, elem.get("bbox"))
            merged_elements.append(new_elem)
            continue

        elem_bbox = elem.get("bbox")
        is_text_elem = elem_type in text_type_set
        elem_unit_key = ("element", elem_idx, None)
        assigned_indices = assigned_by_unit.get(elem_unit_key, [])

        if is_text_elem and elem_bbox and assigned_indices:
            mineru_removed_overlap += 1
            for ocr_idx in assigned_indices:
                consumed_ocr_indices.add(ocr_idx)
                merged_elements.append(_to_text_element(grouped_ocr_elements[ocr_idx], elem))
            continue

        merged_elements.append(elem)

    for ocr_idx in unassigned_ocr:
        consumed_ocr_indices.add(ocr_idx)
        merged_elements.append(grouped_ocr_elements[ocr_idx])

    candidates = len(ocr_elements)
    groups = len(grouped_ocr_elements)
    merged_count = max(0, candidates - groups)

    return merged_elements, {
        "ocr_candidates": candidates,
        "ocr_groups": groups,
        "ocr_merged": merged_count,
        "ocr_added": groups,
        "mineru_removed_overlap": mineru_removed_overlap,
    }


def _collect_mineru_text_units(elements: list[dict[str, Any]], text_type_set: set[str]):
    units = []

    for elem_idx, elem in enumerate(elements):
        elem_type = elem.get("type", "text")

        if elem_type == "list":
            for block_idx, block in enumerate(elem.get("blocks") or []):
                block_bbox = block.get("bbox")
                block_type = block.get("type", "text")
                if block_bbox and block_type in text_type_set:
                    units.append(
                        {
                            "key": ("list_block", elem_idx, block_idx),
                            "bbox": block_bbox,
                            "block": block,
                            "type": block_type,
                        }
                    )
            continue

        elem_bbox = elem.get("bbox")
        if elem_bbox and elem_type in text_type_set:
            units.append(
                {
                    "key": ("element", elem_idx, None),
                    "bbox": elem_bbox,
                    "block": elem,
                    "type": elem_type,
                }
            )

    return units


def _assign_ocr_groups_to_mineru_units(
    grouped_ocr_elements: list[dict[str, Any]],
    mineru_units: list[dict[str, Any]],
    has_overlap: Callable[[list[float], list[float]], bool],
):
    assigned_by_unit = defaultdict(list)
    assigned_ocr = set()

    for ocr_idx, ocr_elem in enumerate(grouped_ocr_elements):
        ocr_bbox = ocr_elem.get("bbox")
        if not ocr_bbox:
            continue

        overlaps = [
            unit
            for unit in mineru_units
            if unit.get("bbox") and has_overlap(ocr_bbox, unit["bbox"])
        ]

        if not overlaps:
            continue

        overlaps.sort(key=lambda u: (u["bbox"][1], u["bbox"][0]))
        target_key = overlaps[0]["key"]
        assigned_by_unit[target_key].append(ocr_idx)
        assigned_ocr.add(ocr_idx)

    unassigned = [idx for idx in range(len(grouped_ocr_elements)) if idx not in assigned_ocr]
    return assigned_by_unit, unassigned


def _to_text_element(ocr_elem: dict[str, Any], elem_template: dict[str, Any]):
    return {
        **ocr_elem,
        "type": elem_template.get("type", ocr_elem.get("type", "text")),
        "source": "ocr",
    }


def _to_list_text_block(ocr_elem: dict[str, Any], block_template: dict[str, Any]):
    ocr_lines = ocr_elem.get("lines", [])

    return {
        "angle": ocr_elem.get("angle", block_template.get("angle", 0)),
        "bbox": ocr_elem.get("bbox", block_template.get("bbox", [0.0, 0.0, 0.0, 0.0])),
        "index": block_template.get("index", ocr_elem.get("index", 0)),
        "is_discarded": block_template.get("is_discarded", False),
        "lines": ocr_lines,
        "type": block_template.get("type", "text"),
        "source": "ocr",
        "ocr_score": ocr_elem.get("ocr_score", 0.0),
    }


def _compute_list_bbox_from_blocks(blocks: list[dict[str, Any]], fallback_bbox):
    bboxes = [block.get("bbox") for block in blocks if block.get("bbox")]
    if not bboxes:
        return fallback_bbox
    return _union_bboxes(bboxes)


def refine_ocr_text_elements(
    ocr_elements: list[dict[str, Any]],
    page_image,
    json_w: float,
    json_h: float,
    font_distance_threshold: float | None = None,
):
    if not ocr_elements:
        return []

    img_h, img_w = page_image.shape[:2]
    refined_elements = []

    for elem in ocr_elements:
        elem_bbox = elem.get("bbox")
        if not elem_bbox:
            refined_elements.append(dict(elem))
            continue

        elem_bbox_px = _json_bbox_to_pixel_bbox(elem_bbox, json_w, json_h, img_w, img_h)
        refined_elem_bbox_px, elem_pad_px = _refine_bbox_vertical(
            page_image,
            elem_bbox_px,
            font_distance_threshold=font_distance_threshold,
        )
        refined_elem_bbox = _pixel_bbox_to_json_bbox(refined_elem_bbox_px, json_w, json_h, img_w, img_h)
        elem_pad_bbox = _pixel_bbox_to_json_bbox(elem_pad_px, json_w, json_h, img_w, img_h)

        original_lines = elem.get("lines", [])
        refined_lines = []
        refined_line_bboxes = []

        for line in original_lines:
            line_bbox = line.get("bbox") or elem_bbox
            line_bbox_px = _json_bbox_to_pixel_bbox(line_bbox, json_w, json_h, img_w, img_h)
            refined_line_bbox_px, _ = _refine_bbox_vertical(
                page_image,
                line_bbox_px,
                font_distance_threshold=font_distance_threshold,
            )
            refined_line_bbox = _pixel_bbox_to_json_bbox(refined_line_bbox_px, json_w, json_h, img_w, img_h)

            new_spans = []
            for span in line.get("spans", []):
                span_bbox = span.get("bbox")
                if span_bbox:
                    new_span_bbox = [
                        refined_line_bbox[0],
                        refined_line_bbox[1],
                        refined_line_bbox[2],
                        refined_line_bbox[3],
                    ]
                else:
                    new_span_bbox = refined_line_bbox

                new_spans.append(
                    {
                        "bbox": new_span_bbox,
                        "content": span.get("content", ""),
                        "type": span.get("type", "text"),
                    }
                )

            if not new_spans:
                new_spans = [{"bbox": refined_line_bbox, "content": "", "type": "text"}]

            refined_lines.append(
                {
                    "bbox": refined_line_bbox,
                    "spans": new_spans,
                }
            )
            refined_line_bboxes.append(refined_line_bbox)

        final_bbox = _union_bboxes(refined_line_bboxes) if refined_line_bboxes else refined_elem_bbox

        new_elem = {
            **elem,
            "bbox": final_bbox,
            "lines": refined_lines if refined_lines else elem.get("lines", []),
            "ocr_bbox_original": elem_bbox,
            "ocr_bbox_pad": elem_pad_bbox,
            "ocr_bbox_refined": final_bbox,
            "source": "ocr",
        }
        refined_elements.append(new_elem)

    return refined_elements


def _group_ocr_elements_by_overlap(
    ocr_elements: list[dict[str, Any]],
    mineru_text_bboxes: list[list[float]],
    has_overlap: Callable[[list[float], list[float]], bool],
):
    if not ocr_elements:
        return []

    valid = [elem for elem in ocr_elements if elem.get("bbox")]
    n = len(valid)
    if n == 0:
        return []

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # OCR-OCR overlap grouping
    for i in range(n):
        bbox_i = valid[i]["bbox"]
        for j in range(i + 1, n):
            bbox_j = valid[j]["bbox"]
            if has_overlap(bbox_i, bbox_j):
                union(i, j)

    # OCR elements overlapping the same MinerU text box are grouped together
    overlaps_by_mineru = defaultdict(list)
    for ocr_idx, ocr_elem in enumerate(valid):
        ocr_bbox = ocr_elem.get("bbox")
        for mineru_idx, mineru_bbox in enumerate(mineru_text_bboxes):
            if has_overlap(ocr_bbox, mineru_bbox):
                overlaps_by_mineru[mineru_idx].append(ocr_idx)

    for overlapping_ocr_indices in overlaps_by_mineru.values():
        if len(overlapping_ocr_indices) < 2:
            continue
        base = overlapping_ocr_indices[0]
        for idx in overlapping_ocr_indices[1:]:
            union(base, idx)

    groups = defaultdict(list)
    for i, elem in enumerate(valid):
        groups[find(i)].append(elem)

    grouped = []
    for _, members in groups.items():
        grouped.append(_combine_ocr_members(members))

    grouped.sort(key=lambda e: (e["bbox"][1], e["bbox"][0]))
    return grouped


def _merge_ocr_line_fragments(ocr_elements: list[dict[str, Any]]):
    if not ocr_elements:
        return []

    sorted_elements = sorted(
        [e for e in ocr_elements if e.get("bbox")],
        key=lambda e: (e["bbox"][1], e["bbox"][0]),
    )
    if not sorted_elements:
        return []

    merged_groups = []
    current_group = [sorted_elements[0]]

    for elem in sorted_elements[1:]:
        prev = current_group[-1]
        if _should_merge_line_fragments(prev.get("bbox"), elem.get("bbox")):
            current_group.append(elem)
        else:
            merged_groups.append(current_group)
            current_group = [elem]

    merged_groups.append(current_group)

    return [_combine_ocr_members(group) for group in merged_groups]


def _should_merge_line_fragments(prev_bbox, curr_bbox):
    if not prev_bbox or not curr_bbox:
        return False

    prev_h = max(1e-6, prev_bbox[3] - prev_bbox[1])
    curr_h = max(1e-6, curr_bbox[3] - curr_bbox[1])
    max_h = max(prev_h, curr_h)

    prev_center_y = (prev_bbox[1] + prev_bbox[3]) / 2
    curr_center_y = (curr_bbox[1] + curr_bbox[3]) / 2
    same_y_band = abs(prev_center_y - curr_center_y) <= max_h * 0.55

    horizontal_gap = curr_bbox[0] - prev_bbox[2]
    close_in_x = -max_h * 0.2 <= horizontal_gap <= max_h * 1.5

    return same_y_band and close_in_x


def _combine_ocr_members(members: list[dict[str, Any]]):
    ordered_members = sorted(members, key=lambda e: (e["bbox"][1], e["bbox"][0]))

    raw_lines = []
    all_bboxes = []
    best_score = 0.0

    for elem in ordered_members:
        elem_bbox = elem.get("bbox")
        if elem_bbox:
            all_bboxes.append(elem_bbox)

        best_score = max(best_score, float(elem.get("ocr_score", 0.0)))

        for line in elem.get("lines", []):
            line_bbox = line.get("bbox") or elem_bbox
            spans = [
                {
                    "bbox": span.get("bbox") or line_bbox,
                    "content": span.get("content", ""),
                    "type": span.get("type", "text"),
                }
                for span in line.get("spans", [])
            ]

            if not spans:
                spans = [{"bbox": line_bbox, "content": "", "type": "text"}]

            raw_lines.append({"bbox": line_bbox, "spans": spans})

    merged_lines = _merge_lines_by_geometry(raw_lines)
    line_bboxes = [line["bbox"] for line in merged_lines if line.get("bbox")]
    final_bbox = _union_bboxes(line_bboxes if line_bboxes else all_bboxes)

    return {
        "angle": 0,
        "bbox": final_bbox,
        "index": 0,
        "is_discarded": False,
        "lines": merged_lines,
        "type": "text",
        "ocr_score": best_score,
        "source": "ocr",
    }


def _merge_lines_by_geometry(lines: list[dict[str, Any]]):
    if not lines:
        return []

    sorted_lines = sorted(lines, key=lambda l: (l["bbox"][1], l["bbox"][0]))
    merged = []

    for line in sorted_lines:
        if not merged:
            merged.append({"bbox": list(line["bbox"]), "spans": list(line.get("spans", []))})
            continue

        prev = merged[-1]
        prev_bbox = prev["bbox"]
        curr_bbox = line["bbox"]

        prev_h = max(1e-6, prev_bbox[3] - prev_bbox[1])
        curr_h = max(1e-6, curr_bbox[3] - curr_bbox[1])
        max_h = max(prev_h, curr_h)

        prev_center_y = (prev_bbox[1] + prev_bbox[3]) / 2
        curr_center_y = (curr_bbox[1] + curr_bbox[3]) / 2
        same_row = abs(prev_center_y - curr_center_y) <= max_h * 0.5

        x_gap = curr_bbox[0] - prev_bbox[2]
        close_x = -max_h * 0.2 <= x_gap <= max_h * 1.4

        if same_row and close_x:
            prev["bbox"] = _union_bboxes([prev_bbox, curr_bbox])
            prev["spans"].extend(line.get("spans", []))
            prev["spans"].sort(key=lambda s: (s.get("bbox", [0, 0, 0, 0])[0], s.get("bbox", [0, 0, 0, 0])[1]))
        else:
            merged.append({"bbox": list(curr_bbox), "spans": list(line.get("spans", []))})

    return merged


def _build_row_font_flags(page_image, x1, x2, y1, y2, font_color, min_pixel_ratio, font_distance_threshold):
    h, w = page_image.shape[:2]
    x1 = max(0, min(int(x1), w))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h))
    y2 = max(0, min(int(y2), h))

    if x2 <= x1 or y2 <= y1:
        return []

    roi = page_image[y1:y2, x1:x2]
    if roi.size == 0:
        return []

    diff = np.linalg.norm(roi.astype(np.float32) - np.array(font_color, dtype=np.float32), axis=2)
    row_match_counts = np.sum(diff < font_distance_threshold, axis=1)
    min_pixels = max(1, int((x2 - x1) * min_pixel_ratio))

    return [count >= min_pixels for count in row_match_counts]


def _build_col_font_flags(page_image, x1, x2, y1, y2, font_color, min_pixel_ratio, font_distance_threshold):
    h, w = page_image.shape[:2]
    x1 = max(0, min(int(x1), w))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h))
    y2 = max(0, min(int(y2), h))

    if x2 <= x1 or y2 <= y1:
        return []

    roi = page_image[y1:y2, x1:x2]
    if roi.size == 0:
        return []

    diff = np.linalg.norm(roi.astype(np.float32) - np.array(font_color, dtype=np.float32), axis=2)
    col_match_counts = np.sum(diff < font_distance_threshold, axis=0)
    min_pixels = max(1, int((y2 - y1) * min_pixel_ratio))

    return [count >= min_pixels for count in col_match_counts]


def _refine_bbox_vertical(
    page_image,
    bbox_px,
    font_distance_threshold: float | None = None,
):
    h, w = page_image.shape[:2]
    x1, y1, x2, y2 = _clamp_pixel_bbox(bbox_px, w, h)
    if x2 <= x1 or y2 <= y1:
        return [x1, y1, x2, y2], [x1, y1, x2, y2]

    pad_bbox = _expand_bbox_with_pad([x1, y1, x2, y2], w, h)
    pad_x1, pad_y1, pad_x2, pad_y2 = pad_bbox

    bg_color = extract_background_color(page_image, [x1, y1, x2, y2])
    font_color, _, _ = extract_font_color(page_image, [x1, y1, x2, y2], bg_color)

    ratio = OCR_REFINE_MIN_PIXEL_RATIO
    font_threshold = OCR_FONT_DISTANCE_THRESHOLD if font_distance_threshold is None else float(font_distance_threshold)
    row_flags = _build_row_font_flags(
        page_image,
        pad_x1,
        pad_x2,
        pad_y1,
        pad_y2,
        font_color,
        ratio,
        font_threshold,
    )
    col_flags = _build_col_font_flags(
        page_image,
        pad_x1,
        pad_x2,
        pad_y1,
        pad_y2,
        font_color,
        ratio,
        font_threshold,
    )

    refined_x1, refined_y1, refined_x2, refined_y2 = x1, y1, x2, y2

    if row_flags:
        orig_top_idx = y1 - pad_y1
        orig_bottom_idx = y2 - pad_y1

        in_box_flags = row_flags[orig_top_idx:orig_bottom_idx]
        true_positions = [idx for idx, has_font in enumerate(in_box_flags) if has_font]

        if true_positions:
            first_idx = orig_top_idx + true_positions[0]
            last_idx = orig_top_idx + true_positions[-1]

            refined_y1 = pad_y1 + first_idx
            refined_y2 = pad_y1 + last_idx + 1

            trimmed_top = refined_y1 > y1
            trimmed_bottom = refined_y2 < y2

            if not trimmed_top:
                scan = first_idx - 1
                while scan >= 0 and row_flags[scan]:
                    refined_y1 = pad_y1 + scan
                    scan -= 1

            if not trimmed_bottom:
                scan = last_idx + 1
                while scan < len(row_flags) and row_flags[scan]:
                    refined_y2 = pad_y1 + scan + 1
                    scan += 1

            while refined_y1 + 1 < refined_y2 and not row_flags[refined_y1 - pad_y1]:
                refined_y1 += 1

            while refined_y1 > pad_y1 and row_flags[refined_y1 - 1 - pad_y1]:
                refined_y1 -= 1

            while refined_y2 - 1 > refined_y1 and not row_flags[refined_y2 - 1 - pad_y1]:
                refined_y2 -= 1

            while refined_y2 < pad_y2 and row_flags[refined_y2 - pad_y1]:
                refined_y2 += 1

    if col_flags:
        orig_left_idx = x1 - pad_x1
        orig_right_idx = x2 - pad_x1

        in_box_flags = col_flags[orig_left_idx:orig_right_idx]
        true_positions = [idx for idx, has_font in enumerate(in_box_flags) if has_font]

        if true_positions:
            first_idx = orig_left_idx + true_positions[0]
            last_idx = orig_left_idx + true_positions[-1]

            refined_x1 = pad_x1 + first_idx
            refined_x2 = pad_x1 + last_idx + 1

            trimmed_left = refined_x1 > x1
            trimmed_right = refined_x2 < x2

            if not trimmed_left:
                scan = first_idx - 1
                while scan >= 0 and col_flags[scan]:
                    refined_x1 = pad_x1 + scan
                    scan -= 1

            if not trimmed_right:
                scan = last_idx + 1
                while scan < len(col_flags) and col_flags[scan]:
                    refined_x2 = pad_x1 + scan + 1
                    scan += 1

            while refined_x1 + 1 < refined_x2 and not col_flags[refined_x1 - pad_x1]:
                refined_x1 += 1

            while refined_x1 > pad_x1 and col_flags[refined_x1 - 1 - pad_x1]:
                refined_x1 -= 1

            while refined_x2 - 1 > refined_x1 and not col_flags[refined_x2 - 1 - pad_x1]:
                refined_x2 -= 1

            while refined_x2 < pad_x2 and col_flags[refined_x2 - pad_x1]:
                refined_x2 += 1

    refined_x1 = max(0, min(refined_x1, w))
    refined_x2 = max(0, min(refined_x2, w))
    refined_y1 = max(0, min(refined_y1, h))
    refined_y2 = max(0, min(refined_y2, h))

    if refined_x2 <= refined_x1 or refined_y2 <= refined_y1:
        return [x1, y1, x2, y2], pad_bbox

    return [refined_x1, refined_y1, refined_x2, refined_y2], pad_bbox


def _expand_bbox_with_pad(bbox_px, img_w, img_h):
    x1, y1, x2, y2 = bbox_px
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    pad_x = max(OCR_BBOX_MIN_PAD_PIXELS, int(round(height * OCR_BBOX_PAD_RATIO)))
    pad_y = max(OCR_BBOX_MIN_PAD_PIXELS, int(round(height * OCR_BBOX_PAD_RATIO)))

    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(img_w, x2 + pad_x),
        min(img_h, y2 + pad_y),
    ]


def _clamp_pixel_bbox(bbox_px, img_w, img_h):
    x1, y1, x2, y2 = bbox_px
    return [
        max(0, min(int(round(x1)), img_w)),
        max(0, min(int(round(y1)), img_h)),
        max(0, min(int(round(x2)), img_w)),
        max(0, min(int(round(y2)), img_h)),
    ]


def _json_bbox_to_pixel_bbox(bbox_json, json_w, json_h, img_w, img_h):
    x1 = bbox_json[0] * (img_w / json_w)
    y1 = bbox_json[1] * (img_h / json_h)
    x2 = bbox_json[2] * (img_w / json_w)
    y2 = bbox_json[3] * (img_h / json_h)
    return _clamp_pixel_bbox([x1, y1, x2, y2], img_w, img_h)


def _scale_pixel_bbox(bbox_px, scale_x: float, scale_y: float, img_w: int, img_h: int):
    x1, y1, x2, y2 = bbox_px
    scaled = [
        x1 * scale_x,
        y1 * scale_y,
        x2 * scale_x,
        y2 * scale_y,
    ]
    return _clamp_pixel_bbox(scaled, img_w, img_h)


def _pixel_bbox_to_json_bbox(bbox_px, json_w, json_h, img_w, img_h):
    x1 = bbox_px[0] * (json_w / img_w)
    y1 = bbox_px[1] * (json_h / img_h)
    x2 = bbox_px[2] * (json_w / img_w)
    y2 = bbox_px[3] * (json_h / img_h)
    return [x1, y1, x2, y2]


def _union_bboxes(bboxes):
    valid = [b for b in bboxes if b]
    if not valid:
        return [0.0, 0.0, 0.0, 0.0]

    x1 = min(b[0] for b in valid)
    y1 = min(b[1] for b in valid)
    x2 = max(b[2] for b in valid)
    y2 = max(b[3] for b in valid)
    return [x1, y1, x2, y2]


def _iter_ocr_results(raw_result):
    structured_items = list(_iter_structured_ocr_results(raw_result))
    if structured_items:
        for item in structured_items:
            yield item
        return

    entries = []
    _collect_legacy_ocr_entries(raw_result, entries)
    for entry in entries:
        points, info = entry
        if not isinstance(info, (list, tuple)) or not info:
            continue

        text = str(info[0]).strip() if len(info) >= 1 else ""
        score = float(info[1]) if len(info) >= 2 else 0.0

        if text:
            yield points, text, score


def _iter_structured_ocr_results(node):
    if node is None:
        return

    if isinstance(node, (list, tuple)):
        for item in node:
            for parsed in _iter_structured_ocr_results(item):
                yield parsed
        return

    if not hasattr(node, "get"):
        return

    rec_texts = _to_list(node.get("rec_texts"))
    rec_scores = _to_list(node.get("rec_scores"))
    polygons = _to_list(node.get("dt_polys"))
    if not polygons:
        polygons = _to_list(node.get("rec_polys"))
    if not polygons:
        polygons = _to_list(node.get("rec_boxes"))

    if not rec_texts or not polygons:
        return

    for i, text_value in enumerate(rec_texts):
        if i >= len(polygons):
            break

        text = str(text_value).strip()
        if not text:
            continue

        score_value = rec_scores[i] if i < len(rec_scores) else 0.0
        try:
            score = float(score_value)
        except (TypeError, ValueError):
            score = 0.0

        yield polygons[i], text, score


def _collect_legacy_ocr_entries(node, out_entries):
    if isinstance(node, list):
        if len(node) == 2 and _is_points(node[0]):
            out_entries.append(node)
            return

        for item in node:
            _collect_legacy_ocr_entries(item, out_entries)


def _is_xy_point(value):
    try:
        if len(value) < 2:
            return False
        float(value[0])
        float(value[1])
        return True
    except Exception:
        return False


def _is_points(value):
    if not hasattr(value, "__len__") or len(value) == 0:
        return False

    first = value[0]
    return _is_xy_point(first)


def _to_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    return []


def _polygon_to_bbox(points, img_w: int, img_h: int):
    xs = []
    ys = []
    for point in points:
        if not _is_xy_point(point):
            continue
        xs.append(float(point[0]))
        ys.append(float(point[1]))

    if not xs or not ys:
        return None

    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(img_w), max(xs))
    y2 = min(float(img_h), max(ys))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]
