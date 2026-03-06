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

OCR_BBOX_PAD_RATIO = 0.10
OCR_BBOX_MIN_PAD_PIXELS = 2
ROW_FONT_DISTANCE_THRESHOLD = 60.0
ROW_FONT_MIN_PIXEL_RATIO = 0.005
ROW_NON_BG_DISTANCE_THRESHOLD = 55.0


class PaddleOCREngine:
    _MODEL_ROOT_ENV = "MINERU_OCR_MODEL_ROOT"
    _MODEL_REQUIRED_DIRS = ("det", "rec", "cls")
    _MODEL_FILE_CANDIDATES = ("inference.pdmodel", "inference.json")
    _PARAM_FILE_CANDIDATES = ("inference.pdiparams",)

    def __init__(
        self,
        lang: str = "ch",
        use_angle_cls: bool = True,
        device_policy: str = "auto",
        model_root: str | None = None,
        offline_only: bool = True,
    ):
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.device_policy = device_policy
        self.model_root = model_root
        self.offline_only = offline_only
        self._ocr = None

        if self.device_policy not in {"auto", "gpu", "cpu"}:
            raise ValueError("device_policy must be one of: auto, gpu, cpu")

    def _resolve_device_order(self):
        if self.device_policy == "gpu":
            return ["gpu"]
        if self.device_policy == "cpu":
            return ["cpu"]
        return ["gpu", "cpu"]

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

        meipass_root = getattr(sys, "_MEIPASS", None)
        if meipass_root:
            bundled_root = Path(meipass_root).resolve() / "models" / "paddleocr"
            if bundled_root.exists():
                return bundled_root, "pyinstaller:_MEIPASS/models/paddleocr"

        exe_parent = Path(sys.executable).resolve().parent
        exe_internal_root = exe_parent / "_internal" / "models" / "paddleocr"
        if exe_internal_root.exists():
            return exe_internal_root, "executable:_internal/models/paddleocr"

        exe_root = exe_parent / "models" / "paddleocr"
        if exe_root.exists():
            return exe_root, "executable:models/paddleocr"

        source_root = Path(__file__).resolve().parents[1] / "models" / "paddleocr"
        return source_root, "source:models/paddleocr"

    def _assert_local_models_integrity(self, model_root: Path):
        if not model_root.exists() or not model_root.is_dir():
            raise RuntimeError(
                f"[OCR] Local model root not found: {model_root}. "
                "Expected a models/paddleocr directory with per-language det/rec/cls subfolders."
            )

        lang_root = model_root / self.lang
        if not lang_root.exists() or not lang_root.is_dir():
            raise RuntimeError(
                f"[OCR] Missing language model directory: {lang_root}. "
                f"Expected layout: {model_root}/<lang>/det|rec|cls"
            )

        missing_dirs = [name for name in self._MODEL_REQUIRED_DIRS if not (lang_root / name).is_dir()]
        if missing_dirs:
            raise RuntimeError(
                f"[OCR] Missing model subdirectories for lang='{self.lang}': {missing_dirs}. "
                f"Expected under: {lang_root}"
            )

        for subdir in self._MODEL_REQUIRED_DIRS:
            model_dir = lang_root / subdir
            has_model_file = any((model_dir / name).exists() for name in self._MODEL_FILE_CANDIDATES)
            has_param_file = any((model_dir / name).exists() for name in self._PARAM_FILE_CANDIDATES)
            if not has_model_file or not has_param_file:
                raise RuntimeError(
                    f"[OCR] Incomplete local model files in {model_dir}. "
                    f"Required one of {self._MODEL_FILE_CANDIDATES} and one of {self._PARAM_FILE_CANDIDATES}."
                )

    def _build_model_dirs(self, model_root: Path):
        lang_root = model_root / self.lang
        return {
            "text_detection_model_dir": str(lang_root / "det"),
            "text_recognition_model_dir": str(lang_root / "rec"),
            "textline_orientation_model_dir": str(lang_root / "cls"),
        }

    def _constructor_attempts_for_device(self, device: str, model_dirs: dict[str, str]):
        base_kwargs = {
            "enable_mkldnn": False,
            "enable_hpi": False,
            "use_tensorrt": False,
            "enable_cinn": False,
            "return_word_box": True,
            **model_dirs,
        }

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

        device_variants = [{"device": device}]

        for device_kwargs in device_variants:
            for attempt in compatibility_attempts:
                yield {**base_kwargs, **device_kwargs, **attempt}

        if self.offline_only and "download" in base_kwargs:
            fallback_base = dict(base_kwargs)
            fallback_base.pop("download", None)
            for device_kwargs in device_variants:
                for attempt in compatibility_attempts:
                    yield {**fallback_base, **device_kwargs, **attempt}

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
        self._assert_local_models_integrity(model_root)
        model_dirs = self._build_model_dirs(model_root)

        requested_order = self._resolve_device_order()
        gpu_available = self._is_gpu_available()
        if not gpu_available and "gpu" in requested_order:
            if self.device_policy == "gpu":
                raise RuntimeError("[OCR] GPU device policy requested but CUDA GPU is not available.")
            print("[OCR] GPU not available; skipping GPU initialization and using CPU.")
        device_order = [d for d in requested_order if d != "gpu" or gpu_available]

        print(f"[OCR] model_root={model_root} ({model_root_source})")
        print(f"[OCR] device_order={device_order}")

        last_error = None
        gpu_error = None
        gpu_failed = False
        for device in device_order:
            for kwargs in self._constructor_attempts_for_device(device, model_dirs):
                try:
                    self._ocr = PaddleOCR(**kwargs)
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

    def extract_text_elements(self, page_image, json_w: float, json_h: float):
        self._ensure_initialized()

        if page_image is None or page_image.size == 0:
            return []

        img_h, img_w = page_image.shape[:2]
        if img_h == 0 or img_w == 0:
            return []

        raw_result = self._run_ocr(page_image)

        raw_elements = []
        for idx, (points, text, score) in enumerate(_iter_ocr_results(raw_result), start=1):
            if not text:
                continue

            bbox_px = _polygon_to_bbox(points, img_w, img_h)
            if not bbox_px:
                continue

            bbox_json = _pixel_bbox_to_json_bbox(bbox_px, json_w, json_h, img_w, img_h)
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
        refined_elements = refine_ocr_text_elements(merged_line_elements, page_image, json_w, json_h)

        for idx, elem in enumerate(refined_elements, start=1):
            elem["index"] = idx

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
        refined_elem_bbox_px, elem_pad_px = _refine_bbox_vertical(page_image, elem_bbox_px)
        refined_elem_bbox = _pixel_bbox_to_json_bbox(refined_elem_bbox_px, json_w, json_h, img_w, img_h)
        elem_pad_bbox = _pixel_bbox_to_json_bbox(elem_pad_px, json_w, json_h, img_w, img_h)

        original_lines = elem.get("lines", [])
        refined_lines = []
        refined_line_bboxes = []

        for line in original_lines:
            line_bbox = line.get("bbox") or elem_bbox
            line_bbox_px = _json_bbox_to_pixel_bbox(line_bbox, json_w, json_h, img_w, img_h)
            refined_line_bbox_px, _ = _refine_bbox_vertical(page_image, line_bbox_px)
            refined_line_bbox = _pixel_bbox_to_json_bbox(refined_line_bbox_px, json_w, json_h, img_w, img_h)

            new_spans = []
            for span in line.get("spans", []):
                span_bbox = span.get("bbox")
                if span_bbox:
                    new_span_bbox = [
                        span_bbox[0],
                        refined_line_bbox[1],
                        span_bbox[2],
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


def _build_row_font_flags(page_image, x1, x2, y1, y2, font_color):
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
    row_match_counts = np.sum(diff < ROW_FONT_DISTANCE_THRESHOLD, axis=1)
    min_pixels = max(1, int((x2 - x1) * ROW_FONT_MIN_PIXEL_RATIO))

    return [count >= min_pixels for count in row_match_counts]


def _refine_bbox_vertical(page_image, bbox_px):
    h, w = page_image.shape[:2]
    x1, y1, x2, y2 = _clamp_pixel_bbox(bbox_px, w, h)
    if x2 <= x1 or y2 <= y1:
        return [x1, y1, x2, y2], [x1, y1, x2, y2]

    pad_bbox = _expand_bbox_with_pad([x1, y1, x2, y2], w, h)
    row_flags = _build_row_font_cache(page_image, [x1, y1, x2, y2], pad_bbox)
    if not row_flags:
        return [x1, y1, x2, y2], pad_bbox

    bg_color = extract_background_color(page_image, [x1, y1, x2, y2])
    font_color, _, _ = extract_font_color(page_image, [x1, y1, x2, y2], bg_color)

    pad_y1 = pad_bbox[1]
    orig_top_idx = y1 - pad_y1
    orig_bottom_idx = y2 - pad_y1

    in_box_flags = row_flags[orig_top_idx:orig_bottom_idx]
    true_positions = [idx for idx, has_font in enumerate(in_box_flags) if has_font]
    if not true_positions:
        return [x1, y1, x2, y2], pad_bbox

    first_idx = orig_top_idx + true_positions[0]
    last_idx = orig_top_idx + true_positions[-1]

    refined_y1 = pad_y1 + first_idx
    refined_y2 = pad_y1 + last_idx + 1

    trimmed_top = refined_y1 > y1
    trimmed_bottom = refined_y2 < y2

    # second wave extend for untouched side only
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

    # Remove margin-based expansion and align boundaries with precomputed row flags.
    full_row_flags = _build_row_font_flags(page_image, x1, x2, pad_bbox[1], pad_bbox[3], font_color)

    while refined_y1 + 1 < refined_y2 and not full_row_flags[refined_y1 - pad_bbox[1]]:
        refined_y1 += 1

    while refined_y1 > pad_bbox[1] and full_row_flags[refined_y1 - 1 - pad_bbox[1]]:
        refined_y1 -= 1

    while refined_y2 - 1 > refined_y1 and not full_row_flags[refined_y2 - 1 - pad_bbox[1]]:
        refined_y2 -= 1

    while refined_y2 < pad_bbox[3] and full_row_flags[refined_y2 - pad_bbox[1]]:
        refined_y2 += 1

    refined_y1 = max(0, min(refined_y1, h))
    refined_y2 = max(0, min(refined_y2, h))
    if refined_y2 <= refined_y1:
        return [x1, y1, x2, y2], pad_bbox

    return [x1, refined_y1, x2, refined_y2], pad_bbox


def _build_row_font_cache(page_image, original_bbox_px, pad_bbox_px):
    ox1, oy1, ox2, oy2 = original_bbox_px
    px1, py1, px2, py2 = pad_bbox_px

    if px2 <= px1 or py2 <= py1:
        return []

    bg_color = extract_background_color(page_image, original_bbox_px)
    font_color, _, _ = extract_font_color(page_image, original_bbox_px, bg_color)
    return _build_row_font_flags(page_image, px1, px2, py1, py2, font_color)


def _expand_bbox_with_pad(bbox_px, img_w, img_h):
    x1, y1, x2, y2 = bbox_px
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    pad_x = max(OCR_BBOX_MIN_PAD_PIXELS, int(round(width * OCR_BBOX_PAD_RATIO)))
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
