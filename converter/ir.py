from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

IR_ELEMENT_TYPES = {"text", "image"}
IR_ALIGNMENTS = {"left", "center", "right", "justify"}


@dataclass(frozen=True)
class TextRunIR:
    text: str
    bbox: list[float]
    line_index: int = 0
    style: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextIR:
    type: str
    bbox: list[float]
    text: str
    source: str
    order: list[float]
    style: dict[str, Any]
    is_discarded: bool = False
    is_watermark: bool = False
    group_id: str | None = None
    text_runs: list[TextRunIR] | None = None
    lines: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class ImageIR:
    type: str
    bbox: list[float]
    source: str
    order: list[float]
    style: dict[str, Any]
    is_discarded: bool = False
    is_watermark: bool = False
    group_id: str | None = None
    text_elements: list[TextIR] = field(default_factory=list)
    crop_pixels: Any | None = None


ElementIR = TextIR | ImageIR


@dataclass(frozen=True)
class PageIR:
    page_index: int
    page_size: tuple[float, float] | None
    elements: list[ElementIR]


@dataclass(frozen=True)
class DocumentIR:
    pages: list[PageIR]


def default_style() -> dict[str, Any]:
    return {
        "bold": False,
        "font_size": None,
        "align": "left",
    }


def normalize_bbox(bbox: Any) -> list[float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox: {bbox}")

    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid bbox values: {bbox}") from exc

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox geometry: {bbox}")

    return [x1, y1, x2, y2]


def normalize_style(style: Any) -> dict[str, Any]:
    style_dict = dict(style or {})
    result = default_style()

    if "bold" in style_dict:
        result["bold"] = bool(style_dict["bold"])

    if "font_size" in style_dict and style_dict["font_size"] is not None:
        try:
            font_size = float(style_dict["font_size"])
            result["font_size"] = font_size if font_size > 0 else None
        except (TypeError, ValueError):
            result["font_size"] = None

    if "align" in style_dict and style_dict["align"] is not None:
        align = str(style_dict["align"]).lower()
        result["align"] = align if align in IR_ALIGNMENTS else "left"

    return result


def _fallback_order(bbox: list[float]) -> list[float]:
    return [bbox[1], bbox[0]]


def compose_text_from_lines_or_spans(
    lines: Any,
    spans: Any,
    fallback_text: Any = None,
) -> str:
    if isinstance(fallback_text, str) and fallback_text:
        return fallback_text

    if isinstance(lines, list) and lines:
        line_texts: list[str] = []
        for line in lines:
            if not isinstance(line, dict):
                continue
            line_spans = line.get("spans") or []
            line_text = "".join(str(span.get("content", "")) for span in line_spans if isinstance(span, dict))
            if line_text:
                line_texts.append(line_text)
        if line_texts:
            return "\n".join(line_texts)

    if isinstance(spans, list) and spans:
        text = "".join(str(span.get("content", "")) for span in spans if isinstance(span, dict))
        if text:
            return text

    return str(fallback_text or "")


def _coerce_text_run(run: Any, index: int = 0) -> TextRunIR:
    if isinstance(run, TextRunIR):
        return TextRunIR(
            text=str(run.text),
            bbox=normalize_bbox(run.bbox),
            line_index=int(run.line_index),
            style=normalize_style(run.style),
        )

    if not isinstance(run, dict):
        raise ValueError("TextRunIR must be a dict or TextRunIR")
    if run.get("bbox") is None:
        raise ValueError("TextRunIR.bbox is required")

    return TextRunIR(
        text=str(run.get("text", "")),
        bbox=normalize_bbox(run.get("bbox")),
        line_index=int(run.get("line_index", index)),
        style=normalize_style(run.get("style") or {}),
    )


def rebuild_text_from_runs(text_runs: list[TextRunIR]) -> str:
    if not text_runs:
        return ""

    normalized_runs = [_coerce_text_run(run, index) for index, run in enumerate(text_runs)]

    grouped: dict[int, list[TextRunIR]] = {}
    for run in normalized_runs:
        grouped.setdefault(int(run.line_index), []).append(run)

    line_texts: list[str] = []
    for line_index in sorted(grouped.keys()):
        runs = sorted(
            grouped[line_index],
            key=lambda run: (run.bbox[0], run.bbox[1]),
        )
        line_texts.append("".join(str(run.text) for run in runs))

    return "\n".join(line_texts)


def normalize_text_runs(text_runs: Any) -> list[TextRunIR] | None:
    if text_runs is None:
        return None

    if not isinstance(text_runs, list):
        raise ValueError("TextIR.text_runs must be a list or None")

    normalized = [_coerce_text_run(run, index) for index, run in enumerate(text_runs)]
    normalized.sort(key=lambda run: (run.line_index, run.bbox[1], run.bbox[0]))
    return normalized


def _build_text_runs_from_lines_or_spans(
    element: dict[str, Any],
    bbox: list[float],
    style: dict[str, Any],
) -> list[TextRunIR] | None:
    lines = element.get("lines")
    if isinstance(lines, list) and lines:
        runs: list[TextRunIR] = []
        for line_index, line in enumerate(lines):
            if not isinstance(line, dict):
                continue
            line_bbox = normalize_bbox(line.get("bbox") or bbox)
            spans = line.get("spans") or []
            if not spans:
                line_text = str(line.get("text", ""))
                if line_text:
                    runs.append(
                        TextRunIR(
                            text=line_text,
                            bbox=line_bbox,
                            line_index=line_index,
                            style=style,
                        )
                    )
                continue
            for span in spans:
                if not isinstance(span, dict):
                    continue
                span_text = str(span.get("content", ""))
                if not span_text:
                    continue
                span_bbox = normalize_bbox(span.get("bbox") or line_bbox)
                span_style = normalize_style(span.get("style") or style)
                runs.append(
                    TextRunIR(
                        text=span_text,
                        bbox=span_bbox,
                        line_index=line_index,
                        style=span_style,
                    )
                )
        if runs:
            return normalize_text_runs(runs)

    spans = element.get("spans")
    if isinstance(spans, list) and spans:
        span_runs: list[TextRunIR] = []
        for span in spans:
            if not isinstance(span, dict):
                continue
            span_text = str(span.get("content", ""))
            if not span_text:
                continue
            span_bbox = normalize_bbox(span.get("bbox") or bbox)
            span_runs.append(
                TextRunIR(
                    text=span_text,
                    bbox=span_bbox,
                    line_index=0,
                    style=normalize_style(span.get("style") or style),
                )
            )
        if span_runs:
            return normalize_text_runs(span_runs)

    return None


def _normalize_lines(lines: Any, fallback_bbox: list[float]) -> list[dict[str, Any]] | None:
    if lines is None:
        return None
    if not isinstance(lines, list):
        raise ValueError("TextIR.lines must be a list or None")

    normalized_lines: list[dict[str, Any]] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        line_bbox = normalize_bbox(line.get("bbox") or fallback_bbox)
        raw_spans = line.get("spans") or []
        normalized_spans: list[dict[str, Any]] = []
        for span in raw_spans:
            if not isinstance(span, dict):
                continue
            span_text = str(span.get("content", ""))
            span_bbox = normalize_bbox(span.get("bbox") or line_bbox)
            normalized_spans.append(
                {
                    "bbox": span_bbox,
                    "content": span_text,
                    "type": str(span.get("type", "text") or "text"),
                    "style": normalize_style(span.get("style") or {}),
                }
            )

        if normalized_spans:
            normalized_lines.append({"bbox": line_bbox, "spans": normalized_spans})
        else:
            normalized_lines.append({"bbox": line_bbox, "spans": []})

    return normalized_lines


def _normalize_image_text_elements(raw_text_elements: Any) -> list[TextIR]:
    if raw_text_elements is None:
        return []
    if not isinstance(raw_text_elements, list):
        raise ValueError("ImageIR.text_elements must be a list")

    normalized: list[TextIR] = []
    for entry in raw_text_elements:
        normalized_entry = normalize_element_ir(entry)
        if not isinstance(normalized_entry, TextIR):
            raise ValueError("ImageIR.text_elements must contain text IR elements")
        normalized.append(normalized_entry)
    return normalized


def materialize_text_runs_for_element(element: Any) -> ElementIR:
    normalized = normalize_element_ir(element)
    if isinstance(normalized, ImageIR):
        return normalized

    if isinstance(normalized.text_runs, list):
        return normalized

    source_element = element if isinstance(element, dict) else {}
    materialized_runs = _build_text_runs_from_lines_or_spans(
        source_element,
        normalized.bbox,
        normalized.style or default_style(),
    )

    if not materialized_runs:
        text_value = str(normalized.text or "")
        if not text_value:
            return normalized
        line_texts = text_value.split("\n")
        materialized_runs = [
            TextRunIR(
                text=line_text,
                bbox=list(normalized.bbox),
                line_index=idx,
                style=dict(normalized.style or {}),
            )
            for idx, line_text in enumerate(line_texts)
            if line_text
        ]

    if not materialized_runs:
        return normalized

    rebuilt = rebuild_text_from_runs(materialized_runs)
    return replace(normalized, text_runs=materialized_runs, text=rebuilt)


def materialize_text_runs_for_elements(elements: list[Any]) -> list[ElementIR]:
    return [materialize_text_runs_for_element(element) for element in elements]


def normalize_element_ir(element: Any) -> ElementIR:
    if isinstance(element, TextIR):
        text_runs = normalize_text_runs(element.text_runs)
        text = str(element.text or "")
        if text_runs is not None and not text:
            text = rebuild_text_from_runs(text_runs)

        normalized_bbox = normalize_bbox(element.bbox)
        normalized_lines = _normalize_lines(element.lines, normalized_bbox)
        text_ir = TextIR(
            type="text",
            bbox=normalized_bbox,
            text=text,
            source=str(element.source or "unknown"),
            order=[float(element.order[0]), float(element.order[1])] if len(element.order) >= 2 else _fallback_order(normalized_bbox),
            style=normalize_style(element.style),
            is_discarded=bool(element.is_discarded),
            is_watermark=bool(getattr(element, "is_watermark", False)),
            group_id=str(element.group_id) if element.group_id is not None else None,
            text_runs=text_runs,
            lines=normalized_lines,
        )

        if not text_ir.text and not text_ir.text_runs:
            raise ValueError("Text IR element requires text or text_runs")
        return text_ir

    if isinstance(element, ImageIR):
        return ImageIR(
            type="image",
            bbox=normalize_bbox(element.bbox),
            source=str(element.source or "unknown"),
            order=[float(element.order[0]), float(element.order[1])] if len(element.order) >= 2 else _fallback_order(normalize_bbox(element.bbox)),
            style=normalize_style(element.style),
            is_discarded=bool(element.is_discarded),
            is_watermark=bool(getattr(element, "is_watermark", False)),
            group_id=str(element.group_id) if element.group_id is not None else None,
            text_elements=_normalize_image_text_elements(element.text_elements),
            crop_pixels=element.crop_pixels,
        )

    if not isinstance(element, dict):
        raise ValueError("IR element must be a dict/TextIR/ImageIR")

    elem_type = element.get("type")
    if elem_type not in IR_ELEMENT_TYPES:
        raise ValueError(f"Unsupported IR element type: {elem_type}")

    bbox = normalize_bbox(element.get("bbox"))
    style = normalize_style(element.get("style"))

    raw_order = element.get("order")
    if isinstance(raw_order, (list, tuple)) and len(raw_order) >= 2:
        order = [float(raw_order[0]), float(raw_order[1])]
    else:
        order = _fallback_order(bbox)

    source = str(element.get("source", "unknown"))
    group_id = element.get("group_id")
    is_discarded = bool(element.get("is_discarded", False))
    is_watermark = bool(element.get("is_watermark", False))

    if elem_type == "text":
        has_text_runs_key = "text_runs" in element
        if has_text_runs_key:
            text_runs = normalize_text_runs(element.get("text_runs"))
        else:
            text_runs = _build_text_runs_from_lines_or_spans(element, bbox, style)

        has_text_field = "text" in element and str(element.get("text") or "") != ""
        text = compose_text_from_lines_or_spans(
            lines=element.get("lines"),
            spans=element.get("spans"),
            fallback_text=element.get("text"),
        )

        if text_runs is not None and not has_text_field:
            text = rebuild_text_from_runs(text_runs)

        text_ir = TextIR(
            type="text",
            bbox=bbox,
            text=str(text),
            source=source,
            order=order,
            style=style,
            is_discarded=is_discarded,
            is_watermark=is_watermark,
            group_id=str(group_id) if group_id is not None else None,
            text_runs=text_runs,
            lines=_normalize_lines(element.get("lines"), bbox),
        )

        if not text_ir.text and not text_ir.text_runs:
            raise ValueError("Text IR element requires text or text_runs")

        return text_ir

    return ImageIR(
        type="image",
        bbox=bbox,
        source=source,
        order=order,
        style=style,
        is_discarded=is_discarded,
        is_watermark=is_watermark,
        group_id=str(group_id) if group_id is not None else None,
        text_elements=_normalize_image_text_elements(element.get("text_elements", [])),
        crop_pixels=element.get("crop_pixels"),
    )


def normalize_elements(elements: list[Any]) -> list[ElementIR]:
    return [normalize_element_ir(elem) for elem in elements]


def validate_ir_elements(
    elements: list[Any],
    require_text_runs_consistency: bool = False,
) -> list[ElementIR]:
    normalized = normalize_elements(elements)
    if not require_text_runs_consistency:
        return normalized

    for element in normalized:
        if not isinstance(element, TextIR):
            continue
        text_runs = element.text_runs
        if not isinstance(text_runs, list):
            continue

        rebuilt = rebuild_text_from_runs(text_runs)
        if element.text != rebuilt:
            raise ValueError(
                "TextIR.text is inconsistent with TextIR.text_runs rebuild result"
            )

    return normalized


def sort_elements(elements: list[ElementIR]) -> list[ElementIR]:
    return sorted(
        elements,
        key=lambda elem: (
            (elem.order if len(elem.order) >= 2 else [elem.bbox[1], elem.bbox[0]])[0],
            (elem.order if len(elem.order) >= 2 else [elem.bbox[1], elem.bbox[0]])[1],
        ),
    )


def build_page_ir(page_index: int, page_size: tuple[float, float] | None, elements: list[Any]) -> PageIR:
    normalized = normalize_elements(elements)
    sorted_elements = sort_elements(normalized)
    return PageIR(page_index=page_index, page_size=page_size, elements=sorted_elements)


def build_document_ir(pages: list[PageIR]) -> DocumentIR:
    return DocumentIR(pages=pages)
