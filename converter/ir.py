from __future__ import annotations

from dataclasses import dataclass, field
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
    group_id: str | None = None
    text_runs: list[TextRunIR] | None = None


@dataclass(frozen=True)
class ImageIR:
    type: str
    bbox: list[float]
    source: str
    order: list[float]
    style: dict[str, Any]
    is_discarded: bool = False
    group_id: str | None = None
    text_elements: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class PageIR:
    page_index: int
    page_size: tuple[float, float] | None
    elements: list[dict[str, Any]]


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


def rebuild_text_from_runs(text_runs: list[dict[str, Any]]) -> str:
    if not text_runs:
        return ""

    grouped: dict[int, list[dict[str, Any]]] = {}
    for run in text_runs:
        line_index = int(run.get("line_index", 0))
        grouped.setdefault(line_index, []).append(run)

    line_texts: list[str] = []
    for line_index in sorted(grouped.keys()):
        runs = sorted(
            grouped[line_index],
            key=lambda run: (
                (run.get("bbox") or [0.0, 0.0, 0.0, 0.0])[0],
                (run.get("bbox") or [0.0, 0.0, 0.0, 0.0])[1],
            ),
        )
        line_texts.append("".join(str(run.get("text", "")) for run in runs))

    return "\n".join(line_texts)


def normalize_text_runs(text_runs: Any) -> list[dict[str, Any]] | None:
    if text_runs is None:
        return None

    if not isinstance(text_runs, list):
        raise ValueError("TextIR.text_runs must be a list or None")

    normalized: list[dict[str, Any]] = []
    for index, run in enumerate(text_runs):
        if not isinstance(run, dict):
            raise ValueError("TextRunIR must be a dict")
        if run.get("bbox") is None:
            raise ValueError("TextRunIR.bbox is required")

        normalized_run = {
            "text": str(run.get("text", "")),
            "bbox": normalize_bbox(run.get("bbox")),
            "line_index": int(run.get("line_index", index)),
            "style": normalize_style(run.get("style") or {}),
        }
        normalized.append(normalized_run)

    normalized.sort(key=lambda run: (run["line_index"], run["bbox"][1], run["bbox"][0]))
    return normalized


def _to_text_run_models(text_runs: list[dict[str, Any]] | None) -> list[TextRunIR] | None:
    if text_runs is None:
        return None
    return [
        TextRunIR(
            text=run["text"],
            bbox=list(run["bbox"]),
            line_index=int(run.get("line_index", 0)),
            style=dict(run.get("style") or {}),
        )
        for run in text_runs
    ]


def _build_text_runs_from_lines_or_spans(
    element: dict[str, Any],
    bbox: list[float],
    style: dict[str, Any],
) -> list[dict[str, Any]] | None:
    lines = element.get("lines")
    if isinstance(lines, list) and lines:
        runs: list[dict[str, Any]] = []
        for line_index, line in enumerate(lines):
            if not isinstance(line, dict):
                continue
            line_bbox = normalize_bbox(line.get("bbox") or bbox)
            spans = line.get("spans") or []
            if not spans:
                line_text = str(line.get("text", ""))
                if line_text:
                    runs.append(
                        {
                            "text": line_text,
                            "bbox": line_bbox,
                            "line_index": line_index,
                            "style": style,
                        }
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
                    {
                        "text": span_text,
                        "bbox": span_bbox,
                        "line_index": line_index,
                        "style": span_style,
                    }
                )
        if runs:
            return normalize_text_runs(runs)

    spans = element.get("spans")
    if isinstance(spans, list) and spans:
        span_runs: list[dict[str, Any]] = []
        for span in spans:
            if not isinstance(span, dict):
                continue
            span_text = str(span.get("content", ""))
            if not span_text:
                continue
            span_bbox = normalize_bbox(span.get("bbox") or bbox)
            span_runs.append(
                {
                    "text": span_text,
                    "bbox": span_bbox,
                    "line_index": 0,
                    "style": normalize_style(span.get("style") or style),
                }
            )
        if span_runs:
            return normalize_text_runs(span_runs)

    return None


def materialize_text_runs_for_element(element: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_element_ir(element)
    if normalized.get("type") != "text":
        return normalized

    if isinstance(normalized.get("text_runs"), list):
        return normalized

    materialized_runs = _build_text_runs_from_lines_or_spans(
        normalized,
        normalized["bbox"],
        normalized.get("style") or default_style(),
    )

    if not materialized_runs:
        text_value = str(normalized.get("text") or "")
        if not text_value:
            return normalized
        line_texts = text_value.split("\n")
        materialized_runs = [
            {
                "text": line_text,
                "bbox": normalized["bbox"],
                "line_index": idx,
                "style": dict(normalized.get("style") or {}),
            }
            for idx, line_text in enumerate(line_texts)
            if line_text
        ]

    if not materialized_runs:
        return normalized

    rebuilt = rebuild_text_from_runs(materialized_runs)
    return {
        **normalized,
        "text_runs": materialized_runs,
        "text": rebuilt,
    }


def materialize_text_runs_for_elements(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [materialize_text_runs_for_element(element) for element in elements]


def normalize_element_ir(element: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(element, dict):
        raise ValueError("IR element must be a dict")

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
            group_id=str(group_id) if group_id is not None else None,
            text_runs=_to_text_run_models(text_runs),
        )

        normalized: dict[str, Any] = {
            "type": text_ir.type,
            "bbox": text_ir.bbox,
            "text": text_ir.text,
            "source": text_ir.source,
            "order": text_ir.order,
            "style": text_ir.style,
            "is_discarded": text_ir.is_discarded,
            "group_id": text_ir.group_id,
            "text_runs": [
                {
                    "text": run.text,
                    "bbox": run.bbox,
                    "line_index": run.line_index,
                    "style": run.style,
                }
                for run in (text_ir.text_runs or [])
            ]
            if text_ir.text_runs is not None
            else None,
        }

        if not normalized.get("text") and not normalized.get("text_runs"):
            raise ValueError("Text IR element requires text or text_runs")

        return normalized

    image_ir = ImageIR(
        type="image",
        bbox=bbox,
        source=source,
        order=order,
        style=style,
        is_discarded=is_discarded,
        group_id=str(group_id) if group_id is not None else None,
        text_elements=list(element.get("text_elements", [])),
    )

    return {
        "type": image_ir.type,
        "bbox": image_ir.bbox,
        "source": image_ir.source,
        "order": image_ir.order,
        "style": image_ir.style,
        "is_discarded": image_ir.is_discarded,
        "group_id": image_ir.group_id,
        "text_elements": image_ir.text_elements,
    }


def normalize_elements(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_element_ir(elem) for elem in elements]


def validate_ir_elements(
    elements: list[dict[str, Any]],
    require_text_runs_consistency: bool = False,
) -> list[dict[str, Any]]:
    normalized = normalize_elements(elements)
    if not require_text_runs_consistency:
        return normalized

    for element in normalized:
        if element.get("type") != "text":
            continue
        text_runs = element.get("text_runs")
        if not isinstance(text_runs, list):
            continue

        rebuilt = rebuild_text_from_runs(text_runs)
        if element.get("text", "") != rebuilt:
            raise ValueError(
                "TextIR.text is inconsistent with TextIR.text_runs rebuild result"
            )

    return normalized


def sort_elements(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        elements,
        key=lambda elem: (
            elem.get("order", [elem["bbox"][1], elem["bbox"][0]])[0],
            elem.get("order", [elem["bbox"][1], elem["bbox"][0]])[1],
        ),
    )


def build_page_ir(page_index: int, page_size: tuple[float, float] | None, elements: list[dict[str, Any]]) -> PageIR:
    normalized = normalize_elements(elements)
    sorted_elements = sort_elements(normalized)
    return PageIR(page_index=page_index, page_size=page_size, elements=sorted_elements)


def build_document_ir(pages: list[PageIR]) -> DocumentIR:
    return DocumentIR(pages=pages)
