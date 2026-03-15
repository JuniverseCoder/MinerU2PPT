from __future__ import annotations

from dataclasses import replace
from functools import cmp_to_key
from typing import Any, Callable

from .ir import ElementIR, ImageIR, TextIR, TextRunIR, rebuild_text_from_runs, validate_ir_elements


def _bbox_overlap(a: list[float], b: list[float]) -> bool:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return x1 < x2 and y1 < y2


def _union_bboxes(bboxes: list[list[float]]) -> list[float]:
    if not bboxes:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]


def _bbox_same_line(a: list[float], b: list[float]) -> bool:
    y1 = max(a[1], b[1])
    y2 = min(a[3], b[3])
    overlap = y2 - y1
    if overlap <= 0:
        return False
    ha = max(1e-6, a[3] - a[1])
    hb = max(1e-6, b[3] - b[1])
    ratio = overlap / min(ha, hb)
    return ratio >= 0.55


def _sort_compare(a: ElementIR, b: ElementIR) -> int:
    bbox_a = a.bbox or [0.0, 0.0, 0.0, 0.0]
    bbox_b = b.bbox or [0.0, 0.0, 0.0, 0.0]

    if _bbox_same_line(bbox_a, bbox_b):
        if bbox_a[0] < bbox_b[0]:
            return -1
        if bbox_a[0] > bbox_b[0]:
            return 1
        return 0

    if bbox_a[1] < bbox_b[1]:
        return -1
    if bbox_a[1] > bbox_b[1]:
        return 1
    return 0


def _split_members_into_lines(ordered: list[TextIR]) -> list[list[TextIR]]:
    line_groups: list[list[TextIR]] = []
    current_members: list[TextIR] = []
    current_bbox: list[float] | None = None

    for elem in ordered:
        elem_bbox = elem.bbox
        if not elem_bbox:
            continue
        if not current_members:
            current_members = [elem]
            current_bbox = list(elem_bbox)
            continue
        if current_bbox is not None and _bbox_same_line(current_bbox, elem_bbox):
            current_members.append(elem)
            current_bbox = _union_bboxes([current_bbox, elem_bbox])
        else:
            line_groups.append(current_members)
            current_members = [elem]
            current_bbox = list(elem_bbox)

    if current_members:
        line_groups.append(current_members)

    return line_groups


def _merge_lines_by_geometry(line_groups: list[list[TextIR]]) -> list[dict[str, Any]]:
    merged_lines: list[dict[str, Any]] = []

    for line_members in line_groups:
        spans: list[dict[str, Any]] = []
        line_bboxes: list[list[float]] = []
        for elem in line_members:
            if elem.bbox:
                line_bboxes.append(elem.bbox)
            for line in _extract_elem_lines(elem):
                line_bbox = line.get("bbox") or elem.bbox
                raw_spans = line.get("spans", []) or []
                if not raw_spans:
                    raw_spans = [{"bbox": line_bbox, "content": "", "type": "text", "style": {}}]
                for span in raw_spans:
                    span_bbox = span.get("bbox") or line_bbox
                    spans.append(
                        {
                            "bbox": span_bbox,
                            "content": span.get("content", ""),
                            "type": span.get("type", "text"),
                            "style": span.get("style") or {},
                        }
                    )

        if spans:
            spans.sort(
                key=lambda span: (
                    (span.get("bbox") or [0, 0, 0, 0])[0],
                    (span.get("bbox") or [0, 0, 0, 0])[1],
                )
            )
            line_bbox = _union_bboxes([span.get("bbox") or [0, 0, 0, 0] for span in spans])
        else:
            line_bbox = _union_bboxes(line_bboxes)
            spans = [{"bbox": line_bbox, "content": "", "type": "text", "style": {}}]

        merged_lines.append({"bbox": line_bbox, "spans": spans})

    return merged_lines


def _extract_elem_lines(elem: TextIR) -> list[dict[str, Any]]:
    lines = elem.lines
    bbox = elem.bbox

    if isinstance(lines, list) and lines:
        extracted: list[dict[str, Any]] = []
        for line in lines:
            if not isinstance(line, dict):
                continue
            line_bbox = line.get("bbox") or bbox
            if not line_bbox:
                continue

            spans: list[dict[str, Any]] = []
            for span in line.get("spans", []) or []:
                if not isinstance(span, dict):
                    continue
                span_bbox = span.get("bbox") or line_bbox
                spans.append(
                    {
                        "bbox": span_bbox,
                        "content": span.get("content", ""),
                        "type": span.get("type", "text"),
                        "style": span.get("style") or {},
                    }
                )

            if not spans:
                spans = [{"bbox": line_bbox, "content": "", "type": "text", "style": {}}]
            extracted.append({"bbox": line_bbox, "spans": spans})

        if extracted:
            return extracted

    if isinstance(elem.text_runs, list) and elem.text_runs:
        grouped: dict[int, list[TextRunIR]] = {}
        for run in elem.text_runs:
            grouped.setdefault(int(run.line_index), []).append(run)

        extracted_from_runs: list[dict[str, Any]] = []
        for line_index in sorted(grouped.keys()):
            line_runs = sorted(grouped[line_index], key=lambda run: (run.bbox[0], run.bbox[1]))
            line_bbox = [
                min(run.bbox[0] for run in line_runs),
                min(run.bbox[1] for run in line_runs),
                max(run.bbox[2] for run in line_runs),
                max(run.bbox[3] for run in line_runs),
            ]
            spans = [
                {
                    "bbox": run.bbox,
                    "content": run.text,
                    "type": "text",
                    "style": run.style,
                }
                for run in line_runs
            ]
            extracted_from_runs.append({"bbox": line_bbox, "spans": spans})

        if extracted_from_runs:
            return extracted_from_runs

    text = str(elem.text or "")
    if bbox and text:
        return [{"bbox": bbox, "spans": [{"bbox": bbox, "content": text, "type": "text", "style": elem.style}]}]

    return []


def _build_text_runs_from_merged_lines(
    merged_lines: list[dict[str, Any]],
    fallback_style: dict[str, Any],
) -> list[TextRunIR]:
    text_runs: list[TextRunIR] = []

    for line_index, line in enumerate(merged_lines):
        line_bbox = line.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        for span in line.get("spans", []) or []:
            span_text = str(span.get("content", ""))
            if not span_text:
                continue
            span_style = span.get("style") if isinstance(span, dict) else None
            text_runs.append(
                TextRunIR(
                    text=span_text,
                    bbox=list(span.get("bbox") or line_bbox),
                    line_index=line_index,
                    style=dict(span_style or fallback_style or {}),
                )
            )

    return text_runs


def _sync_text_and_runs(elem: ElementIR) -> ElementIR:
    if not isinstance(elem, TextIR):
        return elem

    if isinstance(elem.text_runs, list):
        return replace(
            elem,
            text=rebuild_text_from_runs(elem.text_runs),
        )
    return elem


def _inherit_overlay_bold_from_base(
    base_elem: TextIR,
    overlay_elem: TextIR,
) -> TextIR:
    inherited_bold = bool((base_elem.style or {}).get("bold", False))

    merged_style = dict(overlay_elem.style or {})
    merged_style["bold"] = inherited_bold

    merged_runs = overlay_elem.text_runs
    if isinstance(overlay_elem.text_runs, list):
        merged_runs = [
            replace(
                run,
                style={
                    **(run.style or {}),
                    "bold": inherited_bold,
                },
            )
            for run in overlay_elem.text_runs
        ]

    return replace(
        overlay_elem,
        style=merged_style,
        text_runs=merged_runs,
        is_discarded=bool(base_elem.is_discarded or overlay_elem.is_discarded),
        is_watermark=bool(base_elem.is_watermark or overlay_elem.is_watermark),
    )


def _combine_overlay_members(members: list[TextIR]) -> TextIR:
    ordered = sorted(
        [member for member in members if member.bbox],
        key=cmp_to_key(_sort_compare),
    )
    template = ordered[0]

    all_bboxes: list[list[float]] = []
    for elem in ordered:
        if elem.bbox:
            all_bboxes.append(elem.bbox)

    line_groups = _split_members_into_lines(ordered)
    merged_lines = _merge_lines_by_geometry(line_groups)
    line_bboxes = [line["bbox"] for line in merged_lines if line.get("bbox")]
    final_bbox = _union_bboxes(line_bboxes if line_bboxes else all_bboxes)

    merged_text_runs = _build_text_runs_from_merged_lines(merged_lines, template.style or {})

    if merged_text_runs:
        text = rebuild_text_from_runs(merged_text_runs)
    else:
        text = "\n".join(
            "".join(span.get("content", "") for span in line.get("spans", []))
            for line in merged_lines
        ).strip()

    return replace(
        template,
        bbox=final_bbox,
        lines=merged_lines,
        text=text,
        text_runs=merged_text_runs if merged_text_runs else None,
        order=[final_bbox[1], final_bbox[0]],
        source="ocr",
    )


def _merge_overlay_text_fragments_by_base_overlap(
    base_elements: list[ElementIR],
    overlay_elements: list[ElementIR],
    has_overlap: Callable[[list[float], list[float]], bool],
):
    consumed_overlay: set[int] = set()
    synthesized_overlays: list[TextIR] = []

    for base in base_elements:
        if not isinstance(base, TextIR):
            continue

        base_bbox = base.bbox
        if not base_bbox:
            continue

        overlapping_indices: list[int] = []
        for index, overlay in enumerate(overlay_elements):
            if index in consumed_overlay:
                continue
            if not isinstance(overlay, TextIR):
                continue
            overlay_bbox = overlay.bbox
            if not overlay_bbox:
                continue
            if has_overlap(overlay_bbox, base_bbox):
                overlapping_indices.append(index)

        if len(overlapping_indices) < 2:
            continue

        members = [overlay_elements[index] for index in overlapping_indices if isinstance(overlay_elements[index], TextIR)]
        if not members:
            continue

        synthesized_overlays.append(_combine_overlay_members(members))
        consumed_overlay.update(overlapping_indices)

    merged_overlay_elements = [
        overlay_elements[index]
        for index in range(len(overlay_elements))
        if index not in consumed_overlay
    ] + synthesized_overlays

    merged_overlay_elements.sort(key=cmp_to_key(_sort_compare))
    return merged_overlay_elements, {
        "overlay_fragment_groups": len(synthesized_overlays),
        "overlay_fragment_consumed": len(consumed_overlay),
    }


def merge_ir_elements(
    base_elements: list[Any],
    overlay_elements: list[Any],
    has_overlap: Callable[[list[float], list[float]], bool] = _bbox_overlap,
):
    """
    Merge multiple adapter IR outputs with OCR/text overlay semantics.

    - Overlay elements with group_id replace matching base elements by group_id.
    - Remaining overlay text replaces overlapping base text.
    - Non-overlapping overlays are appended.
    """

    base_elements = validate_ir_elements(base_elements)
    overlay_elements = validate_ir_elements(overlay_elements)

    merged: list[ElementIR] = list(base_elements)
    candidates = len(overlay_elements)

    overlay_elements, fragment_stats = _merge_overlay_text_fragments_by_base_overlap(
        base_elements,
        overlay_elements,
        has_overlap,
    )

    by_group: dict[str, list[int]] = {}
    for index, elem in enumerate(merged):
        if elem.group_id:
            by_group.setdefault(str(elem.group_id), []).append(index)

    consumed_overlay: set[int] = set()
    group_replaced = 0

    for overlay_index, overlay_elem in enumerate(overlay_elements):
        if not overlay_elem.group_id:
            continue

        target_indices = by_group.get(str(overlay_elem.group_id), [])
        if not target_indices:
            continue

        replaced_any = False
        for target_index in target_indices:
            target_elem = merged[target_index]
            if isinstance(target_elem, TextIR) and isinstance(overlay_elem, TextIR):
                merged[target_index] = _inherit_overlay_bold_from_base(target_elem, overlay_elem)
                replaced_any = True
                group_replaced += 1

        if replaced_any:
            consumed_overlay.add(overlay_index)

    overlap_replaced = 0
    appended = 0

    for overlay_index, overlay_elem in enumerate(overlay_elements):
        if overlay_index in consumed_overlay:
            continue

        if not isinstance(overlay_elem, TextIR) or not overlay_elem.bbox:
            merged.append(overlay_elem)
            appended += 1
            continue

        replaced = False
        for target_index, target_elem in enumerate(merged):
            if not isinstance(target_elem, TextIR):
                continue
            if not target_elem.bbox:
                continue
            if has_overlap(overlay_elem.bbox, target_elem.bbox):
                merged[target_index] = _inherit_overlay_bold_from_base(target_elem, overlay_elem)
                replaced = True
                overlap_replaced += 1
                break

        if not replaced:
            merged.append(overlay_elem)
            appended += 1

    merged = [_sync_text_and_runs(elem) for elem in merged]
    merged = validate_ir_elements(merged, require_text_runs_consistency=True)
    merged.sort(key=cmp_to_key(_sort_compare))

    stats = {
        "overlay_candidates": candidates,
        "overlay_fragment_groups": fragment_stats["overlay_fragment_groups"],
        "overlay_fragment_consumed": fragment_stats["overlay_fragment_consumed"],
        "group_replaced": group_replaced,
        "overlap_replaced": overlap_replaced,
        "overlay_added": appended,
    }
    return merged, stats
