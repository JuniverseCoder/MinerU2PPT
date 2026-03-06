from __future__ import annotations

from typing import Any, Callable

from .ir import rebuild_text_from_runs, validate_ir_elements


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


def _sort_key(elem: dict[str, Any]):
    # Keep deterministic rendering order by geometry fallback: y then x.
    bbox = elem.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    return (float(bbox[1]), float(bbox[0]))


def _should_merge_line_fragments(prev_bbox: list[float], curr_bbox: list[float]) -> bool:
    prev_h = max(1e-6, prev_bbox[3] - prev_bbox[1])
    curr_h = max(1e-6, curr_bbox[3] - curr_bbox[1])
    max_h = max(prev_h, curr_h)

    prev_center_y = (prev_bbox[1] + prev_bbox[3]) / 2
    curr_center_y = (curr_bbox[1] + curr_bbox[3]) / 2
    same_y_band = abs(prev_center_y - curr_center_y) <= max_h * 0.55

    horizontal_gap = curr_bbox[0] - prev_bbox[2]
    close_in_x = -max_h * 0.2 <= horizontal_gap <= max_h * 1.5

    return same_y_band and close_in_x


def _merge_lines_by_geometry(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not lines:
        return []

    sorted_lines = sorted(lines, key=lambda l: (l["bbox"][1], l["bbox"][0]))
    merged: list[dict[str, Any]] = []

    for line in sorted_lines:
        if not merged:
            merged.append({"bbox": list(line["bbox"]), "spans": list(line.get("spans", []))})
            continue

        prev = merged[-1]
        prev_bbox = prev["bbox"]
        curr_bbox = line["bbox"]

        if _should_merge_line_fragments(prev_bbox, curr_bbox):
            merged_bbox = _union_bboxes([prev_bbox, curr_bbox])
            merged_spans = list(prev.get("spans", [])) + list(line.get("spans", []))
            merged_spans.sort(key=lambda s: ((s.get("bbox") or [0, 0, 0, 0])[0], (s.get("bbox") or [0, 0, 0, 0])[1]))
            merged[-1] = {"bbox": merged_bbox, "spans": merged_spans}
        else:
            merged.append({"bbox": list(curr_bbox), "spans": list(line.get("spans", []))})

    return merged


def _extract_elem_lines(elem: dict[str, Any]) -> list[dict[str, Any]]:
    lines = elem.get("lines")
    bbox = elem.get("bbox")

    if isinstance(lines, list) and lines:
        extracted: list[dict[str, Any]] = []
        for line in lines:
            line_bbox = line.get("bbox") or bbox
            if not line_bbox:
                continue
            spans = []
            for span in line.get("spans", []) or []:
                span_bbox = span.get("bbox") or line_bbox
                spans.append(
                    {
                        "bbox": span_bbox,
                        "content": span.get("content", ""),
                        "type": span.get("type", "text"),
                    }
                )
            if not spans:
                spans = [{"bbox": line_bbox, "content": "", "type": "text"}]
            extracted.append({"bbox": line_bbox, "spans": spans})
        if extracted:
            return extracted

    text = str(elem.get("text", ""))
    if bbox and text:
        return [{"bbox": bbox, "spans": [{"bbox": bbox, "content": text, "type": "text"}]}]
    return []


def _build_text_runs_from_merged_lines(
    merged_lines: list[dict[str, Any]],
    fallback_style: dict[str, Any],
) -> list[dict[str, Any]]:
    text_runs: list[dict[str, Any]] = []

    for line_index, line in enumerate(merged_lines):
        for span in line.get("spans", []) or []:
            span_text = str(span.get("content", ""))
            if not span_text:
                continue
            span_style = span.get("style") if isinstance(span, dict) else None
            text_runs.append(
                {
                    "text": span_text,
                    "bbox": span.get("bbox") or line.get("bbox"),
                    "line_index": line_index,
                    "style": dict(span_style or fallback_style or {}),
                }
            )

    return text_runs


def _sync_text_and_runs(elem: dict[str, Any]) -> dict[str, Any]:
    if elem.get("type") != "text":
        return elem

    text_runs = elem.get("text_runs")
    if isinstance(text_runs, list):
        return {
            **elem,
            "text": rebuild_text_from_runs(text_runs),
        }
    return elem


def _combine_overlay_members(members: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted([m for m in members if m.get("bbox")], key=_sort_key)
    template = ordered[0]

    raw_lines: list[dict[str, Any]] = []
    all_bboxes: list[list[float]] = []
    for elem in ordered:
        elem_bbox = elem.get("bbox")
        if elem_bbox:
            all_bboxes.append(elem_bbox)
        raw_lines.extend(_extract_elem_lines(elem))

    merged_lines = _merge_lines_by_geometry(raw_lines)
    line_bboxes = [line["bbox"] for line in merged_lines if line.get("bbox")]
    final_bbox = _union_bboxes(line_bboxes if line_bboxes else all_bboxes)

    merged_text_runs = _build_text_runs_from_merged_lines(merged_lines, template.get("style") or {})

    if merged_text_runs:
        text = rebuild_text_from_runs(merged_text_runs)
    else:
        text = "\n".join(
            "".join(span.get("content", "") for span in line.get("spans", []))
            for line in merged_lines
        ).strip()

    return {
        **template,
        "bbox": final_bbox,
        "lines": merged_lines,
        "text": text,
        "text_runs": merged_text_runs if merged_text_runs else None,
        "order": [final_bbox[1], final_bbox[0]],
        "source": "ocr",
    }


def _merge_overlay_text_fragments_by_base_overlap(
    base_elements: list[dict[str, Any]],
    overlay_elements: list[dict[str, Any]],
    has_overlap: Callable[[list[float], list[float]], bool],
):
    consumed_overlay: set[int] = set()
    synthesized_overlays: list[dict[str, Any]] = []

    for base in base_elements:
        if base.get("type") != "text":
            continue
        base_bbox = base.get("bbox")
        if not base_bbox:
            continue

        overlapping_indices = []
        for idx, overlay in enumerate(overlay_elements):
            if idx in consumed_overlay:
                continue
            if overlay.get("type") != "text":
                continue
            overlay_bbox = overlay.get("bbox")
            if not overlay_bbox:
                continue
            if has_overlap(overlay_bbox, base_bbox):
                overlapping_indices.append(idx)

        if len(overlapping_indices) < 2:
            continue

        members = [overlay_elements[idx] for idx in overlapping_indices]
        synthesized_overlays.append(_combine_overlay_members(members))
        consumed_overlay.update(overlapping_indices)

    merged_overlay_elements = [
        overlay_elements[idx]
        for idx in range(len(overlay_elements))
        if idx not in consumed_overlay
    ] + synthesized_overlays

    merged_overlay_elements.sort(key=_sort_key)
    return merged_overlay_elements, {
        "overlay_fragment_groups": len(synthesized_overlays),
        "overlay_fragment_consumed": len(consumed_overlay),
    }


def merge_ir_elements(
    base_elements: list[dict[str, Any]],
    overlay_elements: list[dict[str, Any]],
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

    merged = list(base_elements)
    candidates = len(overlay_elements)

    overlay_elements, fragment_stats = _merge_overlay_text_fragments_by_base_overlap(
        base_elements,
        overlay_elements,
        has_overlap,
    )

    # Phase 1: group_id replacement
    by_group: dict[str, list[int]] = {}
    for idx, elem in enumerate(merged):
        group_id = elem.get("group_id")
        if group_id:
            by_group.setdefault(str(group_id), []).append(idx)

    consumed_overlay: set[int] = set()
    group_replaced = 0

    for o_idx, o_elem in enumerate(overlay_elements):
        o_group = o_elem.get("group_id")
        if not o_group:
            continue
        o_group_key = str(o_group)
        target_indices = by_group.get(o_group_key, [])
        if not target_indices:
            continue

        replaced_any = False
        for t_idx in target_indices:
            t_elem = merged[t_idx]
            if t_elem.get("type") == o_elem.get("type") == "text":
                merged[t_idx] = o_elem
                replaced_any = True
                group_replaced += 1
        if replaced_any:
            consumed_overlay.add(o_idx)

    # Phase 2: overlap text replacement
    overlap_replaced = 0
    appended = 0

    for o_idx, o_elem in enumerate(overlay_elements):
        if o_idx in consumed_overlay:
            continue

        o_type = o_elem.get("type")
        o_bbox = o_elem.get("bbox")
        if o_type != "text" or not o_bbox:
            merged.append(o_elem)
            appended += 1
            continue

        replaced = False
        for t_idx, t_elem in enumerate(merged):
            if t_elem.get("type") != "text":
                continue
            t_bbox = t_elem.get("bbox")
            if not t_bbox:
                continue
            if has_overlap(o_bbox, t_bbox):
                merged[t_idx] = o_elem
                replaced = True
                overlap_replaced += 1
                break

        if not replaced:
            merged.append(o_elem)
            appended += 1

    merged = [_sync_text_and_runs(elem) for elem in merged]
    merged = validate_ir_elements(merged, require_text_runs_consistency=True)
    merged.sort(key=_sort_key)

    stats = {
        "overlay_candidates": candidates,
        "overlay_fragment_groups": fragment_stats["overlay_fragment_groups"],
        "overlay_fragment_consumed": fragment_stats["overlay_fragment_consumed"],
        "group_replaced": group_replaced,
        "overlap_replaced": overlap_replaced,
        "overlay_added": appended,
    }
    return merged, stats
