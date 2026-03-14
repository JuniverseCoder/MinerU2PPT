from __future__ import annotations

from typing import Any

from ..ir import TextIR, TextRunIR, build_page_ir, normalize_bbox, normalize_element_ir


class OCRAdapter:
    """Map OCR output into unified IR elements."""

    def __init__(self, ocr_engine):
        self.ocr_engine = ocr_engine

    def extract_page_elements(
        self,
        page_image,
        json_w: float,
        json_h: float,
        page_context: Any | None = None,
    ) -> list[TextIR]:
        stage_elements = self.ocr_engine.extract_text_elements(
            page_image,
            json_w,
            json_h,
            return_stage_elements=True,
        )

        if isinstance(stage_elements, list):
            before_refined = stage_elements
            after_refined = stage_elements
        else:
            before_refined = stage_elements.get("before_refined_elements", [])
            after_refined = stage_elements.get("after_refined_elements", [])

        before_refined_ir = [
            self._to_ir_text_element(elem)
            for elem in before_refined
            if isinstance(elem, dict) and elem.get("bbox")
        ]
        after_refined_ir = [
            self._to_ir_text_element(elem)
            for elem in after_refined
            if isinstance(elem, dict) and elem.get("bbox")
        ]

        if page_context is not None:
            page_index = page_context.page_index
            page_size = (float(json_w), float(json_h))
            page_context.register_stage_page_ir(
                "ocr_before_refined_elements",
                build_page_ir(page_index=page_index, page_size=page_size, elements=before_refined_ir),
            )
            page_context.register_stage_page_ir(
                "ocr_after_refined_elements",
                build_page_ir(page_index=page_index, page_size=page_size, elements=after_refined_ir),
            )

        return after_refined_ir

    def _to_ir_text_element(self, elem: dict[str, Any]) -> TextIR:
        bbox = normalize_bbox(elem.get("bbox"))
        font_size = max(6.0, float(bbox[3]) - float(bbox[1]))

        text_runs: list[TextRunIR] = []
        line_texts: list[str] = []
        normalized_lines: list[dict[str, Any]] = []
        lines = elem.get("lines", []) or []
        for line_index, line in enumerate(lines):
            line_spans = line.get("spans", []) if isinstance(line, dict) else []
            line_bbox = normalize_bbox((line.get("bbox") if isinstance(line, dict) else None) or bbox)
            ordered_spans = sorted(
                [span for span in line_spans if isinstance(span, dict)],
                key=lambda span: (span.get("bbox") or line_bbox)[0],
            )

            line_text = ""
            normalized_spans: list[dict[str, Any]] = []
            for span in ordered_spans:
                span_text = str(span.get("content", ""))
                if not span_text:
                    continue
                span_bbox = normalize_bbox(span.get("bbox") or line_bbox)
                run_style = {
                    "bold": False,
                    "font_size": font_size,
                    "align": "left",
                }
                text_runs.append(
                    TextRunIR(
                        text=span_text,
                        bbox=span_bbox,
                        line_index=line_index,
                        style=run_style,
                    )
                )
                normalized_spans.append(
                    {
                        "bbox": span_bbox,
                        "content": span_text,
                        "type": "text",
                        "style": run_style,
                    }
                )
                line_text += span_text

            if line_text:
                line_texts.append(line_text)
            if normalized_spans:
                normalized_lines.append({"bbox": line_bbox, "spans": normalized_spans})

        if not text_runs:
            text_value = str(elem.get("text") or "")
            if not text_value and isinstance(lines, list):
                text_value = "\n".join(
                    "".join(str(span.get("content", "")) for span in (line.get("spans") or []) if isinstance(span, dict))
                    for line in lines
                    if isinstance(line, dict)
                )
            if not text_value:
                text_value = ""

            text_runs = [
                TextRunIR(
                    text=text_value,
                    bbox=bbox,
                    line_index=0,
                    style={
                        "bold": False,
                        "font_size": font_size,
                        "align": "left",
                    },
                )
            ]
            text = text_value
        else:
            text = "\n".join(line_texts)

        if not normalized_lines and text_runs:
            grouped: dict[int, list[TextRunIR]] = {}
            for run in text_runs:
                grouped.setdefault(int(run.line_index), []).append(run)
            for line_idx in sorted(grouped.keys()):
                line_runs = sorted(grouped[line_idx], key=lambda run: run.bbox[0])
                line_bbox = [
                    min(run.bbox[0] for run in line_runs),
                    min(run.bbox[1] for run in line_runs),
                    max(run.bbox[2] for run in line_runs),
                    max(run.bbox[3] for run in line_runs),
                ]
                normalized_lines.append(
                    {
                        "bbox": line_bbox,
                        "spans": [
                            {
                                "bbox": run.bbox,
                                "content": run.text,
                                "type": "text",
                                "style": run.style,
                            }
                            for run in line_runs
                        ],
                    }
                )

        ir_elem = {
            "type": "text",
            "bbox": bbox,
            "lines": normalized_lines,
            "text": text,
            "text_runs": text_runs,
            "source": "ocr",
            "is_discarded": False,
            "is_watermark": False,
            "group_id": elem.get("group_id"),
            "order": [bbox[1], bbox[0]],
            "style": {
                "bold": False,
                "font_size": font_size,
                "align": "left",
            },
        }
        normalized_text = normalize_element_ir(ir_elem)
        if not isinstance(normalized_text, TextIR):
            raise ValueError("Expected TextIR from OCR normalization")
        return normalized_text
