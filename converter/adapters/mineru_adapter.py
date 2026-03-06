from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..ir import normalize_bbox, normalize_element_ir

TEXT_TYPES = {"text", "title", "caption", "footnote", "footer", "header", "page_number", "list"}
IMAGE_TYPES = {"image", "table", "figure", "formula"}


@dataclass(frozen=True)
class MinerUPageData:
    para_blocks: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    discarded_blocks: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MinerUPageData":
        return cls(
            para_blocks=list(data.get("para_blocks", []) or []),
            images=list(data.get("images", []) or []),
            tables=list(data.get("tables", []) or []),
            discarded_blocks=list(data.get("discarded_blocks", []) or []),
        )


class MinerUAdapter:
    """Map MinerU page data object into unified IR elements."""

    def extract_page_elements(self, page_data: MinerUPageData, include_text_runs: bool = False) -> list[dict[str, Any]]:
        elements: list[dict[str, Any]] = []

        for item in page_data.para_blocks:
            elements.extend(self._to_ir_elements(item, is_discarded=False, include_text_runs=include_text_runs))

        for item in page_data.images:
            elements.extend(self._to_ir_elements(item, is_discarded=False, include_text_runs=include_text_runs))

        for item in page_data.tables:
            elements.extend(self._to_ir_elements(item, is_discarded=False, include_text_runs=include_text_runs))

        for item in page_data.discarded_blocks:
            elements.extend(self._to_ir_elements(item, is_discarded=True, include_text_runs=include_text_runs))

        return elements

    def _to_ir_elements(self, item: dict[str, Any], is_discarded: bool, include_text_runs: bool = False) -> list[dict[str, Any]]:
        if not item or not item.get("bbox"):
            return []

        elem_type = item.get("type", "text")
        if elem_type == "list":
            return self._list_to_text_elements(item, is_discarded, include_text_runs=include_text_runs)

        if elem_type in IMAGE_TYPES:
            return self._image_like_to_elements(item, is_discarded, include_text_runs=include_text_runs)

        if elem_type in TEXT_TYPES:
            return [self._text_element_from_block(item, is_discarded, include_text_runs=include_text_runs)]

        return [self._text_element_from_block(item, is_discarded, include_text_runs=include_text_runs)]

    def _list_to_text_elements(self, item: dict[str, Any], is_discarded: bool, include_text_runs: bool = False) -> list[dict[str, Any]]:
        blocks = item.get("blocks") or []
        if not blocks:
            return [self._text_element_from_block(item, is_discarded, include_text_runs=include_text_runs)]

        group_id = item.get("group_id") or f"mineru-list-{item.get('index', 0)}"
        converted = []
        for block in blocks:
            if not block or not block.get("bbox"):
                continue
            converted.append(
                self._text_element_from_block(
                    block,
                    is_discarded=is_discarded,
                    group_id=group_id,
                    force_bold=False,
                    include_text_runs=include_text_runs,
                )
            )
        return converted

    def _image_like_to_elements(self, item: dict[str, Any], is_discarded: bool, include_text_runs: bool = False) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        blocks = item.get("blocks") or []

        image_bbox = item.get("bbox")
        for block in blocks:
            block_type = block.get("type")
            if block_type == "image_body" and block.get("bbox"):
                image_bbox = block["bbox"]
                break

        image_element = {
            "type": "image",
            "bbox": image_bbox,
            "source": "mineru",
            "is_discarded": is_discarded,
            "group_id": item.get("group_id"),
            "order": [image_bbox[1], image_bbox[0]] if image_bbox else None,
            "style": {},
            "text_elements": [],
        }
        results.append(normalize_element_ir(image_element))

        for block in blocks:
            if block.get("type") == "image_caption" and block.get("bbox"):
                results.append(
                    self._text_element_from_block(
                        block,
                        is_discarded=is_discarded,
                        group_id=item.get("group_id"),
                        force_bold=False,
                        include_text_runs=include_text_runs,
                    )
                )

        return results

    def _text_element_from_block(
        self,
        block: dict[str, Any],
        is_discarded: bool,
        group_id: str | None = None,
        force_bold: bool | None = None,
        include_text_runs: bool = False,
    ) -> dict[str, Any]:
        block_type = block.get("type", "text")
        style = {
            "bold": bool(force_bold if force_bold is not None else block_type == "title"),
            "font_size": None,
            "align": "left",
        }

        bbox = block.get("bbox")
        text_runs = self._build_text_runs_from_block(block) if include_text_runs else None

        element = {
            "type": "text",
            "bbox": bbox,
            "lines": block.get("lines"),
            "spans": block.get("spans"),
            "text": block.get("text"),
            "text_runs": text_runs,
            "source": "mineru",
            "is_discarded": is_discarded,
            "group_id": group_id or block.get("group_id"),
            "order": [bbox[1], bbox[0]] if bbox else None,
            "style": style,
        }
        return normalize_element_ir(element)

    def _build_text_runs_from_block(self, block: dict[str, Any]) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        lines = block.get("lines") or []

        for line_index, line in enumerate(lines):
            if not isinstance(line, dict):
                continue
            line_bbox = line.get("bbox") or block.get("bbox")
            if not line_bbox:
                continue
            line_bbox = normalize_bbox(line_bbox)

            spans = line.get("spans") or []
            if not spans:
                line_text = block.get("text") or ""
                if line_text:
                    runs.append(
                        {
                            "text": str(line_text),
                            "bbox": line_bbox,
                            "line_index": line_index,
                            "style": {},
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
                runs.append(
                    {
                        "text": span_text,
                        "bbox": span_bbox,
                        "line_index": line_index,
                        "style": {},
                    }
                )

        if runs:
            return runs

        block_text = str(block.get("text") or "")
        block_bbox = block.get("bbox")
        if block_text and block_bbox:
            return [
                {
                    "text": block_text,
                    "bbox": normalize_bbox(block_bbox),
                    "line_index": 0,
                    "style": {},
                }
            ]

        return runs
