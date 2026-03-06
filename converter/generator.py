import json
import os
import shutil

import cv2
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Pt

from .adapters import MinerUAdapter, MinerUPageData, OCRAdapter
from .ir import materialize_text_runs_for_elements, validate_ir_elements
from .ir_merge import merge_ir_elements
from .ocr_merge import PaddleOCREngine
from .utils import extract_background_color, extract_font_color, fill_bbox_with_bg


class PageContext:
    def __init__(self, page_image, coords, slide):
        self.slide = slide
        self.original_image = page_image.copy()
        self.background_image = page_image.copy()
        self.coords = coords
        self.elements = []

    def add_element_bbox_for_cleanup(self, bbox, margin_px=0, margin_ratio=0.0, min_margin_px=1):
        """Register a bounding box to be inpainted on the background image."""
        if bbox:
            px_box = [int(v * (self.coords['img_w'] / self.coords['json_w'] if i % 2 == 0 else self.coords['img_h'] / self.coords['json_h'])) for i, v in enumerate(bbox)]
            x1, y1, x2, y2 = px_box

            if margin_ratio > 0:
                height = max(1, y2 - y1)
                margin = max(min_margin_px, int(round(height * margin_ratio)))
                px_box = [x1 - margin, y1 - margin, x2 + margin, y2 + margin]
            elif margin_px > 0:
                px_box = [x1 - margin_px, y1 - margin_px, x2 + margin_px, y2 + margin_px]

            fill_bbox_with_bg(self.background_image, px_box)

    def add_processed_element(self, elem_type, data):
        """Store a fully processed element ready for rendering."""
        self.elements.append({'type': elem_type, 'data': data})

    def generate_debug_images(self, page_index, generator_instance):
        """Generate and save debug images for the page."""
        text_bboxes = [
            elem['data']['bbox']
            for elem in self.elements
            if elem.get('type') == 'text' and elem.get('data', {}).get('bbox')
        ]
        generator_instance._draw_text_bboxes_for_page(
            self.original_image,
            text_bboxes,
            self.coords,
            f"tmp/page_{page_index}_text_boxes.png",
        )

    def render_to_slide(self, generator_instance):
        """Render all processed elements onto the PowerPoint slide."""
        # 1. Render the cleaned background
        bg_path = f"temp_bg_{id(self.slide)}.png"
        cv2.imwrite(bg_path, cv2.cvtColor(self.background_image, cv2.COLOR_RGB2BGR))
        w_pts, h_pts = generator_instance.prs.slide_width, generator_instance.prs.slide_height
        self.slide.shapes.add_picture(bg_path, Pt(0), Pt(0), w_pts, h_pts)
        os.remove(bg_path)

        # 2. Render all image elements first
        for elem in self.elements:
            if elem['type'] == 'image':
                generator_instance._add_picture_from_bbox(self.slide, elem['data']['bbox'], self.original_image, self.coords, elem['data']['text_elements'])

        # 3. Render all text elements on top
        for elem in self.elements:
            if elem['type'] == 'text':
                generator_instance._render_text_from_data(self.slide, elem['data'])


TEXT_CLEANUP_MARGIN_RATIO = 0.05
TEXT_CLEANUP_MIN_MARGIN_PX = 1


class PPTGenerator:
    def __init__(
        self,
        output_path,
        remove_watermark=True,
        ocr_engine=None,
        ocr_device_policy="auto",
        ocr_model_root=None,
        ocr_offline_only=True,
    ):
        self.prs = Presentation()
        self.output_path = output_path
        self.remove_watermark = remove_watermark
        self.ocr_engine = ocr_engine
        self.ocr_device_policy = ocr_device_policy
        self.ocr_model_root = ocr_model_root
        self.ocr_offline_only = ocr_offline_only
        self.debug_images = False # Will be set in process_page
        for i in range(len(self.prs.slides) - 1, -1, -1):
            rId = self.prs.slides._sldIdLst[i].rId
            self.prs.part.drop_rel(rId)
            del self.prs.slides._sldIdLst[i]

    def cap_size(self, w_pts, h_pts):
        MAX_PTS = 56 * 72
        if w_pts > MAX_PTS or h_pts > MAX_PTS:
            scale = MAX_PTS / max(w_pts, h_pts)
            w_pts, h_pts = w_pts * scale, h_pts * scale
        return w_pts, h_pts

    def set_slide_size(self, width_px, height_px, dpi=72):
        w_pts, h_pts = self.cap_size(width_px * 72 / dpi, height_px * 72 / dpi)
        self.prs.slide_width, self.prs.slide_height = Pt(w_pts), Pt(h_pts)

    def add_slide(self):
        return self.prs.slides.add_slide(self.prs.slide_layouts[6])

    def _get_bbox_intersection(self, bbox1, bbox2):
        x1, y1 = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])
        x2, y2 = min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])
        return [x1, y1, x2, y2] if x1 < x2 and y1 < y2 else None

    def _has_bbox_overlap(self, bbox1, bbox2):
        return self._get_bbox_intersection(bbox1, bbox2) is not None

    def _to_px_bbox(self, bbox, coords):
        return [
            int(bbox[0] * coords['img_w'] / coords['json_w']),
            int(bbox[1] * coords['img_h'] / coords['json_h']),
            int(bbox[2] * coords['img_w'] / coords['json_w']),
            int(bbox[3] * coords['img_h'] / coords['json_h']),
        ]

    def _line_center_y(self, line):
        bbox = line.get('bbox')
        if not bbox:
            return 0.0
        return (bbox[1] + bbox[3]) / 2

    def _create_textbox(self, slide, bbox, coords):
        x1, y1, x2, y2 = bbox
        return slide.shapes.add_textbox(
            Pt(x1 * coords['scale_x']), Pt(y1 * coords['scale_y']),
            Pt((x2 - x1) * coords['scale_x']), Pt((y2 - y1) * coords['scale_y'])
        )

    def _draw_text_bboxes_for_page(self, image, text_bboxes, coords, output_path):
        """Draw text-level bounding boxes for debugging."""
        debug_img = image.copy()
        for bbox in text_bboxes:
            px_box = [
                int(bbox[0] * coords['img_w'] / coords['json_w']),
                int(bbox[1] * coords['img_h'] / coords['json_h']),
                int(bbox[2] * coords['img_w'] / coords['json_w']),
                int(bbox[3] * coords['img_h'] / coords['json_h'])
            ]
            cv2.rectangle(debug_img, (px_box[0], px_box[1]), (px_box[2], px_box[3]), (0, 255, 0), 2)
        cv2.imwrite(output_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))

    def _build_text_runs_from_ir_runs(self, context, elem, ir_runs):
        bbox = elem.get("bbox")
        if not bbox or not ir_runs:
            return [], False

        ordered_runs = sorted(
            ir_runs,
            key=lambda run: (
                int(run.get("line_index", 0)),
                (run.get("bbox") or bbox)[0],
                (run.get("bbox") or bbox)[1],
            ),
        )

        runs = []
        current_line_index = None
        line_indexes = set()

        for run in ordered_runs:
            run_text = str(run.get("text", "")).replace("\\%", "%")
            if not run_text:
                continue

            run_bbox = run.get("bbox") or bbox
            line_index = int(run.get("line_index", 0))
            line_indexes.add(line_index)

            if current_line_index is not None and line_index != current_line_index and runs:
                prev_font = runs[-1].get("font", {})
                runs.append({"text": "\n", "font": prev_font})

            run_px_bbox = self._to_px_bbox(run_bbox, context.coords)
            bg_color = extract_background_color(context.original_image, run_px_bbox)
            color, _, _ = extract_font_color(context.original_image, run_px_bbox, bg_color)

            run_style = run.get("style") or {}
            elem_style = elem.get("style") or {}
            style_font_size = run_style.get("font_size") or elem_style.get("font_size")
            if style_font_size:
                font_size_pts = max(6.0, float(style_font_size) * context.coords['scale_y'])
            else:
                font_size_pts = max(6.0, (run_bbox[3] - run_bbox[1]) * context.coords['scale_y'])

            font_info = {
                "name": "Microsoft YaHei",
                "size": Pt(int(font_size_pts)),
                "bold": bool(run_style.get("bold", elem_style.get("bold", False))),
                "color": RGBColor(*color),
            }

            runs.append({"text": run_text, "font": font_info})
            current_line_index = line_index

        if not runs:
            return [], False

        is_single_line = len(line_indexes) <= 1
        return runs, is_single_line

    def _estimate_single_line_height(self, elem):
        lines = elem.get("lines")
        line_heights = []
        if isinstance(lines, list):
            for line in lines:
                line_bbox = line.get("bbox") if isinstance(line, dict) else None
                if (
                    isinstance(line_bbox, (list, tuple))
                    and len(line_bbox) == 4
                    and line_bbox[3] > line_bbox[1]
                ):
                    line_heights.append(float(line_bbox[3] - line_bbox[1]))

        if line_heights:
            return float(np.median(line_heights))

        bbox = elem.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and bbox[3] > bbox[1]:
            return float(bbox[3] - bbox[1])

        return 1.0

    def _compute_text_cleanup_margin_px(self, context, elem):
        json_h = context.coords.get("json_h", 1)
        img_h = context.coords.get("img_h", 1)
        scale_y = (img_h / json_h) if json_h else 1.0

        single_line_height = self._estimate_single_line_height(elem)
        single_line_height_px = max(1.0, single_line_height * scale_y)
        return max(TEXT_CLEANUP_MIN_MARGIN_PX, int(round(single_line_height_px * TEXT_CLEANUP_MARGIN_RATIO)))

    def _process_text(self, context, elem):
        bbox = elem.get("bbox")
        if not bbox:
            return

        cleanup_margin_px = self._compute_text_cleanup_margin_px(context, elem)
        context.add_element_bbox_for_cleanup(
            bbox,
            margin_px=cleanup_margin_px,
            min_margin_px=TEXT_CLEANUP_MIN_MARGIN_PX,
        )

        ir_runs = elem.get("text_runs")
        if isinstance(ir_runs, list):
            text_runs, is_single_line = self._build_text_runs_from_ir_runs(context, elem, ir_runs)
            if text_runs:
                context.add_processed_element("text", {"bbox": bbox, "text_runs": text_runs, "is_single_line": is_single_line})
                return

        text_content = elem.get("text", "")
        if text_content:
            text_runs, is_single_line = self._build_text_runs_from_ir_runs(
                context,
                {
                    **elem,
                    "text_runs": [{
                        "text": str(text_content),
                        "bbox": bbox,
                        "line_index": 0,
                        "style": elem.get("style") or {},
                    }],
                },
                [{
                    "text": str(text_content),
                    "bbox": bbox,
                    "line_index": 0,
                    "style": elem.get("style") or {},
                }],
            )
            if text_runs:
                context.add_processed_element("text", {"bbox": bbox, "text_runs": text_runs, "is_single_line": is_single_line})

    def _render_text_from_data(self, slide, text_data):
        """Renders a text element from processed data onto a slide."""
        bbox = text_data['bbox']
        text_runs = text_data.get('text_runs', [])
        is_single_line = text_data.get('is_single_line', False)

        # If it's a single line, widen the textbox to prevent wrapping due to font differences.
        if is_single_line:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            new_x2 = x1 + width * 1.2
            render_bbox = [x1, y1, new_x2, y2]
        else:
            render_bbox = bbox

        txBox = self._create_textbox(slide, render_bbox, self.coords_for_render)
        tf = txBox.text_frame
        tf.clear()
        tf.margin_bottom = tf.margin_top = tf.margin_left = tf.margin_right = Pt(0)
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT

        for run_data in text_runs:
            run = p.add_run()
            run.text = run_data['text']
            font = run.font
            font_info = run_data.get('font', {})
            font.name = font_info.get('name', "Microsoft YaHei")
            if 'size' in font_info: font.size = font_info['size']
            if 'color' in font_info: font.color.rgb = font_info['color']
            if 'bold' in font_info: font.bold = font_info['bold']

    def _add_picture_from_bbox(self, slide, bbox, page_image, coords, text_elements):
        if not bbox: return
        x1, y1, x2, y2 = bbox;
        left, top, w, h = Pt(x1 * coords['scale_x']), Pt(y1 * coords['scale_y']), Pt((x2 - x1) * coords['scale_x']), Pt(
            (y2 - y1) * coords['scale_y'])
        px_box = [int(x1 * coords['img_w'] / coords['json_w']), int(y1 * coords['img_h'] / coords['json_h']),
                  int(x2 * coords['img_w'] / coords['json_w']), int(y2 * coords['img_h'] / coords['json_h'])]
        crop = page_image[px_box[1]:px_box[3], px_box[0]:px_box[2]].copy()

        # This cleanup logic is now less critical due to the global background inpainting,
        # but can still be useful for images that contain text not defined as a separate text element.
        for txt_e in text_elements:
            txt_box = txt_e.get("bbox")
            if txt_box and self._get_bbox_intersection(bbox, txt_box):
                px_txt_box = [int(v * (
                    coords['img_w'] / coords['json_w'] if i % 2 == 0 else coords['img_h'] / coords['json_h'])) for
                              i, v in enumerate(txt_box)]
                inter = self._get_bbox_intersection(px_box, px_txt_box)
                if inter:
                    local_inter = [inter[0] - px_box[0], inter[1] - px_box[1], inter[2] - px_box[0],
                                   inter[3] - px_box[1]]
                    fill_bbox_with_bg(crop, local_inter)

        if crop.size > 0:
            path = f"temp_crop_img_{x1}_{y1}.png";
            cv2.imwrite(path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            slide.shapes.add_picture(path, left, top, w, h);
            os.remove(path)

    def _process_image(self, context, elem, text_elements):
        context.add_element_bbox_for_cleanup(elem.get("bbox"))
        context.add_processed_element('image', {'bbox': elem.get("bbox"), 'text_elements': text_elements})

    def _process_element(self, context, elem, all_text_elements):
        cat = elem.get("type", "text")
        if cat == "text":
            self._process_text(context, elem)
        elif cat == "image":
            self._process_image(context, elem, all_text_elements)

    def process_page(self, slide, elements, page_image, page_size=None, page_index=0, debug_images=False):
        self.debug_images = debug_images
        elements = validate_ir_elements(elements)
        elements = materialize_text_runs_for_elements(elements)
        elements = validate_ir_elements(elements, require_text_runs_consistency=True)

        img_h, img_w = page_image.shape[:2]
        json_w, json_h = page_size if page_size and all(page_size) else (img_w * 72 / 300, img_h * 72 / 300)
        w_pts, h_pts = self.cap_size(json_w, json_h)
        self.prs.slide_width, self.prs.slide_height = Pt(w_pts), Pt(h_pts)
        coords = {'scale_x': w_pts / json_w, 'scale_y': h_pts / json_h, 'img_w': img_w, 'img_h': img_h,
                  'json_w': json_w, 'json_h': json_h}
        self.coords_for_render = coords

        context = PageContext(page_image, coords, slide)

        all_text_elements = [e for e in elements if e.get("type") == "text"]

        for elem in elements:
            is_discarded = elem.get('is_discarded', False)
            if not is_discarded or (is_discarded and self.remove_watermark):
                context.add_element_bbox_for_cleanup(elem.get("bbox"))

        for elem in elements:
            if elem.get('is_discarded') and self.remove_watermark:
                continue
            self._process_element(context, elem, all_text_elements)

        context.render_to_slide(self)

        if self.debug_images:
            context.generate_debug_images(page_index, self)

    def save(self):
        self.prs.save(self.output_path)


def convert_mineru_to_ppt(
    json_path,
    input_path,
    output_ppt_path,
    remove_watermark=True,
    debug_images=False,
    ocr_engine=None,
    ocr_device_policy="auto",
    ocr_model_root=None,
    ocr_offline_only=True,
):
    from .utils import pdf_to_images
    DPI = 300

    if debug_images:
        if os.path.exists("tmp"):
            shutil.rmtree("tmp")
        os.makedirs("tmp")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if input_path.lower().endswith('.pdf'):
        images = pdf_to_images(input_path, dpi=DPI)
    else:
        try:
            img = Image.open(input_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images = [np.array(img)]
        except Exception as e:
            raise IOError(f"Failed to load image file: {input_path} - {e}")

    if ocr_engine is None:
        ocr_engine = PaddleOCREngine(
            device_policy=ocr_device_policy,
            model_root=ocr_model_root,
            offline_only=ocr_offline_only,
        )

    mineru_adapter = MinerUAdapter()
    ocr_adapter = OCRAdapter(ocr_engine)

    gen = PPTGenerator(
        output_ppt_path,
        remove_watermark=remove_watermark,
        ocr_engine=ocr_engine,
        ocr_device_policy=ocr_device_policy,
        ocr_model_root=ocr_model_root,
        ocr_offline_only=ocr_offline_only,
    )
    pages = data if isinstance(data, list) else next(
        (data[k] for k in ["pdf_info", "pages"] if k in data and isinstance(data[k], list)), [data]
    )
    print(f"[CLEANUP] Found {len(pages)} pages.")
    for i, page_data in enumerate(pages):
        if i >= len(images):
            break
        print(f"Processing page {i + 1}/{len(pages)}...")
        page_img = images[i].copy()
        if i == 0:
            gen.set_slide_size(page_img.shape[1], page_img.shape[0], dpi=DPI)
        slide = gen.add_slide()

        page_size = page_data.get("page_size") or (
            page_data.get("page_info", {}).get("width"),
            page_data.get("page_info", {}).get("height"),
        )
        json_w, json_h = page_size if page_size and all(page_size) else (page_img.shape[1], page_img.shape[0])

        mineru_page = MinerUPageData.from_dict(page_data)
        mineru_elements = validate_ir_elements(mineru_adapter.extract_page_elements(mineru_page, include_text_runs=False))

        try:
            ocr_elements = validate_ir_elements(ocr_adapter.extract_page_elements(page_img, json_w, json_h))
        except Exception as e:
            raise RuntimeError(f"[OCR] Page {i + 1}: OCR extraction failed: {e}") from e

        merged_elements, merge_stats = merge_ir_elements(mineru_elements, ocr_elements, gen._has_bbox_overlap)
        print(
            f"[OCR] Page {i + 1}: "
            f"candidates={merge_stats['overlay_candidates']}, "
            f"group_replaced={merge_stats['group_replaced']}, "
            f"overlap_replaced={merge_stats['overlap_replaced']}, "
            f"added={merge_stats['overlay_added']}"
        )

        gen.process_page(slide, merged_elements, page_img, page_size=page_size, page_index=i, debug_images=debug_images)
    gen.save()
    print(f"Saved to {output_ppt_path}")
