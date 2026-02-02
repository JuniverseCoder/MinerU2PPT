import os
import json
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from .utils import extract_background_color, extract_font_color, fill_bbox_with_bg, get_projection_segments
import cv2
import numpy as np
import string
from PIL import ImageFont


class Character:
    """A simple data class to hold information about a detected character."""

    def __init__(self, bbox, color, line_index):
        self.bbox = bbox
        self.color = color
        self.line_index = line_index
        self.font_size = 0
        self.bold = False
        self.text = ""

    def __repr__(self):
        return (f"Character(text='{self.text}', bbox={self.bbox}, color={self.color}, line={self.line_index}, "
                f"size={self.font_size}, bold={self.bold})")


class PPTGenerator:
    def __init__(self, output_path, perform_cleanup=True):
        self.prs = Presentation()
        self.output_path = output_path
        self.perform_cleanup = perform_cleanup
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

    def _create_textbox(self, slide, bbox, coords):
        x1, y1, x2, y2 = bbox
        return slide.shapes.add_textbox(
            Pt(x1 * coords['scale_x']), Pt(y1 * coords['scale_y']),
            Pt((x2 - x1) * coords['scale_x']), Pt((y2 - y1) * coords['scale_y'])
        )

    def _get_line_ranges(self, page_image, bbox, coords):
        x1, y1, x2, y2 = bbox
        px1, py1, px2, py2 = int(x1 * coords['img_w'] / coords['json_w']), int(
            y1 * coords['img_h'] / coords['json_h']), int(x2 * coords['img_w'] / coords['json_w']), int(
            y2 * coords['img_h'] / coords['json_h'])
        h, w = page_image.shape[:2];
        px1, py1, px2, py2 = max(0, px1), max(0, py1), min(w, px2), min(h, py2)
        if px2 <= px1 or py2 <= py1: return []
        roi = page_image[py1:py2, px1:px2]
        bg_color = extract_background_color(page_image, [px1, py1, px2, py2])
        font_color, _, _ = extract_font_color(page_image, [px1, py1, px2, py2], bg_color)
        initial_lines = get_projection_segments(roi, font_color, axis=1)
        line_infos = []
        scale_y = (y2 - y1) / roi.shape[0] if roi.shape[0] > 0 else 0
        for start_y, end_y in initial_lines:
            line_pixel_bbox = [px1, py1 + start_y, px2, py1 + end_y]
            line_bg = extract_background_color(page_image, line_pixel_bbox)
            line_fg, _, _ = extract_font_color(page_image, line_pixel_bbox, line_bg)
            line_infos.append({'range': [y1 + start_y * scale_y, y1 + end_y * scale_y], 'color': line_fg,
                               'pixel_range': (start_y, end_y)})
        if not line_infos: return []
        avg_line_height = np.mean([info['pixel_range'][1] - info['pixel_range'][0] for info in line_infos])
        recovered_lines = []
        sorted_lines = sorted(line_infos, key=lambda x: x['pixel_range'][0])
        all_gaps = [(0, sorted_lines[0]['pixel_range'][0])] + [
            (sorted_lines[i]['pixel_range'][1], sorted_lines[i + 1]['pixel_range'][0]) for i in
            range(len(sorted_lines) - 1)] + [(sorted_lines[-1]['pixel_range'][1], roi.shape[0])]
        for gap_start, gap_end in all_gaps:
            if (gap_end - gap_start) > avg_line_height * 0.8:
                gap_bbox = [px1, py1 + gap_start, px2, py1 + gap_end]
                gap_bg = extract_background_color(page_image, gap_bbox)
                new_font_color, x_prop, y_prop = extract_font_color(page_image, gap_bbox, gap_bg)
                if y_prop > x_prop * 1.2 and np.linalg.norm(np.array(new_font_color) - np.array(font_color)) > 50:
                    gap_roi = roi[gap_start:gap_end, :];
                    gap_pixels = gap_roi.reshape(-1, 3)
                    gap_diff = np.linalg.norm(gap_pixels - new_font_color, axis=1)
                    gap_mask = (gap_diff < 40).reshape(gap_roi.shape[:2])
                    gap_row_counts = np.sum(gap_mask, axis=1)
                    in_gap_line, gap_line_start = False, 0
                    for y, count in enumerate(gap_row_counts):
                        if count > 1 and not in_gap_line:
                            in_gap_line, gap_line_start = True, y
                        elif count < 1 and in_gap_line:
                            in_gap_line = False
                            if y - gap_line_start > 3:
                                abs_start, abs_end = gap_start + gap_line_start, gap_start + y
                                recovered_lines.append({'range': [y1 + abs_start * scale_y, y1 + abs_end * scale_y],
                                                        'color': new_font_color, 'pixel_range': (abs_start, abs_end)})
                    if in_gap_line:
                        abs_start, abs_end = gap_start + gap_line_start, gap_start + len(gap_row_counts)
                        recovered_lines.append(
                            {'range': [y1 + abs_start * scale_y, y1 + abs_end * scale_y], 'color': new_font_color,
                             'pixel_range': (abs_start, abs_end)})
        if recovered_lines:
            line_infos.extend(recovered_lines)
            line_infos.sort(key=lambda x: x['range'][0])

        if len(line_infos) > 1:
            avg_line_height = np.mean([info['pixel_range'][1] - info['pixel_range'][0] for info in line_infos])
            merged_lines = [line_infos[0]]
            for i in range(1, len(line_infos)):
                prev_line = merged_lines[-1]
                curr_line = line_infos[i]
                gap = curr_line['pixel_range'][0] - prev_line['pixel_range'][1]
                if gap >= 0 and gap <= max(avg_line_height * 0.05, 1):
                    prev_height = prev_line['pixel_range'][1] - prev_line['pixel_range'][0]
                    curr_height = curr_line['pixel_range'][1] - curr_line['pixel_range'][0]
                    new_pixel_range = (prev_line['pixel_range'][0], curr_line['pixel_range'][1])
                    new_range = [prev_line['range'][0], curr_line['range'][1]]
                    new_color = curr_line['color'] if curr_height > prev_height else prev_line['color']
                    merged_lines[-1] = {'range': new_range, 'color': new_color, 'pixel_range': new_pixel_range}
                else:
                    merged_lines.append(curr_line)
            line_infos = merged_lines

        return line_infos

    def _detect_raw_characters(self, page_image, line_infos, bbox, coords):
        char_objects = []
        for i, info in enumerate(line_infos):
            line_bbox = [bbox[0], info['range'][0], bbox[2], info['range'][1]]
            char_objects.extend(self._detect_characters_from_line(page_image, line_bbox, coords, info['color'], i))
        return char_objects

    def _detect_characters_from_line(self, page_image, bbox, coords, line_color, line_index):
        x1, y1, x2, y2 = bbox
        px1, py1, px2, py2 = int(x1 * coords['img_w'] / coords['json_w']), int(
            y1 * coords['img_h'] / coords['json_h']), int(x2 * coords['img_w'] / coords['json_w']), int(
            y2 * coords['img_h'] / coords['json_h'])
        h, w = page_image.shape[:2];
        px1, py1, px2, py2 = max(0, px1), max(0, py1), min(w, px2), min(h, py2)
        if px2 <= px1 or py2 <= py1: return []

        main_roi = page_image[py1:py2, px1:px2]
        all_chars = self._segment_characters_in_roi(main_roi, (px1, py1, px2, py2), line_color, line_index, bbox,
                                                    (px1, py1, px2, py2))

        if not all_chars:
            return []

        sorted_chars = sorted(all_chars, key=lambda c: c.bbox[0])

        gaps = []
        last_x2 = x1
        for char in sorted_chars:
            if char.bbox[0] > last_x2:
                gaps.append((last_x2, char.bbox[0]))
            last_x2 = char.bbox[2]
        if x2 > last_x2:
            gaps.append((last_x2, x2))

        recovered_chars = []
        scale_x_inv = (px2 - px1) / (x2 - x1) if (x2 - x1) > 0 else 0
        min_gap_width = max((py2 - py1) * 2, 6)

        for gap_x1, gap_x2 in gaps:
            gap_px1 = px1 + int((gap_x1 - x1) * scale_x_inv)
            gap_px2 = px1 + int((gap_x2 - x1) * scale_x_inv)

            # Optimization: Skip gaps that are too narrow to contain a character before expensive processing.
            if gap_px2 - gap_px1 < min_gap_width:
                continue

            gap_roi = page_image[py1:py2, gap_px1:gap_px2]
            if gap_roi.size == 0: continue

            gap_bg = extract_background_color(page_image, [gap_px1, py1, gap_px2, py2])
            new_font_color, x_prop, y_prop = extract_font_color(page_image, [gap_px1, py1, gap_px2, py2], gap_bg)

            if (x_prop > 0.5 or y_prop > 0.5) and np.linalg.norm(
                    np.array(new_font_color) - np.array(line_color)) > 50:
                # For text of different colors, which may have a different height,
                # first use y-axis projection to adjust py1. Assumes bottom-alignment.
                segments = get_projection_segments(gap_roi, new_font_color, axis=1)

                if segments:
                    local_py1 = segments[0][0]
                    adjusted_py1 = py1 + local_py1
                    adjusted_gap_roi = page_image[adjusted_py1:py2, gap_px1:gap_px2]

                    if adjusted_gap_roi.size > 0:
                        recovered_chars.extend(
                            self._segment_characters_in_roi(adjusted_gap_roi, (gap_px1, adjusted_py1, gap_px2, py2),
                                                            new_font_color, line_index, bbox, (px1, py1, px2, py2)))

        if recovered_chars:
            all_chars.extend(recovered_chars)
            all_chars.sort(key=lambda c: c.bbox[0])

        return all_chars

    def _segment_characters_in_roi(self, roi, roi_bbox, color, base_line_index, line_json_bbox, line_pixel_bbox):
        local_px1, local_py1, local_px2, local_py2 = roi_bbox
        x1, y1, x2, y2 = line_json_bbox
        px1, py1, px2, py2 = line_pixel_bbox

        initial_chars_px = get_projection_segments(roi, color, axis=0, min_length=2)
        char_objects = []

        # Calculate scales based on the overall line dimensions
        scale_x = (x2 - x1) / (px2 - px1) if (px2 - px1) > 0 else 0
        scale_y = (y2 - y1) / (py2 - py1) if (py2 - py1) > 0 else 0

        # Calculate the JSON y-coordinates corresponding to the ROI's vertical bounds
        char_json_y1 = y1 + (local_py1 - py1) * scale_y
        char_json_y2 = y1 + (local_py2 - py1) * scale_y

        for sx, ex in initial_chars_px:
            # Calculate JSON x-coordinates for the character
            char_json_x1 = x1 + (local_px1 - px1 + sx) * scale_x
            char_json_x2 = x1 + (local_px1 - px1 + ex) * scale_x

            char_bbox = [char_json_x1, char_json_y1, char_json_x2, char_json_y2]
            char_objects.append(Character(bbox=char_bbox, color=color, line_index=base_line_index))

        return char_objects


    def _analyze_and_correct_bboxes(self, char_objects, full_text, coords):
        non_space_chars = [c for c in full_text if c not in " \n"]
        expected_count = len(non_space_chars)
        if not char_objects or len(char_objects) < expected_count:
            return char_objects

        chars = sorted(char_objects, key=lambda c: (c.line_index, c.bbox[0]))
        num_fragments = len(chars)
        num_chars = len(non_space_chars)

        try:
            font = ImageFont.truetype("msyh.ttc", size=30)
            ideal_height = 30
            ideal_char_ratios = [(font.getlength(c) / ideal_height) for c in non_space_chars]
        except IOError:
            return chars

        memo_cost = {}

        def get_merge_cost(start, end, char_idx):
            if (start, end, char_idx) in memo_cost:
                return memo_cost[(start, end, char_idx)]

            merged_bbox = [chars[start].bbox[0],
                           min(c.bbox[1] for c in chars[start:end]),
                           chars[end - 1].bbox[2],
                           max(c.bbox[3] for c in chars[start:end])]

            merged_width = merged_bbox[2] - merged_bbox[0]
            merged_height = merged_bbox[3] - merged_bbox[1]
            if merged_height == 0:
                return float('inf')

            merged_ratio = merged_width / merged_height
            cost = abs(merged_ratio - ideal_char_ratios[char_idx])
            memo_cost[(start, end, char_idx)] = cost
            return cost

        dp = [[float('inf')] * (num_chars + 1) for _ in range(num_fragments + 1)]
        path = [[0] * (num_chars + 1) for _ in range(num_fragments + 1)]
        dp[0][0] = 0

        for j in range(1, num_chars + 1):
            for i in range(1, num_fragments + 1):
                for k in range(i):
                    cost = get_merge_cost(k, i, j - 1)
                    if dp[k][j - 1] + cost < dp[i][j]:
                        dp[i][j] = dp[k][j - 1] + cost
                        path[i][j] = k

        if dp[num_fragments][num_chars] == float('inf'):
            return []

        final_chars = []
        curr_frag = num_fragments
        for curr_char in range(num_chars, 0, -1):
            prev_frag = path[curr_frag][curr_char]

            merged_bbox = [chars[prev_frag].bbox[0],
                           min(c.bbox[1] for c in chars[prev_frag:curr_frag]),
                           chars[curr_frag - 1].bbox[2],
                           max(c.bbox[3] for c in chars[prev_frag:curr_frag])]

            new_char = Character(merged_bbox, chars[prev_frag].color, chars[prev_frag].line_index)
            new_char.text = non_space_chars[curr_char - 1]
            final_chars.append(new_char)
            curr_frag = prev_frag

        return final_chars[::-1]

    def _determine_character_styles(self, final_chars, coords, elem_type):
        for char in final_chars:
            height_pts = (char.bbox[3] - char.bbox[1]) * coords['scale_y']
            char.font_size = int(max(height_pts, 6.0))
            char.bold = elem_type == "title"
        return final_chars

    def _process_text(self, slide, elem, page_image, coords):
        bbox = elem.get("bbox");
        if not bbox: return
        txBox = self._create_textbox(slide, bbox, coords)
        tf = txBox.text_frame;
        tf.clear();
        tf.margin_bottom = tf.margin_top = tf.margin_left = tf.margin_right = Pt(0)
        tf.word_wrap = True;
        p = tf.paragraphs[0];
        p.alignment = PP_ALIGN.LEFT
        all_spans = [s for l in elem.get("lines", []) for s in l.get("spans", [])] if "lines" in elem else elem.get(
            "spans", [])
        if not all_spans: p.text = elem.get("text", ""); return
        if all_spans:
            first_span_content = all_spans[0].get("content", "")
            if first_span_content.lstrip().startswith('-'):
                cleaned_content = first_span_content.lstrip(' \t\n\r\f\v-•*·')
                all_spans[0]["content"] = "• " + cleaned_content
        full_text = "".join([s.get("content", "").replace('\\%', '%') for s in all_spans])
        if not full_text.strip(): return
        print(f"\n--- Processing Text ---\nContent: '{full_text.strip()[:100]}...'")
        try:
            line_infos = self._get_line_ranges(page_image, bbox, coords)
            print(f"Detected lines: {len(line_infos)}")
            if not line_infos: raise ValueError("No lines detected.")
            raw_chars = self._detect_raw_characters(page_image, line_infos, bbox, coords)
            corrected_chars = self._analyze_and_correct_bboxes(raw_chars, full_text, coords)
            non_space_chars = [c for c in full_text if c not in " \n"]
            can_align = len(corrected_chars) == len(non_space_chars)
            print(f"Character alignment successful: {can_align}")
            print(f"Using char-by-char styling (mixed layout support): {can_align}")
            if not can_align:
                self._render_by_line(p, all_spans, line_infos, coords, elem.get("type"))
                return
            final_styles = self._determine_character_styles(corrected_chars, coords, elem.get("type"))
            style_iter = iter(final_styles)
            last_style = None
            for char in full_text:
                run = p.add_run();
                run.text = char;
                font = run.font;
                font.name = "Microsoft YaHei"
                if char not in " \n":
                    style = next(style_iter, None)
                    if style:
                        font.size = Pt(style.font_size);
                        font.color.rgb = RGBColor(*style.color);
                        font.bold = style.bold
                        last_style = style
                elif last_style:
                    font.size = Pt(last_style.font_size);
                    font.color.rgb = RGBColor(*last_style.color);
                    font.bold = last_style.bold
        except Exception:
            sp = txBox.element;
            sp.getparent().remove(sp)
            self._render_spans_in_bbox(slide, all_spans, bbox, page_image, coords, elem_type=elem.get("type"))

    def _render_by_line(self, paragraph, all_spans, line_infos, coords, elem_type):
        span_idx = 0
        for i, info in enumerate(line_infos):
            line_range = info['range'];
            line_spans = []
            while span_idx < len(all_spans):
                span = all_spans[span_idx];
                sbbox = span.get("bbox")
                if sbbox and sbbox[1] < line_range[1] and sbbox[3] > line_range[0]:
                    line_spans.append(span);
                    span_idx += 1
                else:
                    break
            if not line_spans: continue
            line_text = "".join([s.get("content", "").replace('\\%', '%') for s in line_spans])
            if not line_text.strip() and i < len(line_infos) - 1: line_text += "\n"
            run = paragraph.add_run();
            run.text = line_text;
            font = run.font
            font.name = "Microsoft YaHei";
            font.color.rgb = RGBColor(*info['color'])
            font_size_pts = (line_range[1] - line_range[0]) * coords['scale_y']
            font.size = Pt(int(max(font_size_pts, 6.0)))
            if elem_type == "title": font.bold = True
            if i < len(line_infos) - 1 and not line_text.endswith('\n'): paragraph.add_run().text = '\n'

    def _render_spans_in_bbox(self, slide, spans, bbox, page_image, coords, elem_type=None):
        if not spans: return
        txBox = self._create_textbox(slide, bbox, coords)
        p = txBox.text_frame.paragraphs[0];
        txBox.text_frame.clear()
        font_size_pts = (bbox[3] - bbox[1]) * coords['scale_y']
        full_text = "".join([s.get("content", "").replace('\\%', '%') for s in spans])
        run = p.add_run();
        run.text = full_text;
        font = run.font
        font.name = "Microsoft YaHei";
        font.size = Pt(int(font_size_pts))
        if elem_type == "title": font.bold = True
        bg_color = extract_background_color(page_image, bbox)
        color, _, _ = extract_font_color(page_image, bbox, bg_color)
        font.color.rgb = RGBColor(*color)

    def _process_list(self, slide, elem, page_image, coords):
        for block in elem.get("blocks", []):
            spans = [s for l in block.get("lines", []) for s in l.get("spans", [])] if "lines" in block else block.get(
                "spans", [])
            if spans:
                spans.sort(key=lambda s: (s.get("bbox", [0, 0, 0, 0])[1], s.get("bbox", [0, 0, 0, 0])[0]))
                spans[0]["content"] = "• " + spans[0].get("content", "").lstrip(' ·-*•')
                self._process_text(slide, block, page_image, coords)

    def _process_image(self, slide, elem, page_image, coords, text_elements):
        bbox = elem.get("bbox")
        if not bbox: return
        x1, y1, x2, y2 = bbox;
        left, top, w, h = Pt(x1 * coords['scale_x']), Pt(y1 * coords['scale_y']), Pt((x2 - x1) * coords['scale_x']), Pt(
            (y2 - y1) * coords['scale_y'])
        px_box = [int(x1 * coords['img_w'] / coords['json_w']), int(y1 * coords['img_h'] / coords['json_h']),
                  int(x2 * coords['img_w'] / coords['json_w']), int(y2 * coords['img_h'] / coords['json_h'])]
        crop = page_image[px_box[1]:px_box[3], px_box[0]:px_box[2]]
        if self.perform_cleanup and text_elements:
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
            path = f"temp_crop_{elem.get('type')}_{x1}_{y1}.png";
            cv2.imwrite(path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            slide.shapes.add_picture(path, left, top, w, h);
            os.remove(path)

    def _process_element(self, slide, elem, page_image, coords, text_elements=None):
        cat = elem.get("type", "text")
        if cat == "list":
            self._process_list(slide, elem, page_image, coords)
        elif cat in ["text", "title", "caption", "footnote", "footer", "header", "page_number"]:
            self._process_text(slide, elem, page_image, coords)
        elif cat in ["image", "table", "formula", "figure"]:
            self._process_image(slide, elem, page_image, coords, text_elements or [])

    def process_page(self, slide, elements, page_image, page_size=None):
        img_h, img_w = page_image.shape[:2]
        json_w, json_h = page_size if page_size and all(page_size) else (img_w * 72 / 300, img_h * 72 / 300)
        w_pts, h_pts = self.cap_size(json_w, json_h)
        self.prs.slide_width, self.prs.slide_height = Pt(w_pts), Pt(h_pts)
        coords = {'scale_x': w_pts / json_w, 'scale_y': h_pts / json_h, 'img_w': img_w, 'img_h': img_h,
                  'json_w': json_w, 'json_h': json_h}
        text_types = ["list", "text", "title", "caption", "footnote", "footer", "header", "page_number"]
        img_types = ["image", "table", "formula", "figure"]
        text_elems = [e for e in elements if e.get("type", "text") in text_types]
        img_elems = [e for e in elements if e.get("type") in img_types]
        original_img = page_image.copy()
        if self.perform_cleanup:
            for elem in elements:
                if elem.get("bbox"):
                    px_box = [int(v * (
                        coords['img_w'] / coords['json_w'] if i % 2 == 0 else coords['img_h'] / coords['json_h'])) for
                              i, v in enumerate(elem["bbox"])]
                    fill_bbox_with_bg(page_image, px_box)
        bg_path = f"temp_bg_{id(slide)}.png";
        cv2.imwrite(bg_path, cv2.cvtColor(page_image, cv2.COLOR_RGB2BGR))
        slide.shapes.add_picture(bg_path, Pt(0), Pt(0), Pt(w_pts), Pt(h_pts));
        os.remove(bg_path)
        for elem in img_elems: self._process_element(slide, elem, original_img, coords, text_elements=text_elems)
        for elem in text_elems: self._process_element(slide, elem, original_img, coords)

    def save(self):
        self.prs.save(self.output_path)


def convert_mineru_to_ppt(json_path, pdf_path, output_ppt_path, debug=False):
    from .utils import pdf_to_images
    DPI = 300
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def _generate(output_path, cleanup):
        images = pdf_to_images(pdf_path, dpi=DPI)
        gen = PPTGenerator(output_path, perform_cleanup=cleanup)
        pages = data if isinstance(data, list) else next(
            (data[k] for k in ["pdf_info", "pages"] if k in data and isinstance(data[k], list)), [data])
        print(f"[{'CLEANUP' if cleanup else 'DEBUG'}] Found {len(pages)} pages.")
        for i, page_data in enumerate(pages):
            if i >= len(images): break
            print(f"Processing page {i + 1}/{len(pages)}...")
            page_img = images[i].copy()
            if i == 0: gen.set_slide_size(page_img.shape[1], page_img.shape[0], dpi=DPI)
            slide = gen.add_slide()
            elements = [item for key in ["para_blocks", "images", "tables"] for item in page_data.get(key, [])]
            page_size = page_data.get("page_size") or (page_data.get("page_info", {}).get("width"),
                                                       page_data.get("page_info", {}).get("height"))
            gen.process_page(slide, elements, page_img, page_size=page_size)
        gen.save()
        print(f"Saved to {output_path}")

    _generate(output_ppt_path, True)
    if debug: _generate(output_ppt_path.replace(".pptx", "_debug.pptx"), False)
