import fitz
import cv2
import numpy as np
from PIL import Image
import io
import math
from collections import Counter

def pdf_to_images(pdf_path, dpi=300):
    """Convert PDF pages to high-quality images."""
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(np.array(img))
    doc.close()
    return images

def extract_background_color(image, bbox):
    """
    Extracts the most likely background color from a bounding box by analyzing its edges.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return (255, 255, 255)

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return (255, 255, 255)

    edges = np.concatenate([roi[0, :], roi[-1, :], roi[:, 0], roi[:, -1]], axis=0)
    return tuple(map(int, np.median(edges, axis=0)))

def extract_font_color(image, bbox, bg_color):
    """
    Extracts font color and projection proportions from a bounding box.
    Returns: (font_color, x_proportion, y_proportion)
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return (0, 0, 0), 0, 0

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return (0, 0, 0), 0, 0

    pixels = roi.reshape(-1, 3)
    diff_bg = np.linalg.norm(pixels - bg_color, axis=1)
    fg_mask = (diff_bg > 60).reshape(roi.shape[:2])

    x_scores, y_scores = Counter(), Counter()
    # Y-axis projection
    for row in range(roi.shape[0]):
        if np.any(fg_mask[row, :]):
            for color in np.unique(roi[row, fg_mask[row, :]], axis=0):
                y_scores[tuple(color)] += 1
    # X-axis projection
    for col in range(roi.shape[1]):
        if np.any(fg_mask[:, col]):
            for color in np.unique(roi[fg_mask[:, col], col], axis=0):
                x_scores[tuple(color)] += 1

    # Combine scores to find the best candidate color
    combined_scores = x_scores + y_scores
    if not combined_scores:
        font_color = tuple(map(int, pixels[np.argmax(diff_bg)])) if len(pixels) > 0 else (0,0,0)
        return font_color, 0, 0

    font_color = combined_scores.most_common(1)[0][0]

    # Calculate proportions
    x_score = x_scores.get(font_color, 0)
    y_score = y_scores.get(font_color, 0)
    total_score = x_score + y_score

    x_proportion = x_score / total_score if total_score > 0 else 0
    y_proportion = y_score / total_score if total_score > 0 else 0

    return tuple(map(int, font_color)), x_proportion, y_proportion


def get_projection_segments(roi, color, axis, threshold=40, min_count=1, min_length=3):
    """
    Analyzes an image region of interest (ROI) to find segments of a specific color
    along a given axis (0 for columns/x-axis, 1 for rows/y-axis).

    Args:
        roi (np.array): The image data of the region.
        color (tuple): The BGR color to segment.
        axis (int): The axis for projection (0 for x-axis, 1 for y-axis).
        threshold (int): The maximum color distance to be considered a match.
        min_count (int): The minimum number of pixels in a projection line to be considered 'on'.
        min_length (int): The minimum length of a segment to be included.

    Returns:
        list of tuples: A list of (start, end) tuples for each detected segment.
    """
    if roi.size == 0:
        return []

    diff = np.linalg.norm(roi.reshape(-1, 3) - color, axis=1)
    fg_mask = (diff < threshold).reshape(roi.shape[:2])
    counts = np.sum(fg_mask, axis=axis)

    segments = []
    in_segment, start = False, 0
    for i, count in enumerate(np.append(counts, 0)):
        if count > min_count and not in_segment:
            in_segment, start = True, i
        elif count <= min_count and in_segment:
            in_segment = False
            if i - start >= min_length:
                segments.append((start, i))
    return segments


def fill_bbox_with_bg(image, bbox):
    """
    Fill the bbox area in the image with its estimated background color.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return image

    bg_color = extract_background_color(image, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), bg_color, -1)
    return image
