import argparse
import sys
from converter.generator import convert_mineru_to_ppt, TEXT_CLEANUP_MARGIN_RATIO
from converter.ocr_merge import OCR_FONT_DISTANCE_THRESHOLD

DEFAULT_OCR_DET_DB_THRESH = None
DEFAULT_OCR_DET_DB_BOX_THRESH = None
DEFAULT_OCR_DET_DB_UNCLIP_RATIO = None

def main():
    parser = argparse.ArgumentParser(description="MinerU PDF/Image to PPT Converter")
    parser.add_argument("--json", required=True, help="Path to MinerU JSON file")
    parser.add_argument("--input", required=True, help="Path to original PDF/Image file")
    parser.add_argument("--output", required=True, help="Path to output PPT file")
    parser.add_argument("--no-watermark", action="store_true", help="Remove watermarks from the output")
    parser.add_argument("--debug-images", action="store_true", help="Generate debug images in output_dir/debug")
    parser.add_argument("--ocr-device", choices=["auto", "gpu", "cpu"], default="auto", help="OCR device policy")
    parser.add_argument("--ocr-model-root", default=None, help="Path to local PaddleOCR models root")
    parser.add_argument(
        "--ocr-model-variant",
        choices=["auto", "lite", "server"],
        default="auto",
        help="OCR model variant (auto/lite/server)",
    )
    parser.add_argument(
        "--page-range",
        default=None,
        help="PDF page range (1-based), e.g. 1,3,5-8",
    )
    parser.add_argument("--ocr-det-db-thresh", type=float, default=DEFAULT_OCR_DET_DB_THRESH, help="PaddleOCR DB threshold (det_db_thresh)")
    parser.add_argument("--ocr-det-db-box-thresh", type=float, default=DEFAULT_OCR_DET_DB_BOX_THRESH, help="PaddleOCR DB box score threshold (det_db_box_thresh)")
    parser.add_argument("--ocr-det-db-unclip-ratio", type=float, default=DEFAULT_OCR_DET_DB_UNCLIP_RATIO, help="PaddleOCR DB unclip ratio (det_db_unclip_ratio)")
    parser.add_argument("--text-cleanup-margin-ratio", type=float, default=TEXT_CLEANUP_MARGIN_RATIO, help="Text cleanup margin ratio for single-line boxes")
    parser.add_argument(
        "--ocr-font-distance-threshold",
        type=float,
        default=None,
        help="Font sensitivity (higher = larger text box). Defaults depend on model variant.",
    )
    parser.add_argument(
        "--ocr-lang",
        default="ch",
        choices=["ch", "en"],
        help="OCR language (default: ch). Use 'en' for English, 'ch' for Chinese/mixed Chinese-English.",
    )

    args = parser.parse_args()

    print(f"Converting {args.input} to {args.output}...")
    try:
        convert_mineru_to_ppt(
            args.json,
            args.input,
            args.output,
            remove_watermark=args.no_watermark,
            debug_images=args.debug_images,
            ocr_device_policy=args.ocr_device,
            ocr_model_root=args.ocr_model_root,
            ocr_model_variant=args.ocr_model_variant,
            ocr_offline_only=False,
            ocr_det_db_thresh=args.ocr_det_db_thresh,
            ocr_det_db_box_thresh=args.ocr_det_db_box_thresh,
            ocr_det_db_unclip_ratio=args.ocr_det_db_unclip_ratio,
            page_range=args.page_range,
            text_cleanup_margin_ratio=args.text_cleanup_margin_ratio,
            ocr_font_distance_threshold=args.ocr_font_distance_threshold,
            ocr_lang=args.ocr_lang,
        )
        print("Conversion successful.")
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
