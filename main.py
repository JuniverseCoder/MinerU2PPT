import argparse
import os
import sys
import shutil
from converter.generator import convert_mineru_to_ppt

def main():
    parser = argparse.ArgumentParser(description="MinerU PDF/Image to PPT Converter")
    parser.add_argument("--json", required=True, help="Path to MinerU JSON file")
    parser.add_argument("--input", required=True, help="Path to original PDF/Image file")
    parser.add_argument("--output", required=True, help="Path to output PPT file")
    parser.add_argument("--no-watermark", action="store_true", help="Remove watermarks from the output")
    parser.add_argument("--debug-images", action="store_true", help="Generate debug images in the tmp/ directory")

    args = parser.parse_args()

    if args.debug_images:
        if os.path.exists("tmp"):
            shutil.rmtree("tmp")
        os.makedirs("tmp")

    print(f"Converting {args.input} to {args.output}...")
    try:
        convert_mineru_to_ppt(args.json, args.input, args.output, remove_watermark=args.no_watermark, debug_images=args.debug_images)
        print("Conversion successful.")
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
