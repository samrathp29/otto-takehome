import cv2
import glob
import os
import json
import numpy as np
import sys

# Allow imports from src/ when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from segment_lines import segment_lines
from axis_extraction import extract_axis_limits
from postprocess_steps import enforce_monotonicity


def extract_km_curves(img_path):
    """
    Full pipeline: segment lines -> extract axes -> map to data coords -> enforce monotonicity.
    No ground-truth files are read. All information comes from the image alone.
    """
    # Step 1: Get raw pixel contours of colored curves
    raw_contours = segment_lines(img_path)
    if not raw_contours:
        return []

    # Step 2: Get axis limits from the image
    axes = extract_axis_limits(img_path)
    if axes is None:
        return []

    x_min_px = axes["x_min_px"]
    x_max_px = axes["x_max_px"]
    y_min_px = axes["y_min_px"]
    y_max_px = axes["y_max_px"]
    max_t = axes["x_max_val"]

    if x_max_px == x_min_px or y_max_px == y_min_px:
        return []

    # Step 3: Map pixel contours to data coordinates
    curves = []
    for pts in raw_contours:
        mapped_pts = []
        for p in pts:
            px, py = p
            x_val = max(0, min(max_t, (px - x_min_px) / (x_max_px - x_min_px) * max_t))
            # y is inverted: py=y_min_px -> y_val=1.0, py=y_max_px -> y_val=0.0
            y_val = max(0.0, min(1.0, 1.0 - (py - y_min_px) / (y_max_px - y_min_px)))
            mapped_pts.append([float(x_val), float(y_val)])

        # Step 4: Enforce monotonicity via postprocess_steps
        mapped_pts = enforce_monotonicity(mapped_pts)
        curves.append(mapped_pts)

    return curves


def process_all(images_dir="dataset/images", preds_dir="preds"):
    os.makedirs(preds_dir, exist_ok=True)
    images = sorted(glob.glob(f"{images_dir}/*.png"))

    success_count = 0
    for img_path in images:
        try:
            curves = extract_km_curves(img_path)
            base = os.path.basename(img_path).replace(".png", ".json")
            out_path = os.path.join(preds_dir, base)

            with open(out_path, "w") as f:
                json.dump({"curves": curves}, f)
            success_count += 1
        except Exception as e:
            print(f"FAILED: {img_path}: {e}", file=sys.stderr)

    print(f"Successfully processed {success_count}/{len(images)} images.")
    return success_count, len(images)


if __name__ == "__main__":
    process_all()
