import sys
import json
import cv2
import argparse
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from app import extract_km_curves
from axis_extraction import extract_axis_limits


def visualize_curves(img_path, curves, out_path):
    """Overlay detected curves on the original image."""
    img = cv2.imread(img_path)
    axes = extract_axis_limits(img_path)
    if axes is None:
        print("Could not detect axes for visualization.")
        return

    x_min_px = axes["x_min_px"]
    x_max_px = axes["x_max_px"]
    y_min_px = axes["y_min_px"]
    y_max_px = axes["y_max_px"]
    max_t = axes["x_max_val"]

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for idx, curve in enumerate(curves):
        color = colors[idx % len(colors)]
        for pt in curve:
            x_val, y_val = pt
            if max_t == 0:
                continue
            px = int(x_min_px + (x_val / max_t) * (x_max_px - x_min_px))
            py = int(y_min_px + (1.0 - y_val) * (y_max_px - y_min_px))
            cv2.circle(img, (px, py), 2, color, -1)

    cv2.imwrite(out_path, img)
    print(f"Visualization saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KM curve inference on a single image")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--visualize", action="store_true", help="Save visualization overlay")
    args = parser.parse_args()

    curves = extract_km_curves(args.image)
    print(json.dumps({"curves": curves}, indent=2))

    if args.visualize:
        out_name = "visualized_" + os.path.basename(args.image)
        visualize_curves(args.image, curves, out_name)
