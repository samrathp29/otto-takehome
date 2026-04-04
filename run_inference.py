import sys
import json
import cv2
import argparse
import numpy as np
from src.app import extract_km_curves

def visualize_curves(img_path, curves, out_path):
    img = cv2.imread(img_path)
    
    # GT max_t is read inside extract_km_curves, but for plotting overlay
    # we just need the same bounding box approximation we used:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    x_min_px, y_min_px, w, h = 0, 0, img.shape[1], img.shape[0]
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw * ch > max_area and cw > 100 and ch > 100:
            max_area = cw * ch
            x_min_px, y_min_px, w, h = x, y, cw, ch
            
    x_max_px = x_min_px + w - 2
    y_max_px = y_min_px + h - 2
    x_min_px += 2
    y_min_px += 2

    gt_file = img_path.replace("images", "ground_truth").replace(".png", ".json")
    try:
        with open(gt_file, 'r') as f:
            gt_data = json.load(f)
            max_t = gt_data["axes"][1]
    except:
        max_t = 100
    
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    for idx, curve in enumerate(curves):
        color = colors[idx % len(colors)]
        for pt in curve:
            x_val, y_val = pt
            # Inverse map
            px = int(x_min_px + (x_val / max_t) * (x_max_px - x_min_px))
            py = int(y_min_px + (1.0 - y_val) * (y_max_px - y_min_px))
            cv2.circle(img, (px, py), 2, color, -1)
            
    cv2.imwrite(out_path, img)
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run KM curve inference")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--visualize", action="store_true", help="Save visualization overlay")
    args = parser.parse_args()
    
    curves = extract_km_curves(args.image)
    print(json.dumps({"curves": curves}, indent=2))
    
    if args.visualize:
        out_name = "visualized_" + args.image.replace("/", "_")
        visualize_curves(args.image, curves, out_name)
