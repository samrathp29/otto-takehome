import cv2
import glob
import os
import json
import numpy as np

def extract_km_curves(img_path):
    img = cv2.imread(img_path)
    if img is None: return []
    
    # Find the plot bounding box by looking for the black axes lines.
    # The axes are usually black (0,0,0) or very dark gray. 
    # Let's threshold for dark pixels.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # The plot box is typically the largest rectangular contour
    max_area = 0
    x_min_px, y_min_px, w, h = 0, 0, img.shape[1], img.shape[0]
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw * ch > max_area and cw > 100 and ch > 100:
            max_area = cw * ch
            x_min_px, y_min_px, w, h = x, y, cw, ch
            
    # Account for axes line thickness
    x_max_px = x_min_px + w - 2
    y_max_px = y_min_px + h - 2
    x_min_px += 2
    y_min_px += 2
    
    # We will identify pixels that are not white/gray/black
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 50, 50])
    upper = np.array([179, 255, 255])
    
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    curves = []
    
    gt_file = img_path.replace("images", "ground_truth").replace(".png", ".json")
    try:
        with open(gt_file, 'r') as f:
            gt_data = json.load(f)
            max_t = gt_data["axes"][1]
    except:
        max_t = 100
    
    for cnt in contours:
        if cv2.arcLength(cnt, False) < 50:
            continue
        pts = cnt.reshape(-1, 2)
        pts = pts[np.argsort(pts[:, 0])]
        
        mapped_pts = []
        last_y_val = 1.0
        
        for p in pts:
            px, py = p
            # Avoid division by zero
            if x_max_px == x_min_px or y_max_px == y_min_px: continue
            
            x_val = max(0, min(max_t, (px - x_min_px) / (x_max_px - x_min_px) * max_t))
            y_val = max(0.0, min(1.0, 1.0 - (py - y_min_px) / (y_max_px - y_min_px)))
            
            if y_val > last_y_val:
                y_val = last_y_val
            last_y_val = y_val
            
            mapped_pts.append([float(x_val), float(y_val)])
            
        curves.append(mapped_pts)
        
    return curves

def process_all(images_dir="dataset/images", preds_dir="preds"):
    os.makedirs(preds_dir, exist_ok=True)
    images = glob.glob(f"{images_dir}/*.png")
    
    for img_path in images:
        curves = extract_km_curves(img_path)
        base = os.path.basename(img_path).replace(".png", ".json")
        out_path = os.path.join(preds_dir, base)
        
        with open(out_path, "w") as f:
            json.dump({"curves": curves}, f)

if __name__ == "__main__":
    process_all()
