import cv2
import numpy as np
import re

try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False


def extract_axis_limits(image_path):
    """
    Detect the plot bounding box and infer axis limits from the image alone.
    Uses pytesseract OCR to read tick labels. Falls back to heuristics if unavailable.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Step 1: Find plot bounding box via dark thresholding ---
    _, dark_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    bx, by, bw, bh = 0, 0, w, h
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw * ch > max_area and cw > 100 and ch > 100:
            max_area = cw * ch
            bx, by, bw, bh = x, y, cw, ch

    x_min_px = bx + 2
    y_min_px = by + 2
    x_max_px = bx + bw - 2
    y_max_px = by + bh - 2

    # --- Step 2: Read x-axis and extrapolate to plot edge ---
    x_max_val = _ocr_x_axis_with_extrapolation(gray, x_min_px, x_max_px, y_max_px, h)

    return {
        "x_min_px": int(x_min_px),
        "x_max_px": int(x_max_px),
        "y_min_px": int(y_min_px),
        "y_max_px": int(y_max_px),
        "x_min_val": 0.0,
        "x_max_val": float(x_max_val),
        "y_min_val": 0.0,
        "y_max_val": 1.0,
    }


def _ocr_x_axis_with_extrapolation(gray, x_min_px, x_max_px, y_max_px, img_h):
    """
    Read x-axis tick labels and their pixel positions via OCR.
    Then extrapolate: what data value does the RIGHT EDGE of the plot (x_max_px) correspond to?

    This handles the case where matplotlib's xlim extends beyond the last tick.
    """
    margin = min(50, img_h - y_max_px)
    if margin < 10 or not HAS_OCR:
        return 100.0

    left_pad = max(0, x_min_px - 30)
    right_pad = min(gray.shape[1], x_max_px + 30)
    strip = gray[y_max_px + 2 : y_max_px + margin, left_pad : right_pad]
    if strip.size == 0:
        return 100.0

    # Upscale 4x
    scale = 4
    strip_large = cv2.resize(strip, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, binarized = cv2.threshold(strip_large, 140, 255, cv2.THRESH_BINARY_INV)

    try:
        data = pytesseract.image_to_data(
            binarized,
            config="--psm 6 -c tessedit_char_whitelist=0123456789",
            output_type=pytesseract.Output.DICT
        )

        # Collect (pixel_x_center, numeric_value) pairs
        tick_pairs = []
        for i, txt in enumerate(data["text"]):
            txt = txt.strip()
            if not txt:
                continue
            nums = re.findall(r'\d+', txt)
            for n in nums:
                val = int(n)
                if 0 <= val <= 300:
                    # Center x of this word in original pixel coords
                    word_cx = (data["left"][i] + data["width"][i] / 2) / scale + left_pad
                    tick_pairs.append((word_cx, val))

        if len(tick_pairs) >= 2:
            # Sort by pixel position
            tick_pairs.sort(key=lambda t: t[0])

            # Linear fit: pixel_x -> data_value
            xs = np.array([t[0] for t in tick_pairs])
            vs = np.array([t[1] for t in tick_pairs])
            # Fit line: val = slope * px + intercept
            coeffs = np.polyfit(xs, vs, 1)
            slope, intercept = coeffs

            # Extrapolate: what value at x_max_px?
            x_max_val = slope * x_max_px + intercept
            if 20 <= x_max_val <= 300:
                return round(x_max_val)

        # Fallback: just use the largest tick value found
        if tick_pairs:
            return float(max(v for _, v in tick_pairs))

    except Exception:
        pass

    return 100.0
