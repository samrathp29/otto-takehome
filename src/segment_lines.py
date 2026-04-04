import cv2
import numpy as np


def segment_lines(image_path):
    """
    Extract raw pixel contours of colored curves from a KM plot image.
    Uses HSV color masking to isolate non-gray/non-black/non-white pixels
    (i.e., the actual plotted lines), then finds contours.

    Returns a list of contours, where each contour is an Nx2 numpy array
    of (x, y) pixel coordinates sorted by x.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Isolate saturated, colored pixels (excludes white, gray, black)
    lower = np.array([0, 50, 50])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    curves = []
    for cnt in contours:
        if cv2.arcLength(cnt, False) < 50:
            continue
        pts = cnt.reshape(-1, 2)
        # Sort by x coordinate (left to right = increasing time)
        pts = pts[np.argsort(pts[:, 0])]
        curves.append(pts)

    return curves
