from __future__ import annotations

import cv2
import numpy as np


def preprocess_frame(frame, invert: bool = False, threshold_mode: str = "otsu"):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if invert:
        gray = 255 - gray

    if threshold_mode == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_mode == "fixed_127":
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    elif threshold_mode == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
    else:
        raise ValueError(f"Unsupported threshold_mode: {threshold_mode}")

    return binary


def postprocess_binary(binary, morph_open: bool = False, morph_close: bool = False):
    out = binary.copy()
    kernel = np.ones((3, 3), np.uint8)

    if morph_open:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    if morph_close:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)

    return out


def detect_point_blobs(binary_img, min_area: int = 3, max_area: int = 500):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img)

    points = []
    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            cx, cy = centroids[i]
            points.append((float(cx), float(cy), int(area)))

    return points


def sort_points_topdown(points):
    return sorted(points, key=lambda p: (p[1], p[0]))


def score_points(points, expected_min: int = 8, expected_max: int = 25):
    n = len(points)
    if n == 0:
        return -9999.0

    # Prefer plausible point counts
    if expected_min <= n <= expected_max:
        count_score = 100.0 - abs(n - ((expected_min + expected_max) / 2))
    else:
        count_score = -abs(n - expected_max) * 5.0

    # Prefer moderate point areas (not too tiny/noisy, not too huge/merged)
    areas = [p[2] for p in points]
    area_score = -np.std(areas) if len(areas) > 1 else 0.0

    return float(count_score + area_score)


def extract_points_single_mode(
    frame,
    invert: bool = False,
    threshold_mode: str = "otsu",
    min_area: int = 3,
    max_area: int = 500,
    morph_open: bool = False,
    morph_close: bool = False,
):
    binary = preprocess_frame(frame, invert=invert, threshold_mode=threshold_mode)
    binary = postprocess_binary(binary, morph_open=morph_open, morph_close=morph_close)
    points = detect_point_blobs(binary, min_area=min_area, max_area=max_area)
    ordered = sort_points_topdown(points)

    return {
        "binary": binary,
        "points": ordered,
        "mode": {
            "invert": invert,
            "threshold_mode": threshold_mode,
            "min_area": min_area,
            "max_area": max_area,
            "morph_open": morph_open,
            "morph_close": morph_close,
        }
    }


def extract_points_from_frame(frame):
    """
    Try multiple extraction modes and keep the best-scoring one.
    """
    candidate_modes = [
        {"invert": False, "threshold_mode": "otsu",      "min_area": 2, "max_area": 120, "morph_open": False, "morph_close": False},
        {"invert": True,  "threshold_mode": "otsu",      "min_area": 2, "max_area": 120, "morph_open": False, "morph_close": False},
        {"invert": False, "threshold_mode": "fixed_127", "min_area": 2, "max_area": 120, "morph_open": False, "morph_close": False},
        {"invert": True,  "threshold_mode": "fixed_127", "min_area": 2, "max_area": 120, "morph_open": False, "morph_close": False},
        {"invert": False, "threshold_mode": "adaptive",  "min_area": 2, "max_area": 120, "morph_open": True,  "morph_close": False},
        {"invert": True,  "threshold_mode": "adaptive",  "min_area": 2, "max_area": 120, "morph_open": True,  "morph_close": False},
        {"invert": False, "threshold_mode": "otsu",      "min_area": 5, "max_area": 250, "morph_open": True,  "morph_close": False},
        {"invert": True,  "threshold_mode": "otsu",      "min_area": 5, "max_area": 250, "morph_open": True,  "morph_close": False},
    ]

    best = None
    best_score = -1e18

    for mode in candidate_modes:
        res = extract_points_single_mode(frame, **mode)
        score = score_points(res["points"])

        if score > best_score:
            best = res
            best_score = score

    best["score"] = float(best_score)
    return best