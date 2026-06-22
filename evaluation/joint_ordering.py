from __future__ import annotations

import numpy as np


def points_to_xy(points):
    if not points:
        return np.zeros((0, 2), dtype=float)
    return np.array([[p[0], p[1]] for p in points], dtype=float)


def split_left_right(points_xy, center_x):
    left = points_xy[points_xy[:, 0] < center_x]
    right = points_xy[points_xy[:, 0] >= center_x]

    left = left[np.argsort(left[:, 1])] if len(left) > 0 else left
    right = right[np.argsort(right[:, 1])] if len(right) > 0 else right
    return left, right


def safe_pick(arr, idx_from_top=0, fallback=None):
    if len(arr) > idx_from_top:
        return arr[idx_from_top]
    return fallback if fallback is not None else np.array([np.nan, np.nan], dtype=float)


def order_points_functional_schema(points):
    """
    Convert unordered point blobs to a 9-point functional body schema.

    Output order:
    0 head
    1 upper_center
    2 lower_center
    3 left_upper
    4 right_upper
    5 left_mid
    6 right_mid
    7 left_foot
    8 right_foot
    """
    points_xy = points_to_xy(points)

    if len(points_xy) == 0:
        return np.full((9, 2), np.nan, dtype=float)

    # sort by y (top to bottom)
    order_y = np.argsort(points_xy[:, 1])
    points_xy = points_xy[order_y]

    head = points_xy[0]

    # feet candidates: two lowest points
    if len(points_xy) >= 2:
        feet_candidates = points_xy[np.argsort(points_xy[:, 1])[-2:]]
        feet_candidates = feet_candidates[np.argsort(feet_candidates[:, 0])]
        left_foot = feet_candidates[0]
        right_foot = feet_candidates[1]
    else:
        left_foot = np.array([np.nan, np.nan], dtype=float)
        right_foot = np.array([np.nan, np.nan], dtype=float)

    # center axis
    feet_mid = np.nanmean(np.vstack([left_foot, right_foot]), axis=0)
    if np.isnan(feet_mid).any():
        center_x = float(np.nanmean(points_xy[:, 0]))
    else:
        center_x = float((head[0] + feet_mid[0]) / 2.0)

    y_min = float(np.min(points_xy[:, 1]))
    y_max = float(np.max(points_xy[:, 1]))
    y_range = max(y_max - y_min, 1e-6)

    # vertical bands
    upper_band = points_xy[(points_xy[:, 1] > y_min + 0.10 * y_range) & (points_xy[:, 1] <= y_min + 0.40 * y_range)]
    middle_band = points_xy[(points_xy[:, 1] > y_min + 0.40 * y_range) & (points_xy[:, 1] <= y_min + 0.70 * y_range)]

    # center points = closest to body center x
    def pick_center(band, default):
        if len(band) == 0:
            return default
        idx = np.argmin(np.abs(band[:, 0] - center_x))
        return band[idx]

    upper_center = pick_center(upper_band, head)
    lower_center = pick_center(middle_band, upper_center)

    # split left/right
    left_upper_candidates, right_upper_candidates = split_left_right(upper_band, center_x)
    left_mid_candidates, right_mid_candidates = split_left_right(middle_band, center_x)

    left_upper = safe_pick(left_upper_candidates, 0, fallback=upper_center)
    right_upper = safe_pick(right_upper_candidates, 0, fallback=upper_center)
    left_mid = safe_pick(left_mid_candidates, -1 if len(left_mid_candidates) > 0 else 0, fallback=lower_center)
    right_mid = safe_pick(right_mid_candidates, -1 if len(right_mid_candidates) > 0 else 0, fallback=lower_center)

    ordered = np.vstack([
        head,
        upper_center,
        lower_center,
        left_upper,
        right_upper,
        left_mid,
        right_mid,
        left_foot,
        right_foot,
    ])

    return ordered