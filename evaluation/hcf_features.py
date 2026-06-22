from __future__ import annotations

import numpy as np


# 9-point functional schema
HEAD = 0
UPPER_CENTER = 1
LOWER_CENTER = 2
LEFT_UPPER = 3
RIGHT_UPPER = 4
LEFT_MID = 5
RIGHT_MID = 6
LEFT_FOOT = 7
RIGHT_FOOT = 8


def _dist(a, b):
    return np.linalg.norm(a - b, axis=-1)


def _safe_std(x):
    return float(np.std(x)) if len(x) > 0 else 0.0


def _safe_mean(x):
    return float(np.mean(x)) if len(x) > 0 else 0.0


def compute_hcf_features(frames_array: np.ndarray) -> dict:
    """
    frames_array: (T, 9, 3)
    Use x,y only for point/stick figure.
    """
    xy = frames_array[:, :, :2]  # (T, 9, 2)
    T = xy.shape[0]

    head = xy[:, HEAD]
    upper = xy[:, UPPER_CENTER]
    lower = xy[:, LOWER_CENTER]
    lu = xy[:, LEFT_UPPER]
    ru = xy[:, RIGHT_UPPER]
    lm = xy[:, LEFT_MID]
    rm = xy[:, RIGHT_MID]
    lf = xy[:, LEFT_FOOT]
    rf = xy[:, RIGHT_FOOT]

    # Body lengths / configuration
    head_upper_dist = _dist(head, upper)
    upper_lower_dist = _dist(upper, lower)
    lower_left_foot_dist = _dist(lower, lf)
    lower_right_foot_dist = _dist(lower, rf)
    left_upper_to_lower = _dist(lu, lower)
    right_upper_to_lower = _dist(ru, lower)

    # Spreads
    shoulder_width_proxy = np.abs(ru[:, 0] - lu[:, 0])
    hip_width_proxy = np.abs(rm[:, 0] - lm[:, 0])
    foot_separation = np.abs(rf[:, 0] - lf[:, 0])

    # Vertical structure
    body_height_proxy = np.abs(lf[:, 1] - head[:, 1]) + np.abs(rf[:, 1] - head[:, 1]) / 2.0
    torso_height_proxy = np.abs(lower[:, 1] - upper[:, 1])

    # Compactness / openness
    upper_spread = np.abs(ru[:, 0] - upper[:, 0]) + np.abs(lu[:, 0] - upper[:, 0])
    lower_spread = np.abs(rf[:, 0] - lower[:, 0]) + np.abs(lf[:, 0] - lower[:, 0])

    # Left-right asymmetry
    upper_asym = np.abs(np.abs(lu[:, 0] - upper[:, 0]) - np.abs(ru[:, 0] - upper[:, 0]))
    lower_asym = np.abs(np.abs(lf[:, 0] - lower[:, 0]) - np.abs(rf[:, 0] - lower[:, 0]))

    # Torso tilt proxy
    torso_dx = upper[:, 0] - lower[:, 0]
    torso_dy = upper[:, 1] - lower[:, 1]
    torso_tilt = np.arctan2(torso_dx, torso_dy + 1e-6)

    return {
        # body configuration
        "hcf_head_upper_dist_mean": _safe_mean(head_upper_dist),
        "hcf_head_upper_dist_std": _safe_std(head_upper_dist),
        "hcf_upper_lower_dist_mean": _safe_mean(upper_lower_dist),
        "hcf_upper_lower_dist_std": _safe_std(upper_lower_dist),

        # posture / openness
        "hcf_shoulder_width_mean": _safe_mean(shoulder_width_proxy),
        "hcf_shoulder_width_std": _safe_std(shoulder_width_proxy),
        "hcf_hip_width_mean": _safe_mean(hip_width_proxy),
        "hcf_hip_width_std": _safe_std(hip_width_proxy),
        "hcf_foot_separation_mean": _safe_mean(foot_separation),
        "hcf_foot_separation_std": _safe_std(foot_separation),

        # body size proxies
        "hcf_body_height_mean": _safe_mean(body_height_proxy),
        "hcf_body_height_std": _safe_std(body_height_proxy),
        "hcf_torso_height_mean": _safe_mean(torso_height_proxy),
        "hcf_torso_height_std": _safe_std(torso_height_proxy),

        # openness / compactness
        "hcf_upper_spread_mean": _safe_mean(upper_spread),
        "hcf_upper_spread_std": _safe_std(upper_spread),
        "hcf_lower_spread_mean": _safe_mean(lower_spread),
        "hcf_lower_spread_std": _safe_std(lower_spread),

        # asymmetry
        "hcf_upper_asym_mean": _safe_mean(upper_asym),
        "hcf_upper_asym_std": _safe_std(upper_asym),
        "hcf_lower_asym_mean": _safe_mean(lower_asym),
        "hcf_lower_asym_std": _safe_std(lower_asym),

        # torso tilt
        "hcf_torso_tilt_mean": _safe_mean(torso_tilt),
        "hcf_torso_tilt_std": _safe_std(torso_tilt),

        # limb-to-center structure
        "hcf_left_upper_to_lower_mean": _safe_mean(left_upper_to_lower),
        "hcf_right_upper_to_lower_mean": _safe_mean(right_upper_to_lower),
        "hcf_left_leg_length_proxy_mean": _safe_mean(lower_left_foot_dist),
        "hcf_right_leg_length_proxy_mean": _safe_mean(lower_right_foot_dist),

        "hcf_n_frames": float(T),
        "hcf_n_joints": float(xy.shape[1]),
    }