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


def _safe_mean(x):
    return float(np.mean(x)) if len(x) > 0 else 0.0


def _safe_std(x):
    return float(np.std(x)) if len(x) > 0 else 0.0


def _velocity(xy_seq):
    if len(xy_seq) < 2:
        return np.zeros((1, xy_seq.shape[1], xy_seq.shape[2]), dtype=float)
    return np.diff(xy_seq, axis=0)


def _speed_from_vel(v):
    return np.linalg.norm(v, axis=-1)


def _smoothness(signal):
    if len(signal) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(signal))))


def compute_kinematic_features(frames_array: np.ndarray) -> dict:
    """
    frames_array: (T, 9, 3)
    """
    xy = frames_array[:, :, :2]
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

    centroid = np.mean(xy, axis=1)  # (T, 2)

    vel = _velocity(xy)
    joint_speed = _speed_from_vel(vel)  # (T-1, 9)

    centroid_vel = _velocity(centroid[:, None, :])[:, 0, :]
    centroid_speed = np.linalg.norm(centroid_vel, axis=-1)

    # vertical bounce / lateral sway
    head_vertical = head[:, 1]
    upper_vertical = upper[:, 1]
    lower_vertical = lower[:, 1]

    head_lateral = head[:, 0]
    upper_lateral = upper[:, 0]
    lower_lateral = lower[:, 0]

    # limb amplitudes
    left_upper_amp = np.sqrt((lu[:, 0] - upper[:, 0])**2 + (lu[:, 1] - upper[:, 1])**2)
    right_upper_amp = np.sqrt((ru[:, 0] - upper[:, 0])**2 + (ru[:, 1] - upper[:, 1])**2)

    left_foot_amp = np.sqrt((lf[:, 0] - lower[:, 0])**2 + (lf[:, 1] - lower[:, 1])**2)
    right_foot_amp = np.sqrt((rf[:, 0] - lower[:, 0])**2 + (rf[:, 1] - lower[:, 1])**2)

    # symmetry
    upper_amp_diff = np.abs(left_upper_amp - right_upper_amp)
    foot_amp_diff = np.abs(left_foot_amp - right_foot_amp)

    # foot gait dynamics
    foot_sep = np.abs(rf[:, 0] - lf[:, 0])
    foot_y_diff = np.abs(rf[:, 1] - lf[:, 1])

    # temporal smoothness / regularity proxies
    centroid_speed_smoothness = _smoothness(centroid_speed)
    foot_sep_smoothness = _smoothness(foot_sep)
    upper_amp_smoothness = _smoothness((left_upper_amp + right_upper_amp) / 2.0)

    # motion energy
    mean_joint_speed = _safe_mean(joint_speed.flatten())
    std_joint_speed = _safe_std(joint_speed.flatten())

    # upper vs lower activity
    upper_speed = joint_speed[:, [HEAD, UPPER_CENTER, LEFT_UPPER, RIGHT_UPPER]]
    lower_speed = joint_speed[:, [LOWER_CENTER, LEFT_MID, RIGHT_MID, LEFT_FOOT, RIGHT_FOOT]]

    return {
        # global locomotion
        "kin_centroid_speed_mean": _safe_mean(centroid_speed),
        "kin_centroid_speed_std": _safe_std(centroid_speed),
        "kin_joint_speed_mean": mean_joint_speed,
        "kin_joint_speed_std": std_joint_speed,

        # vertical bounce
        "kin_head_vertical_std": _safe_std(head_vertical),
        "kin_upper_vertical_std": _safe_std(upper_vertical),
        "kin_lower_vertical_std": _safe_std(lower_vertical),

        # lateral sway
        "kin_head_lateral_std": _safe_std(head_lateral),
        "kin_upper_lateral_std": _safe_std(upper_lateral),
        "kin_lower_lateral_std": _safe_std(lower_lateral),

        # upper-limb dynamics
        "kin_left_upper_amp_mean": _safe_mean(left_upper_amp),
        "kin_right_upper_amp_mean": _safe_mean(right_upper_amp),
        "kin_upper_amp_diff_mean": _safe_mean(upper_amp_diff),

        # lower-limb dynamics
        "kin_left_foot_amp_mean": _safe_mean(left_foot_amp),
        "kin_right_foot_amp_mean": _safe_mean(right_foot_amp),
        "kin_foot_amp_diff_mean": _safe_mean(foot_amp_diff),

        # stride / gait proxies
        "kin_foot_sep_mean": _safe_mean(foot_sep),
        "kin_foot_sep_std": _safe_std(foot_sep),
        "kin_foot_y_diff_mean": _safe_mean(foot_y_diff),
        "kin_foot_y_diff_std": _safe_std(foot_y_diff),

        # activity ratio
        "kin_upper_speed_mean": _safe_mean(upper_speed.flatten()),
        "kin_lower_speed_mean": _safe_mean(lower_speed.flatten()),

        # smoothness
        "kin_centroid_speed_smoothness": centroid_speed_smoothness,
        "kin_foot_sep_smoothness": foot_sep_smoothness,
        "kin_upper_amp_smoothness": upper_amp_smoothness,

        "kin_n_frames": float(T),
    }