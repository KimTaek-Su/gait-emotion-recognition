from __future__ import annotations

import numpy as np


def frames_to_keypoints_payload(frames_array: np.ndarray) -> dict:
    keypoints = []
    for frame in frames_array:
        for joint in frame:
            x, y, z = [float(v) for v in joint]
            keypoints.append([x, y, z])

    return {
        "keypoints": keypoints,
        "n_joints": int(frames_array.shape[1]),
    }


def frames_to_skeleton_payload(frames_array: np.ndarray) -> dict:
    skeleton_data = []
    for frame in frames_array:
        for joint in frame:
            x, y, z = [float(v) for v in joint]
            skeleton_data.append(f"{x},{y},{z}")

    return {
        "skeleton_data": skeleton_data,
        "n_joints": int(frames_array.shape[1]),
    }