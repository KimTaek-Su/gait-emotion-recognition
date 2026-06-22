from __future__ import annotations

import numpy as np


def compute_point_sequence_quality(sequence: np.ndarray, expected_joints: int):
    """
    sequence: (T, J, 2)
    """
    T, J, C = sequence.shape
    nan_ratio = float(np.mean(np.isnan(sequence))) if np.isnan(sequence).any() else 0.0

    diffs = np.diff(sequence, axis=0)
    finite_diffs = diffs[np.isfinite(diffs)]
    jitter = float(np.mean(np.abs(finite_diffs))) if finite_diffs.size else 0.0

    joint_count_score = min(1.0, J / max(expected_joints, 1))
    quality_score = max(0.0, 1.0 - nan_ratio) * 0.6 + joint_count_score * 0.4

    return {
        "point_nan_ratio": nan_ratio,
        "point_jitter": jitter,
        "point_joint_count": int(J),
        "point_quality_score": float(quality_score),
    }