from __future__ import annotations

import numpy as np


def compute_feature_quality(frames_array: np.ndarray, valid_ratio: float) -> dict:
    T, J, C = frames_array.shape

    missing_ratio = float(np.mean(frames_array == 0.0))
    jitter = float(np.mean(np.abs(np.diff(frames_array, axis=0)))) if T > 1 else 0.0

    gait_suitability = 0.5 * valid_ratio + 0.5 * max(0.0, min(1.0, T / 50.0))

    return {
        "quality_valid_ratio": float(valid_ratio),
        "quality_missing_ratio": missing_ratio,
        "quality_jitter": jitter,
        "quality_gait_suitability": gait_suitability,
    }