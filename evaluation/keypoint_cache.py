from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def save_keypoint_cache(cache_path: str, result: dict) -> None:
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

    serializable = dict(result)
    if "frames_array" in serializable:
        serializable["frames_array"] = np.asarray(serializable["frames_array"]).tolist()

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)


def load_keypoint_cache(cache_path: str) -> dict:
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "frames_array" in data:
        data["frames_array"] = np.array(data["frames_array"], dtype=float)

    return data