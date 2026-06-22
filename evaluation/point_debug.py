from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def draw_points_on_frame(frame, points, color=(0, 0, 255)):
    out = frame.copy()
    for idx, p in enumerate(points):
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.circle(out, (x, y), 4, color, -1)
        cv2.putText(
            out, str(idx),
            (x + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )
    return out


def binary_to_bgr(binary):
    if len(binary.shape) == 2:
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return binary


def make_debug_panel(frame, binary, overlay):
    h, w = frame.shape[:2]
    binary_bgr = binary_to_bgr(binary)
    binary_bgr = cv2.resize(binary_bgr, (w, h))
    overlay = cv2.resize(overlay, (w, h))

    top = np.hstack([frame, binary_bgr, overlay])
    return top


def save_debug_image(path: str, panel):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, panel)