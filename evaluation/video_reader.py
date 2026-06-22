from __future__ import annotations

import cv2


def read_video_frames(
    video_path: str,
    max_frames: int | None = None,
    sample_every: int = 1,
    resize_width: int | None = None,
    resize_height: int | None = None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "success": False,
            "error_message": f"Cannot open video: {video_path}",
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            if resize_width and resize_height:
                frame = cv2.resize(frame, (resize_width, resize_height))
            frames.append(frame)

        frame_idx += 1
        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()

    if len(frames) == 0:
        return {
            "success": False,
            "error_message": f"No readable sampled frames: {video_path}",
        }

    return {
        "success": True,
        "frames": frames,
        "fps": float(fps) if fps else 0.0,
        "frame_count_sampled": len(frames),
    }