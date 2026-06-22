from __future__ import annotations

import numpy as np

from evaluation.video_reader import read_video_frames
from evaluation.point_figure_extractor import extract_points_from_frame
from evaluation.point_tracking import to_xy_array, nearest_neighbor_match, fill_missing_linear
from evaluation.point_quality import compute_point_sequence_quality
from evaluation.joint_ordering import order_points_functional_schema


def extract_keypoints_from_video(
    video_path: str,
    pose_backend_name: str = "point_figure",
    max_frames: int | None = None,
    sample_every: int = 1,
    resize_width: int | None = None,
    resize_height: int | None = None,
    min_detected_frames: int = 10,
    invert: bool = False,
    expected_joints: int = 9,
) -> dict:
    video_result = read_video_frames(
        video_path=video_path,
        max_frames=max_frames,
        sample_every=sample_every,
        resize_width=resize_width,
        resize_height=resize_height,
    )
    if not video_result["success"]:
        return {
            "success": False,
            "error_type": "video_read_failed",
            "error_message": video_result["error_message"],
        }

    frames = video_result["frames"]

    raw_points = []
    mode_history = []
    for frame in frames:
        res = extract_points_from_frame(frame)
        raw_points.append(res["points"])
        mode_history.append(res["mode"])

    detected_counts = [len(p) for p in raw_points]
    n_detected_frames = sum(1 for n in detected_counts if n > 0)

    if n_detected_frames < min_detected_frames:
        return {
            "success": False,
            "error_type": "too_few_detected_frames",
            "error_message": f"Detected frames {n_detected_frames} < {min_detected_frames}",
            "n_frames_total": len(frames),
            "n_frames_detected": n_detected_frames,
            "valid_ratio": n_detected_frames / max(len(frames), 1),
            "detected_counts": detected_counts,
            "mode_history": mode_history,
        }

    init_ordered = None
    for pts in raw_points:
        if len(pts) > 0:
            init_ordered = order_points_functional_schema(pts)
            break

    if init_ordered is None or len(init_ordered) == 0:
        return {
            "success": False,
            "error_type": "no_points_found",
            "error_message": "No point blobs found in any frame",
            "n_frames_total": len(frames),
            "n_frames_detected": 0,
            "valid_ratio": 0.0,
            "mode_history": mode_history,
        }

    sequence = []
    prev_xy = init_ordered

    for pts in raw_points:
        curr_xy_raw = to_xy_array(pts)
        if len(curr_xy_raw) == 0:
            matched = np.full_like(prev_xy, np.nan, dtype=float)
        else:
            curr_xy = order_points_functional_schema(pts)
            matched = nearest_neighbor_match(prev_xy, curr_xy, max_dist=60.0)

        sequence.append(matched)
        prev_xy = np.where(np.isnan(matched), prev_xy, matched)

    sequence = np.array(sequence, dtype=float)
    sequence = fill_missing_linear(sequence)

    z = np.zeros((sequence.shape[0], sequence.shape[1], 1), dtype=float)
    frames_array = np.concatenate([sequence, z], axis=2)

    quality = compute_point_sequence_quality(sequence, expected_joints=expected_joints)

    return {
        "success": True,
        "frames_array": frames_array,
        "n_frames_total": len(frames),
        "n_frames_detected": n_detected_frames,
        "valid_ratio": n_detected_frames / max(len(frames), 1),
        "n_joints": int(frames_array.shape[1]),
        "fps": video_result["fps"],
        "detected_counts": detected_counts,
        "mode_history": mode_history,
        **quality,
    }