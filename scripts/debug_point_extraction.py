from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from evaluation.point_figure_extractor import extract_points_from_frame
from evaluation.point_debug import draw_points_on_frame, make_debug_panel, save_debug_image


def load_samples(labels_csv: str, videos_dir: str):
    df = pd.read_csv(labels_csv)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "sample_id": str(row["sample_id"]),
            "file_name": str(row["file_name"]).strip(),
            "label": str(row["label"]).strip(),
            "video_path": str(Path(videos_dir) / str(row["file_name"]).strip()),
        })
    return rows


def read_selected_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"success": False, "error": f"Cannot open video: {video_path}"}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sorted(set([0, max(0, frame_count // 2), max(0, frame_count - 1)]))

    saved = []
    current = 0
    ptr = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if ptr < len(idxs) and current == idxs[ptr]:
            saved.append((current, frame.copy()))
            ptr += 1
        current += 1
        if ptr >= len(idxs):
            break

    cap.release()
    return {"success": True, "frames": saved, "frame_count": frame_count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    samples = load_samples(args.labels_csv, args.videos_dir)

    if args.sample_ids:
        wanted = set(args.sample_ids)
        samples = [s for s in samples if s["sample_id"] in wanted]
    else:
        samples = samples[:args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        print(f"[DEBUG] sample_id={sample['sample_id']} file={sample['file_name']}")
        result = read_selected_frames(sample["video_path"])
        if not result["success"]:
            print(f"  [FAIL] {result['error']}")
            continue

        for frame_idx, frame in result["frames"]:
            ext = extract_points_from_frame(frame)
            overlay = draw_points_on_frame(frame, ext["points"])
            panel = make_debug_panel(frame, ext["binary"], overlay)

            out_path = output_dir / f"{sample['sample_id']}_frame_{frame_idx}.png"
            save_debug_image(str(out_path), panel)

            print(
                f"  saved={out_path} "
                f"n_points={len(ext['points'])} "
                f"score={ext['score']:.2f} "
                f"mode={ext['mode']}"
            )


if __name__ == "__main__":
    main()