from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd


def load_samples(labels_csv: str, videos_dir: str):
    df = pd.read_csv(labels_csv)

    required = ["sample_id", "file_name", "label"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "sample_id": str(row["sample_id"]),
            "file_name": str(row["file_name"]).strip(),
            "label": str(row["label"]).strip(),
            "video_path": str(Path(videos_dir) / str(row["file_name"]).strip()),
        })
    return rows


def save_debug_frames(video_path: str, output_dir: Path, sample_id: str, max_frames_to_save: int = 3):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {
            "success": False,
            "error": f"Cannot open video: {video_path}"
        }

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    indices = []
    if frame_count > 0:
        indices = sorted(set([
            0,
            max(0, frame_count // 2),
            max(0, frame_count - 1),
        ]))
    else:
        indices = [0, 1, 2]

    saved_files = []
    current_idx = 0
    target_ptr = 0

    ret = True
    while ret and target_ptr < len(indices):
        ret, frame = cap.read()
        if not ret:
            break

        if current_idx == indices[target_ptr]:
            out_path = output_dir / f"{sample_id}_frame_{current_idx}.png"
            cv2.imwrite(str(out_path), frame)
            saved_files.append(str(out_path))
            target_ptr += 1

        current_idx += 1

    cap.release()

    return {
        "success": True,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "saved_files": saved_files,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.labels_csv, args.videos_dir)

    for idx, sample in enumerate(samples[:args.limit], start=1):
        print(f"\n[DEBUG] sample_id={sample['sample_id']} file={sample['file_name']} label={sample['label']}")
        result = save_debug_frames(
            video_path=sample["video_path"],
            output_dir=output_dir,
            sample_id=sample["sample_id"],
            max_frames_to_save=3,
        )

        if not result["success"]:
            print(f"  [FAIL] {result['error']}")
            continue

        print(f"  frame_count={result['frame_count']}, fps={result['fps']:.2f}, size=({result['width']}x{result['height']})")
        for f in result["saved_files"]:
            print(f"  saved: {f}")


if __name__ == "__main__":
    main()