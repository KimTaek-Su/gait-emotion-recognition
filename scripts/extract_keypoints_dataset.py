from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.config import load_config
from evaluation.keypoint_extractor import extract_keypoints_from_video
from evaluation.keypoint_cache import save_keypoint_cache
from evaluation.utils import ensure_dir
from evaluation.video_dataset_loader import load_video_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dir(cfg.keypoint_cache_dir)

    samples = load_video_dataset(cfg.labels_csv, cfg.videos_dir, cfg.video_extensions)

    n_done = 0
    n_success = 0
    n_fail = 0

    for idx, sample in enumerate(samples):
        if args.limit is not None and idx >= args.limit:
            break

        if not sample["file_exists"] or not sample["ext_allowed"]:
            print(f"[SKIP] {sample['sample_id']} missing file or bad extension")
            continue

        result = extract_keypoints_from_video(
            video_path=sample["video_path"],
            pose_backend_name=cfg.pose_backend,
            max_frames=cfg.max_frames,
            sample_every=cfg.sample_every,
            resize_width=cfg.resize_width,
            resize_height=cfg.resize_height,
            min_detected_frames=cfg.min_detected_frames,
        )

        cache_path = str(Path(cfg.keypoint_cache_dir) / f"{sample['sample_id']}.json")
        payload = {
            "sample_id": sample["sample_id"],
            "file_name": sample["file_name"],
            "label": sample["label"],
            "subject_id": sample.get("subject_id", ""),
            "view": sample.get("view", ""),
            "data_group": sample.get("data_group", ""),
            **result,
        }
        save_keypoint_cache(cache_path, payload)

        n_done += 1
        if result.get("success", False):
            n_success += 1
            print(
                f"[OK] {sample['sample_id']} "
                f"detected={result.get('n_frames_detected')} "
                f"/ total={result.get('n_frames_total')} "
                f"valid_ratio={result.get('valid_ratio', 0.0):.3f} "
                f"-> {cache_path}"
            )
        else:
            n_fail += 1
            print(
                f"[FAIL] {sample['sample_id']} "
                f"error_type={result.get('error_type', '')} "
                f"error_message={result.get('error_message', '')} "
                f"-> {cache_path}"
            )

    print(
        f"[SUMMARY] processed={n_done}, success={n_success}, fail={n_fail}"
    )


if __name__ == "__main__":
    main()