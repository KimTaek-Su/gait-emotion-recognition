from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from evaluation.config import load_config
from evaluation.keypoint_cache import load_keypoint_cache
from evaluation.hcf_features import compute_hcf_features
from evaluation.kinematic_features import compute_kinematic_features
from evaluation.feature_quality import compute_feature_quality
from evaluation.report import save_dataframe
from evaluation.video_dataset_loader import load_video_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    samples = load_video_dataset(cfg.labels_csv, cfg.videos_dir, cfg.video_extensions)

    rows = []
    n_cache_found = 0
    n_success = 0
    n_failed = 0

    for sample in samples:
        cache_path = Path(cfg.keypoint_cache_dir) / f"{sample['sample_id']}.json"
        if not cache_path.exists():
            print(f"[SKIP] cache not found: {cache_path}")
            continue

        n_cache_found += 1
        data = load_keypoint_cache(str(cache_path))

        if not data.get("success", False):
            n_failed += 1
            print(
                f"[FAIL] sample_id={sample['sample_id']} "
                f"error_type={data.get('error_type', '')} "
                f"error_message={data.get('error_message', '')}"
            )
            continue

        frames_array = data["frames_array"]
        if frames_array is None or len(frames_array) == 0:
            n_failed += 1
            print(f"[FAIL] sample_id={sample['sample_id']} empty frames_array")
            continue

        row = {
            "sample_id": sample["sample_id"],
            "file_name": sample["file_name"],
            "video_path": sample["video_path"],
            "label": sample["label"],
            "subject_id": sample.get("subject_id", ""),
            "view": sample.get("view", ""),
            "data_group": sample.get("data_group", ""),
        }
        row.update(compute_hcf_features(frames_array))
        row.update(compute_kinematic_features(frames_array))
        row.update(compute_feature_quality(frames_array, data.get("valid_ratio", 0.0)))

        rows.append(row)
        n_success += 1

    print(f"[SUMMARY] cache_found={n_cache_found}, success={n_success}, failed={n_failed}")

    if not rows:
        raise RuntimeError(
            "No valid feature rows were generated. "
            "This usually means pose extraction failed for all samples. "
            "Check pose backend config and extracted_keypoints/*.json files."
        )

    df = pd.DataFrame(rows)
    out_path = str(Path(cfg.output_dir) / "features.csv")
    save_dataframe(df, out_path)
    print(f"[DONE] feature table saved: {out_path}")


if __name__ == "__main__":
    main()