from __future__ import annotations

from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = ["sample_id", "file_name", "label"]


def load_video_dataset(labels_csv: str, videos_dir: str, allowed_exts: list[str]) -> list[dict]:
    df = pd.read_csv(labels_csv)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    allowed_exts = {e.lower() for e in allowed_exts}
    videos_dir = Path(videos_dir)

    samples = []
    for _, row in df.iterrows():
        file_name = str(row["file_name"]).strip()
        video_path = videos_dir / file_name

        sample = {
            "sample_id": str(row["sample_id"]),
            "file_name": file_name,
            "video_path": str(video_path),
            "file_ext": video_path.suffix.lower(),
            "label": str(row["label"]).strip(),
            "subject_id": str(row["subject_id"]).strip() if "subject_id" in df.columns and pd.notna(row.get("subject_id")) else "",
            "view": str(row["view"]).strip() if "view" in df.columns and pd.notna(row.get("view")) else "",
            "data_group": str(row["data_group"]).strip() if "data_group" in df.columns and pd.notna(row.get("data_group")) else "",
            "file_exists": video_path.exists(),
        }
        sample["ext_allowed"] = sample["file_ext"] in allowed_exts
        samples.append(sample)

    return samples