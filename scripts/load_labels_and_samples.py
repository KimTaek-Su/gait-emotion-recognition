import pandas as pd
from pathlib import Path

def load_samples(labels_csv: str, videos_dir: str):
    df = pd.read_csv(labels_csv)

    samples = []
    for _, row in df.iterrows():
        samples.append({
            "sample_id": str(row["sample_id"]),
            "file_name": str(Path(videos_dir) / row["file_name"]),
            "label": row["label"]
        })
    return samples

samples = load_samples(
    labels_csv="D:/datasets/gait_eval_avi/labels.csv",
    videos_dir="D:/datasets/gait_eval_avi/videos"
)

print(samples[:3])
# [{'sample_id': '1', 'file_name': 'D:/datasets/gait_eval_avi/videos/sample1.avi', 'label': 'Anger'}, ...]