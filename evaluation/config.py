from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ExperimentConfig:
    dataset_name: str
    labels_csv: str
    videos_dir: str
    keypoint_cache_dir: str
    output_dir: str

    video_extensions: list[str] = field(default_factory=lambda: [".avi"])
    pose_backend: str = "yolo_pose"   # yolo_pose | mediapipe
    max_frames: Optional[int] = None
    sample_every: int = 1
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None
    trim_start_ratio: float = 0.0
    trim_end_ratio: float = 0.0

    min_detected_frames: int = 10
    evaluation_mode: str = "4class"
    data_group: str = "avi_main"      # avi_main | mov_probe
    label_mapping: dict = field(default_factory=dict)

    classifier_name: str = "logreg"   # logreg | svm | rf
    random_state: int = 42
    n_splits: int = 5
    use_group_split: bool = True
    exclude_feature_families: list[str] = field(default_factory=list)
    exclude_feature_names: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not Path(self.labels_csv).exists():
            raise FileNotFoundError(f"labels_csv not found: {self.labels_csv}")
        if not Path(self.videos_dir).exists():
            raise FileNotFoundError(f"videos_dir not found: {self.videos_dir}")


def load_config(config_path: str) -> ExperimentConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = ExperimentConfig(**data)
    cfg.validate()
    return cfg