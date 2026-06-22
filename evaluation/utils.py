from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def apply_label_mapping(label: str, mapping: dict) -> str:
    if not mapping:
        return label
    return mapping.get(label, label)