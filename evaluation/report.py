from __future__ import annotations

from pathlib import Path
import pandas as pd

from evaluation.utils import ensure_dir, save_json


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_summary(summary: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_json(summary, path)


def write_markdown_report(lines: list[str], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))