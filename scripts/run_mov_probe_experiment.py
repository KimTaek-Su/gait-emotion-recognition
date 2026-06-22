from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from evaluation.config import load_config
from evaluation.report import save_summary, write_markdown_report
from evaluation.utils import ensure_dir, now_run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(cfg.output_dir) / f"run_{now_run_id()}"
    ensure_dir(str(run_dir))

    feature_path = Path(cfg.output_dir) / "features.csv"
    df = pd.read_csv(feature_path)

    summary = {
        "dataset_name": cfg.dataset_name,
        "n_samples": int(len(df)),
        "avg_valid_ratio": float(df["quality_valid_ratio"].mean()) if "quality_valid_ratio" in df else None,
        "avg_gait_suitability": float(df["quality_gait_suitability"].mean()) if "quality_gait_suitability" in df else None,
        "note": "Probe-only experiment. Do not pool with AVI main results.",
    }

    save_summary(summary, str(run_dir / "summary.json"))
    lines = [
        "# MOV Probe Experiment",
        "",
        f"- dataset: {cfg.dataset_name}",
        f"- n_samples: {summary['n_samples']}",
        f"- avg_valid_ratio: {summary['avg_valid_ratio']}",
        f"- avg_gait_suitability: {summary['avg_gait_suitability']}",
        "",
        "## Interpretation",
        "- This probe is intended for domain mismatch analysis only.",
        "- These results should not be merged with the AVI main mechanism experiment.",
    ]
    write_markdown_report(lines, str(run_dir / "report.md"))
    print(f"[DONE] probe results saved to {run_dir}")


if __name__ == "__main__":
    main()