from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.base import clone

from evaluation.config import load_config
from evaluation.classifier import create_classifier
from evaluation.metrics import classification_summary
from evaluation.mechanism_analysis import compute_feature_summary
from evaluation.report import save_dataframe, save_summary, write_markdown_report
from evaluation.utils import ensure_dir, now_run_id, apply_label_mapping


META_COLS = ["sample_id", "file_name", "video_path", "label", "subject_id", "view", "data_group"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(cfg.output_dir) / f"run_{now_run_id()}"
    ensure_dir(str(run_dir))

    feature_path = Path(cfg.output_dir) / "features.csv"
    df = pd.read_csv(feature_path)

    if df.empty:
        raise RuntimeError("features.csv is empty.")

    df["label_eval"] = df["label"].apply(lambda x: apply_label_mapping(x, cfg.label_mapping))

    feature_cols = [c for c in df.columns if c not in META_COLS + ["label_eval"]]
    if not feature_cols:
        raise RuntimeError("No usable feature columns found.")

    X = df[feature_cols].copy()
    y = df["label_eval"].copy()

    if len(df) < 4:
        raise RuntimeError(
            f"Too few samples for cross-validation: n={len(df)}. "
            "Extract more valid samples first."
        )

    clf = create_classifier(cfg.classifier_name, random_state=cfg.random_state)

    y_true_all = []
    y_pred_all = []

    use_group_split = False
    split_iter = None

    if cfg.use_group_split and "subject_id" in df.columns:
        groups = df["subject_id"].fillna("").astype(str)
        unique_groups = [g for g in groups.unique().tolist() if g != ""]
        if len(unique_groups) >= cfg.n_splits:
            use_group_split = True
            splitter = GroupKFold(n_splits=cfg.n_splits)
            split_iter = splitter.split(X, y, groups=groups)
            print(f"[INFO] Using GroupKFold with {len(unique_groups)} groups.")
        else:
            print(
                f"[WARN] Group split requested but only {len(unique_groups)} valid groups found. "
                f"Falling back to StratifiedKFold."
            )

    if split_iter is None:
        class_counts = y.value_counts()
        min_class_count = int(class_counts.min())
        effective_splits = min(cfg.n_splits, min_class_count)

        if effective_splits < 2:
            raise RuntimeError(
                f"Not enough samples per class for StratifiedKFold. "
                f"Minimum class count is {min_class_count}."
            )

        splitter = StratifiedKFold(
            n_splits=effective_splits,
            shuffle=True,
            random_state=cfg.random_state
        )
        split_iter = splitter.split(X, y)
        print(f"[INFO] Using StratifiedKFold with n_splits={effective_splits}.")

    for fold_idx, (tr_idx, te_idx) in enumerate(split_iter, start=1):
        model = clone(clf)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = model.predict(X.iloc[te_idx])

        y_true_all.extend(y.iloc[te_idx].tolist())
        y_pred_all.extend(pred.tolist())
        print(f"[FOLD {fold_idx}] done")

    labels = sorted(y.unique().tolist())
    summary = classification_summary(y_true_all, y_pred_all, labels=labels)
    summary["classifier"] = cfg.classifier_name
    summary["dataset_name"] = cfg.dataset_name
    summary["feature_count"] = len(feature_cols)
    summary["n_samples"] = int(len(df))
    summary["used_group_split"] = bool(use_group_split)

    group_stats = compute_feature_summary(df, label_col="label_eval")

    pred_df = pd.DataFrame({
        "y_true": y_true_all,
        "y_pred": y_pred_all,
    })

    save_dataframe(pred_df, str(run_dir / "predictions.csv"))
    save_dataframe(group_stats, str(run_dir / "group_stats.csv"))
    save_summary(summary, str(run_dir / "summary.json"))

    lines = [
        "# AVI Mechanism Experiment",
        "",
        f"- dataset: {cfg.dataset_name}",
        f"- classifier: {cfg.classifier_name}",
        f"- n_samples: {summary['n_samples']}",
        f"- used_group_split: {summary['used_group_split']}",
        f"- accuracy: {summary['accuracy']:.4f}",
        f"- balanced_accuracy: {summary['balanced_accuracy']:.4f}",
        f"- macro_f1: {summary['macro_f1']:.4f}",
        f"- weighted_f1: {summary['weighted_f1']:.4f}",
        "",
        "## Notes",
        "- This is the main gait mechanism experiment.",
        "- Fallback to StratifiedKFold is applied if subject groups are insufficient.",
    ]
    write_markdown_report(lines, str(run_dir / "report.md"))
    print(f"[DONE] results saved to {run_dir}")


if __name__ == "__main__":
    main()