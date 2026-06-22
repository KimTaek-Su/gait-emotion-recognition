import numpy as np
import pandas as pd

from evaluation.kinematic_features import compute_kinematic_features
from evaluation.mechanism_analysis import (
    build_feature_family_map,
    compute_domain_shift,
    compute_feature_effects,
    compute_group_statistics,
    cross_validate_predictions,
    run_family_ablation,
    select_feature_columns,
)


def make_mechanism_df() -> pd.DataFrame:
    rows = []
    labels = ["happy", "sad", "happy", "sad", "angry", "angry"]
    subjects = ["S1", "S2", "S3", "S4", "S5", "S6"]
    for index, (label, subject) in enumerate(zip(labels, subjects), start=1):
        base = float(index)
        rows.append({
            "sample_id": str(index),
            "file_name": f"sample_{index}.avi",
            "label": label,
            "subject_id": subject,
            "data_group": "avi_main",
            "kin_centroid_speed_mean": base + (2.0 if label == "happy" else 0.0),
            "kin_lateral_sway": base / 2.0 + (1.5 if label == "sad" else 0.0),
            "kin_vertical_bounce": base / 3.0 + (1.0 if label == "angry" else 0.0),
            "quality_valid_ratio": 0.9,
        })
    return pd.DataFrame(rows)


def test_compute_kinematic_features_exposes_posture_proxies():
    frames = np.array([
        [[0.0, 0.0, 0.0], [2.0, 4.0, 0.0]],
        [[1.0, 0.5, 0.0], [3.0, 4.5, 0.0]],
        [[2.0, 1.0, 0.0], [4.0, 5.0, 0.0]],
    ])
    features = compute_kinematic_features(frames)
    assert "kin_posture_ratio_mean" in features
    assert "kin_body_width_mean" in features
    assert features["kin_centroid_speed_mean"] > 0


def test_build_feature_family_map_handles_legacy_and_canonical_names():
    family_map = build_feature_family_map([
        "kin_centroid_speed_mean",
        "avg_speed",
        "y_variance",
        "quality_valid_ratio",
        "hcf_n_frames",
    ])
    assert family_map["kin_centroid_speed_mean"] == "speed"
    assert family_map["avg_speed"] == "speed"
    assert family_map["y_variance"] == "vertical_motion"
    assert family_map["quality_valid_ratio"] == "quality"
    assert family_map["hcf_n_frames"] == "capture"


def test_select_feature_columns_excludes_nuisance_families():
    df = make_mechanism_df().assign(hcf_n_frames=[90, 95, 100, 105, 110, 115])
    selected = select_feature_columns(df, exclude_families={"quality", "capture"})
    assert "quality_valid_ratio" not in selected
    assert "hcf_n_frames" not in selected
    assert "kin_centroid_speed_mean" in selected


def test_mechanism_analysis_outputs_non_empty_tables():
    df = make_mechanism_df()
    predictions_df, fold_metrics_df = cross_validate_predictions(
        df=df,
        feature_cols=["kin_centroid_speed_mean", "kin_lateral_sway", "kin_vertical_bounce"],
        groups=df["subject_id"].to_numpy(),
        classifier_name="logreg",
        n_splits=3,
    )
    group_stats = compute_group_statistics(df)
    feature_effects = compute_feature_effects(df)
    family_ablation = run_family_ablation(
        df=df,
        groups=df["subject_id"].to_numpy(),
        classifier_name="logreg",
        n_splits=3,
    )

    assert len(predictions_df) == len(df)
    assert not fold_metrics_df.empty
    assert not group_stats.empty
    assert not feature_effects.empty
    assert "all_features" in set(family_ablation["feature_set"])


def test_compute_domain_shift_uses_shared_numeric_features_only():
    reference_df = make_mechanism_df()
    probe_df = make_mechanism_df().rename(columns={"kin_vertical_bounce": "kin_vertical_bounce"}).copy()
    probe_df["kin_centroid_speed_mean"] += 3.0
    shift_df = compute_domain_shift(reference_df, probe_df)
    assert not shift_df.empty
    assert "kin_centroid_speed_mean" in set(shift_df["feature"])