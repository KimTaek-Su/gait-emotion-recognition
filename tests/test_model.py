import pytest
import numpy as np

from src.feature_extractor import extract_features_from_skeleton, parse_skeleton_data
from src.main import convert_keypoints_to_skeleton_data, pad_features_if_needed


def build_valid_skeleton_data(n_frames: int = 5, n_joints: int = 17):
    skeleton_data = []
    for frame_idx in range(n_frames):
        for joint_idx in range(n_joints):
            x = 0.5 + frame_idx * 0.01 + joint_idx * 0.02
            y = 0.3 + frame_idx * 0.015 + joint_idx * 0.01
            z = 0.1 + frame_idx * 0.005
            skeleton_data.append(f"{x},{y},{z}")
    return skeleton_data


def test_parse_skeleton_data_returns_expected_shape():
    data = parse_skeleton_data(build_valid_skeleton_data(n_frames=3, n_joints=17), n_joints=17)
    assert data.shape == (3, 17, 3)


def test_extract_features_from_skeleton_returns_14_features():
    features = extract_features_from_skeleton(build_valid_skeleton_data(n_frames=4, n_joints=17), n_joints=17)
    assert isinstance(features, np.ndarray)
    assert features.shape == (14,)


def test_convert_keypoints_to_skeleton_data_validates_shape():
    with pytest.raises(ValueError):
        convert_keypoints_to_skeleton_data([[0.1, 0.2]], n_joints=1)


def test_pad_features_if_needed_pads_to_model_dimension():
    class DummyModel:
        n_features_in_ = 20

    padded = pad_features_if_needed([1.0] * 14, DummyModel())
    assert len(padded) == 20
    assert padded[14:] == [0.0] * 6
