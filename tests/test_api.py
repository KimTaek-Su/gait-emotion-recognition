"""
API 테스트 코드

현재 src.main 구현 기준으로 FastAPI 엔드포인트를 테스트합니다.
"""

import numpy as np

from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


# 17개 관절 x 5프레임 더미 skeleton_data 생성
# 좌표는 단순 선형 증가값으로 구성하여 feature 추출이 가능하도록 함.
def build_valid_skeleton_data(n_frames: int = 5, n_joints: int = 17):
    skeleton_data = []
    for frame_idx in range(n_frames):
        for joint_idx in range(n_joints):
            x = 0.5 + frame_idx * 0.01 + joint_idx * 0.02
            y = 0.3 + frame_idx * 0.015 + joint_idx * 0.01
            z = 0.1 + frame_idx * 0.005
            skeleton_data.append(f"{x},{y},{z}")
    return skeleton_data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "gait-emotion-recognition"
    assert data["version"] == "2.0.0"
    assert "prediction_logging" in data


def test_frontend_index_is_served_from_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "걸음걸이 감정 인식" in response.text


def test_frontend_javascript_is_served():
    response = client.get("/app.js")
    assert response.status_code == 200
    assert "buildApiUrl" in response.text


def test_predict_emotion_with_valid_skeleton_data():
    response = client.post(
        "/predict_emotion",
        json={
            "skeleton_data": build_valid_skeleton_data(),
            "n_joints": 17,
        },
    )

    # 모델이 정상 로드되면 200, 없으면 503
    assert response.status_code in [200, 503]
    data = response.json()

    if response.status_code == 200:
        assert "emotion" in data
        assert "confidence" in data
        assert "confidence_level" in data
        assert "probabilities" in data
        assert "features" in data
        assert "message" in data
        assert isinstance(data["emotion"], str)
        assert 0 <= data["confidence"] <= 1
        assert data["confidence_level"] in ["high", "medium", "low"]
        assert len(data["features"]) >= 14
    else:
        assert "detail" in data
        assert "모델 파일 로드 실패" in data["detail"]


def test_predict_emotion_with_minimal_skeleton_data_padding_case():
    response = client.post(
        "/predict_emotion",
        json={
            "skeleton_data": build_valid_skeleton_data(n_frames=2),
            "n_joints": 17,
        },
    )

    assert response.status_code in [200, 503]
    data = response.json()

    if response.status_code == 200:
        assert "features" in data
        assert len(data["features"]) >= 14
    else:
        assert "detail" in data


def test_predict_emotion_with_invalid_skeleton_data_format():
    response = client.post(
        "/predict_emotion",
        json={
            "skeleton_data": ["0.5,0.3", "0.6,0.4,0.1"],
            "n_joints": 1,
        },
    )

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_emotion_with_missing_input_field():
    response = client.post(
        "/predict_emotion",
        json={"invalid_field": []},
    )

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_emotion_with_empty_keypoints():
    response = client.post(
        "/predict_emotion",
        json={"keypoints": []},
    )

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_emotion_with_valid_keypoints_array():
    # 13개 관절 x 4프레임 = 52개 좌표
    keypoints = []
    for frame_idx in range(4):
        for joint_idx in range(13):
            keypoints.append([
                0.5 + frame_idx * 0.01 + joint_idx * 0.02,
                0.2 + frame_idx * 0.01 + joint_idx * 0.03,
                0.1,
            ])

    response = client.post(
        "/predict_emotion",
        json={
            "keypoints": keypoints,
            "n_joints": 13,
        },
    )

    assert response.status_code in [200, 503]
    data = response.json()

    if response.status_code == 200:
        assert "emotion" in data
        assert "features" in data
        assert len(data["features"]) >= 14
    else:
        assert "detail" in data


def test_predict_emotion_with_33_joint_skeleton_data():
    skeleton_data = []
    for frame_idx in range(4):
        for joint_idx in range(33):
            skeleton_data.append(
                f"{0.1 + frame_idx * 0.01 + joint_idx * 0.001},"
                f"{0.2 + frame_idx * 0.01 + joint_idx * 0.001},"
                f"{0.3 + frame_idx * 0.005}"
            )

    response = client.post(
        "/predict_emotion",
        json={"skeleton_data": skeleton_data, "n_joints": 33},
    )

    assert response.status_code in [200, 503]
    data = response.json()
    assert "detail" in data or "emotion" in data


def test_cors_preflight_endpoint():
    response = client.options(
        "/predict_emotion",
        headers={
            "Origin": "http://localhost:5500",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200


class DummyPredictionLogStore:
    def __init__(self):
        self.calls = []

    def save_prediction(self, **kwargs):
        self.calls.append(kwargs)
        return True


class DummyModel:
    n_features_in_ = 4

    def predict_proba(self, X):
        return np.array([[0.72, 0.08, 0.05, 0.05, 0.04, 0.06]], dtype=float)


def test_successful_prediction_is_logged(monkeypatch):
    dummy_store = DummyPredictionLogStore()

    monkeypatch.setattr("src.main.prediction_log_store", dummy_store)
    monkeypatch.setattr("src.main.extract_features_from_request", lambda payload: [0.1, 0.2, 0.3, 0.4])
    monkeypatch.setattr("src.main.fusion_model", DummyModel())

    response = client.post(
        "/predict_emotion",
        json={
            "keypoints": [[0.1, 0.2, 0.0] for _ in range(26)],
            "n_joints": 13,
        },
        headers={"X-Request-ID": "req-success-1"},
    )

    assert response.status_code == 200
    assert len(dummy_store.calls) == 1

    log_entry = dummy_store.calls[0]
    assert log_entry["request_id"] == "req-success-1"
    assert log_entry["status_code"] == 200
    assert log_entry["input_type"] == "keypoints"
    assert log_entry["joint_count"] == 13
    assert log_entry["frame_count"] == 2
    assert log_entry["feature_count"] == 4
    assert log_entry["predicted_emotion"] == "happy"
    assert log_entry["confidence_level"] == "medium"


def test_failed_prediction_is_logged(monkeypatch):
    dummy_store = DummyPredictionLogStore()

    def raise_feature_error(payload):
        raise ValueError("bad payload")

    monkeypatch.setattr("src.main.prediction_log_store", dummy_store)
    monkeypatch.setattr("src.main.extract_features_from_request", raise_feature_error)

    response = client.post(
        "/predict_emotion",
        json={"keypoints": []},
        headers={"X-Request-ID": "req-fail-1"},
    )

    assert response.status_code == 422
    assert len(dummy_store.calls) == 1

    log_entry = dummy_store.calls[0]
    assert log_entry["request_id"] == "req-fail-1"
    assert log_entry["status_code"] == 422
    assert log_entry["input_type"] == "keypoints"
    assert "특징 추출 실패" in log_entry["error_message"]
