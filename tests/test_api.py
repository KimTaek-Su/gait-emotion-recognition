"""
API 테스트 코드

현재 src.main 구현 기준으로 FastAPI 엔드포인트를 테스트합니다.
"""

from fastapi.testclient import TestClient
from src.main import app
import src.main as main_module
from src.main import MODEL_PATH

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
    assert "model" in data
    assert data["model"]["mode"] in ["trained", "fallback"]


def test_predict_emotion_with_valid_skeleton_data():
    response = client.post(
        "/predict_emotion",
        json={
            "skeleton_data": build_valid_skeleton_data(),
            "n_joints": 17,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "emotion" in data
    assert "confidence" in data
    assert "confidence_level" in data
    assert "probabilities" in data
    assert "features" in data
    assert "message" in data
    assert "model" in data
    assert isinstance(data["emotion"], str)
    assert 0 <= data["confidence"] <= 1
    assert data["confidence_level"] in ["high", "medium", "low"]
    assert len(data["features"]) >= 14
    assert data["model"]["mode"] in ["trained", "fallback"]


def test_predict_emotion_with_minimal_skeleton_data_padding_case():
    response = client.post(
        "/predict_emotion",
        json={
            "skeleton_data": build_valid_skeleton_data(n_frames=2),
            "n_joints": 17,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert len(data["features"]) >= 14


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

    assert response.status_code == 200
    data = response.json()
    assert "emotion" in data
    assert "features" in data
    assert len(data["features"]) >= 14


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

    assert response.status_code == 200
    data = response.json()
    assert "emotion" in data
    assert data["model"]["mode"] in ["trained", "fallback"]


def test_predict_emotion_uses_fallback_model_when_configured(monkeypatch):
    class TrackingFallbackModel(main_module.FallbackEmotionModel):
        def __init__(self):
            self.called = False

        def predict_proba(self, X):
            self.called = True
            return super().predict_proba(X)

    fallback_model = TrackingFallbackModel()

    monkeypatch.setattr(main_module, "fusion_model", fallback_model)
    monkeypatch.setattr(
        main_module,
        "model_runtime_info",
        {
            "mode": "fallback",
            "source": "in_repo_demo",
            "path": MODEL_PATH,
            "n_features_in": 14,
            "fallback_reason": "test override",
        },
    )

    response = client.post(
        "/predict_emotion",
        json={
            "skeleton_data": build_valid_skeleton_data(),
            "n_joints": 17,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model"]["mode"] == "fallback"
    assert data["model"]["source"] == "in_repo_demo"
    assert fallback_model.called is True


def test_cors_preflight_endpoint():
    response = client.options(
        "/predict_emotion",
        headers={
            "Origin": "http://localhost:5500",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
