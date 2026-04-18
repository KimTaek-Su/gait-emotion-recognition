"""
API 테스트 코드

pytest를 사용하여 FastAPI 엔드포인트를 테스트합니다.
"""

import pytest
from fastapi.testclient import TestClient
from src.main import app

# TestClient 생성
client = TestClient(app)


def test_root_endpoint():
    """
    루트 엔드포인트 테스트

    GET / 요청이 정상적으로 작동하는지 확인합니다.
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["version"] == "2.0.0"


def test_health_endpoint():
    """
    헬스 체크 엔드포인트 테스트

    GET /health 요청이 정상적으로 작동하는지 확인합니다.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "gait-emotion-recognition"
    assert data["version"] == "2.0.0"


def test_predict_emotion_with_skeleton_data():
    """
    skeleton_data 형식으로 감정 예측 테스트

    17개 관절 x 5 프레임 데이터가 정상적으로 처리되는지 확인합니다.
    """
    # 17개 관절 x 5 프레임 = 85개의 좌표
    skeleton_data = []
    for frame_idx in range(5):
        for joint_idx in range(17):
            x = 0.5 + frame_idx * 0.01 + joint_idx * 0.02
            y = 0.3 + frame_idx * 0.01 + joint_idx * 0.03
            z = 0.1 + frame_idx * 0.005
            skeleton_data.append(f"{x},{y},{z}")

    response = client.post(
        "/predict_emotion",
        json={
            "skeleton_data": skeleton_data,
            "n_joints": 17,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # 응답 데이터 검증
    assert "emotion" in data
    assert "confidence" in data
    assert "confidence_level" in data
    assert "probabilities" in data
    assert "features" in data
    assert "message" in data

    # 감정이 유효한 값인지 확인
    assert isinstance(data["emotion"], str)
    assert len(data["emotion"]) > 0

    # 신뢰도가 0~1 범위인지 확인
    assert 0 <= data["confidence"] <= 1

    # 신뢰도 수준이 유효한지 확인
    assert data["confidence_level"] in ["high", "medium", "low"]


def test_predict_emotion_with_minimal_skeleton_data():
    """
    최소한의 skeleton_data로 감정 예측 테스트

    4프레임 미만의 데이터도 패딩을 통해 처리되는지 확인합니다.
    """
    # 17개 관절 x 2 프레임 = 34개의 좌표
    skeleton_data = []
    for frame_idx in range(2):
        for joint_idx in range(17):
            x = 0.5 + frame_idx * 0.01 + joint_idx * 0.02
            y = 0.3 + frame_idx * 0.01 + joint_idx * 0.03
            z = 0.1 + frame_idx * 0.005
            skeleton_data.append(f"{x},{y},{z}")

    response = client.post(
        "/predict_emotion",
        json={
            "skeleton_data": skeleton_data,
            "n_joints": 17,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "features" in data


def test_predict_emotion_with_empty_data():
    """
    빈 데이터로 감정 예측 테스트

    빈 요청을 보낼 때 422 에러를 반환하는지 확인합니다.
    """
    response = client.post(
        "/predict_emotion",
        json={},
    )

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_emotion_with_invalid_skeleton_data():
    """
    잘못된 skeleton_data 형식으로 감정 예측 테스트

    파싱 에러가 적절히 처리되는지 확인합니다.
    """
    skeleton_data = [
        "0.5,0.3",  # z 좌표 누락
        "0.6,0.4,0.1",
    ]

    response = client.post(
        "/predict_emotion",
        json={
            "skeleton_data": skeleton_data,
            "n_joints": 1,
        },
    )

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_emotion_probabilities_sum():
    """
    확률 합계 테스트

    모든 감정의 확률 합이 대략 1에 가까운지 확인합니다.
    """
    skeleton_data = []
    for frame_idx in range(5):
        for joint_idx in range(17):
            x = 0.5 + frame_idx * 0.01 + joint_idx * 0.02
            y = 0.3 + frame_idx * 0.01 + joint_idx * 0.03
            z = 0.1 + frame_idx * 0.005
            skeleton_data.append(f"{x},{y},{z}")

    response = client.post(
        "/predict_emotion",
        json={"skeleton_data": skeleton_data, "n_joints": 17},
    )

    assert response.status_code == 200
    data = response.json()

    total_prob = sum(data["probabilities"].values())
    assert 0.9 <= total_prob <= 1.1


def test_cors_headers():
    """
    CORS 헤더 테스트

    프론트엔드에서 API를 호출할 수 있도록 CORS 헤더가 설정되어 있는지 확인합니다.
    """
    response = client.options(
        "/predict_emotion",
        headers={
            "Origin": "http://localhost:5500",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert "access-control-allow-origin" in response.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
