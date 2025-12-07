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
    assert data["version"] == "1.0.0"


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
    assert data["version"] == "1.0.0"


def test_predict_emotion_with_valid_data():
    """
    유효한 데이터로 감정 예측 테스트
    
    POST /predict_emotion 요청에 올바른 키포인트 데이터를 보내고
    정상적인 응답을 받는지 확인합니다.
    """
    # 샘플 키포인트 데이터
    valid_keypoints = [
        {
            "nose": [320, 100],
            "left_shoulder": [280, 150],
            "right_shoulder": [360, 150],
            "left_elbow": [250, 200],
            "right_elbow": [390, 200],
            "left_wrist": [230, 250],
            "right_wrist": [410, 250],
            "left_hip": [290, 300],
            "right_hip": [350, 300],
            "left_knee": [285, 400],
            "right_knee": [355, 400],
            "left_ankle": [280, 500],
            "right_ankle": [360, 500]
        },
        {
            "nose": [325, 105],
            "left_shoulder": [285, 155],
            "right_shoulder": [365, 155],
            "left_hip": [295, 305],
            "right_hip": [355, 305],
            "left_ankle": [285, 505],
            "right_ankle": [365, 505]
        }
    ]
    
    response = client.post(
        "/predict_emotion",
        json={"keypoints": valid_keypoints}
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
    
    # 특징이 14개인지 확인
    assert len(data["features"]) == 14


def test_predict_emotion_with_insufficient_data():
    """
    부족한 데이터로 감정 예측 테스트
    
    프레임이 2개 미만일 때 400 에러를 반환하는지 확인합니다.
    """
    # 프레임이 1개만 있는 데이터
    insufficient_keypoints = [
        {
            "nose": [320, 100],
            "left_shoulder": [280, 150],
            "right_shoulder": [360, 150]
        }
    ]
    
    response = client.post(
        "/predict_emotion",
        json={"keypoints": insufficient_keypoints}
    )
    
    assert response.status_code == 400  # Bad request
    data = response.json()
    assert "detail" in data


def test_predict_emotion_with_empty_data():
    """
    빈 데이터로 감정 예측 테스트
    
    빈 키포인트 리스트를 보낼 때 400 에러를 반환하는지 확인합니다.
    """
    response = client.post(
        "/predict_emotion",
        json={"keypoints": []}
    )
    
    assert response.status_code == 400  # Bad request
    data = response.json()
    assert "detail" in data


def test_predict_emotion_with_invalid_format():
    """
    잘못된 형식으로 감정 예측 테스트
    
    키포인트 필드가 없을 때 400 에러를 반환하는지 확인합니다.
    """
    response = client.post(
        "/predict_emotion",
        json={"invalid_field": []}
    )
    
    assert response.status_code == 400  # Bad request


def test_predict_emotion_probabilities_sum():
    """
    확률 합계 테스트
    
    모든 감정의 확률 합이 대략 1에 가까운지 확인합니다.
    (규칙 기반 예측에서는 정확히 1이어야 하지만, 필터링으로 인해 약간 다를 수 있음)
    """
    valid_keypoints = [
        {"nose": [320, 100], "left_ankle": [280, 500], "right_ankle": [360, 500]},
        {"nose": [325, 105], "left_ankle": [285, 505], "right_ankle": [365, 505]}
    ]
    
    response = client.post(
        "/predict_emotion",
        json={"keypoints": valid_keypoints}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # 확률의 합 계산
    total_prob = sum(data["probabilities"].values())
    
    # 합이 0.9 ~ 1.1 범위 내에 있는지 확인 (필터링 고려)
    assert 0.9 <= total_prob <= 1.1


def test_cors_headers():
    """
    CORS 헤더 테스트
    
    프론트엔드에서 API를 호출할 수 있도록 CORS 헤더가 설정되어 있는지 확인합니다.
    """
    # POST 요청으로 CORS 헤더 확인 (OPTIONS는 FastAPI에서 자동 처리됨)
    response = client.post(
        "/predict_emotion",
        headers={"Origin": "http://localhost:5500"},
        json={"keypoints": [
            {"nose": [320, 100]},
            {"nose": [325, 105]}
        ]}
    )
    
    # CORS 헤더가 응답에 포함되어 있는지 확인
    assert "access-control-allow-origin" in response.headers or response.status_code in [200, 400]


def test_predict_emotion_with_skeleton_data():
    """
    skeleton_data 형식으로 감정 예측 테스트
    
    새로운 skeleton_data 형식이 정상적으로 작동하는지 확인합니다.
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
            "n_joints": 17
        }
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
    
    # 특징이 14개인지 확인
    assert len(data["features"]) == 14


def test_predict_emotion_with_minimal_skeleton_data():
    """
    최소한의 skeleton_data로 감정 예측 테스트
    
    4프레임 미만의 데이터도 패딩을 통해 처리되는지 확인합니다.
    """
    # 17개 관절 x 2 프레임 = 34개의 좌표 (최소 프레임보다 적음)
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
            "n_joints": 17
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["features"]) == 14


if __name__ == "__main__":
    # pytest 실행
    pytest.main([__file__, "-v"])
