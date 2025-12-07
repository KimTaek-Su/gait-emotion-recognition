"""
모델 로딩 및 예측 테스트

EmotionModel 클래스의 모델 로딩 기능을 테스트합니다.
"""

import pytest
import numpy as np
import joblib
import os
import tempfile
from src.model import EmotionModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder


def test_load_dictionary_based_model():
    """
    딕셔너리 형태로 저장된 모델 파일을 올바르게 로드하는지 테스트
    """
    model = EmotionModel("models/rf_emotion_model.joblib")
    
    # 모델이 올바르게 로드되었는지 확인
    assert model.model is not None
    assert model.scaler is not None
    assert model.label_encoder is not None
    assert model.classes is not None
    assert model.feature_dim == 14
    assert model.use_fallback == False
    
    # 감정 클래스가 6개인지 확인
    assert len(model.classes) == 6
    assert set(model.classes) == {'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad'}


def test_model_prediction_with_dictionary_model():
    """
    딕셔너리 형태 모델로 예측이 정상적으로 작동하는지 테스트
    """
    model = EmotionModel("models/rf_emotion_model.joblib")
    
    # 랜덤 특징으로 예측 수행
    features = np.random.rand(14)
    result = model.predict_emotion(features)
    
    # 결과 검증
    assert "emotion" in result
    assert "confidence" in result
    assert "probabilities" in result
    
    # 예측된 감정이 유효한 클래스인지 확인
    assert result["emotion"] in model.classes
    
    # 신뢰도가 0~1 범위인지 확인
    assert 0 <= result["confidence"] <= 1
    
    # 모든 감정의 확률이 있는지 확인
    assert len(result["probabilities"]) == 6
    for emotion in model.classes:
        assert emotion in result["probabilities"]
        assert 0 <= result["probabilities"][emotion] <= 1


def test_model_prediction_with_scaler():
    """
    스케일러가 올바르게 적용되는지 테스트
    """
    model = EmotionModel("models/rf_emotion_model.joblib")
    
    # 정상 범위의 특징 (0~2 범위)
    features = np.array([1.2, 0.5, 2.8, 0.3, 0.4, 0.0, 0.5, 0.3, 1.0, 0.8, 0.7, 1.2, 0.6, 0.9])
    
    # 예측 수행 (스케일러가 내부적으로 적용됨)
    result = model.predict_emotion(features)
    
    # 예측이 성공했는지 확인
    assert result["emotion"] in model.classes
    assert result["confidence"] > 0


def test_load_legacy_model():
    """
    직접 모델 객체로 저장된 구 형식도 지원하는지 테스트
    """
    # 임시 파일에 구 형식 모델 저장
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
        tmp_path = tmp.name
        
        # 간단한 RandomForest 모델 생성 및 저장
        legacy_model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = np.random.rand(100, 14)
        y_train = np.random.choice(['Happy', 'Sad', 'Angry'], 100)
        legacy_model.fit(X_train, y_train)
        
        joblib.dump(legacy_model, tmp_path)
    
    try:
        # 구 형식 모델 로드
        model = EmotionModel(tmp_path)
        
        # 모델이 로드되었는지 확인
        assert model.model is not None
        assert model.use_fallback == False
        
        # 스케일러와 인코더는 없어야 함
        assert model.scaler is None
        assert model.label_encoder is None
        
        # 예측이 가능한지 확인
        features = np.random.rand(14)
        result = model.predict_emotion(features)
        assert "emotion" in result
        assert "confidence" in result
        
    finally:
        # 임시 파일 삭제
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_fallback_when_model_file_missing():
    """
    모델 파일이 없을 때 규칙 기반 예측으로 전환하는지 테스트
    """
    model = EmotionModel("models/nonexistent_model.joblib")
    
    # fallback 모드가 활성화되어야 함
    assert model.use_fallback == True
    assert model.model is None
    
    # 규칙 기반 예측이 작동하는지 확인
    features = np.random.rand(14)
    result = model.predict_emotion(features)
    
    assert "emotion" in result
    assert "confidence" in result
    assert "probabilities" in result


def test_fallback_when_model_key_missing():
    """
    딕셔너리에 'model' 키가 없을 때 규칙 기반 예측으로 전환하는지 테스트
    """
    # 임시 파일에 'model' 키가 없는 딕셔너리 저장
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
        tmp_path = tmp.name
        invalid_dict = {
            'scaler': StandardScaler(),
            'label_encoder': LabelEncoder(),
            'classes': ['Happy', 'Sad']
            # 'model' 키가 없음
        }
        joblib.dump(invalid_dict, tmp_path)
    
    try:
        # 잘못된 딕셔너리 모델 로드
        model = EmotionModel(tmp_path)
        
        # fallback 모드가 활성화되어야 함
        assert model.use_fallback == True
        
        # 규칙 기반 예측이 작동하는지 확인
        features = np.random.rand(14)
        result = model.predict_emotion(features)
        assert "emotion" in result
        
    finally:
        # 임시 파일 삭제
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_probability_sum():
    """
    모든 감정의 확률 합이 대략 1에 가까운지 테스트
    """
    model = EmotionModel("models/rf_emotion_model.joblib")
    
    features = np.random.rand(14)
    result = model.predict_emotion(features)
    
    # 확률의 합 계산
    total_prob = sum(result["probabilities"].values())
    
    # 합이 0.99 ~ 1.01 범위 내에 있는지 확인 (부동소수점 오차 고려)
    assert 0.99 <= total_prob <= 1.01


def test_multiple_predictions():
    """
    여러 번 예측을 수행해도 일관되게 작동하는지 테스트
    """
    model = EmotionModel("models/rf_emotion_model.joblib")
    
    # 동일한 특징으로 여러 번 예측
    features = np.array([1.2, 0.5, 2.8, 0.3, 0.4, 0.0, 0.5, 0.3, 1.0, 0.8, 0.7, 1.2, 0.6, 0.9])
    
    results = []
    for _ in range(5):
        result = model.predict_emotion(features)
        results.append(result)
    
    # 모든 예측이 동일한 결과를 반환하는지 확인 (결정론적)
    first_emotion = results[0]["emotion"]
    for result in results[1:]:
        assert result["emotion"] == first_emotion


if __name__ == "__main__":
    # pytest 실행
    pytest.main([__file__, "-v"])
