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


EMOTION_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]


@pytest.fixture
def dictionary_model_path():
    """딕셔너리 형태로 저장된 모델 픽스처"""
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    X_train = np.random.RandomState(42).rand(200, 14)
    y_train = np.random.RandomState(42).choice(EMOTION_CLASSES, 200)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    clf.fit(X_scaled, y_train)

    le = LabelEncoder()
    le.fit(EMOTION_CLASSES)

    model_dict = {
        "model": clf,
        "scaler": scaler,
        "label_encoder": le,
        "classes": EMOTION_CLASSES,
        "feature_dim": 14,
    }

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name
        joblib.dump(model_dict, tmp_path)

    yield tmp_path

    if os.path.exists(tmp_path):
        os.remove(tmp_path)


@pytest.fixture
def legacy_model_path():
    """직접 모델 객체로 저장된 레거시 형식 픽스처"""
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    X_train = np.random.RandomState(42).rand(100, 14)
    y_train = np.random.RandomState(42).choice(["Happy", "Sad", "Angry"], 100)
    clf.fit(X_train, y_train)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name
        joblib.dump(clf, tmp_path)

    yield tmp_path

    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_load_dictionary_based_model(dictionary_model_path):
    """딕셔너리 형태로 저장된 모델 파일을 올바르게 로드하는지 테스트"""
    model = EmotionModel(dictionary_model_path)

    assert model.model is not None
    assert model.scaler is not None
    assert model.label_encoder is not None
    assert model.classes is not None
    assert model.feature_dim == 14
    assert model.use_fallback is False

    assert len(model.classes) == 6
    assert set(model.classes) == set(EMOTION_CLASSES)


def test_model_prediction_with_dictionary_model(dictionary_model_path):
    """딕셔너리 형태 모델로 예측이 정상적으로 작동하는지 테스트"""
    model = EmotionModel(dictionary_model_path)

    features = np.random.rand(14)
    result = model.predict_emotion(features)

    assert "emotion" in result
    assert "confidence" in result
    assert "probabilities" in result

    assert result["emotion"] in model.classes
    assert 0 <= result["confidence"] <= 1
    assert len(result["probabilities"]) == 6

    for emotion in model.classes:
        assert emotion in result["probabilities"]
        assert 0 <= result["probabilities"][emotion] <= 1


def test_model_prediction_with_scaler(dictionary_model_path):
    """스케일러가 올바르게 적용되는지 테스트"""
    model = EmotionModel(dictionary_model_path)

    features = np.array([1.2, 0.5, 2.8, 0.3, 0.4, 0.0, 0.5, 0.3, 1.0, 0.8, 0.7, 1.2, 0.6, 0.9])
    result = model.predict_emotion(features)

    assert result["emotion"] in model.classes
    assert result["confidence"] > 0


def test_load_legacy_model(legacy_model_path):
    """직접 모델 객체로 저장된 구 형식도 지원하는지 테스트"""
    model = EmotionModel(legacy_model_path)

    assert model.model is not None
    assert model.use_fallback is False
    assert model.scaler is None
    assert model.label_encoder is None

    features = np.random.rand(14)
    result = model.predict_emotion(features)
    assert "emotion" in result
    assert "confidence" in result


def test_fallback_when_model_file_missing():
    """모델 파일이 없을 때 규칙 기반 예측으로 전환하는지 테스트"""
    model = EmotionModel("models/nonexistent_model.joblib")

    assert model.use_fallback is True
    assert model.model is None

    features = np.random.rand(14)
    result = model.predict_emotion(features)

    assert "emotion" in result
    assert "confidence" in result
    assert "probabilities" in result


def test_fallback_when_model_key_missing():
    """딕셔너리에 'model' 키가 없을 때 규칙 기반 예측으로 전환하는지 테스트"""
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name
        invalid_dict = {
            "scaler": StandardScaler(),
            "label_encoder": LabelEncoder(),
            "classes": ["Happy", "Sad"],
        }
        joblib.dump(invalid_dict, tmp_path)

    try:
        model = EmotionModel(tmp_path)
        assert model.use_fallback is True

        features = np.random.rand(14)
        result = model.predict_emotion(features)
        assert "emotion" in result
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_probability_sum(dictionary_model_path):
    """모든 감정의 확률 합이 대략 1에 가까운지 테스트"""
    model = EmotionModel(dictionary_model_path)

    features = np.random.rand(14)
    result = model.predict_emotion(features)

    total_prob = sum(result["probabilities"].values())
    assert 0.99 <= total_prob <= 1.01


def test_multiple_predictions(dictionary_model_path):
    """여러 번 예측을 수행해도 일관되게 작동하는지 테스트"""
    model = EmotionModel(dictionary_model_path)

    features = np.array([1.2, 0.5, 2.8, 0.3, 0.4, 0.0, 0.5, 0.3, 1.0, 0.8, 0.7, 1.2, 0.6, 0.9])

    results = [model.predict_emotion(features) for _ in range(5)]

    first_emotion = results[0]["emotion"]
    for result in results[1:]:
        assert result["emotion"] == first_emotion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
