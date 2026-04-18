"""
감정 예측 모델 모듈 (EmotionModel)

모델 로딩, 예측, 폴백 처리를 담당합니다.
"""

import numpy as np
import joblib
import os


EMOTION_LABELS = ["happy", "sad", "fear", "disgust", "angry", "neutral"]


class EmotionModel:
    """
    걸음걸이 감정 인식 모델 래퍼 클래스

    - joblib 파일에서 모델을 로드합니다.
    - 딕셔너리 형태(model, scaler, label_encoder, classes)와 레거시(직접 모델 객체) 형식 모두 지원합니다.
    - 모델 로드 실패 시 규칙 기반 폴백 예측을 제공합니다.
    """

    def __init__(self, model_path: str):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.classes = None
        self.feature_dim = 14
        self.use_fallback = False

        if not os.path.exists(model_path):
            print(f"[WARN] 모델 파일 없음: {model_path} → 폴백 모드 활성화")
            self.use_fallback = True
            return

        try:
            loaded = joblib.load(model_path)

            if isinstance(loaded, dict):
                if "model" not in loaded or loaded["model"] is None:
                    print(f"[WARN] 딕셔너리에 'model' 키 없음 → 폴백 모드 활성화")
                    self.use_fallback = True
                    return

                self.model = loaded["model"]
                self.scaler = loaded.get("scaler", None)
                self.label_encoder = loaded.get("label_encoder", None)
                self.classes = loaded.get("classes", None)
                self.feature_dim = loaded.get("feature_dim", 14)
                print(f"[MODEL] 딕셔너리 형태 모델 로드 완료: {model_path}")
            else:
                # 레거시 형식: 모델 객체 직접 저장
                self.model = loaded
                if hasattr(loaded, "classes_"):
                    self.classes = list(loaded.classes_)
                print(f"[MODEL] 레거시 모델 로드 완료: {model_path}")

        except Exception as e:
            print(f"[ERROR] 모델 로드 실패: {repr(e)} → 폴백 모드 활성화")
            self.use_fallback = True

    def predict_emotion(self, features: np.ndarray) -> dict:
        """
        특징 벡터로부터 감정을 예측합니다.

        Args:
            features: 14개 특징값 numpy 배열

        Returns:
            {"emotion": str, "confidence": float, "probabilities": dict}
        """
        if self.use_fallback or self.model is None:
            return self._fallback_predict(features)

        try:
            X = np.array(features).reshape(1, -1)

            if self.scaler is not None:
                X = self.scaler.transform(X)

            pred_probs = self.model.predict_proba(X)[0]
            pred_idx = int(np.argmax(pred_probs))

            if self.classes is not None:
                classes = list(self.classes)
            elif hasattr(self.model, "classes_"):
                classes = list(self.model.classes_)
            else:
                classes = EMOTION_LABELS

            if self.label_encoder is not None:
                emotion = classes[pred_idx]
            else:
                emotion = classes[pred_idx]

            confidence = float(np.max(pred_probs))
            probabilities = {cls: float(prob) for cls, prob in zip(classes, pred_probs)}

            return {
                "emotion": emotion,
                "confidence": confidence,
                "probabilities": probabilities,
            }

        except Exception as e:
            print(f"[WARN] 모델 예측 실패: {repr(e)} → 폴백 사용")
            return self._fallback_predict(features)

    def _fallback_predict(self, features: np.ndarray) -> dict:
        """
        규칙 기반 폴백 예측

        특징값의 패턴에 따라 간단한 규칙으로 감정을 추정합니다.
        """
        f = np.array(features, dtype=float)
        avg_speed = f[0] if len(f) > 0 else 0.5

        if avg_speed > 1.5:
            emotion = "happy"
            probs = [0.5, 0.05, 0.05, 0.05, 0.3, 0.05]
        elif avg_speed < 0.3:
            emotion = "sad"
            probs = [0.05, 0.5, 0.1, 0.1, 0.05, 0.2]
        else:
            emotion = "neutral"
            probs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]

        classes = EMOTION_LABELS
        confidence = max(probs)

        return {
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": {cls: float(p) for cls, p in zip(classes, probs)},
        }
