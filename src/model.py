"""
감정 예측 모델 모듈 (Emotion Prediction Model)

이 파일은 학습된 머신러닝 모델을 로드하고, 걸음걸이 특징으로부터 감정을 예측합니다.
모델 파일이 없을 경우 규칙 기반(Rule-based) 예측으로 대체됩니다.
"""

import os
import logging
import numpy as np
from typing import Dict, Tuple
import joblib

# 로깅 설정
logger = logging.getLogger(__name__)

# 감정 클래스 (6가지)
EMOTION_CLASSES = ["Happy", "Sad", "Fear", "Disgust", "Angry", "Neutral"]


class EmotionModel:
    """
    감정 예측 모델 클래스
    
    학습된 Random Forest 모델을 로드하거나,
    모델이 없을 경우 규칙 기반 예측을 사용합니다.
    """
    
    def __init__(self, model_path: str = "models/rf_emotion_model.joblib"):
        """
        모델 초기화
        
        Args:
            model_path: 학습된 모델 파일의 경로
        """
        self.model_path = model_path
        self.model = None
        self.use_fallback = False
        
        # 모델 로드 시도
        self._load_model()
    
    def _load_model(self):
        """
        joblib으로 저장된 모델 파일을 로드합니다.
        모델 파일이 없으면 fallback 모드로 전환합니다.
        """
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"모델을 성공적으로 로드했습니다: {self.model_path}")
                self.use_fallback = False
            except Exception as e:
                logger.warning(f"모델 로드 실패: {e}. 규칙 기반 예측을 사용합니다.")
                self.use_fallback = True
        else:
            logger.warning(f"모델 파일이 없습니다: {self.model_path}. 규칙 기반 예측을 사용합니다.")
            self.use_fallback = True
    
    def predict_emotion(self, features: np.ndarray) -> Dict[str, any]:
        """
        걸음걸이 특징으로부터 감정을 예측합니다.
        
        Args:
            features: 14개의 걸음걸이 특징값 (numpy 배열)
        
        Returns:
            예측 결과 딕셔너리
            {
                "emotion": "happy",  # 예측된 감정
                "confidence": 0.85,  # 예측 신뢰도 (0~1)
                "probabilities": {   # 각 감정의 확률
                    "happy": 0.85,
                    "sad": 0.10,
                    "angry": 0.05
                }
            }
        """
        if self.use_fallback:
            return self._rule_based_prediction(features)
        else:
            return self._model_based_prediction(features)
    
    def _model_based_prediction(self, features: np.ndarray) -> Dict[str, any]:
        """
        학습된 모델을 사용한 예측
        
        Args:
            features: 14개의 걸음걸이 특징값
            
        Returns:
            예측 결과 딕셔너리
        """
        try:
            # 특징을 2D 배열로 변환 (모델 입력 형식)
            features_2d = features.reshape(1, -1)
            
            # 예측 수행
            prediction = self.model.predict(features_2d)[0]
            
            # 확률 예측 (모델이 predict_proba를 지원하는 경우)
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_2d)[0]
                emotion_labels = self.model.classes_
                
                # 확률을 딕셔너리로 변환
                prob_dict = {
                    label: float(prob) 
                    for label, prob in zip(emotion_labels, probabilities)
                }
                
                confidence = float(max(probabilities))
            else:
                # predict_proba가 없는 경우 기본값 설정
                prob_dict = {prediction: 1.0}
                confidence = 1.0
            
            return {
                "emotion": str(prediction),
                "confidence": confidence,
                "probabilities": prob_dict
            }
            
        except Exception as e:
            logger.error(f"모델 예측 중 오류 발생: {e}. 규칙 기반 예측으로 전환합니다.")
            return self._rule_based_prediction(features)
    
    def _rule_based_prediction(self, features: np.ndarray) -> Dict[str, any]:
        """
        규칙 기반 예측 (Fallback)
        
        모델이 없을 때 사용하는 간단한 규칙 기반 감정 예측입니다.
        실제 머신러닝 모델만큼 정확하지는 않지만, 데모 목적으로 사용할 수 있습니다.
        
        Args:
            features: 14개의 걸음걸이 특징값
        
        Returns:
            예측 결과 딕셔너리
            
        규칙 설명:
        - Happy (행복): 빠른 속도 + 큰 보폭 + 높은 에너지 + 자연스러운 팔 스윙
        - Sad (슬픔): 느린 속도 + 작은 보폭 + 낮은 에너지 + 머리 숙임
        - Fear (공포): 중간 속도 + 불규칙한 걸음 + 경직된 자세 + 높은 긴장도
        - Disgust (혐오): 느린 속도 + 움츠린 자세 + 낮은 자세 개방도
        - Angry (분노): 빠른 속도 + 불규칙한 걸음 + 높은 에너지 + 경직된 자세
        - Neutral (중립): 위 조건에 모두 해당하지 않는 경우
        """
        # 특징 추출 (인덱스 기반)
        avg_speed = features[0]           # 평균 속도
        stride_length = features[1]       # 보폭
        step_frequency = features[2]      # 발걸음 빈도
        arm_swing = features[4]           # 팔 스윙
        head_tilt = features[5]           # 머리 기울기
        posture_openness = features[6]    # 자세 개방도
        vertical_bounce = features[7]     # 수직 움직임
        step_regularity = features[10]    # 걸음 규칙성
        energy = features[11]             # 에너지
        
        # 감정별 점수 계산
        happy_score = 0.0
        sad_score = 0.0
        fear_score = 0.0
        disgust_score = 0.0
        angry_score = 0.0
        neutral_score = 0.5  # 기본값
        
        # Happy 조건: 빠르고 경쾌한 걸음
        if avg_speed > 1.5 and energy > 1.2:
            happy_score += 0.3
        if arm_swing > 0.4:
            happy_score += 0.2
        if vertical_bounce > 0.3:
            happy_score += 0.2
        if stride_length > 0.6:
            happy_score += 0.15
        if step_regularity > 0.7:
            happy_score += 0.15
        
        # Sad 조건: 느리고 무거운 걸음
        if avg_speed < 0.8 and energy < 0.8:
            sad_score += 0.3
        if arm_swing < 0.2:
            sad_score += 0.2
        if head_tilt < -0.1:  # 머리를 숙임
            sad_score += 0.2
        if stride_length < 0.4:
            sad_score += 0.15
        if vertical_bounce < 0.15:
            sad_score += 0.15
        
        # Fear 조건: 경계하는 걸음
        if 0.9 < avg_speed < 1.4 and step_regularity < 0.65:
            fear_score += 0.3
        if posture_openness < 0.45:  # 경직되고 움츠린 자세
            fear_score += 0.25
        if arm_swing < 0.3:  # 팔 움직임 제한적
            fear_score += 0.2
        if energy > 0.9 and energy < 1.3:  # 중간 긴장도
            fear_score += 0.15
        if stride_length < 0.5:  # 짧은 보폭
            fear_score += 0.1
        
        # Disgust 조건: 움츠리고 거부하는 걸음
        if avg_speed < 1.0 and posture_openness < 0.4:
            disgust_score += 0.3
        if head_tilt < -0.05:  # 약간 머리를 숙이거나 돌림
            disgust_score += 0.2
        if arm_swing < 0.25:  # 제한된 팔 움직임
            disgust_score += 0.2
        if stride_length < 0.45:  # 작은 보폭
            disgust_score += 0.15
        if vertical_bounce < 0.2:  # 낮은 수직 움직임
            disgust_score += 0.15
        
        # Angry 조건: 빠르지만 불규칙한 걸음
        if avg_speed > 1.3 and step_regularity < 0.6:
            angry_score += 0.3
        if energy > 1.5:
            angry_score += 0.2
        if posture_openness < 0.4:  # 경직된 자세
            angry_score += 0.2
        if arm_swing < 0.25:  # 팔 움직임 제한적
            angry_score += 0.15
        if step_frequency > 3.0:
            angry_score += 0.15
        
        # 가장 높은 점수를 가진 감정 선택
        scores = {
            "Happy": happy_score,
            "Sad": sad_score,
            "Fear": fear_score,
            "Disgust": disgust_score,
            "Angry": angry_score,
            "Neutral": neutral_score
        }
        
        # 점수 정규화 (합이 1이 되도록)
        total_score = sum(scores.values())
        if total_score > 0:
            probabilities = {
                emotion: score / total_score 
                for emotion, score in scores.items()
            }
        else:
            # 모든 점수가 0인 경우 균등 분포
            probabilities = {
                "Happy": 1.0/6,
                "Sad": 1.0/6,
                "Fear": 1.0/6,
                "Disgust": 1.0/6,
                "Angry": 1.0/6,
                "Neutral": 1.0/6
            }
        
        # 가장 확률이 높은 감정 선택
        predicted_emotion = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted_emotion]
        
        return {
            "emotion": predicted_emotion,
            "confidence": float(confidence),
            "probabilities": {k: float(v) for k, v in probabilities.items()}
        }


# 전역 모델 인스턴스 (싱글톤 패턴)
_model_instance = None


def get_model(model_path: str = "models/rf_emotion_model.joblib") -> EmotionModel:
    """
    모델 인스턴스를 반환합니다 (싱글톤 패턴).
    
    Args:
        model_path: 모델 파일 경로
        
    Returns:
        EmotionModel 인스턴스
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = EmotionModel(model_path)
    return _model_instance


def predict_emotion(features: np.ndarray, model_path: str = "models/rf_emotion_model.joblib") -> Dict[str, any]:
    """
    걸음걸이 특징으로부터 감정을 예측하는 편의 함수
    
    Args:
        features: 14개의 걸음걸이 특징값 (numpy 배열)
        model_path: 모델 파일 경로 (선택사항)
    
    Returns:
        예측 결과 딕셔너리
    """
    model = get_model(model_path)
    return model.predict_emotion(features)
