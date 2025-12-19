from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any
import numpy as np
import joblib

# ====== 1. Pydantic 모델 정의 ======
class PredictionRequest(BaseModel):
    keypoints: List[List[float]] = Field(..., description="프레임별 3차원 좌표 리스트 (예: [[x, y, z], ...])")

    @validator("keypoints", each_item=True)
    def validate_keypoint_item(cls, v, values, **kwargs):
        if not isinstance(v, list):
            raise ValueError(f"좌표 {v}가 리스트가 아닙니다.")
        if len(v) != 3:
            raise ValueError(f"좌표에 3개의 값이 필요합니다. 현재: {len(v)}개")
        if not all(isinstance(val, (int, float)) for val in v):
            raise ValueError(f"좌표 {v}의 값이 int나 float 형식이 아닙니다.")
        return v

# ====== 2. FastAPI 앱 객체 생성 ======
app = FastAPI(
    title="걸음걸이 감정 인식 API",
    description="Bi-LSTM HCF Fusion 기반의 실시간 감정 예측 시스템",
    version="2.0.0"
)

# ====== 3. 모델 로드 (Bi-LSTM_HCF_Fusion_final_results.pkl) ======
try:
    fusion_model = joblib.load('./models/deployment/gait_emotion_api_model.joblib')
except Exception as e:
    fusion_model = None
    print(f"[!] 모델 로드 실패: {e}")

# 감정 라벨 예시
emotion_labels = ["happy", "sad", "fear", "disgust", "angry", "neutral"]

# ====== 4. 헬스 체크 ======
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "gait-emotion-recognition",
        "version": "2.0.0"
    }

# ====== 5. 감정 예측 엔드포인트 ======
@app.post("/predict_emotion")
async def predict_emotion_endpoint(request: PredictionRequest):
    """
    걸음걸이 키포인트 데이터를 받아 감정을 예측합니다.

    Args:
        request: PredictionRequest 객체 (키포인트 데이터 포함)

    Returns:
        예측 결과 JSON
    Raises:
        HTTPException: 데이터 처리 또는 모델 에러 시
    """
    # --------- (1) 입력 데이터 검증 및 파싱 ----------
    try:
        kp_data = np.array(request.keypoints)          # (frames, 3)
        if kp_data.shape[1] != 3:
            raise ValueError("각 키포인트에 3개의 좌표[x, y, z]가 필요합니다.")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"입력 좌표 파싱 실패: {str(e)}")

    # --------- (2) 특징벡터 추출 (간단 예시, 실제는 src/feature_extractor.py 사용) ----------
    # 아래는 더미 예시: 실제 시스템에는 신체역학 14D 벡터 등 실사용 코드 필요
    def extract_hcf_features(kp_arr: np.ndarray) -> List[float]:
        # 실제에서는 보폭/각도 등 계산 (여기선 평균, 표준편차 등 예시)
        mean_vec = np.mean(kp_arr, axis=0).tolist()
        std_vec = np.std(kp_arr, axis=0).tolist()
        return mean_vec + std_vec  # 6개 예시 (실제는 14개)

    features_14d = extract_hcf_features(kp_data)   # 실제: 14개, 여기선 6개 예시

    # --------- (3) 모델 예측 ----------
    if fusion_model is None:
        raise HTTPException(status_code=503, detail="모델 파일 로드 실패. 서버 점검 필요.")
    try:
        # 입력: (batch, features) 로 reshape 필요
        features_input = np.array(features_14d).reshape(1, -1)
        pred_probs = fusion_model.predict_proba(features_input)[0]   # (6,)
        emotion_idx = int(np.argmax(pred_probs))
        emotion = emotion_labels[emotion_idx]
        confidence = float(np.max(pred_probs))

        # 신뢰도 구간 결정 (예: 0.8이상 high, 0.5이상 medium, 이하 low)
        if confidence > 0.8:
            confidence_level = "high"
        elif confidence > 0.5:
            confidence_level = "medium"
        else:
            confidence_level = "low"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류가 발생했습니다: {str(e)}")

    # --------- (4) 결과 반환 ----------
    return {
        "emotion": emotion,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "probabilities": {label: float(prob) for label, prob in zip(emotion_labels, pred_probs)},
        "features": features_14d,
        "features_shape": list(features_input.shape),
        "message": "감정이 성공적으로 예측되었습니다."
    }
