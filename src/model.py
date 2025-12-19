from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import numpy as np
import joblib
import os

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

# ====== 2. FastAPI 앱 객체 생성 및 CORS 설정 ======
app = FastAPI(
    title="걸음걸이 감정 인식 API",
    description="Bi-LSTM HCF Fusion 기반의 실시간 감정 예측 시스템",
    version="2.0.0"
)

# 반드시 app 생성 이후에 CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 개발 중에는 "*" 사용. 배포 시 도메인 명시 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 임시 preflight 핸들러 (CORS 문제 디버그용, CORS 미들웨어가 정상 동작하면 제거 가능)
@app.options("/predict_emotion")
async def preflight():
    return Response(status_code=200, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization"
    })

# ====== 3. 모델 로드 및 기대 차원 로그 ======
MODEL_PATH = os.path.join("models", "deployment", "gait_emotion_api_model.joblib")
fusion_model = None
try:
    fusion_model = joblib.load(MODEL_PATH)
    expected_features = getattr(fusion_model, "n_features_in_", None)
    print("[MODEL] loaded from:", os.path.abspath(MODEL_PATH))
    print("[MODEL] expected n_features_in_ =", expected_features)
except Exception as e:
    fusion_model = None
    print(f"[!] 모델 로드 실패: {repr(e)}")

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
    # 입력 검증 및 파싱
    try:
        kp_data = np.array(request.keypoints, dtype=float)  # shape: (frames, 3)
        if kp_data.ndim != 2 or kp_data.shape[1] != 3:
            raise ValueError("각 키포인트에 3개의 좌표[x, y, z]가 필요합니다.")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"입력 좌표 파싱 실패: {str(e)}")

    # 특징벡터 추출 (여기서는 예시; 실제로는 원본 feature_extractor를 사용해야 함)
    def extract_hcf_features(kp_arr: np.ndarray) -> List[float]:
        mean_vec = np.mean(kp_arr, axis=0).tolist()
        std_vec = np.std(kp_arr, axis=0).tolist()
        return mean_vec + std_vec  # 현재 6개 예시

    features = extract_hcf_features(kp_data)  # 현재 예시로 6개 생성

    # 임시 패딩: 모델이 기대하는 차원까지 0으로 채움 (디버그/확인용)
    expected = getattr(fusion_model, "n_features_in_", None)
    if expected is not None and len(features) < expected:
        print(f"[TEMP PAD] features len {len(features)} -> pad to {expected}")
        features = features + [0.0] * (expected - len(features))

    # 모델 존재 확인
    if fusion_model is None:
        raise HTTPException(status_code=503, detail="모델 파일 로드 실패. 서버 점검 필요.")

    # 디버그 로그: 생성된 features와 모델 기대치 출력
    expected = getattr(fusion_model, "n_features_in_", None)
    print("DEBUG: generated features len =", len(features), "features =", features)
    print("DEBUG: model expected n_features_in_ =", expected)

    # 특징 차원 불일치 처리
    if expected is not None and len(features) != expected:
        # 권장: 명확한 에러 반환
        raise HTTPException(
            status_code=422,
            detail=f"특징 차원 불일치: 생성된 features 길이={len(features)}인데 모델은 {expected}차원을 기대합니다."
        )
        # 임시 패딩으로 동작 확인을 원하면 아래 주석을 해제하세요 (주의: 결과 신뢰 불가)
        # if len(features) < expected:
        #     features = features + [0.0] * (expected - len(features))
        # elif len(features) > expected:
        #     features = features[:expected]

    # 모델 예측
    try:
        X = np.array(features).reshape(1, -1)
        pred_probs = fusion_model.predict_proba(X)[0]
        emotion_idx = int(np.argmax(pred_probs))
        emotion = emotion_labels[emotion_idx]
        confidence = float(np.max(pred_probs))
        if confidence > 0.8:
            confidence_level = "high"
        elif confidence > 0.5:
            confidence_level = "medium"
        else:
            confidence_level = "low"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류가 발생했습니다: {str(e)}")

    return {
        "emotion": emotion,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "probabilities": {label: float(prob) for label, prob in zip(emotion_labels, pred_probs)},
        "features": features,
        "features_shape": list(X.shape),
        "message": "감정이 성공적으로 예측되었습니다."
    }
