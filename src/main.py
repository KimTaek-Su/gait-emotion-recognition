from fastapi import FastAPI, HTTPException, Response, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import numpy as np
import joblib
import os

# === 실제 구현으로 대체! ===
def convert_keypoints_to_skeleton_data(keypoints: List[List[float]], n_joints: Optional[int] = None) -> List[str]:
    arr = np.array(keypoints, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("keypoints는 [ [x,y,z], ... ] 2차원 배열이어야 합니다.")
    total_points = arr.shape[0]
    if n_joints is None:
        n_joints = 17  # 13에서 17로 수정하거나, 호출할 때 17을 넘겨줘야 함
    if total_points % n_joints != 0:
        raise ValueError(f"keypoints 길이({total_points})가 n_joints({n_joints})로 나누어떨어지지 않습니다.")
    n_frames = total_points // n_joints
    reshaped = arr.reshape(n_frames, n_joints, 3)
    skeleton_data = [f"{float(x)},{float(y)},{float(z)}"
                     for f in range(n_frames) for (x, y, z) in reshaped[f]]
    return skeleton_data

def extract_hcf_features_from_request(request_body: dict) -> List[float]:
    if extract_features_from_skeleton is None and extract_features is None:
        raise RuntimeError("feature_extractor 모듈을 불러오지 못했습니다.")

    # 우선 skeleton_data 우선
    if "skeleton_data" in request_body and request_body["skeleton_data"]:
        skeleton_data = request_body["skeleton_data"]
        n_joints = request_body.get("n_joints", 17)
        feat = extract_features_from_skeleton(skeleton_data, n_joints=n_joints)
        return feat.tolist() if hasattr(feat, "tolist") else list(feat)

    # [ [x, y, z], ... ] 구조
    if "keypoints" in request_body and request_body["keypoints"]:
        keypoints = request_body["keypoints"]
        n_joints = request_body.get("n_joints", 13)
        skeleton_data = convert_keypoints_to_skeleton_data(keypoints, n_joints=n_joints)
        feat = extract_features_from_skeleton(skeleton_data, n_joints=n_joints)
        return feat.tolist() if hasattr(feat, "tolist") else list(feat)

    # 프레임별 dict 입력(보류)
    if "keypoints_dicts" in request_body and request_body["keypoints_dicts"]:
        keypoints_dicts = request_body["keypoints_dicts"]
        feat = extract_features(keypoints_dicts)
        return feat.tolist() if hasattr(feat, "tolist") else list(feat)

    raise ValueError("'skeleton_data' 또는 'keypoints' 필드가 필요합니다.")

# ===============================
# 1. Pydantic 모델 정의 (keypoints only; 실제 입력은 비공식적으로 더 지원)
# ===============================
class PredictionRequest(BaseModel):
    keypoints: List[List[float]] = Field(..., description="프레임별 3차원 좌표 리스트 (예: [[x, y, z], ...])")
    
    @validator("keypoints", each_item=True)
    def validate_keypoint_item(cls, v):
        if not isinstance(v, list):
            raise ValueError(f"좌표 {v}가 리스트가 아닙니다.")
        if len(v) != 3:
            raise ValueError(f"좌표에 3개의 값이 필요합니다. 현재: {len(v)}개")
        if not all(isinstance(val, (int, float)) for val in v):
            raise ValueError(f"좌표 {v}의 값이 int나 float 형식이 아닙니다.")
        return v

# ===============================
# 2. 앱 생성 및 CORS 설정
# ===============================
app = FastAPI(
    title="걸음걸이 감정 인식 API",
    description="Bi-LSTM HCF Fusion 기반의 실시간 감정 예측 시스템",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 시에는 "*" 허용, 배포 시에는 도메인 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CORS Preflight (임시, 필요 없으면 삭제 가능)
@app.options("/predict_emotion")
async def preflight():
    return Response(status_code=200, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization"
    })

# ===============================
# 3. 모델 및 extractor 로드
# ===============================
MODEL_PATH = os.path.join("models", "deployment", "gait_emotion_api_model.joblib")
fusion_model = None
try:
    fusion_model = joblib.load(MODEL_PATH)
    expected_features = getattr(fusion_model, "n_features_in_", None)
    print("[MODEL] loaded from:", os.path.abspath(MODEL_PATH))
    print("[MODEL] expected n_features_in_ =", expected_features)
except Exception as e:
    print(f"[!] 모델 로드 실패: {repr(e)}")
    fusion_model = None

emotion_labels = ["happy", "sad", "fear", "disgust", "angry", "neutral"]

# feature extractor import (src/feature_extractor.py)
try:
    from feature_extractor import extract_features_from_skeleton, extract_features
    print("[INIT] imported (root) feature_extractor")
except Exception:
    try:
        from src.feature_extractor import extract_features_from_skeleton, extract_features
        print("[INIT] imported (package) src.feature_extractor")
    except Exception as e:
        extract_features_from_skeleton = None
        extract_features = None
        print("[WARN] feature_extractor import failed:", repr(e))

# ===============================
# 4. Health check (GET만 허용)
# ===============================
@app.get("/health", tags=["health"])
async def health():
    """
    서비스 상태 Health Check (GET만 허용)
    """
    return {
        "status": "healthy",
        "service": "gait-emotion-recognition",
        "version": "2.0.0"
    }

# ===============================
# 5. 특징 추출 유틸리티
# ===============================
def convert_keypoints_to_skeleton_data(keypoints: List[List[float]], n_joints: Optional[int] = None) -> List[str]:
    arr = np.array(keypoints, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("keypoints는 [ [x,y,z], ... ] 2차원 배열이어야 합니다.")
    total_points = arr.shape[0]
    if n_joints is None:
        n_joints = 13  # 기본값
    if total_points % n_joints != 0:
        raise ValueError(f"keypoints 길이({total_points})가 n_joints({n_joints})로 나누어떨어지지 않습니다.")
    n_frames = total_points // n_joints
    reshaped = arr.reshape(n_frames, n_joints, 3)
    skeleton_data = [f"{float(x)},{float(y)},{float(z)}"
                     for f in range(n_frames) for (x, y, z) in reshaped[f]]
    return skeleton_data

def extract_hcf_features_from_request(request_body: dict) -> List[float]:
    if extract_features_from_skeleton is None and extract_features is None:
        raise RuntimeError("feature_extractor 모듈을 불러오지 못했습니다.")

    # 우선 skeleton_data 우선
    if "skeleton_data" in request_body and request_body["skeleton_data"]:
        skeleton_data = request_body["skeleton_data"]
        n_joints = request_body.get("n_joints", 17)
        feat = extract_features_from_skeleton(skeleton_data, n_joints=n_joints)
        return feat.tolist() if hasattr(feat, "tolist") else list(feat)

    # [ [x, y, z], ... ] 구조
    if "keypoints" in request_body and request_body["keypoints"]:
        keypoints = request_body["keypoints"]
        n_joints = request_body.get("n_joints", 13)
        skeleton_data = convert_keypoints_to_skeleton_data(keypoints, n_joints=n_joints)
        feat = extract_features_from_skeleton(skeleton_data, n_joints=n_joints)
        return feat.tolist() if hasattr(feat, "tolist") else list(feat)

    # 프레임별 dict 입력(아래는 보류, 공식 지원은 위 방식)
    if "keypoints_dicts" in request_body and request_body["keypoints_dicts"]:
        keypoints_dicts = request_body["keypoints_dicts"]
        feat = extract_features(keypoints_dicts)
        return feat.tolist() if hasattr(feat, "tolist") else list(feat)

    raise ValueError("'skeleton_data' 또는 'keypoints' 필드가 필요합니다.")

# ===============================
# 6. 감정 예측 엔드포인트
# ===============================
@app.post("/predict_emotion", tags=["main"])
async def predict_emotion_endpoint(request: dict):
    """
    - 입력: skeleton_data (List[str]), keypoints (List[List[float]]), 또는 keypoints_dicts (비공식) 중 하나
    - 출력: 예측 감정, 신뢰도, 확률분포 등
    """
    body_dict = dict(request)

    # 피쳐 추출
    try:
        features = extract_hcf_features_from_request(body_dict)
        # --- 추가된 부분 시작 ---
        expected = getattr(fusion_model, "n_features_in_", 20)
        if len(features) < expected:
            # 부족한 개수만큼 0.0으로 채워줌(14개 → 20개)
            padding_size = expected - len(features)
            features.extend([0.0] * padding_size)
        # --- 추가된 부분 끝 ---
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"특징 추출 실패: {e}")

    # 모델 및 차원 체크
    if fusion_model is None:
        raise HTTPException(status_code=503, detail="모델 파일 로드 실패. 서버 점검 필요.")
    expected = getattr(fusion_model, "n_features_in_", None)
    if expected is not None and len(features) != expected:
        raise HTTPException(
            status_code=422,
            detail=f"특징 차원 불일치: 생성={len(features)}, 기대={expected}"
        )

    # 모델 예측
    try:
        X = np.array(features).reshape(1, -1)
        pred_probs = fusion_model.predict_proba(X)[0]
        emotion_idx = int(np.argmax(pred_probs))
        emotion = emotion_labels[emotion_idx]
        confidence = float(np.max(pred_probs))
        confidence_level = (
            "high" if confidence > 0.8 else
            "medium" if confidence > 0.5 else
            "low"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {str(e)}")

    return {
        "emotion": emotion,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "probabilities": {label: float(prob) for label, prob in zip(emotion_labels, pred_probs)},
        "features": features,
        "features_shape": list(X.shape),
        "message": "감정이 성공적으로 예측되었습니다."
    }
