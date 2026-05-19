from typing import List

import joblib
import numpy as np
import os

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware


APP_VERSION = "2.0.0"
MODEL_PATH = os.path.join("models", "deployment", "gait_emotion_api_model.joblib")
DEFAULT_KEYPOINT_JOINTS = 13
DEFAULT_SKELETON_JOINTS = 17
EMOTION_LABELS = ["happy", "sad", "fear", "disgust", "angry", "neutral"]


app = FastAPI(
    title="걸음걸이 감정 인식 API",
    description="Bi-LSTM HCF Fusion 기반의 실시간 감정 예측 시스템",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.options("/predict_emotion")
async def preflight():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        },
    )


def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        print("[MODEL] loaded from:", os.path.abspath(MODEL_PATH))
        print("[MODEL] expected n_features_in_ =", getattr(model, "n_features_in_", None))
        return model
    except Exception as e:
        print(f"[!] 모델 로드 실패: {repr(e)}")
        return None


fusion_model = load_model()


try:
    from src.feature_extractor import extract_features_from_skeleton
    print("[INIT] imported src.feature_extractor")
except Exception as e:
    extract_features_from_skeleton = None
    print("[WARN] feature_extractor import failed:", repr(e))


@app.get("/health", tags=["health"])
async def health():
    return {
        "status": "healthy",
        "service": "gait-emotion-recognition",
        "version": APP_VERSION,
    }


def convert_keypoints_to_skeleton_data(
    keypoints: List[List[float]],
    n_joints: int,
) -> List[str]:
    try:
        arr = np.array(keypoints, dtype=float)
    except Exception:
        raise ValueError("keypoints는 숫자로 구성된 2차원 배열이어야 합니다.")

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("keypoints는 [[x, y, z], ...] 형식의 2차원 배열이어야 합니다.")

    total_points = arr.shape[0]
    if total_points == 0:
        raise ValueError("keypoints가 비어 있습니다.")

    if total_points % n_joints != 0:
        raise ValueError(
            f"keypoints 길이({total_points})가 n_joints({n_joints})로 나누어떨어지지 않습니다."
        )

    n_frames = total_points // n_joints
    reshaped = arr.reshape(n_frames, n_joints, 3)

    return [
        f"{float(x)},{float(y)},{float(z)}"
        for frame in reshaped
        for (x, y, z) in frame
    ]


def extract_features_from_request(request_body: dict) -> List[float]:
    if extract_features_from_skeleton is None:
        raise RuntimeError("feature_extractor 모듈을 불러오지 못했습니다.")

    if request_body.get("skeleton_data"):
        skeleton_data = request_body["skeleton_data"]
        n_joints = int(request_body.get("n_joints", DEFAULT_SKELETON_JOINTS))
        feat = extract_features_from_skeleton(skeleton_data, n_joints=n_joints)
        return feat.tolist() if hasattr(feat, "tolist") else list(feat)

    if request_body.get("keypoints"):
        keypoints = request_body["keypoints"]
        n_joints = int(request_body.get("n_joints", DEFAULT_KEYPOINT_JOINTS))
        skeleton_data = convert_keypoints_to_skeleton_data(keypoints, n_joints=n_joints)
        feat = extract_features_from_skeleton(skeleton_data, n_joints=n_joints)
        return feat.tolist() if hasattr(feat, "tolist") else list(feat)

    raise ValueError("'skeleton_data' 또는 'keypoints' 필드가 필요합니다.")


def pad_features_if_needed(features: List[float], model) -> List[float]:
    expected = getattr(model, "n_features_in_", None)
    if expected is None:
        return features

    if len(features) > expected:
        raise ValueError(f"특징 차원이 모델 기대값보다 큽니다: 생성={len(features)}, 기대={expected}")

    if len(features) < expected:
        features = features + [0.0] * (expected - len(features))

    return features


@app.post("/predict_emotion", tags=["main"])
async def predict_emotion_endpoint(request: dict):
    try:
        features = extract_features_from_request(request)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"특징 추출 실패: {e}")

    if fusion_model is None:
        raise HTTPException(status_code=503, detail="모델 파일 로드 실패. 서버 점검 필요.")

    try:
        features = pad_features_if_needed(features, fusion_model)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        X = np.array(features, dtype=float).reshape(1, -1)
        pred_probs = fusion_model.predict_proba(X)[0]
        emotion_idx = int(np.argmax(pred_probs))
        emotion = EMOTION_LABELS[emotion_idx]
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
        "probabilities": {
            label: float(prob) for label, prob in zip(EMOTION_LABELS, pred_probs)
        },
        "features": features,
        "features_shape": list(X.shape),
        "message": "감정이 성공적으로 예측되었습니다.",
    }
