from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import numpy as np
import os

# ===============================
# 1. 앱 생성 및 CORS 설정
# ===============================
APP_VERSION = "2.0.0"

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

# ===============================
# 2. 모델 및 특징 추출기 로드
# ===============================
MODEL_PATH = os.path.join("models", "deployment", "gait_emotion_api_model.joblib")
fusion_model = None
try:
    import joblib
    fusion_model = joblib.load(MODEL_PATH)
    expected_features = getattr(fusion_model, "n_features_in_", None)
    print(f"[MODEL] 로드 완료: {os.path.abspath(MODEL_PATH)}")
    print(f"[MODEL] 기대 특징 수: {expected_features}")
except Exception as e:
    print(f"[WARN] 모델 로드 실패: {repr(e)}")
    fusion_model = None

emotion_labels = ["happy", "sad", "fear", "disgust", "angry", "neutral"]

# feature_extractor import
extract_features_from_skeleton = None
extract_features = None
try:
    from feature_extractor import extract_features_from_skeleton, extract_features
    print("[INIT] feature_extractor 로드 완료 (root)")
except ImportError:
    try:
        from src.feature_extractor import extract_features_from_skeleton, extract_features
        print("[INIT] feature_extractor 로드 완료 (src 패키지)")
    except ImportError as e:
        print(f"[WARN] feature_extractor import 실패: {repr(e)}")

# ===============================
# 3. 유틸리티 함수
# ===============================
def convert_keypoints_to_skeleton_data(
    keypoints: List[List[float]], n_joints: Optional[int] = None
) -> List[str]:
    """
    [[x,y,z], ...] 형태의 키포인트를 ["x,y,z", ...] 문자열 리스트로 변환합니다.
    """
    arr = np.array(keypoints, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("keypoints는 [[x,y,z], ...] 2차원 배열이어야 합니다.")
    total_points = arr.shape[0]
    if n_joints is None:
        n_joints = 13
    if total_points % n_joints != 0:
        raise ValueError(
            f"keypoints 길이({total_points})가 n_joints({n_joints})로 나누어떨어지지 않습니다."
        )
    n_frames = total_points // n_joints
    reshaped = arr.reshape(n_frames, n_joints, 3)
    return [
        f"{float(x)},{float(y)},{float(z)}"
        for f in range(n_frames)
        for (x, y, z) in reshaped[f]
    ]


def extract_hcf_features_from_request(request_body: dict) -> List[float]:
    """
    요청 본문에서 skeleton_data, keypoints, keypoints_dicts 중 하나를 읽어
    14개 HCF 특징 벡터를 추출합니다.
    """
    if extract_features_from_skeleton is None and extract_features is None:
        raise RuntimeError("feature_extractor 모듈을 불러오지 못했습니다.")

    # skeleton_data 우선 처리
    if "skeleton_data" in request_body and request_body["skeleton_data"]:
        skeleton_data = request_body["skeleton_data"]
        n_joints = request_body.get("n_joints", 17)
        feat = extract_features_from_skeleton(skeleton_data, n_joints=n_joints)
        return feat.tolist() if hasattr(feat, "tolist") else list(feat)

    # [[x, y, z], ...] 구조
    if "keypoints" in request_body and request_body["keypoints"]:
        keypoints = request_body["keypoints"]
        n_joints = request_body.get("n_joints", 13)
        skeleton_data = convert_keypoints_to_skeleton_data(keypoints, n_joints=n_joints)
        feat = extract_features_from_skeleton(skeleton_data, n_joints=n_joints)
        return feat.tolist() if hasattr(feat, "tolist") else list(feat)

    # 프레임별 딕셔너리 입력
    if "keypoints_dicts" in request_body and request_body["keypoints_dicts"]:
        keypoints_dicts = request_body["keypoints_dicts"]
        feat = extract_features(keypoints_dicts)
        return feat.tolist() if hasattr(feat, "tolist") else list(feat)

    raise ValueError("'skeleton_data' 또는 'keypoints' 필드가 필요합니다.")


# ===============================
# 4. 루트 엔드포인트
# ===============================
@app.get("/", tags=["info"])
async def root():
    """API 기본 정보를 반환합니다."""
    return {
        "message": "걸음걸이 감정 인식 API에 오신 것을 환영합니다!",
        "version": APP_VERSION,
        "docs": "/docs",
    }


# ===============================
# 5. 헬스 체크
# ===============================
@app.get("/health", tags=["health"])
async def health():
    """서비스 상태 확인"""
    return {
        "status": "healthy",
        "service": "gait-emotion-recognition",
        "version": APP_VERSION,
    }


# ===============================
# 6. 감정 예측 엔드포인트
# ===============================
@app.post("/predict_emotion", tags=["prediction"])
async def predict_emotion_endpoint(request: dict):
    """
    걸음걸이 데이터로부터 감정을 예측합니다.

    입력: skeleton_data, keypoints, 또는 keypoints_dicts 중 하나
    출력: 예측 감정, 신뢰도, 확률분포, 특징 벡터
    """
    body_dict = dict(request)

    # 특징 추출
    try:
        features = extract_hcf_features_from_request(body_dict)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"특징 추출 실패: {e}")

    # 모델 존재 확인
    if fusion_model is None:
        raise HTTPException(status_code=503, detail="모델 파일 로드 실패. 서버 점검 필요.")

    # 특징 차원 보정 (모델이 기대하는 수보다 부족하면 0으로 패딩)
    expected = getattr(fusion_model, "n_features_in_", None)
    if expected is not None and len(features) < expected:
        features.extend([0.0] * (expected - len(features)))
    if expected is not None and len(features) != expected:
        raise HTTPException(
            status_code=422,
            detail=f"특징 차원 불일치: 생성={len(features)}, 기대={expected}",
        )

    # 모델 예측
    try:
        X = np.array(features).reshape(1, -1)
        pred_probs = fusion_model.predict_proba(X)[0]
        emotion_idx = int(np.argmax(pred_probs))
        emotion = emotion_labels[emotion_idx]
        confidence = float(np.max(pred_probs))
        confidence_level = (
            "high" if confidence > 0.8
            else "medium" if confidence > 0.5
            else "low"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {str(e)}")

    return {
        "emotion": emotion,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "probabilities": {
            label: float(prob) for label, prob in zip(emotion_labels, pred_probs)
        },
        "features": features,
        "features_shape": list(X.shape),
        "message": "감정이 성공적으로 예측되었습니다.",
    }
