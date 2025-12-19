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
    # request: PredictionRequest (기존) 대신 raw dict를 받도록 변경하거나,
    # 현재 Pydantic 모델을 유지하려면 request.dict()로 변환해 사용
    body_dict = request.dict()  # 또는 request.__dict__ 형태

    # features 생성 (통합 함수 사용)
    try:
        features = extract_hcf_features_from_request(body_dict)  # 14D 벡터 기대
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"특징 추출 실패: {str(e)}")

    # 이후 기존 로직: 모델 존재 확인, expected 비교 등
    expected = getattr(fusion_model, "n_features_in_", None)
    print("DEBUG: generated features len =", len(features), "features =", features)
    print("DEBUG: model expected n_features_in_ =", expected)
    if expected is not None and len(features) != expected:
        raise HTTPException(status_code=422, detail=f"특징 차원 불일치: 생성된 features 길이={len(features)}인데 모델은 {expected}차원을 기대합니다.")

    # 특징벡터 추출 (여기서는 예시; 실제로는 원본 feature_extractor를 사용해야 함)
    def extract_hcf_features(kp_arr: np.ndarray) -> List[float]:
        # ---------- feature_extractor 통합 래퍼 ----------
        pass  # 실제 구현 필요

    # src/feature_extractor.py 의 함수들을 안전히 불러와 사용합니다.
    try:
        # 같은 디렉터리(src)에서 실행되는 경우
        from feature_extractor import extract_features_from_skeleton, extract_features
        print("[INIT] imported extract_features_from_skeleton, extract_features from feature_extractor")
    except Exception:
        try:
            # 패키지화된 경우
            from src.feature_extractor import extract_features_from_skeleton, extract_features
            print("[INIT] imported from src.feature_extractor")
        except Exception as e:
            extract_features_from_skeleton = None
            extract_features = None
            print("[WARN] feature_extractor import failed:", repr(e))

    def convert_keypoints_to_skeleton_data(keypoints: List[List[float]], n_joints: Optional[int] = None) -> List[str]:
        """
        keypoints가 다음 두 가지 중 하나일 때 처리:
        1) keypoints: list of dicts per frame (handled elsewhere)
        2) keypoints: flat list of [x,y,z] entries length = n_frames * n_joints
        이 함수는 (n_frames, n_joints, 3) 형태로 재구성한 뒤 "x,y,z" 문자열 리스트로 반환합니다.
        """
        arr = np.array(keypoints, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("keypoints는 각 항목이 [x,y,z] 형태의 2차원 배열이어야 합니다.")
        total_points = arr.shape[0]

        # n_joints가 주어지지 않으면 README/docs의 기본 관절 수(예: 13) 사용
        if n_joints is None:
            # feature_extractor.extract_features uses 13 joint_names in its extract_features wrapper
            n_joints = 13

        if total_points % n_joints != 0:
            raise ValueError(f"keypoints 길이({total_points})가 n_joints({n_joints})로 나누어떨어지지 않습니다. 프레임 재구성이 불가합니다.")

        n_frames = total_points // n_joints
        reshaped = arr.reshape(n_frames, n_joints, 3)  # (n_frames, n_joints, 3)

        # flatten to list of "x,y,z" strings in frame-major, joint-major order
        skeleton_data = []
        for f in range(n_frames):
            for j in range(n_joints):
                x, y, z = reshaped[f, j]
                skeleton_data.append(f"{float(x)},{float(y)},{float(z)}")
        return skeleton_data

    def extract_hcf_features_from_request(request_body: dict) -> List[float]:
        """
        request_body: dict that may contain:
          - 'skeleton_data': List[str]  (preferred)
          - 'keypoints': List[List[float]]  (flat list of [x,y,z] entries)
          - optionally 'n_joints': int
        Returns: 1D list of features (length 14 expected)
        """
        if extract_features_from_skeleton is None and extract_features is None:
            raise RuntimeError("feature_extractor 모듈을 불러오지 못했습니다.")

        # 우선 skeleton_data 우선 처리
        if "skeleton_data" in request_body and request_body["skeleton_data"]:
            skeleton_data = request_body["skeleton_data"]
            n_joints = request_body.get("n_joints", None)
            # call extract_features_from_skeleton
            feat = extract_features_from_skeleton(skeleton_data, n_joints=n_joints if n_joints is not None else 17)
            return feat.tolist() if hasattr(feat, "tolist") else list(feat)

        # 다음으로 keypoints(flat [x,y,z] entries) 처리
        if "keypoints" in request_body and request_body["keypoints"]:
            keypoints = request_body["keypoints"]
            n_joints = request_body.get("n_joints", None)
            # try to convert to skeleton_data
            skeleton_data = convert_keypoints_to_skeleton_data(keypoints, n_joints=n_joints)
            feat = extract_features_from_skeleton(skeleton_data, n_joints=(n_joints if n_joints is not None else 13))
            return feat.tolist() if hasattr(feat, "tolist") else list(feat)

        # 마지막으로 dict-per-frame 형식(keypoints as list of dicts) 처리 (extract_features wrapper)
        if "keypoints_dicts" in request_body and request_body["keypoints_dicts"]:
            keypoints_dicts = request_body["keypoints_dicts"]
            feat = extract_features(keypoints_dicts)
            return feat.tolist() if hasattr(feat, "tolist") else list(feat)

        raise ValueError("요청에 'skeleton_data' 또는 'keypoints' (또는 'keypoints_dicts')가 필요합니다.")

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
