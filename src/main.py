from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional
import uuid

import joblib
import numpy as np

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.prediction_logging import prediction_log_store


APP_VERSION = "2.0.0"
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "deployment" / "gait_emotion_api_model.joblib"
FRONTEND_DIR = BASE_DIR / "frontend"
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
        print("[MODEL] loaded from:", MODEL_PATH.resolve())
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
        "prediction_logging": prediction_log_store.health_payload(),
    }


def parse_optional_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def detect_input_type(request_body: Dict[str, Any]) -> str:
    if request_body.get("skeleton_data") is not None:
        return "skeleton_data"
    if request_body.get("keypoints") is not None:
        return "keypoints"
    return "unknown"


def resolve_joint_count(request_body: Dict[str, Any], input_type: str) -> Optional[int]:
    if "n_joints" in request_body:
        return parse_optional_int(request_body.get("n_joints"))

    if input_type == "skeleton_data":
        return DEFAULT_SKELETON_JOINTS

    if input_type == "keypoints":
        return DEFAULT_KEYPOINT_JOINTS

    return None


def estimate_frame_count(request_body: Dict[str, Any], joint_count: Optional[int]) -> Optional[int]:
    if joint_count is None or joint_count <= 0:
        return None

    sequence = None
    if request_body.get("skeleton_data") is not None:
        sequence = request_body.get("skeleton_data")
    elif request_body.get("keypoints") is not None:
        sequence = request_body.get("keypoints")

    if not isinstance(sequence, list) or not sequence:
        return None

    total_points = len(sequence)
    if total_points % joint_count != 0:
        return None

    return total_points // joint_count


def build_request_preview(request_body: Dict[str, Any], sample_limit: int = 5) -> Dict[str, Any]:
    preview: Dict[str, Any] = {
        "keys": sorted(request_body.keys()),
    }

    extra_fields = {}
    for key, value in request_body.items():
        if key in {"keypoints", "skeleton_data"}:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            extra_fields[key] = value

    if extra_fields:
        preview["extra_fields"] = extra_fields

    keypoints = request_body.get("keypoints")
    if keypoints is not None:
        if isinstance(keypoints, list):
            preview["keypoints_count"] = len(keypoints)
            preview["keypoints_sample"] = keypoints[:sample_limit]
        else:
            preview["keypoints_type"] = type(keypoints).__name__

    skeleton_data = request_body.get("skeleton_data")
    if skeleton_data is not None:
        if isinstance(skeleton_data, list):
            preview["skeleton_data_count"] = len(skeleton_data)
            preview["skeleton_data_sample"] = skeleton_data[:sample_limit]
        else:
            preview["skeleton_data_type"] = type(skeleton_data).__name__

    return preview


def resolve_client_host(http_request: Request) -> Optional[str]:
    forwarded_for = http_request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()

    if http_request.client is not None:
        return http_request.client.host

    return None


def resolve_request_id(http_request: Request) -> str:
    request_id = http_request.headers.get("x-request-id", "").strip()
    return request_id or str(uuid.uuid4())


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
async def predict_emotion_endpoint(payload: dict, http_request: Request):
    started_at = perf_counter()
    request_id = resolve_request_id(http_request)
    input_type = detect_input_type(payload)
    joint_count = resolve_joint_count(payload, input_type)
    frame_count = estimate_frame_count(payload, joint_count)
    request_preview = build_request_preview(payload)
    client_host = resolve_client_host(http_request)

    def write_prediction_log(
        *,
        status_code: int,
        feature_count: Optional[int] = None,
        predicted_emotion: Optional[str] = None,
        confidence: Optional[float] = None,
        confidence_level: Optional[str] = None,
        probabilities: Optional[Dict[str, float]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        prediction_log_store.save_prediction(
            request_id=request_id,
            route=http_request.url.path,
            client_host=client_host,
            input_type=input_type,
            joint_count=joint_count,
            frame_count=frame_count,
            feature_count=feature_count,
            status_code=status_code,
            predicted_emotion=predicted_emotion,
            confidence=confidence,
            confidence_level=confidence_level,
            latency_ms=round((perf_counter() - started_at) * 1000, 3),
            request_preview=request_preview,
            probabilities=probabilities,
            error_message=error_message,
        )

    try:
        features = extract_features_from_request(payload)
    except Exception as e:
        detail = f"특징 추출 실패: {e}"
        write_prediction_log(status_code=422, error_message=detail)
        raise HTTPException(status_code=422, detail=detail)

    if fusion_model is None:
        detail = "모델 파일 로드 실패. 서버 점검 필요."
        write_prediction_log(
            status_code=503,
            feature_count=len(features),
            error_message=detail,
        )
        raise HTTPException(status_code=503, detail=detail)

    try:
        features = pad_features_if_needed(features, fusion_model)
    except Exception as e:
        detail = str(e)
        write_prediction_log(
            status_code=422,
            feature_count=len(features),
            error_message=detail,
        )
        raise HTTPException(status_code=422, detail=detail)

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
        detail = f"서버 내부 오류: {str(e)}"
        write_prediction_log(
            status_code=500,
            feature_count=len(features),
            error_message=detail,
        )
        raise HTTPException(status_code=500, detail=detail)

    response_payload = {
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

    write_prediction_log(
        status_code=200,
        feature_count=len(features),
        predicted_emotion=emotion,
        confidence=confidence,
        confidence_level=confidence_level,
        probabilities=response_payload["probabilities"],
    )

    return response_payload


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
