"""
걸음걸이 감정 인식 API 서버 (Gait Emotion Recognition API)

이 파일은 FastAPI를 사용하여 걸음걸이 데이터로부터 감정을 예측하는 REST API를 제공합니다.
비유: 이 서버는 마치 "걸음걸이 분석 전문가"처럼 동작합니다.
      사용자가 걸음걸이 데이터를 보내면, 전문가가 분석하여 감정 상태를 알려줍니다.
"""

import logging
import os
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 프로젝트 내부 모듈
from src.feature_extractor import extract_features
from src.model import predict_emotion

# 환경 변수 로드
load_dotenv()

# 로깅 설정 (print 대신 logging 사용)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# 데이터 모델 정의 (Pydantic Models)
# ============================================================================

class KeypointData(BaseModel):
    """
    단일 프레임의 키포인트 데이터
    
    각 신체 부위의 (x, y) 좌표를 포함합니다.
    예) {"nose": [100, 200], "left_shoulder": [90, 220], ...}
    """
    nose: List[float] = Field(default=None, description="코 좌표 [x, y]")
    left_shoulder: List[float] = Field(default=None, description="왼쪽 어깨 좌표 [x, y]")
    right_shoulder: List[float] = Field(default=None, description="오른쪽 어깨 좌표 [x, y]")
    left_elbow: List[float] = Field(default=None, description="왼쪽 팔꿈치 좌표 [x, y]")
    right_elbow: List[float] = Field(default=None, description="오른쪽 팔꿈치 좌표 [x, y]")
    left_wrist: List[float] = Field(default=None, description="왼쪽 손목 좌표 [x, y]")
    right_wrist: List[float] = Field(default=None, description="오른쪽 손목 좌표 [x, y]")
    left_hip: List[float] = Field(default=None, description="왼쪽 엉덩이 좌표 [x, y]")
    right_hip: List[float] = Field(default=None, description="오른쪽 엉덩이 좌표 [x, y]")
    left_knee: List[float] = Field(default=None, description="왼쪽 무릎 좌표 [x, y]")
    right_knee: List[float] = Field(default=None, description="오른쪽 무릎 좌표 [x, y]")
    left_ankle: List[float] = Field(default=None, description="왼쪽 발목 좌표 [x, y]")
    right_ankle: List[float] = Field(default=None, description="오른쪽 발목 좌표 [x, y]")
    
    model_config = {
        # None 값을 가진 필드는 자동으로 제외
        "exclude_none": True
    }


class PredictionRequest(BaseModel):
    """
    감정 예측 요청 데이터
    
    여러 프레임의 키포인트 데이터를 포함합니다.
    최소 2개 이상의 프레임이 필요합니다.
    """
    keypoints: List[Dict] = Field(
        ..., 
        description="키포인트 데이터 리스트 (각 프레임별)",
        min_length=2
    )


class EmotionThresholds:
    """
    감정 예측 신뢰도 임계값 관리 클래스
    
    하드코딩된 값들을 한 곳에서 관리하여 유지보수를 쉽게 합니다.
    """
    HIGH_CONFIDENCE = 0.5      # 높은 신뢰도 기준 (50% 이상)
    LOW_PROBABILITY = 0.05     # 낮은 확률 기준 (5% 이하)
    MIN_DIFF_THRESHOLD = 0.2   # 최소 확률 차이 기준 (20% 이상)


# ============================================================================
# FastAPI 앱 생성 및 설정
# ============================================================================

app = FastAPI(
    title="걸음걸이 감정 인식 API",
    description="걸음걸이 데이터로부터 감정을 예측하는 REST API",
    version="1.0.0",
    docs_url="/docs",      # Swagger UI 문서 경로
    redoc_url="/redoc"     # ReDoc 문서 경로
)

# CORS 설정 (Cross-Origin Resource Sharing)
# 프론트엔드에서 API를 호출할 수 있도록 허용
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5500,http://127.0.0.1:5500").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # 허용할 도메인 목록
    allow_credentials=True,
    allow_methods=["*"],            # 모든 HTTP 메서드 허용 (GET, POST, etc.)
    allow_headers=["*"],            # 모든 헤더 허용
)


# ============================================================================
# API 엔드포인트 정의
# ============================================================================

@app.get("/")
async def root():
    """
    루트 엔드포인트
    
    API가 정상적으로 작동하는지 확인하는 간단한 엔드포인트입니다.
    """
    return {
        "message": "걸음걸이 감정 인식 API에 오신 것을 환영합니다!",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """
    헬스 체크 엔드포인트
    
    서버가 정상적으로 작동 중인지 확인합니다.
    Docker나 쿠버네티스에서 컨테이너 상태를 모니터링할 때 사용됩니다.
    """
    return {
        "status": "healthy",
        "service": "gait-emotion-recognition",
        "version": "1.0.0"
    }


@app.post("/predict_emotion")
async def predict_emotion_endpoint(request: PredictionRequest):
    """
    감정 예측 엔드포인트
    
    걸음걸이 키포인트 데이터를 받아서 감정을 예측합니다.
    
    Args:
        request: PredictionRequest 객체 (키포인트 데이터 포함)
    
    Returns:
        예측 결과 JSON
        {
            "emotion": "happy",           # 예측된 감정
            "confidence": 0.85,           # 신뢰도 (0~1)
            "confidence_level": "high",   # 신뢰도 수준 (high/medium/low)
            "probabilities": {            # 각 감정별 확률
                "happy": 0.85,
                "sad": 0.10,
                "angry": 0.05
            },
            "features": [...],            # 추출된 14개 특징값
            "message": "감정이 성공적으로 예측되었습니다."
        }
    
    Raises:
        HTTPException: 데이터 처리 중 오류 발생 시
    """
    try:
        logger.info(f"감정 예측 요청 수신: {len(request.keypoints)}개 프레임")
        
        # 1단계: 키포인트 데이터 검증
        if not request.keypoints or len(request.keypoints) < 2:
            logger.warning("키포인트 데이터 부족")
            raise HTTPException(
                status_code=400,
                detail="최소 2개 이상의 프레임이 필요합니다."
            )
        
        # 2단계: 특징 추출
        # feature_extractor.py의 extract_features 함수 사용
        logger.info("특징 추출 시작...")
        features = extract_features(request.keypoints)
        logger.info(f"특징 추출 완료: {features.shape}")
        
        # 3단계: 감정 예측
        # model.py의 predict_emotion 함수 사용
        logger.info("감정 예측 시작...")
        model_path = os.getenv("MODEL_PATH", "models/rf_emotion_model.joblib")
        prediction = predict_emotion(features, model_path)
        logger.info(f"감정 예측 완료: {prediction['emotion']} (신뢰도: {prediction['confidence']:.2f})")
        
        # 4단계: 신뢰도 수준 판단
        confidence = prediction["confidence"]
        if confidence >= EmotionThresholds.HIGH_CONFIDENCE:
            confidence_level = "high"       # 높은 신뢰도 (50% 이상)
        elif confidence >= 0.3:
            confidence_level = "medium"     # 중간 신뢰도 (30~50%)
        else:
            confidence_level = "low"        # 낮은 신뢰도 (30% 미만)
        
        # 5단계: 확률이 너무 낮은 감정 필터링
        # 5% 미만의 확률을 가진 감정은 표시하지 않음
        filtered_probabilities = {
            emotion: prob 
            for emotion, prob in prediction["probabilities"].items()
            if prob >= EmotionThresholds.LOW_PROBABILITY
        }
        
        # 6단계: 응답 생성
        response = {
            "emotion": prediction["emotion"],
            "confidence": round(prediction["confidence"], 3),
            "confidence_level": confidence_level,
            "probabilities": {
                emotion: round(prob, 3) 
                for emotion, prob in filtered_probabilities.items()
            },
            "features": features.tolist(),  # numpy 배열을 list로 변환 (JSON 직렬화)
            "features_shape": list(features.shape),  # shape도 list로 변환
            "message": "감정이 성공적으로 예측되었습니다."
        }
        
        # 7단계: 추가 메시지 (신뢰도가 낮을 경우 경고)
        if confidence_level == "low":
            response["warning"] = "예측 신뢰도가 낮습니다. 더 많은 프레임 데이터를 제공하면 정확도가 향상됩니다."
        
        # 8단계: 감정 간 확률 차이가 작을 경우 경고
        probs = list(filtered_probabilities.values())
        if len(probs) >= 2:
            sorted_probs = sorted(probs, reverse=True)
            if sorted_probs[0] - sorted_probs[1] < EmotionThresholds.MIN_DIFF_THRESHOLD:
                response["warning"] = "여러 감정이 비슷한 확률로 예측되었습니다. 결과가 불확실할 수 있습니다."
        
        logger.info("응답 전송 완료")
        return JSONResponse(content=response)
    
    except HTTPException:
        # HTTPException은 그대로 전달
        raise
    
    except Exception as e:
        # 예상치 못한 오류 처리
        logger.error(f"감정 예측 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"서버 내부 오류가 발생했습니다: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    전역 예외 처리기
    
    처리되지 않은 모든 예외를 캐치하여 로그에 기록하고,
    사용자에게 친절한 오류 메시지를 반환합니다.
    """
    logger.error(f"처리되지 않은 예외 발생: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "서버에서 예상치 못한 오류가 발생했습니다.",
            "error": str(exc)
        }
    )


# ============================================================================
# 서버 시작 (개발 환경에서만 사용)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # 환경 변수에서 호스트와 포트 읽기
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"서버 시작: {host}:{port}")
    
    # Uvicorn 서버 실행
    # reload=True: 코드 변경 시 자동 재시작 (개발 환경에서만 사용)
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=True,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
