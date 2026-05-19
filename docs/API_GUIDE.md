# API 가이드

걸음걸이 감정 인식 API의 실행/요청 형식을 현재 구현(`src/main.py`) 기준으로 정리한 문서입니다.

## 📡 기본 정보

- **Base URL**: `http://localhost:8000`
- **API 버전**: `2.0.0`
- **Content-Type**: `application/json`

## 🔗 엔드포인트

### 1) 헬스 체크

`GET /health`

응답 예시:

```json
{
  "status": "healthy",
  "service": "gait-emotion-recognition",
  "version": "2.0.0"
}
```

### 2) 감정 예측

`POST /predict_emotion`

요청은 아래 둘 중 하나를 포함해야 합니다.

#### A. `skeleton_data` 형식

```json
{
  "skeleton_data": [
    "0.5,0.3,0.1",
    "0.48,0.35,0.12",
    "0.52,0.35,0.12"
  ],
  "n_joints": 17
}
```

- `skeleton_data`: `"x,y,z"` 문자열 배열
- `n_joints`: 관절 수 (생략 시 기본값 17)

#### B. `keypoints` 형식

```json
{
  "keypoints": [
    [0.5, 0.3, 0.1],
    [0.48, 0.35, 0.12],
    [0.52, 0.35, 0.12]
  ],
  "n_joints": 13
}
```

- `keypoints`: `[[x, y, z], ...]` 배열
- `n_joints`: 관절 수 (생략 시 기본값 13)
- `len(keypoints) % n_joints == 0` 이어야 합니다.

### 성공 응답 (200)

```json
{
  "emotion": "happy",
  "confidence": 0.97,
  "confidence_level": "high",
  "probabilities": {
    "happy": 0.97,
    "sad": 0.01,
    "fear": 0.01,
    "disgust": 0.0,
    "angry": 0.0,
    "neutral": 0.01
  },
  "features": [0.1, 0.2, 0.3],
  "features_shape": [1, 20],
  "message": "감정이 성공적으로 예측되었습니다."
}
```

> `features`는 추출된 14개 특징에서 시작해, 모델 입력 차원(`n_features_in_`)에 맞게 0으로 패딩될 수 있습니다.

### 에러 응답

- `422`: 입력 형식 오류 / 특징 추출 실패 / 특징 차원 오류
- `503`: 모델 로드 실패
- `500`: 예측 단계 내부 오류

## 💡 cURL 예시

```bash
curl -X POST "http://localhost:8000/predict_emotion" \
  -H "Content-Type: application/json" \
  -d '{
    "skeleton_data": ["0.5,0.3,0.1", "0.48,0.35,0.12", "0.52,0.35,0.12"],
    "n_joints": 17
  }'
```
