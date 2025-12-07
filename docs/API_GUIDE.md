# API 가이드

걸음걸이 감정 인식 API의 상세 사용 가이드입니다.

## 📡 기본 정보

- **Base URL**: `http://localhost:8000`
- **API 버전**: 1.0.0
- **Content-Type**: `application/json`

## 🔗 엔드포인트

### 1. 루트 엔드포인트

**설명**: API 서버의 기본 정보를 반환합니다.

```
GET /
```

**응답 예시**:
```json
{
  "message": "걸음걸이 감정 인식 API에 오신 것을 환영합니다!",
  "version": "1.0.0",
  "docs": "/docs"
}
```

---

### 2. 헬스 체크

**설명**: 서버가 정상적으로 작동 중인지 확인합니다.

```
GET /health
```

**응답 예시**:
```json
{
  "status": "healthy",
  "service": "gait-emotion-recognition",
  "version": "1.0.0"
}
```

---

### 3. 감정 예측

**설명**: 걸음걸이 키포인트 데이터로부터 감정을 예측합니다.

```
POST /predict_emotion
```

#### 📌 두 가지 입력 형식 지원

##### 형식 1: keypoints (딕셔너리 형식)
API 테스트 및 간단한 사용에 적합

```json
{
  "keypoints": [
    {
      "nose": [320, 100],
      "left_shoulder": [280, 150],
      "right_shoulder": [360, 150],
      "left_elbow": [250, 200],
      "right_elbow": [390, 200],
      "left_wrist": [230, 250],
      "right_wrist": [410, 250],
      "left_hip": [290, 300],
      "right_hip": [350, 300],
      "left_knee": [285, 400],
      "right_knee": [355, 400],
      "left_ankle": [280, 500],
      "right_ankle": [360, 500]
    },
    {
      "nose": [325, 105],
      "left_shoulder": [285, 155],
      "right_shoulder": [365, 155],
      ...
    }
  ]
}
```

##### 형식 2: skeleton_data (문자열 배열 형식)
MediaPipe Pose 웹캠 데이터에 적합

```json
{
  "skeleton_data": [
    "0.5,0.3,0.1",
    "0.48,0.35,0.12",
    "0.52,0.35,0.12",
    ...
  ],
  "n_joints": 17
}
```

#### 요청 필드 설명

##### keypoints 형식 (형식 1)

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| keypoints | Array | ✅* | 키포인트 데이터 배열 (최소 2개 프레임) |
| keypoints[].nose | [float, float] | ❌ | 코 좌표 [x, y] |
| keypoints[].left_shoulder | [float, float] | ❌ | 왼쪽 어깨 좌표 [x, y] |
| keypoints[].right_shoulder | [float, float] | ❌ | 오른쪽 어깨 좌표 [x, y] |
| keypoints[].left_elbow | [float, float] | ❌ | 왼쪽 팔꿈치 좌표 [x, y] |
| keypoints[].right_elbow | [float, float] | ❌ | 오른쪽 팔꿈치 좌표 [x, y] |
| keypoints[].left_wrist | [float, float] | ❌ | 왼쪽 손목 좌표 [x, y] |
| keypoints[].right_wrist | [float, float] | ❌ | 오른쪽 손목 좌표 [x, y] |
| keypoints[].left_hip | [float, float] | ❌ | 왼쪽 엉덩이 좌표 [x, y] |
| keypoints[].right_hip | [float, float] | ❌ | 오른쪽 엉덩이 좌표 [x, y] |
| keypoints[].left_knee | [float, float] | ❌ | 왼쪽 무릎 좌표 [x, y] |
| keypoints[].right_knee | [float, float] | ❌ | 오른쪽 무릎 좌표 [x, y] |
| keypoints[].left_ankle | [float, float] | ❌ | 왼쪽 발목 좌표 [x, y] |
| keypoints[].right_ankle | [float, float] | ❌ | 오른쪽 발목 좌표 [x, y] |

*keypoints 또는 skeleton_data 중 하나는 필수

##### skeleton_data 형식 (형식 2)

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| skeleton_data | Array[string] | ✅* | "x,y,z" 형식의 문자열 배열 |
| n_joints | int | ❌ | 관절 개수 (기본값: 17) |

*keypoints 또는 skeleton_data 중 하나는 필수

#### 성공 응답 (200 OK)

```json
{
  "emotion": "Happy",
  "confidence": 0.850,
  "confidence_level": "high",
  "probabilities": {
    "Happy": 0.650,
    "Sad": 0.100,
    "Fear": 0.080,
    "Disgust": 0.070,
    "Angry": 0.050,
    "Neutral": 0.050
  },
  "features": [1.5, 0.6, 2.5, 0.15, 0.4, 0.05, 0.7, 0.3, 0.1, 0.05, 0.85, 1.2, 0.4, 0.35],
  "features_shape": [14],
  "message": "감정이 성공적으로 예측되었습니다."
}
```

#### 응답 필드 설명

| 필드 | 타입 | 설명 |
|------|------|------|
| emotion | string | 예측된 감정 (Happy, Sad, Fear, Disgust, Angry, Neutral) |
| confidence | float | 예측 신뢰도 (0.0 ~ 1.0) |
| confidence_level | string | 신뢰도 수준 (high, medium, low) |
| probabilities | object | 각 감정별 확률 분포 (6가지 감정) |
| features | array | 추출된 14개 특징값 |
| features_shape | array | 특징 배열의 shape |
| message | string | 결과 메시지 |
| warning | string | 경고 메시지 (선택적) |

#### 에러 응답

##### 400 Bad Request - 데이터 부족 또는 잘못된 형식
```json
{
  "detail": "keypoints 또는 skeleton_data 중 하나를 제공해야 합니다."
}
```

```json
{
  "detail": "최소 2개 이상의 프레임이 필요합니다."
}
```

##### 500 Internal Server Error - 서버 오류
```json
{
  "detail": "서버 내부 오류가 발생했습니다: 좌표 0 파싱 실패: invalid,data"
}
```

---

## 📊 감정 종류

현재 시스템이 지원하는 감정:

| 감정 | 영문 코드 | 설명 |
|------|-----------|------|
| 행복 | Happy | 빠르고 경쾌한 걸음, 큰 보폭, 활발한 팔 움직임 |
| 슬픔 | Sad | 느리고 무거운 걸음, 작은 보폭, 머리를 숙임 |
| 공포 | Fear | 경계하는 걸음, 경직된 자세, 불규칙한 걸음 |
| 혐오 | Disgust | 느린 속도, 움츠린 자세, 제한된 움직임 |
| 분노 | Angry | 빠르지만 불규칙한 걸음, 경직된 자세 |
| 중립 | Neutral | 평범하고 일정한 걸음걸이 |

---

## 🔍 특징 설명 (14가지 물리 기반 특징)

### A. Kinematics (5개)
1. **평균 속도**: 전체 움직임의 평균 속도
2. **최대 속도**: 관찰된 최대 속도
3. **속도 표준편차**: 속도 변화의 일관성
4. **평균 가속도**: 속도 변화율
5. **평균 저크**: 가속도 변화율 (부드러움 측정)

### B. Body parts (3개)
6. **손 움직임**: 손목(관절 5, 6)의 평균 속도
7. **발 움직임**: 발목(관절 11, 12)의 평균 속도
8. **좌우 대칭성**: 좌우 움직임의 차이

### C. Volume (1개)
9. **바운딩 박스 부피**: 신체가 차지하는 공간의 크기

### D. Posture (2개)
10. **머리 기울기**: 머리의 수직 위치
11. **척추 길이**: 어깨-엉덩이 거리

### E. Joint variance (3개)
12-14. **관절 분산 상위 3개**: 가장 많이 움직이는 관절들

---

## 💡 사용 예시

### cURL

```bash
curl -X POST "http://localhost:8000/predict_emotion" \
  -H "Content-Type: application/json" \
  -d '{
    "keypoints": [
      {
        "nose": [320, 100],
        "left_ankle": [280, 500],
        "right_ankle": [360, 500]
      },
      {
        "nose": [325, 105],
        "left_ankle": [285, 505],
        "right_ankle": [365, 505]
      }
    ]
  }'
```

### Python (requests)

```python
import requests

url = "http://localhost:8000/predict_emotion"
data = {
    "keypoints": [
        {
            "nose": [320, 100],
            "left_ankle": [280, 500],
            "right_ankle": [360, 500]
        },
        {
            "nose": [325, 105],
            "left_ankle": [285, 505],
            "right_ankle": [365, 505]
        }
    ]
}

response = requests.post(url, json=data)
result = response.json()
print(f"예측된 감정: {result['emotion']}")
print(f"신뢰도: {result['confidence']:.2%}")
```

### JavaScript (fetch) - keypoints 형식

```javascript
const url = 'http://localhost:8000/predict_emotion';
const data = {
  keypoints: [
    {
      nose: [320, 100],
      left_ankle: [280, 500],
      right_ankle: [360, 500]
    },
    {
      nose: [325, 105],
      left_ankle: [285, 505],
      right_ankle: [365, 505]
    }
  ]
};

fetch(url, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(data)
})
  .then(response => response.json())
  .then(result => {
    console.log('예측된 감정:', result.emotion);
    console.log('신뢰도:', result.confidence);
  });
```

### JavaScript (fetch) - skeleton_data 형식 (웹캠용)

```javascript
const url = 'http://localhost:8000/predict_emotion';

// 5 프레임 x 17 관절 = 85개 좌표
const skeleton_data = [];
for (let frame = 0; frame < 5; frame++) {
  for (let joint = 0; joint < 17; joint++) {
    const x = 0.5 + frame * 0.01 + joint * 0.02;
    const y = 0.3 + frame * 0.01 + joint * 0.03;
    const z = 0.1 + frame * 0.005;
    skeleton_data.push(`${x},${y},${z}`);
  }
}

fetch(url, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    skeleton_data: skeleton_data,
    n_joints: 17
  })
})
  .then(response => response.json())
  .then(result => {
    console.log('예측된 감정:', result.emotion);
    console.log('신뢰도:', result.confidence);
  });
```

---

## 🔒 보안 고려사항

- 프로덕션 환경에서는 HTTPS를 사용하세요
- API 키 인증을 추가하는 것을 권장합니다
- Rate Limiting을 설정하여 남용을 방지하세요
- 입력 데이터 검증을 강화하세요

---

## 📈 성능 최적화 팁

1. **프레임 수**: 
   - keypoints 형식: 최소 2개, 권장 10~30개
   - skeleton_data 형식: 최소 4프레임 자동 패딩, 권장 30~300개 (1~10초)
2. **키포인트**: 더 많은 키포인트를 제공할수록 정확도 향상
3. **데이터 품질**: 깨끗하고 일관된 데이터 사용
4. **배치 처리**: 여러 예측을 순차적으로 처리하는 것보다 배치로 처리

## 🎥 MediaPipe Pose 통합

프론트엔드에서 MediaPipe Pose를 사용할 때:

1. **관절 매핑**: MediaPipe의 33개 관절을 17개로 매핑
   ```javascript
   const MEDIAPIPE_TO_17_JOINTS = [
     0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 5, 2, 7, 8
   ];
   ```

2. **프레임 수집**: 최소 30프레임 (약 1초) 수집 권장

3. **데이터 전송**: skeleton_data 형식으로 전송
   ```javascript
   {
     skeleton_data: allFrames.flat(),  // 모든 프레임을 하나의 배열로
     n_joints: 17
   }
   ```

---

## 🐛 문제 해결

### 문제: 연결 거부 (Connection Refused)
**해결**: 서버가 실행 중인지 확인하세요
```bash
docker-compose up
# 또는
python -m uvicorn src.main:app --reload
```

### 문제: CORS 오류
**해결**: `.env` 파일에서 `ALLOWED_ORIGINS` 설정 확인

### 문제: 낮은 정확도
**해결**: 
- 더 많은 프레임 제공
- 키포인트 데이터의 품질 확인
- 학습된 모델 파일(.joblib) 사용

---

## 📞 지원

- **문서**: `/docs` (Swagger UI)
- **ReDoc**: `/redoc`
- **GitHub**: [프로젝트 저장소]

---

*마지막 업데이트: 2025-12-07*
