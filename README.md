# 🚶 gait-emotion-recognition 실행 가이드 (검증 기반)

이 문서는 **실제 저장소 상태를 점검하고 실행 경로를 확인한 뒤, 현재 코드 기준으로 정리한 실행 가이드**입니다.
요약하면, 이 저장소의 **실제 백엔드 실행 엔트리포인트는 `src/main.py`**이며, 핵심 검증 대상도 이 경로를 기준으로 설명합니다.

---

## 1) 저장소 구성과 구조

### 최상위 구조

```text
gait-emotion-recognition/
├── src/
│   ├── main.py                # 실제 API 엔트리포인트 (주 실행 경로)
│   ├── feature_extractor.py   # skeleton_data → 14개 특징 추출 로직
│   └── model.py               # 구버전/실험 성격 API 코드 (현재 실행 경로 아님)
├── models/
│   ├── deployment/
│   │   └── gait_emotion_api_model.joblib  # API 예측에 사용하는 배포 모델
│   └── research/              # 연구/실험 모델 관련 폴더
├── frontend/
│   ├── index.html             # 브라우저 데모 UI
│   └── app.js                 # API 호출, 웹캠(MediaPipe JS) 처리
├── tests/
│   ├── test_api.py            # API 엔드포인트 테스트
│   └── test_model.py          # feature 변환/패딩 등 유닛 테스트
├── scripts/
│   ├── extract_gait_keypoints.py
│   └── gait_emotion_predct.py   # 원본 파일명(typo 포함)
├── docs/
│   └── API_GUIDE.md
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

### 구조를 어떻게 읽으면 되는가

이 저장소는 크게 4개 층으로 보면 이해가 쉽습니다.

1. **API 서비스 층**
   - `src/main.py`
   - FastAPI 앱 생성, 모델 로드, `/health`, `/predict_emotion` 제공

2. **특징 추출 층**
   - `src/feature_extractor.py`
   - 입력 스켈레톤 데이터를 14개 특징으로 변환

3. **모델 자산 층**
   - `models/deployment/gait_emotion_api_model.joblib`
   - 서버가 시작될 때 로드되는 실제 배포 모델

4. **사용/데모 층**
   - `frontend/`: 브라우저에서 API를 호출하는 정적 데모
   - `tests/`: 최소 API/특징 변환 검증
   - `scripts/`: 실험/수동 작업용 스크립트

---

## 2) 현재 구현된 기능과 역할

### ✅ 구현/핵심 경로로 확인되는 기능

1. **헬스체크 API**: `GET /health`
2. **감정 예측 API**: `POST /predict_emotion`
   - 입력 형식 A: `skeleton_data` (`["x,y,z", ...]`)
   - 입력 형식 B: `keypoints` (`[[x, y, z], ...]`)
3. **특징 추출 파이프라인**
   - `src/feature_extractor.py`에서 14개 특징 생성
   - 모델 입력 차원(`n_features_in_`)과 다르면 `src/main.py`에서 0 패딩
4. **배포 모델 로딩**
   - 경로: `models/deployment/gait_emotion_api_model.joblib`
5. **브라우저 데모**
   - `frontend/index.html` + `frontend/app.js`
   - 텍스트 입력 기반 API 테스트
   - 웹캠 기반 실시간 수집/전송 UI

### ⚠️ 참고(핵심 실행 경로 외)

- `src/model.py`는 현재 기본 실행 문맥(`uvicorn src.main:app`)에서 사용되지 않습니다.
- `src/model.py`에는 또 다른 FastAPI 앱과 예시성 특징 추출 코드가 있으나, **현재 저장소의 주 실행 경로로 보면 안 됩니다.**
- `scripts/`의 스크립트는 실험/수동 작업 성격이며, 하드코딩 경로/입력 전제가 있어 README의 핵심 실행 검증 범위에 포함하지 않습니다.

---

## 3) 실행 가능 여부 검증 결과

이 저장소를 볼 때 가장 중요한 기준은 **무엇이 실제로 확인되었고, 무엇이 아직 확인되지 않았는가**입니다.

### ✅ 확인된 실행 경로

아래는 현재 코드 구조상 **실행 경로로 신뢰할 수 있는 부분**입니다.

- `uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload`
- `GET /health`
- `POST /predict_emotion`
- `python -m pytest -q`
- `frontend/` 정적 서빙 후 백엔드 호출

### 🟡 부분적으로만 신뢰할 수 있는 경로

- `frontend/`의 웹캠 분석 기능
  - 브라우저에서 MediaPipe JavaScript를 사용합니다.
  - 백엔드 API와 연결 구조는 맞지만, 실제 정확도/입력 품질은 웹캠 환경에 크게 의존합니다.
- 33개 관절 입력 경로
  - `frontend/app.js`는 웹캠 수집 시 33개 관절을 서버로 전송합니다.
  - `src/main.py`와 `src/feature_extractor.py`는 `n_joints`를 받도록 되어 있어 구조상 처리 가능하지만, 실제 모델 학습 분포와 완전히 일치하는지는 별도 검토가 필요합니다.

### ❌ 이 README의 “검증 완료” 범위에 포함하지 않는 경로

- `src/model.py`를 직접 서버 엔트리포인트로 쓰는 실행
- `scripts/extract_gait_keypoints.py`, `scripts/gait_emotion_predct.py`의 end-to-end 실행
- Python 3.12 환경에서 Python 패키지 `mediapipe`를 직접 사용하는 경로

---

## 4) 실행을 막던 문제(원인)

초기 상태에서 가장 큰 문제는 **설치/실행 기준이 README와 실제 코드 사이에서 혼동될 수 있다는 점**입니다.

핵심 원인은 다음과 같습니다.

1. **실제 엔트리포인트 혼동**
   - 현재 저장소에서 실제 엔트리포인트는 `src/main.py`인데,
   - `src/model.py`도 별도 FastAPI 앱처럼 보여 사용자가 잘못 실행할 수 있습니다.

2. **의존성 버전 문제 가능성**
   - `numpy==1.24.4` 같은 구버전은 Python 3.12에서 설치 실패 가능성이 있습니다.
   - OpenCV/MediaPipe도 Python 버전과 플랫폼에 따라 제약이 있습니다.

3. **외부 자산 의존성**
   - `models/deployment/gait_emotion_api_model.joblib` 파일이 없거나 손상되어도,
     서버는 내장 fallback 모델로 `/predict_emotion`을 계속 제공합니다.
   - Git LFS로 관리되는 환경이라면 모델 파일이 비어 있거나 내려받아지지 않을 수 있으며,
     이 경우 응답 `model.mode`가 `fallback`으로 표시됩니다.

---

## 5) 실행 가능하도록 실제로 반영/정리해야 하는 핵심 포인트

현재 코드 기준으로, 사용자가 실제로 실행할 때 가장 중요한 사실은 아래 5가지입니다.

1. **실행 명령은 `uvicorn src.main:app` 기준으로 봐야 합니다.**
2. **모델 파일이 정상 로드되면 실제 학습 모델을 쓰고, 실패 시 내장 fallback 모델을 사용합니다.**
3. **프론트엔드 웹캠 기능은 Python `mediapipe`가 아니라 브라우저용 MediaPipe JS를 사용합니다.**
4. **`src/model.py`는 레거시/실험 성격으로 보고, 주 실행 경로로 사용하지 않는 것이 안전합니다.**
5. **`scripts/`는 실험용이므로 README의 공식 실행 절차와 분리해서 봐야 합니다.**

---

## 6) 처음부터 실행하는 정확한 절차

## 6-1. 사전 준비물

- OS: Linux/macOS/Windows
- Python: 3.10 이상 권장
- Git
- (권장) Git LFS
- (선택) Docker / Docker Compose

모델 파일이 LFS로 관리되는 경우를 대비해, 아래 명령을 권장합니다.
fallback이 있어도 실제 학습 모델 추론을 원하면 LFS 파일을 정상 확보해야 합니다.

```bash
git lfs install
git lfs pull
```

---

## 6-2. 로컬(권장) 실행 절차

```bash
# 1) 저장소 클론
git clone https://github.com/KimTaek-Su/gait-emotion-recognition.git
cd gait-emotion-recognition

# 2) (권장) 가상환경 생성
python -m venv .venv

# 3) 가상환경 활성화
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# 4) 의존성 설치
pip install -r requirements.txt

# 5) 서버 실행
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

실행 후 접속:
- Swagger: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

---

## 6-3. 동작 확인 명령 (복붙용)

### Health 확인

```bash
curl http://localhost:8000/health
```

예상 응답:

```json
{
  "status":"healthy",
  "service":"gait-emotion-recognition",
  "version":"2.0.0",
  "model":{
    "mode":"trained",
    "source":"artifact"
  }
}
```

### 감정 예측 확인 (`skeleton_data`)

```bash
curl -X POST "http://localhost:8000/predict_emotion" \
  -H "Content-Type: application/json" \
  -d '{
    "skeleton_data": [
      "0.5,0.3,0.1", "0.52,0.31,0.1", "0.54,0.32,0.1", "0.56,0.33,0.1",
      "0.58,0.34,0.1", "0.6,0.35,0.1", "0.62,0.36,0.1", "0.64,0.37,0.1",
      "0.66,0.38,0.1", "0.68,0.39,0.1", "0.7,0.4,0.1", "0.72,0.41,0.1",
      "0.74,0.42,0.1", "0.76,0.43,0.1", "0.78,0.44,0.1", "0.8,0.45,0.1",
      "0.82,0.46,0.1", "0.51,0.3,0.1", "0.53,0.31,0.1", "0.55,0.32,0.1",
      "0.57,0.33,0.1", "0.59,0.34,0.1", "0.61,0.35,0.1", "0.63,0.36,0.1",
      "0.65,0.37,0.1", "0.67,0.38,0.1", "0.69,0.39,0.1", "0.71,0.4,0.1",
      "0.73,0.41,0.1", "0.75,0.42,0.1", "0.77,0.43,0.1", "0.79,0.44,0.1",
      "0.81,0.45,0.1", "0.83,0.46,0.1"
    ],
    "n_joints": 17
  }'
```

예상: `emotion`, `confidence`, `probabilities`와 `model` 메타데이터가 포함된 JSON 반환.
`model.mode`가 `trained`면 배포 모델, `fallback`이면 내장 데모 모델입니다.

---

## 6-4. 테스트 실행

```bash
python -m pytest -q
```

`tests/test_api.py`, `tests/test_model.py`는 현재 `src.main` 중심 경로를 기준으로 작성되어 있습니다.
즉, 테스트가 의미하는 것은 **현재 저장소의 주 실행 경로가 `src/main.py`라는 점**을 다시 확인해준다는 것입니다.

---

## 6-5. 프론트엔드 실행 (정적 파일)

`frontend/`는 별도 빌드 도구 없이 정적 파일로 구성되어 있습니다.

```bash
cd frontend
python -m http.server 5500
```

브라우저에서 `http://localhost:5500` 접속 후,
백엔드(`http://localhost:8000`)가 켜져 있어야 API 호출이 동작합니다.

### 프론트엔드 사용 방식 2가지

1. **텍스트(JSON) 입력 테스트**
   - textarea에 keypoints JSON 입력
   - `predictEmotion()` → `/predict_emotion` 호출

2. **웹캠 분석 데모**
   - 브라우저 MediaPipe JS가 관절을 수집
   - `frontend/app.js`가 이를 `skeleton_data`로 변환해 서버에 전송

### 중요한 점

- 프론트엔드 웹캠 기능은 **브라우저의 MediaPipe JavaScript**를 사용합니다.
- 이것은 `requirements.txt`에 들어가는 **Python 패키지 `mediapipe`와 별개**입니다.
- 따라서 Python 3.12에서 `mediapipe`를 기본 설치하지 않더라도, 브라우저 데모 자체는 동작할 수 있습니다.

---

## 6-6. Docker 실행

```bash
docker-compose up --build
```

- 기본적으로 `uvicorn src.main:app` 실행
- 포트: `8000:8000`

현재 `docker-compose.yml`은 `src/`와 `models/`를 마운트합니다.
즉, Docker 실행 시에도 모델 파일 상태에 따라 `trained`/`fallback` 모드가 자동 결정됩니다.

---

## 7) 데이터셋/모델/사전학습 파일 요구사항

### API 실행에 필요한 파일

- `models/deployment/gait_emotion_api_model.joblib`
  - API가 시작될 때 로드를 시도
  - 없거나 손상되면 fallback 모델로 자동 전환되어 `POST /predict_emotion`은 계속 동작
  - 어떤 모델이 쓰였는지는 `/health`와 `/predict_emotion`의 `model` 필드에서 확인 가능

### fallback → 실제 모델 교체 방법

1. 학습된 `joblib` 모델 파일을 `models/deployment/gait_emotion_api_model.joblib`에 배치합니다.
2. 서버를 재시작합니다.
3. `GET /health`에서 `model.mode`가 `trained`인지 확인합니다.
4. 여전히 `fallback`이면 파일 손상/직렬화 호환성(`predict_proba` 지원 여부)을 점검합니다.

### 입력 데이터 형식 요구사항

#### A. `skeleton_data`
- 형식: `"x,y,z"` 문자열 배열
- 각 문자열은 반드시 3개 좌표를 포함해야 함
- 전체 길이는 사실상 `n_joints * n_frames` 구조여야 함

#### B. `keypoints`
- 형식: `[[x, y, z], ...]`
- 비어 있으면 안 됨
- 모든 원소가 길이 3의 숫자 배열이어야 함
- `len(keypoints) % n_joints == 0` 이어야 함

### 선택 사항

- 영상에서 키포인트 추출 자동화(`scripts/extract_gait_keypoints.py`)는
  `opencv`, `mediapipe` 등 추가 환경 의존성이 있습니다.
- 이 경로는 현재 README의 핵심 실행 검증 범위 밖입니다.

---

## 8) 트러블슈팅

### Q1. `pip install -r requirements.txt`가 실패합니다.

- Python 버전 확인:
  ```bash
  python --version
  ```
- Python 3.12에서는 일부 패키지 호환성 문제가 발생할 수 있습니다.
- 가능하면 Python 3.10 또는 3.11 가상환경에서 먼저 재현하는 것을 권장합니다.
- `mediapipe`가 꼭 필요하지 않다면, 핵심 API 실행은 `src/main.py` 기준으로 먼저 확인하세요.

### Q2. 어떤 모델로 추론 중인지 확인하고 싶습니다.

확인 방법:
- `GET /health` 또는 `POST /predict_emotion` 응답의 `model.mode` 확인
- `trained`면 배포 모델 사용
- `fallback`이면 내장 데모 모델 사용

확인 항목:
```bash
ls -lh models/deployment
```

실제 학습 모델로 전환하려면(선택):
```bash
git lfs pull
```

### Q3. 프론트엔드에서 API 호출 실패(CORS/연결 오류)가 납니다.

- 백엔드가 `http://localhost:8000`에서 실행 중인지 확인
- 프론트엔드를 `file://`로 직접 열지 말고 `python -m http.server`로 서빙
- 브라우저 콘솔에서 `/health` 연결 로그 확인

### Q4. `src/model.py`로 서버를 띄워도 되나요?

권장하지 않습니다.

이 저장소에서 **현재 주 실행 경로는 `src/main.py`** 입니다.
`src/model.py`는 구조상 별도 FastAPI 앱처럼 보이지만,
내부 특징 추출 방식이 현재 주 파이프라인과 다르므로 혼동을 일으킬 수 있습니다.

---

## 9) 현재 확인된 주의사항

이 섹션은 현재 저장소를 사용할 때 가장 중요한 경계 조건을 정리한 것입니다.

### 9-1. 반드시 기억할 점

1. **실제 실행 엔트리포인트는 `src.main:app` 입니다.**
2. **`src/model.py`는 현재 공식 실행 경로로 보지 않는 것이 안전합니다.**
3. **예측 API는 fresh clone에서도 항상 동작하며, 모델 상태는 응답 메타데이터로 구분합니다.**
4. **프론트엔드 웹캠 기능은 브라우저용 MediaPipe JS를 사용합니다.**
5. **`scripts/`는 실험용 성격이 강하므로 README의 공식 실행 절차와 분리해서 이해해야 합니다.**

### 9-2. 검증 범위 구분

#### ✅ 현재 확인된 범위
- `src/main.py` 기반 API 구동
- `/health`
- `/predict_emotion`
- `tests/` 기반 기본 검증
- `frontend/` 정적 파일 서빙 및 API 호출 구조
- fresh clone 환경에서도 fallback 모델을 통한 예측 응답(HTTP 200) 보장

#### 🟡 추가 검토가 필요한 범위
- 웹캠 기반 실제 추론 품질
- 33개 관절 입력이 현재 모델 학습 조건과 얼마나 일치하는지
- Docker 환경에서의 모델 파일/LFS 상태
- `model.mode=fallback`일 때의 감정 분류 정확도(데모용 규칙 기반 확률)

#### 🔴 이 README에서 실행 보장을 하지 않는 범위
- `src/model.py` 직접 사용
- `scripts/`의 end-to-end 자동 처리
- Python용 `mediapipe` 기반 전체 실험 파이프라인

### 9-3. 지금 이 저장소를 처음 실행하는 사람에게 권장하는 순서

처음 실행하는 경우에는 아래 순서를 따르는 것이 가장 안전합니다.

1. **`uvicorn src.main:app`으로 서버를 먼저 띄웁니다.**
2. **`GET /health`로 서버가 정상 기동했는지 확인합니다.**
3. **`POST /predict_emotion`에 예제 요청을 보내 모델 로드와 추론이 되는지 확인합니다.**
4. **그 다음에 `frontend/`를 정적 서버로 띄워 브라우저 데모를 확인합니다.**
5. **`src/model.py`, `scripts/`는 마지막에 참고용/실험용으로만 살펴보는 것이 좋습니다.**

### 9-4. 이 섹션만 빠르게 보고 싶다면

- **서버 실행 기준:** `src.main:app`
- **권장 모델 파일:** `models/deployment/gait_emotion_api_model.joblib` (없으면 fallback 사용)
- **공식 확인 경로:** `/health` → `/predict_emotion`
- **프론트엔드 웹캠:** 브라우저 MediaPipe JS 기반
- **주의할 파일:** `src/model.py`, `scripts/`

---

## 10) 이번 점검 요약

현재 저장소를 사용할 때 가장 중요한 결론은 다음과 같습니다.

- **실행 기준은 `src/main.py`다.**
- **예측 API는 모델 파일이 없어도 fallback으로 동작한다(실모델 보장은 아님).**
- **프론트엔드는 단순 정적 파일이지만, 웹캠 기능은 브라우저 MediaPipe JS에 의존한다.**
- **`src/model.py`, `scripts/`는 현재 공식 실행 절차의 중심이 아니다.**

즉, 처음 실행하는 사용자는 **README의 로컬 실행 절차 → `/health` 확인 → `/predict_emotion` 테스트** 순서로 접근하는 것이 가장 안전합니다.

필요하면 다음 단계로, `src/model.py`를 레거시 파일로 명시적으로 정리하거나 `scripts/`를 현재 API 명세에 맞게 리팩터링할 수 있습니다.
