# 🚶 gait-emotion-recognition 실행 가이드 (검증 기반)

이 문서는 **실제 저장소 상태를 점검하고 직접 실행/테스트한 결과**를 바탕으로 작성되었습니다.
요약하면, 이 저장소는 FastAPI 기반 감정 예측 API와 브라우저 데모를 포함하며, 기본 실행 경로는 `src/main.py`입니다.

---

## 1) 저장소 구성과 구조

### 최상위 구조

```text
gait-emotion-recognition/
├── src/
│   ├── main.py                # 실제 API 엔트리포인트 (검증됨)
│   ├── feature_extractor.py   # skeleton_data → 14개 특징 추출 로직 (검증됨)
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
│   └── gait_emotion_predct.py
├── docs/
│   └── API_GUIDE.md
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## 2) 현재 구현된 기능과 역할

### ✅ 구현/동작 확인된 핵심

1. **헬스체크 API**: `GET /health`
2. **감정 예측 API**: `POST /predict_emotion`
   - 입력 형식 A: `skeleton_data` (`["x,y,z", ...]`)
   - 입력 형식 B: `keypoints` (`[[x, y, z], ...]`)
3. **특징 추출 파이프라인**
   - `src/feature_extractor.py`에서 14개 특징 생성
   - 모델 입력 차원(`n_features_in_`)과 다르면 `src/main.py`에서 0 패딩
4. **배포 모델 로딩**
   - 경로: `models/deployment/gait_emotion_api_model.joblib`

### ⚠️ 참고(핵심 실행 경로 외)

- `src/model.py`는 현재 기본 실행 문맥(`uvicorn src.main:app`)에서 사용되지 않습니다.
- `scripts/`의 스크립트는 실험/수동 작업 성격이며, 하드코딩 경로/입력 전제가 있어 바로 재현 가능한 파이프라인으로 검증하지 않았습니다.

---

## 3) 실행 가능 여부 검증 결과

아래 항목은 실제로 이 저장소에서 실행 확인했습니다.

### ✅ 검증 완료

- 의존성 설치 (`pip install -r requirements.txt`)
- 테스트 실행 (`python -m pytest -q`) → **13 passed**
- API 서버 기동 (`uvicorn src.main:app --host 127.0.0.1 --port 8000`)
- 수동 API 호출
  - `GET /health` 정상
  - `POST /predict_emotion` 정상(예측 JSON 반환)

### ⚠️ 완전 검증하지 못한 항목

- `scripts/extract_gait_keypoints.py`, `scripts/gait_emotion_predct.py`의 end-to-end 실행
  - 외부 영상/입력 파일 및 환경 전제 필요
- Python 3.12 환경에서 Python용 `mediapipe` 직접 실행 경로
  - 본 저장소 핵심 API 실행에는 필요 없으므로 기본 설치 의존성에서 분리

---

## 4) 실행을 막던 문제(원인)

초기 상태에서 `pip install -r requirements.txt`가 실패했습니다.

- 원인 1: `numpy==1.24.4` 고정
  - Python 3.12에서 해당 버전 설치 실패
- 원인 2: OpenCV 고정 버전(`4.7.0.72`)은 Python 3.12 환경에서 호환 문제가 발생할 수 있음
- 원인 3: `mediapipe`는 Python 버전/플랫폼 의존성이 큰데, 핵심 API 실행에 필수는 아님

---

## 5) 실행 가능하도록 실제 변경한 내용

### 변경 파일
- `requirements.txt`

### 변경 사항
1. `numpy`를 Python 버전에 따라 분기
   - `<3.12`: `1.24.4`
   - `>=3.12`: `1.26.4`
2. `opencv-python`, `opencv-contrib-python`도 Python 버전에 따라 분기
   - `<3.12`: `4.7.0.72`
   - `>=3.12`: `4.10.0.84`
3. `mediapipe`는 Python `<3.12`에서만 기본 설치되도록 조건부 지정
   - 핵심 API 실행 경로와 테스트는 mediapipe 없이 동작

즉, **프로젝트 의도(기존 API/모델 추론 흐름)는 유지하면서, 설치 실패만 최소 수정으로 해결**했습니다.

---

## 6) 처음부터 실행하는 정확한 절차

## 6-1. 사전 준비물

- OS: Linux/macOS/Windows
- Python: 3.10 이상 권장 (3.12 포함)
- Git
- (선택) Git LFS
- (선택) Docker / Docker Compose

> 모델 파일이 LFS로 관리되는 경우를 대비해, 아래 명령을 권장합니다.

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
{"status":"healthy","service":"gait-emotion-recognition","version":"2.0.0"}
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

예상: `emotion`, `confidence`, `probabilities` 등이 포함된 JSON 반환.

---

## 6-4. 테스트 실행

```bash
python -m pytest -q
```

예상: 모든 테스트 통과.

---

## 6-5. 프론트엔드 실행 (정적 파일)

`frontend/`는 별도 빌드 도구 없이 정적 파일로 구성되어 있습니다.

```bash
cd frontend
python -m http.server 5500
```

브라우저에서 `http://localhost:5500` 접속 후,
백엔드(`http://localhost:8000`)가 켜져 있어야 API 호출이 동작합니다.

---

## 6-6. Docker 실행

```bash
docker-compose up --build
```

- 기본적으로 `uvicorn src.main:app` 실행
- 포트: `8000:8000`

---

## 7) 데이터셋/모델/사전학습 파일 요구사항

### API 실행에 반드시 필요한 파일

- `models/deployment/gait_emotion_api_model.joblib`
  - API가 시작될 때 로드
  - 없거나 손상되면 `POST /predict_emotion`에서 503 발생

### 입력 데이터 형식 요구사항

- `skeleton_data`: `"x,y,z"` 문자열 배열
- `keypoints`: `[[x,y,z], ...]` 숫자 배열
- `keypoints` 사용 시 `len(keypoints) % n_joints == 0` 이어야 함

### 선택 사항

- 영상에서 키포인트 추출 자동화(`scripts/extract_gait_keypoints.py`)는
  `opencv`, `mediapipe` 등 추가 환경 의존성이 있습니다.

---

## 8) 트러블슈팅

### Q1. `pip install -r requirements.txt`가 실패합니다.

- Python 버전 확인:
  ```bash
  python --version
  ```
- 현재 `requirements.txt`는 Python 3.12에서도 설치 가능하도록 분기되어 있습니다.
- 그래도 실패하면 가상환경을 새로 만들고 재시도하세요.

### Q2. `/predict_emotion`이 503을 반환합니다.

- `models/deployment/gait_emotion_api_model.joblib` 파일 존재/크기 확인
- LFS 사용 저장소라면 `git lfs pull` 실행

### Q3. 프론트엔드에서 API 호출 실패(CORS/연결오류)

- 백엔드가 `http://localhost:8000`에서 실행 중인지 확인
- 프론트엔드를 `file://`로 직접 열지 말고 `python -m http.server`로 서빙

---

## 9) 현재 한계 / 주의사항

- 핵심 API/테스트 경로는 검증 완료했지만, `scripts/`는 연구/실험 스크립트 성격입니다.
- Python 기반 MediaPipe 처리까지 반드시 필요하면 Python 3.10 가상환경을 별도로 두는 것을 권장합니다.
- 실제 예측 품질은 입력 데이터 품질과 모델 학습 데이터 분포에 크게 의존합니다.

---

## 10) 이번 점검에서의 검증 로그 요약

- `pip install -r requirements.txt` ✅
- `python -m pytest -q` → `13 passed` ✅
- `GET /health` 응답 확인 ✅
- `POST /predict_emotion` 실제 응답 확인 ✅

필요하면 다음 단계로, `scripts/`를 현재 API 명세와 완전히 맞추는 리팩터링도 진행할 수 있습니다.
