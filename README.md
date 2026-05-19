# 🚶 걸음걸이 감정 인식 시스템

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.123.9-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Bi-LSTM HCF Fusion 기반, 걸음걸이 패턴의 실시간 감정 인식 API 시스템**

사람의 걸음걸이(Gait)는 감정 상태에 따라 미묘하게 변화합니다.  
이 프로젝트는 걸음걸이 키포인트/스켈레톤 데이터로부터 감정(행복, 슬픔, 분노 등)을 예측하는 FastAPI 기반 REST API를 제공합니다.

---

## 📖 목차

- [프로젝트 소개](#-프로젝트-소개)
- [주요 특징](#-주요-특징)
- [기술 스택](#-기술-스택)
- [시작하기](#-시작하기)
  - [Docker로 실행](#1-docker로-실행)
  - [로컬 환경에서 실행](#2-로컬-환경에서-실행)
- [프로젝트 구조](#-프로젝트-구조)
- [API 사용법](#-api-사용법)
- [테스트](#-테스트)
- [현재 확인된 주의사항](#-현재-확인된-주의사항)

---

## 🎯 프로젝트 소개

이 저장소는 걸음걸이 기반 감정 인식을 위한 API 서버와 프론트엔드 예제를 포함합니다.
현재 서버 엔트리포인트는 `src/main.py`이며, Docker와 로컬 실행 모두 이 엔트리포인트를 기준으로 동작합니다.

### 전체 파이프라인

1. **데이터 입력:** MediaPipe 등에서 추출한 신체 키포인트 또는 skeleton data
2. **특징 추출:** `src/feature_extractor.py`에서 14개 HCF 특징 추출
3. **감정 예측:** 배포 모델 `models/deployment/gait_emotion_api_model.joblib` 로드
4. **결과 반환:** 예측 감정, 신뢰도, 확률 분포를 JSON으로 반환

---

## ✨ 주요 특징

- ✅ **FastAPI 기반 REST API**
- ✅ **Docker / docker-compose 지원**
- ✅ **HCF(14) 특징 추출 로직 포함**
- ✅ **CORS 허용 설정 포함**
- ✅ **프론트엔드 데모 포함**
- ✅ **pytest 테스트 파일 포함**

---

## 🛠 기술 스택

**백엔드**
- Python 3.10
- FastAPI 0.123.9
- Uvicorn 0.38.0
- Pydantic 2.12.5

**머신러닝 / 데이터 처리**
- scikit-learn 1.6.1
- joblib 1.5.2
- numpy 1.24.4
- pandas 2.3.3
- scipy 1.15.3

**컴퓨터 비전**
- OpenCV 4.7.0.72
- MediaPipe 0.10.21

---

## 🚀 시작하기

### 사전 요구사항

- Python 3.10+
- Docker / Docker Compose (선택)
- Git LFS

모델 파일은 Git LFS로 관리되므로 아래 명령이 필요합니다.

```bash
git lfs install
git lfs pull
```

---

### 1. Docker로 실행

```bash
# 저장소 클론
git clone https://github.com/KimTaek-Su/gait-emotion-recognition.git
cd gait-emotion-recognition

# LFS 모델 다운로드
git lfs install
git lfs pull

# 컨테이너 실행
docker-compose up --build
```

서버 실행 후 접속:
- Swagger UI: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

> 현재 Dockerfile은 `uvicorn src.main:app --host 0.0.0.0 --port 8000`로 서버를 실행합니다.

---

### 2. 로컬 환경에서 실행

```bash
# 저장소 클론
git clone https://github.com/KimTaek-Su/gait-emotion-recognition.git
cd gait-emotion-recognition

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows

# 패키지 설치
pip install -r requirements.txt

# LFS 모델 다운로드
git lfs install
git lfs pull

# 서버 실행
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📁 프로젝트 구조

```text
gait-emotion-recognition/
├── Dockerfile
├── docker-compose.yml
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── main.py                  # FastAPI 서버 엔트리포인트
│   ├── feature_extractor.py     # HCF 특징 추출
│   └── model.py                 # 실험/보조 코드
├── models/
│   ├── deployment/
│   │   └── gait_emotion_api_model.joblib
│   └── research/
├── frontend/
├── tests/
└── .gitattributes
```

---

## 🔌 API 사용법

### Health Check

```http
GET /health
```

예시 응답:

```json
{
  "status": "healthy",
  "service": "gait-emotion-recognition",
  "version": "2.0.0"
}
```

### 감정 예측

```http
POST /predict_emotion
```

현재 서버는 아래 두 입력 형식을 처리합니다.

#### 1) `skeleton_data` 사용 예시

```json
{
  "skeleton_data": ["0.1,0.2,0.0", "0.2,0.3,0.0"],
  "n_joints": 17
}
```

#### 2) `keypoints` 사용 예시

```json
{
  "keypoints": [
    [0.1, 0.2, 0.0],
    [0.2, 0.3, 0.0]
  ],
  "n_joints": 13
}
```

예시 응답:

```json
{
  "emotion": "happy",
  "confidence": 0.97,
  "confidence_level": "high",
  "probabilities": {
    "happy": 0.97,
    "sad": 0.01,
    "fear": 0.01,
    "disgust": 0.00,
    "angry": 0.00,
    "neutral": 0.01
  },
  "features": [0.1, 0.2, 0.3],
  "features_shape": [1, 14],
  "message": "감정이 성공적으로 예측되었습니다."
}
```

---

## 🧪 테스트

```bash
pytest tests/
```

> 참고: 현재 저장소의 테스트 코드는 서버 구현과 일부 기대값이 어긋날 수 있으므로, 테스트 실패 시 `tests/test_api.py`와 `src/main.py`를 함께 점검해야 합니다.

---

## ⚠ 현재 확인된 주의사항

- 실제 서버 엔트리포인트는 `src/main.py`입니다.
- README의 실행 방법, 엔드포인트, 응답 예시는 코드 변경 시 함께 업데이트되어야 합니다.
- 배포 ���델 파일은 현재 `models/deployment/gait_emotion_api_model.joblib`입니다.
- `tests/test_api.py`는 일부 구버전 스펙(예: 버전 문자열, 입력 형식)을 기대하고 있을 수 있습니다.
- 프론트엔드와 백엔드의 `n_joints` 처리 규약은 추가 정리가 필요할 수 있습니다.

---

## 📄 라이선스

이 저장소의 라이선스는 저장소 설정 및 LICENSE 파일을 기준으로 확인하세요.
