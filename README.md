# 🚶 걸음걸이 감정 인식 시스템

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

**걸음걸이 패턴으로 감정을 인식하는 AI 기반 시스템**

사람의 걸음걸이(Gait)는 감정 상태에 따라 미묘하게 변화합니다. 이 프로젝트는 머신러닝을 활용하여 걸음걸이 데이터로부터 감정(행복, 슬픔, 분노 등)을 자동으로 예측하는 REST API를 제공합니다.

## 📖 목차

- [프로젝트 소개](#-프로젝트-소개)
- [주요 특징](#-주요-특징)
- [공익적 활용](#-공익적-활용)
- [기술 스택](#-기술-스택)
- [성능 지표](#-성능-지표)
- [시작하기](#-시작하기)
  - [Docker로 실행 (권장)](#1-docker로-실행-권장)
  - [로컬 환경에서 실행](#2-로컬-환경에서-실행)
- [프로젝트 구조](#-프로젝트-구조)
- [API 사용법](#-api-사용법)
- [프론트엔드](#-프론트엔드)
- [테스트](#-테스트)
- [개발 가이드](#-개발-가이드)
- [문제 해결](#-문제-해결)
- [기여하기](#-기여하기)
- [라이선스](#-라이선스)

## 🎯 프로젝트 소개

걸음걸이 감정 인식(Gait-Emotion Recognition)은 사람의 독특한 걸음걸이 패턴을 분석하여 감정 상태를 추론하는 기술입니다. 이 프로젝트는 다음과 같은 과정으로 작동합니다:

1. **데이터 입력**: 비디오나 센서에서 추출한 신체 키포인트(관절 좌표)
2. **특징 추출**: 걸음걸이의 14가지 특징 자동 추출 (속도, 보폭, 팔 스윙 등)
3. **감정 예측**: 머신러닝 모델(Random Forest)로 감정 분류
4. **결과 반환**: 예측된 감정과 신뢰도를 JSON 형태로 반환

### 🔍 걸음걸이로 알 수 있는 것들

- **행복**: 빠르고 경쾌한 걸음, 큰 보폭, 활발한 팔 움직임
- **슬픔**: 느리고 무거운 걸음, 작은 보폭, 머리를 숙임
- **분노**: 빠르지만 불규칙한 걸음, 경직된 자세
- **중립**: 평범하고 일정한 걸음걸이

## ✨ 주요 특징

- ✅ **RESTful API**: FastAPI 기반의 고성능 API
- ✅ **Docker 지원**: 어디서든 동일한 환경에서 실행 가능
- ✅ **머신러닝**: Random Forest 모델로 90.79% 정확도 달성
- ✅ **규칙 기반 Fallback**: 모델 파일 없이도 작동 가능
- ✅ **상세한 주석**: 비전공자도 이해할 수 있는 한글 주석
- ✅ **프론트엔드**: 웹 인터페이스로 쉽게 테스트 가능
- ✅ **자동 테스트**: pytest로 API 엔드포인트 검증
- ✅ **CORS 지원**: 프론트엔드에서 자유롭게 호출 가능

## 🌍 공익적 활용

이 기술은 다양한 공익적 목적으로 활용될 수 있습니다:

### 1. 범죄 예방 및 공공 안전
- **CCTV 분석**: 공공장소에서 이상 행동 감지
- **폭력 예방**: 분노나 공격성 징후를 조기 발견
- **실종자 수색**: 감정 상태로 위험 상황 파악

### 2. 군중 안전 경보 시스템
- **이벤트 모니터링**: 축제, 콘서트 등에서 군중의 감정 상태 파악
- **패닉 감지**: 대규모 인파에서 불안이나 공포 확산 감지
- **대피 유도**: 위험 상황 발생 시 신속한 대응

### 3. 놀이공원 감정 모니터링
- **고객 경험 분석**: 방문객의 만족도를 실시간으로 파악
- **안전 관리**: 불안이나 스트레스를 느끼는 방문객 조기 발견
- **서비스 개선**: 감정 데이터 기반 시설 및 서비스 최적화

### 4. 의료 및 헬스케어
- **우울증 모니터링**: 환자의 감정 상태 추적
- **노인 케어**: 요양원에서 노인의 정서 상태 파악
- **재활 치료**: 치료 진행 상황 객관적 평가

### 5. 기타 활용 분야
- **로봇 상호작용**: 로봇이 사람의 감정을 이해하고 적절히 반응
- **스마트 시티**: 도시 전체의 시민 행복도 측정
- **교육**: 학생들의 학습 스트레스 파악

## 🛠 기술 스택

### 백엔드
- **Python 3.11**: 주 프로그래밍 언어
- **FastAPI 0.104.1**: 고성능 웹 프레임워크
- **Uvicorn 0.24.0**: ASGI 서버
- **Pydantic 2.5.0**: 데이터 검증

### 머신러닝
- **scikit-learn 1.3.2**: Random Forest 분류기
- **NumPy 1.26.4**: 수치 연산
- **joblib 1.3.2**: 모델 저장/로드

### 인프라
- **Docker**: 컨테이너화
- **docker-compose**: 멀티 컨테이너 관리

### 개발 도구
- **pytest 7.4.3**: 테스트 프레임워크
- **httpx 0.25.2**: 비동기 HTTP 클라이언트
- **python-dotenv 1.0.0**: 환경 변수 관리

## 📊 성능 지표

현재 시스템의 성능:

| 지표 | 값 |
|------|-----|
| **정확도** | 90.79% |
| **모델** | Random Forest |
| **특징 수** | 14개 |
| **지원 감정** | 4가지 (happy, sad, angry, neutral) |
| **응답 시간** | < 100ms |

## 🚀 시작하기

### 사전 요구사항

#### Docker 사용 시 (권장)
- Docker 20.10 이상
- docker-compose 1.29 이상

#### 로컬 실행 시
- Python 3.11 이상
- pip

---

### 1. Docker로 실행 (권장)

가장 간단하고 빠른 방법입니다. 모든 의존성이 자동으로 설치됩니다.

```bash
# 1. 저장소 클론
git clone https://github.com/KimTaek-Su/gait-emotion-recognition.git
cd gait-emotion-recognition

# 2. Docker Compose로 실행
docker-compose up --build

# 서버가 http://localhost:8000 에서 실행됩니다
```

**서버 확인**:
```bash
curl http://localhost:8000/health
```

**중지**:
```bash
docker-compose down
```

---

### 2. 로컬 환경에서 실행

Python 가상환경을 사용하는 전통적인 방법입니다.

```bash
# 1. 저장소 클론
git clone https://github.com/KimTaek-Su/gait-emotion-recognition.git
cd gait-emotion-recognition

# 2. 가상환경 생성 및 활성화
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 환경 변수 설정 (선택사항)
cp .env.example .env
# .env 파일을 필요에 따라 수정

# 5. 서버 실행
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# 또는 직접 실행
python -m src.main
```

**서버가 실행되면**:
- API 서버: http://localhost:8000
- Swagger 문서: http://localhost:8000/docs
- ReDoc 문서: http://localhost:8000/redoc

---

## 📁 프로젝트 구조

```
gait-emotion-recognition/
├── README.md                    # 프로젝트 설명서 (이 파일)
├── requirements.txt             # Python 패키지 목록
├── Dockerfile                   # Docker 이미지 설정
├── docker-compose.yml           # Docker 실행 설정
├── .gitignore                   # Git 제외 파일 목록
├── .env.example                 # 환경변수 예시
│
├── src/                         # 백엔드 소스 코드
│   ├── __init__.py
│   ├── main.py                  # FastAPI 서버 (엔드포인트 정의)
│   ├── feature_extractor.py     # 특징 추출 모듈
│   └── model.py                 # ML 모델 로드 및 예측
│
├── models/                      # 학습된 모델 파일 폴더
│   └── .gitkeep                 # (모델 파일은 .gitignore에 포함)
│
├── frontend/                    # 프론트엔드 파일
│   ├── index.html               # 웹 UI
│   └── app.js                   # JavaScript 로직
│
├── tests/                       # 테스트 코드
│   ├── __init__.py
│   └── test_api.py              # API 테스트
│
└── docs/                        # 문서
    └── API_GUIDE.md             # API 상세 가이드
```

### 주요 파일 설명

- **`src/main.py`**: FastAPI 서버의 메인 파일. API 엔드포인트 정의
- **`src/feature_extractor.py`**: 키포인트로부터 14개 특징 추출
- **`src/model.py`**: 머신러닝 모델 로드 및 감정 예측
- **`frontend/index.html`**: 웹 인터페이스 (테스트용)
- **`tests/test_api.py`**: API 엔드포인트 자동 테스트

---

## 🔌 API 사용법

### 1. 헬스 체크

서버 상태 확인:

```bash
curl http://localhost:8000/health
```

**응답**:
```json
{
  "status": "healthy",
  "service": "gait-emotion-recognition",
  "version": "1.0.0"
}
```

---

### 2. 감정 예측

키포인트 데이터로 감정 예측:

```bash
curl -X POST "http://localhost:8000/predict_emotion" \
  -H "Content-Type: application/json" \
  -d '{
    "keypoints": [
      {
        "nose": [320, 100],
        "left_shoulder": [280, 150],
        "right_shoulder": [360, 150],
        "left_ankle": [280, 500],
        "right_ankle": [360, 500]
      },
      {
        "nose": [325, 105],
        "left_shoulder": [285, 155],
        "right_shoulder": [365, 155],
        "left_ankle": [285, 505],
        "right_ankle": [365, 505]
      }
    ]
  }'
```

**응답**:
```json
{
  "emotion": "happy",
  "confidence": 0.850,
  "confidence_level": "high",
  "probabilities": {
    "happy": 0.850,
    "sad": 0.100,
    "angry": 0.050
  },
  "features": [1.5, 0.6, 2.5, ...],
  "features_shape": [14],
  "message": "감정이 성공적으로 예측되었습니다."
}
```

더 자세한 API 사용법은 [API_GUIDE.md](docs/API_GUIDE.md)를 참고하세요.

---

## 🖥 프론트엔드

웹 브라우저에서 직접 테스트할 수 있는 UI를 제공합니다.

### 실행 방법

1. API 서버가 실행 중인지 확인
2. `frontend/index.html` 파일을 브라우저에서 열기
3. "샘플 데이터 로드" 버튼 클릭
4. "감정 분석" 버튼 클릭

또는 Live Server를 사용:

```bash
# VS Code의 Live Server 확장 사용
# 또는 Python HTTP 서버 사용
cd frontend
python -m http.server 5500
```

그 다음 브라우저에서 http://localhost:5500 접속

---

## 🧪 테스트

pytest를 사용한 자동 테스트:

```bash
# 모든 테스트 실행
pytest

# 상세 출력
pytest -v

# 특정 테스트 파일만 실행
pytest tests/test_api.py

# 커버리지와 함께 실행
pytest --cov=src tests/
```

**테스트 항목**:
- ✅ 루트 엔드포인트
- ✅ 헬스 체크
- ✅ 유효한 데이터로 감정 예측
- ✅ 부족한 데이터 처리
- ✅ 빈 데이터 처리
- ✅ 잘못된 형식 처리
- ✅ 확률 합계 검증
- ✅ CORS 헤더 확인

---

## 💻 개발 가이드

### 환경 변수

`.env` 파일에서 설정 가능:

```bash
ENV=development              # 환경 (development/production)
HOST=0.0.0.0                # 서버 호스트
PORT=8000                   # 서버 포트
LOG_LEVEL=DEBUG             # 로그 레벨 (DEBUG/INFO/WARNING/ERROR)
ALLOWED_ORIGINS=http://localhost:5500,http://127.0.0.1:5500
MODEL_PATH=models/rf_emotion_model.joblib  # 모델 파일 경로
```

### 모델 파일 추가

학습된 모델 파일(.joblib)이 있다면:

1. `models/` 폴더에 `rf_emotion_model.joblib` 파일 배치
2. 서버가 자동으로 모델을 로드합니다
3. 모델이 없으면 규칙 기반 예측을 사용합니다

### 코드 스타일

- Python 코드는 PEP 8 준수
- 한글 주석으로 비전공자도 이해 가능하도록 작성
- 함수와 클래스에 docstring 필수

---

## 🐛 문제 해결

### 문제: Docker 빌드 실패

**원인**: Docker 또는 docker-compose가 설치되지 않음

**해결**:
```bash
# Docker 설치 확인
docker --version
docker-compose --version

# 설치: https://docs.docker.com/get-docker/
```

---

### 문제: 포트 충돌 (Port already in use)

**원인**: 8000 포트가 이미 사용 중

**해결**:
```bash
# 다른 포트 사용
docker-compose up -p 8080:8000

# 또는 사용 중인 프로세스 종료
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:8000 | xargs kill -9
```

---

### 문제: CORS 오류

**원인**: 프론트엔드 도메인이 허용되지 않음

**해결**: `.env` 파일 수정
```bash
ALLOWED_ORIGINS=http://localhost:5500,http://127.0.0.1:5500,http://yourdomain.com
```

---

### 문제: 모듈 Import 오류

**원인**: 의존성이 설치되지 않음

**해결**:
```bash
pip install -r requirements.txt
```

---

## 🤝 기여하기

프로젝트에 기여하고 싶으시다면:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자유롭게 사용, 수정, 배포할 수 있습니다.

---

## 📞 연락처

- **프로젝트 링크**: https://github.com/KimTaek-Su/gait-emotion-recognition
- **이슈 제보**: https://github.com/KimTaek-Su/gait-emotion-recognition/issues

---

## 🙏 감사의 말

이 프로젝트는 걸음걸이 패턴 분석 연구를 기반으로 제작되었습니다.

---

## 📚 참고 자료

- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [scikit-learn 문서](https://scikit-learn.org/)
- [Docker 공식 문서](https://docs.docker.com/)

---

**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**

*마지막 업데이트: 2025-12-07*
