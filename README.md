# 🚶 걸음걸이 감정 인식 시스템

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.123.9-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

**최고 정확도 96.42%를 달성한 Bi-LSTM HCF Fusion 기반, 걸음걸이 패턴의 실시간 감정 인식 AI 시스템**

사람의 걸음걸이(Gait)는 감정 상태에 따라 미묘하게 변화합니다.  
이 프로젝트는 **Bi-LSTM_HCF_Fusion 딥러닝 모델**을 활용하여 걸음걸이 데이터로부터 감정(행복, 슬픔, 분노 등)을 자동으로 예측하는 REST API를 제공합니다.

---

## 📖 목차

- [프로젝트 소개](#-프로젝트-소개)
- [연구 정리 한눈에 보기](#-연구-정리-한눈에-보기)
- [주요 특징](#-주요-특징)
- [공익적 활용](#-공익적-활용)
- [기술 스택](#-기술-스택)
- [성능 지표](#-성능-지표)
- [방법론: 14가지 수제 특징 (HCF) + LSTM 시계열](#-방법론-hcf14--bi-lstm-시계열-융합)
- [문서 안내](#-문서-안내)
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

---

## 🎯 프로젝트 소개

걸음걸이 감정 인식(Gait-Emotion Recognition)은 사람의 독특한 걸음걸이 패턴을 분석하여 감정 상태를 추론하는 기술입니다.
이 프로젝트는 **HCF 기반 14차원 특징 벡터와 Raw Keypoint 시계열 입력을 Bi-LSTM(Fusion) 딥러닝 모델**에 투입하여  
최고 정확도 96.42%를 달성했습니다.

### 전체 파이프라인

1. **데이터 입력:** MediaPipe 등에서 추출한 신체 키포인트(관절 좌표)
2. **특징 추출:** 걸음걸이의 **14가지 수제 특징(HCF)** 추출 및 프레임 시계열
3. **감정 예측:** **Bi-LSTM_HCF_Fusion** 딥러닝 모델로 감정 분류 (최종 배포 모델)
4. **결과 반환:** 예측된 감정과 신뢰도를 JSON 형태로 반환

---

## 🧭 연구 정리 한눈에 보기

| 항목 | 핵심 내용 |
|:--|:--|
| 연구 질문 | **걸음걸이만으로도 사람의 정서 상태를 안정적으로 구분할 수 있는가?** |
| 입력 표현 | 신체 키포인트 시계열(`skeleton_data`)과 여기서 파생한 **14개 HCF 특징** |
| 모델 전략 | **Bi-LSTM**으로 시계열 패턴을 읽고, **HCF 특징**과 결합하는 Fusion 구조 사용 |
| 예측 대상 | Happy, Sad, Fear, Disgust, Angry, Neutral |
| 현재 배포 기준 | `models/deployment/`의 배포용 모델을 API에서 로드해 추론 |

### 연구 관점에서 보면

1. **문제 정의**  
   비언어적 행동 신호 중 하나인 걸음걸이에서 정서적 단서를 읽어내는 것이 목표입니다.
2. **표현 학습 + 해석 가능성의 균형**  
   Raw 시계열만 쓰지 않고, 속도·가속도·자세 같은 **해석 가능한 특징(HCF)** 을 함께 사용합니다.
3. **실험 자산 분리**  
   - `models/research/`: 실험 결과 및 연구 자산  
   - `models/deployment/`: 실제 서비스 배포 기준 모델  
   - `src/`: API 및 특징 추출/추론 코드
4. **재정리 시작점**  
   연구 배경, 방법론, 실험 비교, 향후 과제를 한 문서로 다시 보고 싶다면 `docs/RESEARCH_OVERVIEW.md`부터 보는 것을 권장합니다.

---

## ✨ 주요 특징

- ✅ **RESTful API:** FastAPI 기반의 고성능 API
- ✅ **Docker 지원:** 어디서든 동일한 환경 실행
- ✅ **Bi-LSTM HCF Fusion:** 시계열 + 특징 벡터 융합으로 고정확도 딥러닝 예측
- ✅ **실시간 처리:** 빠른 추론 속도 (80~100ms 이내)
- ✅ **상세 한글 주석:** 비전공자도 이해 가능한 코드, 주석 제공
- ✅ **프론트엔드:** 웹 인터페이스 제공
- ✅ **자동 테스트:** pytest로 API 엔드포인트 자동 검증
- ✅ **CORS 지원:** 프론트엔드에서 자유 호출 지원

---

## 🌍 공익적 활용

이 기술은 다음과 같이 다양한 공익적 목적으로 활용될 수 있습니다.

1. **범죄 예방 및 공공 안전**
   - CCTV 행동 감지와 위험 감정 조기 판단
2. **군중 안전 경보 시스템**
   - 실시간 인파 감정 상태 모니터링, 사고 예방
3. **놀이공원, 공공장소 감정 모니터링**
   - 방문객 만족도 실시간 파악, 안전 및 서비스 관리
4. **의료 및 헬스케어**
   - 우울증, 불안 등 감정 상태 분석(고령자/환자 케어 지원)
5. **로봇, 스마트시티 등 첨단 인식**
   - 로봇이 인간 감정에 따라 적응적 반응, 도시 단위 행복도 측정

---

## 🛠 기술 스택

**백엔드**
- Python 3.10
- FastAPI 0.123.9
- Uvicorn 0.38.0
- Pydantic 2.12.5

**딥러닝/머신러닝**
- TensorFlow/Keras 2.x (Bi-LSTM 구현 및 Inference)
- scikit-learn 1.6.1 (전처리 파이프/보완용)
- joblib 1.5.2 (서브모델, 유틸)
- numpy 1.24.4 (수치 연산)
- mediapipe (키포인트 추출에 활용)

**컴퓨터 비전**
- OpenCV 4.7.0.72 (영상/좌표 데이터 처리)

**인프라**
- Docker / docker-compose
- Git LFS (딥러닝 모델/대용량파일)

**개발 도구**
- pytest, httpx

---

## 📊 성능 지표

| 모델 아키텍처             | 사용 특징    | 정확도   | 응답 시간  | 비고             |
|:-------------------------|:----------- |:-------- |:---------- |:-----------------|
| Bi-LSTM HCF Fusion (배포) | Raw+HCF     | 96.42%   | ~90ms      | 최종 배포모델    |
| KNN                      | 14개 HCF    | 96.99%   | 0.048 ms   | 비공개           |
| Bi-LSTM                  | Raw-only    | 94.66%   | ~80ms      | 시계열 only      |
| Random Forest            | 14개 HCF    | 72.81%   | 0.072 ms   | 보조/비교모델    |
| SVM                      | 14개 HCF    | 34.42%   | 약 15ms    | 전통 ML          |

> **현재 API는 `models/deployment/gait_emotion_api_model.joblib`를 로드해 감정 예측에 사용합니다.**

---

## 🧬 방법론: HCF(14) + Bi-LSTM 시계열 융합

- **HCF(Hand Crafted Features):**  
  보폭, 각도 변화율, 관절 움직임 패턴 등 14개 주요 신체 역학 특징 추출
- **Bi-LSTM:**  
  키포인트의 시계열(raw trajectory) 전체 패턴을 동적으로 모델링
- **Fusion Layer:**  
  HCF 벡터와 LSTM 임베딩 특징을 결합해 감정 구분력 극대화

> 구현 상세: `src/feature_extractor.py`, `src/model.py` 참고  
> 딥러닝 아키텍처/재현: `models/research/` 참조

---

## 📚 문서 안내

- [`docs/RESEARCH_OVERVIEW.md`](./docs/RESEARCH_OVERVIEW.md): 연구 배경, 문제 정의, 방법론, 실험 관점 정리
- [`docs/API_GUIDE.md`](./docs/API_GUIDE.md): API 요청/응답과 운영 관점 사용 가이드

---

## 🚀 시작하기

### 환경 정보

- Python 3.10+
- 지원 감정: Happy, Sad, Fear, Disgust, Angry, Neutral (6가지)

### 사전 요구사항

- **Docker:** v20.10 이상 권장, docker-compose v1.29+
- **로컬 실행:** Python 3.10+, pip
- **딥러닝 모델 파일 LFS 관리:**  
  (`git lfs install`, `git lfs pull`)

---

### 1. Docker로 실행 (권장)

모든 의존성 및 실행환경 자동 셋업!

```bash
# 1. 저장소 클론
git clone https://github.com/KimTaek-Su/gait-emotion-recognition.git
cd gait-emotion-recognition

# 2. Docker Compose로 빌드 및 실행
docker-compose up --build

# 3. 서버 접속/확인
# Swagger 테스트: http://localhost:8000/docs
# API 기본주소: http://localhost:8000
```
> **참고:** 배포용 모델(`models/deployment/gait_emotion_api_model.joblib`)이 자동 로드됩니다.

---

### 2. 로컬 환경에서 실행

Python 직접 실행 안내:

```bash
# 1. 저장소 클론
git clone https://github.com/KimTaek-Su/gait-emotion-recognition.git
cd gait-emotion-recognition

# 2. 가상환경 (권장)
python -m venv venv
source venv/bin/activate       # (Linux/macOS)
venv\Scripts\activate          # (Windows)

# 3. 패키지 설치
pip install -r requirements.txt

# 4. (대용량 모델 LFS)
git lfs install
git lfs pull

# 5. 서버 실행
uvicorn src.main:app --reload
```

---

## 📁 프로젝트 구조

```
gait-emotion-recognition/
├── Dockerfile
├── docker-compose.yml
├── README.md
├── docs/
│   ├── API_GUIDE.md
│   └── RESEARCH_OVERVIEW.md
├── requirements.txt
├── src/
│   ├── main.py                     # FastAPI 서버 진입점
│   ├── feature_extractor.py        # 14가지 특징(HCF) 추출 및 시계열/전처리
│   └── model.py                    # 모델 로드/추론 로직
├── scripts/
│   ├── gait_emotion_predict.py      # 예시: 특징 추출/감정예측 유틸
│   ├── extract_gait_keypoints.py    # 예시: 영상 → 키포인트 변환
│   └── ...
├── models/
│   ├── deployment/
│   │   └── gait_emotion_api_model.joblib         # 현재 API가 로드하는 배포용 모델
│   └── research/
│       ├── ... (실험용 모델, 로그 등)
├── frontend/
│   ├── index.html
│   └── ...
├── tests/
│   ├── test_api.py
│   └── ...
└── .gitattributes
```
- **src/main.py**: FastAPI 서버 진입점 (걸음걸이 감정 인식 API)
- **src/**: 특징 추출/모델 관리 등 내부 로직
- **scripts/**: 분석/추출/개별 실행 테스트 스크립트(별도 실행)
- **models/deployment/**: 배포용 모델 파일 저장 위치
- **frontend/**: 웹 데모
- **tests/**: 자동화 테스트
- **.gitattributes**: LFS 설정

---

## 🔌 API 사용법

### 문서 및 UI 테스트

- FastAPI 자동 문서: [`http://localhost:8000/docs`](http://localhost:8000/docs)

### 주요 엔드포인트 예시

#### 감정 예측
```
POST /predict_emotion
```

**입력 예시 (application/json)**
```json
{
  "keypoints": [[[x, y], ...], ...]  // 프레임 순서별 관절 (시계열 2D 배열)
}
```
- 입력은 **(프레임개수, 관절수, 좌표차원)**의 3차원 리스트
- 또는 14차원 HCF 특징벡터 시계열 (internal 변환)

**응답 예시**
```json
{
  "emotion": "happy",
  "confidence": 0.97,
  "probabilities": {
    "happy": 0.97,
    "sad": 0.02,
    ...
  },
  "features": [[1.2, ...14개], ...],        // 시계열의 HCF 14D 백터
  "features_shape": [timesteps, 14],
  "message": "감정이 성공적으로 예측되었습니다."
}
```

#### 서버 상태 체크

```
GET /health
```
응답: `{"status":"healthy","service":"gait-emotion-recognition","version":"2.0.0"}`

---

## 🌐 프론트엔드

- `frontend/` 폴더에 웹 데모(html) 포함
- API 서버와 도메인이 다를 경우 FastAPI에서 CORS 허용

---

## 🧪 테스트

- Pytest 기반 자동 테스트:  
  ```bash
  pytest tests/
  ```
- API, 특징 추출, 모델로드 등 커버

---

## 🛠 개발 가이드

- **한글 주석 & 상세 설명**
- 특징/HCF 추출, Bi-LSTM 모델 inference 등 명확 분리
- 모델 교체시 `models/deployment/` 내 pkl 파일만 대체
- 추가 연구/실험 확장: `scripts/`, `models/research/` 참고

---

## 🐞 문제 해결

- **LFS 파일:**  
  대용량(.pt, .pkl, .h5 등)은 반드시  
  `git lfs install`, `git lfs pull`, `.gitattributes` 확인

- **실행 에러**
  - 패키지 미설치 → `pip install -r requirements.txt`
  - 모델 파일 누락 → `git lfs pull`

- **환경/버전**
  - Python 3.10, Docker 등 최신 권장
  - 상세 문의는 [이슈 트래커](https://github.com/KimTaek-Su/gait-emotion-recognition/issues)

---

## 🤝 기여하기

1. 저장소 Fork → 새 브랜치 생성
2. 기능/수정 개발 및 테스트 코딩
3. Pull Request 제출 (한글/영어 모두 환영)
4. 코드 리뷰 및 병합
5. 문의/제안/에러는 [이슈](https://github.com/KimTaek-Su/gait-emotion-recognition/issues) 사용!

---

## 📄 라이선스

- 본 프로젝트는 MIT License 기반입니다.
- 자유로운 사용·수정·배포가 가능합니다.
- 라이선스 전문: [LICENSE](./LICENSE) 참조

---

**문의/협업:**  
이메일: taeksu880@gmail.com  
이슈 트래커: [https://github.com/KimTaek-Su/gait-emotion-recognition/issues](https://github.com/KimTaek-Su/gait-emotion-recognition/issues)

---
