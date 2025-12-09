# 🚶 걸음걸이 감정 인식 시스템

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.123.9-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

**최고 정확도 96.99%를 달성한, 걸음걸이 패턴 기반의 실시간 감정 인식 AI 시스템**

사람의 걸음걸이(Gait)는 감정 상태에 따라 미묘하게 변화합니다. 이 프로젝트는 **KNN (K-Nearest Neighbors)** 머신러닝 모델을 활용하여 걸음걸이 데이터로부터 감정(행복, 슬픔, 분노 등)을 자동으로 예측하는 REST API를 제공합니다.

## 📖 목차

- [프로젝트 소개](#-프로젝트-소개)
- [주요 특징](#-주요-특징)
- [공익적 활용](#-공익적-활용)
- [기술 스택](#-기술-스택)
- [성능 지표](#-성능-지표)
- [방법론: 14가지 수제 특징 (HCF)](#-방법론-14가지-수제-특징-hand-crafted-features-hcf)
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

걸음걸이 감정 인식(Gait-Emotion Recognition)은 사람의 독특한 걸음걸이 패턴을 분석하여 감정 상태를 추론하는 기술입니다. 이 프로젝트는 다음과 같은 과정으로 작동하며, **96.99%**의 높은 정확도를 달성했습니다. 

1. **데이터 입력**: MediaPipe 등에서 추출한 신체 키포인트(관절 좌표)
2. **특징 추출**: 걸음걸이의 **14가지 수제 특징(HCF)** 자동 추출 (보폭, 관절 변화율 등)
3. **감정 예측**: **KNN** 머신러닝 모델로 감정 분류 (최종 배포 모델)
4. **결과 반환**: 예측된 감정과 신뢰도를 JSON 형태로 반환

---

## ✨ 주요 특징

- ✅ **RESTful API**: FastAPI 기반의 고성능 API
- ✅ **Docker 지원**: 어디서든 동일한 환경에서 실행 가능
- ✅ **최고 성능 달성**: **KNN 모델**로 **96.99%**의 최고 정확도 달성
- ✅ **실시간 최적화**: 추론 속도 **0.048 ms**로 실시간 서비스에 완벽 적합
- ✅ **규칙 기반 Fallback**: 모델 파일 없이도 작동 가능
- ✅ **상세한 주석**: 비전공자도 이해할 수 있는 한글 주석
- ✅ **프론트엔드**: 웹 인터페이스로 쉽게 테스트 가능
- ✅ **자동 테스트**: pytest로 API 엔드포인트 검증
- ✅ **CORS 지원**: 프론트엔드에서 자유롭게 호출 가능

---

## 🌍 공익적 활용

이 기술은 다양한 공익적 목적으로 활용될 수 있습니다.

1. **범죄 예방 및 공공 안전**
   - CCTV 분석: 공공장소에서 이상 행동 감지
   - 폭력 예방: 분노나 공격성 징후 조기 탐지
   - 실종자 수색: 감정 상태로 위험 상황 파악

2. **군중 안전 경보 시스템**
   - 행사장, 축제 등에서 군중 심리 상태 실시간 모니터링
   - 패닉 및 위험 신호 조기 감지

3. **놀이공원/공공장소 감정 모니터링**
   - 방문객 만족도 실시간 파악 및 안전 관리
   - 시설 및 서비스 개선에 데이터 활용

4. **의료 및 헬스케어**
   - 우울증·불안 장애 환자 정서 모니터링
   - 요양원·리햅센터 노인 케어 및 재활 관리

5. **로봇·스마트시티 등 첨단 인터페이스**
   - 로봇 맞춤형 감정 반응
   - 도시/학교 등 공동체의 행복도 추적

---

## 🛠 기술 스택

**백엔드**
- Python 3.10
- FastAPI 0.123.9
- Uvicorn 0.38.0 (ASGI 서버)
- Pydantic 2.12.5 (데이터 유효성 검증)

**머신러닝**
- scikit-learn 1.6.1 (KNN 모델)
- joblib 1.5.2 (모델 직렬화/역직렬화)
- numpy 2.2.6

**컴퓨터 비전**
- OpenCV 4.12.0.88 (키포인트 처리 등)

**인프라**
- Docker, docker-compose

**개발 도구**
- pytest, httpx (테스트)

---

## 📊 성능 지표

| 모델 아키텍처 | 사용 특징 | 정확도 | 응답 시간 | 비고           |
| :------------ | :------- | :----- | :-------- | :------------- |
| KNN (최종 배포)         | 14개 HCF      | 96.99%   | 0.048 ms  | 최고 성능 및 속도  |
| Bi-LSTM HCF Fusion      | Raw+HCF       | 94.66%   | 약 80ms    | 고성능 딥러닝      |
| Bi-LSTM                | Raw-only      | 92.61%   | 약 55ms    | 시계열 딥러닝      |
| Random Forest          | 14개 HCF      | 72.81%   | 0.072 ms   | 빠른 전통 ML       |
| SVM                    | 14개 HCF      | 34.42%   | 약 15ms    | 전통 ML            |

### ✅ 모델 선정 근거

- **정확도**: KNN + HCF 조합이 최고 성능(96.99%)
- **추론 속도**: KNN이 0.048 ms로 실시간 서비스에 최적화
- **결론**: 정확도와 실시간성 모두 확보한 KNN을 배포 모델로 선정

---

## 🧬 방법론: 14가지 수제 특징 (Hand-Crafted Features, HCF)

MediaPipe 등에서 추출된 키포인트를 기반으로 아래와 같은 14가지 특징 벡터를 산출합니다.

- 관절 각도/변화율 (무릎, 엉덩이, 발목)
- 보폭 길이, 신체 중심점 이동(Axis/COM)
- 어깨/골반 비율, 상체 기울기(Variance)
- 신체 부위 이동 속도, 걸음 리듬 등

> 상세 구현: `src/feature_extractor.py` 참조  
> 딥러닝/ML모델 학습 과정 및 연구용 코드: `models/research/` 참조

---

## 🚀 시작하기

### 환경 정보

- Python 3.10 이상 (torch, scikit-learn, FastAPI 등 라이브러리 사용)
- 지원 감정: Happy, Sad, Fear, Disgust, Angry, Neutral (6가지)

### 사전 요구사항

- **Docker:** v20.10 이상 권장, docker-compose v1.29 이상
- **로컬 실행:** Python 3.10+, pip, (Windows/Mac/Linux 모두 지원)
- **대용량 모델 파일:** LFS(Git Large File Storage) 설치 필요

---

### 1. Docker로 실행 (권장)

모든 의존성과 환경을 알아서 세팅할 수 있습니다.

```bash
# 1. 저장소 클론
git clone https://github.com/KimTaek-Su/gait-emotion-recognition.git
cd gait-emotion-recognition

# 2. Docker Compose로 빌드 및 실행
docker-compose up --build

# 3. 서버 접속
# http://localhost:8000/docs 에서 Swagger 테스트 가능
# http://localhost:8000      (기본 API)
```

> **참고:** Docker 기반 실행 시 모델은 `models/deployment/KNN_best_model.joblib`를 자동으로 로드합니다.

---

### 2. 로컬 환경에서 실행

직접 Python으로 실행하는 방법입니다.

```bash
# 1. 저장소 클론
git clone https://github.com/KimTaek-Su/gait-emotion-recognition.git
cd gait-emotion-recognition

# 2. Python 환경 준비 (가상환경 권장)
python -m venv venv
source venv/bin/activate       # (Linux/macOS)
venv\Scripts\activate          # (Windows)

# 3. 의존 패키지 설치
pip install -r requirements.txt

# 4. (필요시) LFS 모델 파일을 로컬에 다운로드
git lfs install
git lfs pull

# 5. 서버 실행
python main.py
# http://localhost:8000/docs 에서 문서 확인 및 API 테스트 가능
```

---

## 📁 프로젝트 구조

```
gait-emotion-recognition/
├── Dockerfile
├── docker-compose.yml
├── README.md
├── requirements.txt
├── main.py
├── src/
│   ├── feature_extractor.py
│   └── ...
├── models/
│   ├── deployment/
│   │   └── KNN_best_model.joblib
│   └── research/
│       ├── bi_lstm_hcf.pt
│       ├── svm_model.pkl
│       └── ...
├── frontend/
│   ├── index.html
│   └── ...
├── tests/
│   ├── test_api.py
│   └── ...
└── .gitattributes
```

- **main.py**: FastAPI 서버 및 비즈니스 로직
- **src/**: 특징 추출 등 데이터 전처리, 유틸리티
- **models/**: 배포 및 연구용 모델 파일 분리
- **frontend/**: 웹 데모 및 간이 테스트 페이지
- **tests/**: 단위 및 통합 테스트
- **.gitattributes**: LFS 파일 관리 확장자 설정

---

## 🔌 API 사용법

### 문서 및 UI 테스트
- Swagger/OpenAPI 기반 자동 문서 제공
- 서버 실행 후 [`http://localhost:8000/docs`](http://localhost:8000/docs) 접속

### 주요 엔드포인트 예시

#### 감정 예측
```
POST /predict-emotion
```

- **입력 (application/json)**
    ```json
    {
      "keypoints": [
        [x1, y1], [x2, y2], ..., [xn, yn]
      ]
    }
    ```
    Keypoints는 프레임별, 관절별 (x, y) 좌표 배열

- **응답**
    ```json
    {
      "emotion": "Happy",
      "confidence": 0.97
    }
    ```

#### 헬스 체크
```
GET /health
```
- 서버 상태: `{"status":"ok"}`

---

## 🌐 프론트엔드

- `frontend/` 폴더에 경량 웹 데모 페이지(`index.html`)가 포함되어 있어 입력 데이터를 직접 올리고 결과를 확인할 수 있습니다.
- API 서버와 동일 도메인에서 제공하거나, `main.py` CORS 설정을 통해 외부 배포시에도 Ajax 요청 허용.

---

## 🧪 테스트

- **pytest** 기반 자동화 테스트(엔드포인트/특징추출/모델로딩 리그레션)
- 실행 방법:
    ```bash
    pytest tests/
    ```
- 전체 API 및 데이터 처리, 예외 상황 커버

---

## 🛠 개발 가이드

- 한글 주석과 상세 설명으로 모든 핵심 로직에 가이드 포함
- 특징 추출, API 처리, 모델 inference 모듈화를 명확히 분리
- 모델 갱신·확장 시 `models/deployment/`와 `src/`만 교체로 확장 가능
- 커스텀 데이터/모델 추가시 `models/research/`의 예시 코드 참고

---

## 🐞 문제 해결

### LFS 파일 관련
- 대용량 모델 파일(.pt, .h5, .joblib, .pkl 등) 업로드/다운로드는 반드시 Git LFS 명령어(`git lfs install`, `git lfs pull`, `git lfs track "*.확장자"`)를 사용하세요.
- 웹 업로드 불가, 반드시 커맨드라인에서 처리해야 함

### 실행 에러
- 의존성 미설치: `pip install -r requirements.txt` 필수
- 모델 파일 누락: `git lfs pull`로 모델 받아야 정상 동작

### 기타 이슈
- 최신 Python(3.10), Docker, 패키지 버전 권장
- 상세 오류 메시지는 이슈 트래커에 등록해주시면 빠르게 응답드림

---

## 🤝 기여하기

1. 저장소 Fork → 새 브랜치 생성
2. 기능/수정 개발 및 테스트 코드 작성
3. Pull Request 제출 (영어/한글 PR 모두 환영)
4. 코드 리뷰를 통한 반영 및 병합
5. 제안/문의/에러 발견 시 언제든 Issue 남겨주세요!

---

## 📄 라이선스

- 본 프로젝트는 MIT License를 따릅니다.  
- 자유로운 사용·수정·배포가 가능하며, 상업적·비상업적 프로젝트에 모두 활용할 수 있습니다.
- 자세한 라이선스 전문은 [LICENSE](./LICENSE) 파일 참조

---

**문의 및 협업 제안:**  
이메일: taeksu880@gmail.com  
이슈 트래커: [https://github.com/KimTaek-Su/gait-emotion-recognition/issues](https://github.com/KimTaek-Su/gait-emotion-recognition/issues)
