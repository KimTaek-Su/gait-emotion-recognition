"""
gait emotion_perdict.py
- gait__keypoints.json 에서 관절좌표 시퀀스를 읽고,
- feature_extractor.py 의 함수로 14개 HCF 벡터를 만들고,
- 감정 예측(API 호출 or 모델 예측)까지 진행하는 예제.

★ 주의 ★
- feature_extractor.py 의 ';'특징 추출 함수'와 '감정 예측 함수'(또는 API 사용 방식)는 프로젝트마다 이름/입력/출력이 다를 수 있음.
- 본인 프로젝트의 함수 이름과 입력양식을 꼭 확인!
"""

import json
import numpy as np

# --------- 1. JSON 에서 키포인트 시퀀스 읽어오기 ---------
with open("gait_keypoints.json", "r", encoding="utf-8") as f:
    keypoints_seq = json.load(f)
# numpy 배열로 변환(예: [프레임][관절][2])
keypoints_seq = np.array(keypoints_seq)
print(f"키포인트 shape: {keypoints_seq.shape}")  # 예: (246, 33, 2)

if keypoints_seq.shape[2] == 2:
    # z값을 0으로 패딩하여 (T, 33, 3)으로 변환
    zero_pad = np.zeros((keypoints_seq.shape[0], keypoints_seq.shape[1], 1), dtype=keypoints_seq.dtype)
    keypoints_seq = np.concatenate([keypoints_seq, zero_pad], axis=2)
    print(f"패딩 후 keypoints shape: {keypoints_seq.shape}")

# --------- 2. feature_extractor.py 임포트 및 특징 추출 ---------
# HINT 1: 실제 함수 이름(예: extract_hcf, extract_features_from_keypoints 등)을 본인 소스 주석/코드에서 꼭 확인!
# HINT 2: 함수의 입력(expected argument)도 꼭 본인 소스 주석/README에서 확인!
from feature_extractor import extract_features
# 예) extract_features_from_keypoints(keypoints_seq: np.ndarray) → list[float]

feature_vector = extract_features(keypoints_seq, n_joints=keypoints_seq.shape[1])
print("특징 벡터:", feature_vector)

# --------- 3. 감정 분류(예측) ---------
# 상황에 따라 분기
# 3-1) 서버 API 호출 방식(API 명세를 확인!):
USE_API = True  #  ← 직접 예측이면 False, api 호출이면 True

if USE_API:
    import requests
    api_url = "http://localhost:8000/predict-emotion"  # 서버 주소/엔드포인트 맞게!
    
    if isinstance(feature_vector, np.ndarray):
        feature_vector = feature_vector.tolist()
    elif isinstance(feature_vector, list) and isinstance(feature_vector[0], np.ndarray):
        # 만약 이중 리스트([ [ ... ] ]) 구조라면 내부 요소도 list로!
        feature_vector = [v.tolist() if hasattr(v, 'tolist') else v for v in feature_vector]        
    
    # HINT 3: key(keypoints/features 등) & 데이터포맷은 API 명세/문서/예제와 꼭 맞춰야 함
    # 만약 feature_vector가 2차원 (예: [[특징1, 특징2, ...]])
    payload = {"keypoints": feature_vector}
    res = requests.post(api_url, json=payload)

    try:
        result = res.json()
        print(f"\n[API 예측결과]\n감정: {result.get('emotion')}\n신뢰도: {result.get('confidence')}")
    except Exception as e:
        print("API 호출 에러:", e)
        print("응답:", res.text)
else:
    # 3-2) 로컬에서 모델 직접 예측 함수가 있는 경우
    # (예시 signature: predict_emotion(feature_vector) → (emotion:str, confidence:float))
    from feature_extractor import predict_emotion
    emotion, confidence = predict_emotion(feature_vector)
    print(f"\n[직접 예측]\n감정: {emotion} / 신뢰도: {confidence}")
    
# --------- 4. 영상 결과 합성/출력 ---------
import cv2

INPUT_VIDEO = 'walking_sample.mp4'     # 실제 영상명 맞추기
OUTPUT_VIDEO = 'output_with_emotion.mp4'

cap = cv2.VideoCapture(INPUT_VIDEO)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

emotion_str = f"{result.get('emotion', 'Unknown')} ({100*float(result.get('confidence', 0)):.2f}%)"

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.putText(frame, emotion_str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 130, 255), 3, cv2.LINE_AA)
    out.write(frame)
cap.release()
out.release()
print(f"[+] 감정 오버레이 영상 저장: {OUTPUT_VIDEO}")

# --------- 5. 결과 저장(예: json/csv) ---------
import json
with open("emotion_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("[+] 예측 결과를 emotion_result.json 파일에 저장했습니다.")

import csv

with open("emotion_result.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    # 컬럼명
    writer.writerow(["Emotion", "Confidence"])
    writer.writerow([result.get("emotion"), result.get("confidence")])
print("[+] 예측 결과를 emotion_result.csv 파일에 저장했습니다.")