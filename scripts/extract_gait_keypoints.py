import cv2
import mediapipe as mp
import numpy as np
import json 
from tqdm import tqdm

# ===== 설정 =====
INPUT_VIDEO = r'D:\research\project1\gait-emotion-recognition_v.1.5\7252275-uhd_3840_2160_25fps.mp4'  # 분석할 영상 파일
OUTPUT_JSON = 'gait_keypoints.json'                                                                   # 키포인트 정보를 저장할 파일
MIN_DETECTIONS = 30                                                                                   # 최소 검출 프레임(충분한 데이터만 저장)

# ===== MediaPipe Pose 세팅 =====
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
drawing = mp.solutions.drawing_utils

# ===== 영상 열기 =====
cap = cv2.VideoCapture(INPUT_VIDEO)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video info: {INPUT_VIDEO} / {total_frames} frames / {fps:1f} fps / {width}x{height}")

all_keypoints = []  # [프레임][관절][x, y] 구조

# ===== 프레임 루프 =====
for _ in tqdm(range(total_frames), desc="Extracting keypoints"):
    success, frame = cap.read()
    if not success:
        break
    
    # BGR to RGB 변환
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        coords = []
        for lm in results.pose_landmarks.landmark:
            coords.append([lm.x, lm.y])  # 정규화된 (x, y) 좌표
        all_keypoints.append(coords)
    else:
        # 추출 실패 프레임은 None (혹은 패딩 등)으로 처리
        all_keypoints.append(None)
        
cap.release()
print(f"성공적으로 추출한 프레임: {sum(x is not None for x in all_keypoints)}")

# ===== (선택) 충분히 인물이 검추뢴 프레임만 저장 =====
filtered_keypoints = [k for k in all_keypoints if k is not None]
if len(filtered_keypoints) < MIN_DETECTIONS:
    print(f"[경고] 사람 인식된 프레임이 {MIN_DETECTIONS}개 미만입니다 - 다른 영상 추천!")
else:
    print(f"총 {len(filtered_keypoints)} 프레임에서 인물 검출 완료.")
    
# ===== 파일로 저장(JSON) =====
# numpy 는 json 직렬화가 어려워 float 로 변환
with open(OUTPUT_JSON, "w", encoding="utf_8") as f:
    json.dump(filtered_keypoints, f, ensure_ascii=False)
    
print(f"키포인트 json 저장 완료: {OUTPUT_JSON}")