"""
걸음걸이 키포인트 추출 스크립트

영상 파일에서 MediaPipe Pose를 사용하여 관절 좌표를 추출하고 JSON으로 저장합니다.

사용법:
    python scripts/extract_gait_keypoints.py --input VIDEO_PATH [--output OUTPUT_JSON] [--min-detections N]
"""

import argparse
import json
import sys

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="영상에서 MediaPipe Pose 키포인트를 추출합니다."
    )
    parser.add_argument(
        "--input", required=True, help="분석할 영상 파일 경로"
    )
    parser.add_argument(
        "--output", default="gait_keypoints.json", help="키포인트 JSON 저장 경로 (기본: gait_keypoints.json)"
    )
    parser.add_argument(
        "--min-detections", type=int, default=30,
        help="최소 검출 프레임 수 (기본: 30)",
    )
    return parser.parse_args()


def extract_keypoints(input_video: str, output_json: str, min_detections: int):
    """영상에서 키포인트를 추출하여 JSON으로 저장합니다."""
    # MediaPipe Pose 세팅
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # 영상 열기
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"[오류] 영상 파일을 열 수 없습니다: {input_video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"영상 정보: {input_video}")
    print(f"  프레임: {total_frames} / FPS: {fps:.1f} / 해상도: {width}x{height}")

    all_keypoints = []

    # 프레임 루프
    for _ in tqdm(range(total_frames), desc="키포인트 추출 중"):
        success, frame = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            coords = [[lm.x, lm.y] for lm in results.pose_landmarks.landmark]
            all_keypoints.append(coords)
        else:
            all_keypoints.append(None)

    cap.release()
    pose.close()

    detected = sum(x is not None for x in all_keypoints)
    print(f"성공적으로 추출한 프레임: {detected}/{total_frames}")

    # 검출된 프레임만 필터링
    filtered_keypoints = [k for k in all_keypoints if k is not None]
    if len(filtered_keypoints) < min_detections:
        print(
            f"[경고] 인물 인식 프레임이 {len(filtered_keypoints)}개로 "
            f"최소 기준({min_detections})에 미달합니다."
        )
    else:
        print(f"총 {len(filtered_keypoints)} 프레임에서 인물 검출 완료.")

    # JSON 저장
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(filtered_keypoints, f, ensure_ascii=False)

    print(f"키포인트 JSON 저장 완료: {output_json}")


if __name__ == "__main__":
    args = parse_args()
    extract_keypoints(args.input, args.output, args.min_detections)