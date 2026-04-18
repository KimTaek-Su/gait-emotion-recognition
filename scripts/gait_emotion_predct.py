"""
걸음걸이 감정 예측 파이프라인 스크립트

키포인트 JSON 파일에서 관절 좌표 시퀀스를 읽고,
feature_extractor.py로 14개 HCF 특징 벡터를 만들어
감정을 예측(API 호출 또는 로컬 모델)합니다.

사용법:
    python scripts/gait_emotion_predct.py --keypoints gait_keypoints.json [--use-api] [--video walking.mp4]
"""

import argparse
import json
import sys
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="키포인트 JSON에서 감정을 예측합니다."
    )
    parser.add_argument(
        "--keypoints", default="gait_keypoints.json",
        help="키포인트 JSON 파일 경로 (기본: gait_keypoints.json)",
    )
    parser.add_argument(
        "--use-api", action="store_true",
        help="서버 API를 통해 예측 (기본: 로컬 feature_extractor 사용)",
    )
    parser.add_argument(
        "--api-url", default="http://localhost:8000/predict_emotion",
        help="API 엔드포인트 URL (기본: http://localhost:8000/predict_emotion)",
    )
    parser.add_argument(
        "--video", default=None,
        help="감정 오버레이 영상을 생성할 원본 영상 경로 (선택)",
    )
    parser.add_argument(
        "--output-video", default="output_with_emotion.mp4",
        help="오버레이 영상 저장 경로 (기본: output_with_emotion.mp4)",
    )
    return parser.parse_args()


def load_keypoints(path: str) -> np.ndarray:
    """키포인트 JSON 파일을 로드하고 numpy 배열로 변환합니다."""
    if not os.path.exists(path):
        print(f"[오류] 키포인트 파일을 찾을 수 없습니다: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        keypoints_seq = json.load(f)

    keypoints_seq = np.array(keypoints_seq)
    print(f"키포인트 shape: {keypoints_seq.shape}")

    # 2D 좌표인 경우 z=0으로 패딩하여 3D로 변환
    if keypoints_seq.ndim == 3 and keypoints_seq.shape[2] == 2:
        zero_pad = np.zeros(
            (keypoints_seq.shape[0], keypoints_seq.shape[1], 1),
            dtype=keypoints_seq.dtype,
        )
        keypoints_seq = np.concatenate([keypoints_seq, zero_pad], axis=2)
        print(f"z좌표 패딩 후 shape: {keypoints_seq.shape}")

    return keypoints_seq


def predict_via_api(keypoints_seq: np.ndarray, api_url: str) -> dict:
    """API 서버를 통해 감정을 예측합니다."""
    import requests

    n_frames, n_joints, _ = keypoints_seq.shape

    # skeleton_data 형식으로 변환
    skeleton_data = []
    for frame in keypoints_seq:
        for joint in frame:
            skeleton_data.append(f"{float(joint[0])},{float(joint[1])},{float(joint[2])}")

    payload = {
        "skeleton_data": skeleton_data,
        "n_joints": n_joints,
    }

    try:
        res = requests.post(api_url, json=payload, timeout=30)
        res.raise_for_status()
        return res.json()
    except requests.ConnectionError:
        print(f"[오류] API 서버에 연결할 수 없습니다: {api_url}")
        sys.exit(1)
    except requests.HTTPError as e:
        print(f"[오류] API 응답 에러: {e}")
        print(f"  응답: {res.text}")
        sys.exit(1)


def predict_local(keypoints_seq: np.ndarray) -> dict:
    """로컬 feature_extractor를 사용하여 감정을 예측합니다."""
    # feature_extractor 경로 추가
    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from feature_extractor import extract_features_from_skeleton

    n_frames, n_joints, _ = keypoints_seq.shape

    # skeleton_data 형식으로 변환
    skeleton_data = []
    for frame in keypoints_seq:
        for joint in frame:
            skeleton_data.append(f"{float(joint[0])},{float(joint[1])},{float(joint[2])}")

    features = extract_features_from_skeleton(skeleton_data, n_joints=n_joints)
    print(f"추출된 특징 벡터 ({len(features)}개): {features}")

    # 간단한 규칙 기반 예측 (모델 없이)
    # 규칙 기반 예측: 평균 속도(features[0])를 기준으로 감정 추정
    # 0.05 이상: 빠른 걸음 → 행복, 0.01 이하: 느린 걸음 → 슬픔, 그 외: 중립
    avg_speed = features[0]
    if avg_speed > 0.05:
        emotion = "happy"
        confidence = 0.6
    elif avg_speed < 0.01:
        emotion = "sad"
        confidence = 0.5
    else:
        emotion = "neutral"
        confidence = 0.7

    return {
        "emotion": emotion,
        "confidence": confidence,
        "features": features.tolist() if hasattr(features, "tolist") else list(features),
    }


def overlay_emotion_on_video(
    input_video: str, output_video: str, emotion: str, confidence: float
):
    """영상에 감정 결과를 오버레이합니다."""
    import cv2

    if not os.path.exists(input_video):
        print(f"[오류] 영상 파일을 찾을 수 없습니다: {input_video}")
        return

    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    emotion_str = f"{emotion} ({confidence * 100:.1f}%)"

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(
            frame, emotion_str, (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 130, 255), 3, cv2.LINE_AA,
        )
        out.write(frame)

    cap.release()
    out.release()
    print(f"감정 오버레이 영상 저장: {output_video}")


def save_result(result: dict):
    """예측 결과를 JSON 및 CSV로 저장합니다."""
    import csv

    with open("emotion_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("예측 결과 저장: emotion_result.json")

    with open("emotion_result.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Emotion", "Confidence"])
        writer.writerow([result.get("emotion"), result.get("confidence")])
    print("예측 결과 저장: emotion_result.csv")


def main():
    args = parse_args()

    # 1. 키포인트 로드
    keypoints_seq = load_keypoints(args.keypoints)

    # 2. 감정 예측
    if args.use_api:
        result = predict_via_api(keypoints_seq, args.api_url)
        print(f"\n[API 예측 결과]")
    else:
        result = predict_local(keypoints_seq)
        print(f"\n[로컬 예측 결과]")

    print(f"  감정: {result.get('emotion')}")
    print(f"  신뢰도: {result.get('confidence')}")

    # 3. 결과 저장
    save_result(result)

    # 4. 영상 오버레이 (선택)
    if args.video:
        overlay_emotion_on_video(
            args.video, args.output_video,
            result.get("emotion", "Unknown"),
            float(result.get("confidence", 0)),
        )


if __name__ == "__main__":
    main()