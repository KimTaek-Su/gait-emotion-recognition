"""
특징 추출 모듈 (Feature Extractor)

이 파일은 걸음걸이 데이터에서 감정을 예측하는 데 필요한 14가지 물리 기반 특징을 추출합니다.
"""

import numpy as np
from typing import List, Dict, Union


def parse_skeleton_data(skeleton_data: List[str], n_joints: int = 17) -> np.ndarray:
    """
    문자열 배열을 3D numpy 배열로 변환
    
    Args:
        skeleton_data: ["x,y,z", "x,y,z", ...] 형식의 문자열 리스트
        n_joints: 관절 개수 (기본값: 17)
    
    Returns:
        (n_frames, n_joints, 3) 형태의 numpy 배열
    
    Raises:
        ValueError: 잘못된 데이터 형식인 경우
    """
    coords = []
    for i, s in enumerate(skeleton_data):
        try:
            parts = s.split(',')
            if len(parts) != 3:
                raise ValueError(f"좌표 {i}에 3개의 값이 필요합니다. 현재: {len(parts)}개")
            coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except (ValueError, IndexError) as e:
            raise ValueError(f"좌표 {i} 파싱 실패: {s}. 오류: {str(e)}")
    
    coords = np.array(coords)
    n_frames = len(coords) // n_joints
    if n_frames == 0:
        raise ValueError(f"데이터가 부족합니다. 최소 {n_joints}개의 좌표가 필요합니다.")
    return coords.reshape(n_frames, n_joints, 3)


def root_centering(data: np.ndarray) -> np.ndarray:
    """
    루트 센터링: 0번 관절(코/머리)을 기준으로 좌표 정규화
    
    Args:
        data: (n_frames, n_joints, 3) 형태의 numpy 배열
    
    Returns:
        루트 센터링된 (n_frames, n_joints, 3) 형태의 numpy 배열
    """
    root = data[:, 0:1, :]  # (n_frames, 1, 3)
    return data - root


def ensure_min_frames(data: np.ndarray, min_frames: int = 4) -> np.ndarray:
    """
    최소 프레임 수 확보 (패딩)
    
    Args:
        data: (n_frames, n_joints, 3) 형태의 numpy 배열
        min_frames: 최소 프레임 수
    
    Returns:
        최소 프레임 수를 만족하는 (n_frames, n_joints, 3) 형태의 numpy 배열
    """
    n_frames = data.shape[0]
    if n_frames >= min_frames:
        return data
    
    # 첫 번째 프레임으로 패딩
    padding = np.tile(data[0:1], (min_frames - n_frames, 1, 1))
    return np.concatenate([padding, data], axis=0)


def extract_features_from_skeleton(skeleton_data: List[str], n_joints: int = 17) -> np.ndarray:
    """
    스켈레톤 데이터에서 14개 물리 기반 특징 추출
    
    14가지 특징:
    A. Kinematics (5개):
       [0] 평균 속도 (mean velocity magnitude)
       [1] 최대 속도 (max velocity magnitude)
       [2] 속도 표준편차 (velocity std)
       [3] 평균 가속도 (mean acceleration)
       [4] 평균 저크 (mean jerk)
    
    B. Body parts (3개):
       [5] 손 움직임 (hand movement - joints 5, 6: wrists)
       [6] 발 움직임 (foot movement - joints 11, 12: ankles)
       [7] 좌우 대칭성 (left-right symmetry)
    
    C. Volume (1개):
       [8] 바운딩 박스 부피 평균
    
    D. Posture (2개):
       [9] 머리 기울기 (head tilt)
       [10] 척추 길이 (spine length)
    
    E. Joint variance (3개):
       [11-13] 상위 3개 관절의 분산
    
    Args:
        skeleton_data: ["x,y,z", ...] 형식의 문자열 리스트
        n_joints: 관절 개수 (기본값: 17)
    
    Returns:
        14개의 특징값을 담은 numpy 배열
    """
    # 1. 데이터 파싱
    data = parse_skeleton_data(skeleton_data, n_joints)
    
    # 2. 루트 센터링
    data = root_centering(data)
    
    # 3. 최소 4프레임 확보
    data = ensure_min_frames(data, min_frames=4)
    
    n_frames = data.shape[0]
    
    # 4. 속도 계산 (velocity = 위치의 1차 미분)
    velocity = np.diff(data, axis=0)  # (n_frames-1, n_joints, 3)
    vel_magnitude = np.linalg.norm(velocity, axis=2)  # (n_frames-1, n_joints)
    
    # 5. 가속도 계산 (acceleration = 속도의 1차 미분)
    acceleration = np.diff(velocity, axis=0)  # (n_frames-2, n_joints, 3)
    acc_magnitude = np.linalg.norm(acceleration, axis=2)
    
    # 6. 저크 계산 (jerk = 가속도의 1차 미분)
    jerk = np.diff(acceleration, axis=0)  # (n_frames-3, n_joints, 3)
    jerk_magnitude = np.linalg.norm(jerk, axis=2)
    
    # 특징 추출
    features = np.zeros(14)
    
    # A. Kinematics (5개)
    features[0] = np.mean(vel_magnitude)  # 평균 속도
    features[1] = np.max(vel_magnitude)   # 최대 속도
    features[2] = np.std(vel_magnitude)   # 속도 표준편차
    features[3] = np.mean(acc_magnitude) if acc_magnitude.size > 0 else 0  # 평균 가속도
    features[4] = np.mean(jerk_magnitude) if jerk_magnitude.size > 0 else 0  # 평균 저크
    
    # B. Body parts (3개)
    # 손 움직임 (관절 5, 6: 왼쪽/오른쪽 손목)
    if n_joints > 6:
        hand_joints = [5, 6]
        features[5] = np.mean(vel_magnitude[:, hand_joints])
    elif n_joints > 1:
        features[5] = np.mean(vel_magnitude[:, -1])  # 마지막 관절 사용
    else:
        features[5] = 0.0
    
    # 발 움직임 (관절 11, 12: 왼쪽/오른쪽 발목)
    if n_joints > 12:
        foot_joints = [11, 12]
        features[6] = np.mean(vel_magnitude[:, foot_joints])
    elif n_joints > 1:
        features[6] = np.mean(vel_magnitude[:, -1])  # 마지막 관절 사용
    else:
        features[6] = 0.0
    
    # 좌우 대칭성 (왼손 vs 오른손 속도 차이)
    if n_joints > 6:
        left_hand_vel = np.mean(vel_magnitude[:, 5])
        right_hand_vel = np.mean(vel_magnitude[:, 6])
        features[7] = abs(left_hand_vel - right_hand_vel)
    else:
        features[7] = 0
    
    # C. Volume (1개)
    # 바운딩 박스 부피
    bbox_volumes = []
    for frame in data:
        x_range = np.max(frame[:, 0]) - np.min(frame[:, 0])
        y_range = np.max(frame[:, 1]) - np.min(frame[:, 1])
        z_range = np.max(frame[:, 2]) - np.min(frame[:, 2])
        bbox_volumes.append(x_range * y_range * max(z_range, 0.001))
    features[8] = np.mean(bbox_volumes)
    
    # D. Posture (2개)
    # 머리 기울기 (0번 관절 y좌표 - 1번 관절 y좌표)
    if n_joints > 1:
        features[9] = np.mean(data[:, 0, 1] - data[:, 1, 1])
    
    # 척추 길이 (어깨 중심 - 엉덩이 중심 거리)
    if n_joints > 8:
        shoulder_center = (data[:, 1, :] + data[:, 2, :]) / 2  # 관절 1, 2: 어깨
        hip_center = (data[:, 7, :] + data[:, 8, :]) / 2       # 관절 7, 8: 엉덩이
        spine_length = np.linalg.norm(shoulder_center - hip_center, axis=1)
        features[10] = np.mean(spine_length)
    
    # E. Joint variance (3개)
    # 각 관절의 움직임 분산, 상위 3개
    joint_variances = np.var(vel_magnitude, axis=0)  # (n_joints,)
    top3_indices = np.argsort(joint_variances)[-3:]
    features[11:14] = joint_variances[top3_indices]
    
    return features


def extract_features(keypoints: List[Dict]) -> np.ndarray:
    """
    딕셔너리 형식의 키포인트에서 특징 추출
    기존 호환성을 위해 유지
    
    Args:
        keypoints: 각 프레임의 신체 키포인트 좌표 리스트
                   예) [{"nose": [x, y], "left_shoulder": [x, y], ...}, ...]
    
    Returns:
        14개의 특징값을 담은 numpy 배열
    """
    if not keypoints or len(keypoints) < 2:
        # 데이터가 부족할 경우 기본값 반환 (중립적인 걸음걸이)
        return np.array([
            1.0,   # avg_speed
            0.5,   # stride_length
            2.0,   # step_frequency
            0.1,   # body_sway
            0.3,   # arm_swing
            0.0,   # head_tilt
            0.5,   # posture_openness
            0.2,   # vertical_bounce
            0.1,   # foot_drag
            0.05,  # asymmetry
            0.8,   # step_regularity
            1.0,   # energy
            0.5,   # gait_phase_duration
            0.3    # center_of_mass_displacement
        ])
    
    # 딕셔너리를 skeleton_data 형식으로 변환
    joint_names = [
        "nose", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle"
    ]
    
    skeleton_data = []
    for frame in keypoints:
        for joint_name in joint_names:
            if joint_name in frame and frame[joint_name]:
                coords = frame[joint_name]
                x, y = coords[0], coords[1]
                z = coords[2] if len(coords) > 2 else 0.0
                skeleton_data.append(f"{x},{y},{z}")
            else:
                skeleton_data.append("0.0,0.0,0.0")
    
    return extract_features_from_skeleton(skeleton_data, n_joints=len(joint_names))
