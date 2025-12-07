"""
특징 추출 모듈 (Feature Extractor)

이 파일은 걸음걸이 데이터에서 감정을 예측하는 데 필요한 14가지 특징을 추출합니다.
비유: 사람의 걸음걸이는 마치 지문처럼 고유한 패턴을 가지고 있으며,
      각각의 특징은 그 패턴의 다른 측면을 나타냅니다.
"""

import numpy as np
from typing import List, Dict


def extract_features(keypoints: List[Dict]) -> np.ndarray:
    """
    걸음걸이 키포인트 데이터에서 14가지 특징을 추출합니다.
    
    Args:
        keypoints: 각 프레임의 신체 키포인트 좌표 리스트
                   예) [{"nose": [x, y], "left_shoulder": [x, y], ...}, ...]
    
    Returns:
        14개의 특징값을 담은 numpy 배열
        
    특징 설명 (14가지):
    -------------------------------------------------------------------------
    1. avg_speed (평균 속도): 걸음걸이의 평균 이동 속도
       - 행복할 때: 빠르고 경쾌한 걸음
       - 슬플 때: 느리고 무거운 걸음
       
    2. stride_length (보폭): 한 걸음의 평균 거리
       - 자신감 있을 때: 큰 보폭
       - 불안할 때: 작은 보폭
       
    3. step_frequency (발걸음 빈도): 단위 시간당 걸음 수
       - 급할 때: 빠른 발걸음
       - 여유로울 때: 느린 발걸음
       
    4. body_sway (상체 흔들림): 걸을 때 상체가 좌우로 흔들리는 정도
       - 자연스러울 때: 적당한 흔들림
       - 경직되었을 때: 흔들림이 적음
       
    5. arm_swing (팔 스윙): 걸을 때 팔을 흔드는 정도
       - 편안할 때: 자연스러운 팔 움직임
       - 긴장했을 때: 팔 움직임이 적음
       
    6. head_tilt (머리 기울기): 머리의 평균 기울기 각도
       - 긍정적일 때: 머리를 들고 걷기
       - 부정적일 때: 머리를 숙이고 걷기
       
    7. posture_openness (자세 개방도): 어깨가 펴진 정도
       - 자신감 있을 때: 어깨를 펴고 걷기
       - 위축되었을 때: 어깨가 움츠러짐
       
    8. vertical_bounce (수직 움직임): 걸을 때 상하로 튀는 정도
       - 경쾌할 때: 발랄한 튀는 걸음
       - 무기력할 때: 평평한 걸음
       
    9. foot_drag (발 끌림): 발을 끄는 정도
       - 피곤할 때: 발을 많이 끌음
       - 활기찰 때: 발을 확실히 들어 올림
       
    10. asymmetry (비대칭성): 좌우 걸음의 차이
        - 정상일 때: 대칭적인 걸음
        - 불편할 때: 비대칭적인 걸음
        
    11. step_regularity (걸음 규칙성): 걸음이 일정한 정도
        - 안정적일 때: 규칙적인 걸음
        - 불안할 때: 불규칙한 걸음
        
    12. energy (에너지): 전체적인 움직임의 에너지 수준
        - 활동적일 때: 높은 에너지
        - 무기력할 때: 낮은 에너지
        
    13. gait_phase_duration (걸음 단계 지속시간): 각 걸음 단계가 지속되는 평균 시간
        - 빠를 때: 짧은 지속 시간
        - 느릴 때: 긴 지속 시간
        
    14. center_of_mass_displacement (무게중심 이동): 무게중심의 이동 거리
        - 역동적일 때: 큰 이동
        - 조심스러울 때: 작은 이동
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
    
    # 실제 특징 추출 로직
    # 여기서는 간단한 계산 예시를 보여줍니다
    # 실제 프로젝트에서는 더 정교한 계산이 필요합니다
    
    frames = len(keypoints)
    
    # 1. avg_speed: 전체 이동 거리 / 시간
    # 코의 위치 변화를 기준으로 속도 계산
    if "nose" in keypoints[0] and "nose" in keypoints[-1]:
        start_pos = np.array(keypoints[0]["nose"])
        end_pos = np.array(keypoints[-1]["nose"])
        total_distance = np.linalg.norm(end_pos - start_pos)
        avg_speed = total_distance / frames if frames > 0 else 0.0
    else:
        avg_speed = 1.0
    
    # 2. stride_length: 발목 간 거리의 평균
    stride_lengths = []
    for frame in keypoints:
        if "left_ankle" in frame and "right_ankle" in frame:
            left = np.array(frame["left_ankle"])
            right = np.array(frame["right_ankle"])
            stride_lengths.append(np.linalg.norm(left - right))
    stride_length = np.mean(stride_lengths) if stride_lengths else 0.5
    
    # 3. step_frequency: 발의 위치 변화 빈도
    step_frequency = min(frames / 10.0, 5.0)  # 임의 정규화
    
    # 4. body_sway: 어깨 중심의 x축 변화량 표준편차
    shoulder_centers = []
    for frame in keypoints:
        if "left_shoulder" in frame and "right_shoulder" in frame:
            left = np.array(frame["left_shoulder"])
            right = np.array(frame["right_shoulder"])
            center = (left + right) / 2
            shoulder_centers.append(center[0])
    body_sway = np.std(shoulder_centers) if shoulder_centers else 0.1
    
    # 5. arm_swing: 손목 위치의 변화량
    wrist_movements = []
    for i in range(1, len(keypoints)):
        curr = keypoints[i]
        prev = keypoints[i-1]
        if "left_wrist" in curr and "left_wrist" in prev:
            movement = np.linalg.norm(
                np.array(curr["left_wrist"]) - np.array(prev["left_wrist"])
            )
            wrist_movements.append(movement)
    arm_swing = np.mean(wrist_movements) if wrist_movements else 0.3
    
    # 6. head_tilt: 머리의 y축 위치 평균
    head_tilts = []
    for frame in keypoints:
        if "nose" in frame:
            head_tilts.append(frame["nose"][1])
    head_tilt = np.mean(head_tilts) / 100.0 if head_tilts else 0.0  # 정규화
    
    # 7. posture_openness: 어깨 간 거리
    shoulder_widths = []
    for frame in keypoints:
        if "left_shoulder" in frame and "right_shoulder" in frame:
            width = np.linalg.norm(
                np.array(frame["left_shoulder"]) - np.array(frame["right_shoulder"])
            )
            shoulder_widths.append(width)
    posture_openness = np.mean(shoulder_widths) / 100.0 if shoulder_widths else 0.5
    
    # 8. vertical_bounce: 엉덩이 y축 변화의 표준편차
    hip_heights = []
    for frame in keypoints:
        if "left_hip" in frame and "right_hip" in frame:
            avg_hip_y = (frame["left_hip"][1] + frame["right_hip"][1]) / 2
            hip_heights.append(avg_hip_y)
    vertical_bounce = np.std(hip_heights) / 10.0 if hip_heights else 0.2
    
    # 9. foot_drag: 발목 y축 변화의 최소값 (낮을수록 끌림)
    ankle_heights = []
    for frame in keypoints:
        if "left_ankle" in frame:
            ankle_heights.append(frame["left_ankle"][1])
        if "right_ankle" in frame:
            ankle_heights.append(frame["right_ankle"][1])
    foot_drag = min(ankle_heights) / 100.0 if ankle_heights else 0.1
    
    # 10. asymmetry: 좌우 발목 높이 차이의 표준편차
    ankle_diffs = []
    for frame in keypoints:
        if "left_ankle" in frame and "right_ankle" in frame:
            diff = abs(frame["left_ankle"][1] - frame["right_ankle"][1])
            ankle_diffs.append(diff)
    asymmetry = np.std(ankle_diffs) / 10.0 if ankle_diffs else 0.05
    
    # 11. step_regularity: 걸음 간격의 일관성 (변동계수의 역수)
    if stride_lengths and len(stride_lengths) > 1:
        cv = np.std(stride_lengths) / (np.mean(stride_lengths) + 1e-6)
        step_regularity = max(0, 1 - cv)
    else:
        step_regularity = 0.8
    
    # 12. energy: 전체 움직임의 크기 (모든 키포인트의 변화량 합)
    total_energy = 0.0
    for i in range(1, len(keypoints)):
        frame_energy = 0.0
        for key in keypoints[i]:
            if key in keypoints[i-1]:
                movement = np.linalg.norm(
                    np.array(keypoints[i][key]) - np.array(keypoints[i-1][key])
                )
                frame_energy += movement
        total_energy += frame_energy
    energy = total_energy / (frames * 10) if frames > 0 else 1.0
    
    # 13. gait_phase_duration: 걸음 한 사이클의 평균 시간
    gait_phase_duration = frames / (step_frequency + 1e-6) / 10.0
    
    # 14. center_of_mass_displacement: 전체 무게중심 이동
    com_positions = []
    for frame in keypoints:
        # 주요 관절들의 평균 위치를 무게중심으로 근사
        positions = []
        for key in ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]:
            if key in frame:
                positions.append(frame[key])
        if positions:
            com = np.mean(positions, axis=0)
            com_positions.append(com)
    
    if len(com_positions) > 1:
        com_displacement = np.linalg.norm(com_positions[-1] - com_positions[0]) / frames
    else:
        com_displacement = 0.3
    
    # 14개 특징을 numpy 배열로 반환
    features = np.array([
        avg_speed,
        stride_length,
        step_frequency,
        body_sway,
        arm_swing,
        head_tilt,
        posture_openness,
        vertical_bounce,
        foot_drag,
        asymmetry,
        step_regularity,
        energy,
        gait_phase_duration,
        com_displacement
    ])
    
    return features
