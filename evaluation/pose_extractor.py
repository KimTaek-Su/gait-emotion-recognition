import cv2
import numpy as np

def extract_pose_sequence(video_path, max_frames=300):
    """
    Point-Light Display 영상에서 밝은 점들의 (X, Y) 좌표를 추출합니다.
    """
    cap = cv2.VideoCapture(video_path)
    sequence = []
    
    while cap.isOpened() and len(sequence) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. 영상을 흑백으로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. 이진화 (밝기 200 이상인 픽셀만 하얀색으로)
        # ※ 영상의 밝기에 따라 200이라는 숫자를 150이나 220으로 조절해야 할 수 있습니다.
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # 3. 윤곽선(점들) 찾기
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        frame_points = []
        for cnt in contours:
            # 너무 작은 노이즈(먼지)는 제외 (면적 기준)
            if cv2.contourArea(cnt) < 2: 
                continue
                
            # 4. 점의 무게중심(Center X, Center Y) 계산
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                frame_points.append([cX, cY])
        
        # 5. 점들을 Y좌표(위에서 아래) 기준으로 정렬하여 순서 일관성 부여
        if len(frame_points) > 0:
            frame_points = sorted(frame_points, key=lambda p: p[1])
            sequence.append(frame_points)
        else:
            # 점이 하나도 안 잡힌 프레임은 빈 리스트 추가 (또는 이전 프레임 복사)
            sequence.append([])

    cap.release()
    
    # 추출된 좌표 시퀀스를 반환 (결과 형태: [프레임수, 점의 개수, 2])
    return sequence

# 테스트용 코드 (직접 실행 시 동작 확인)
if __name__ == "__main__":
    # 영상 경로 하나를 넣어 직접 점이 잘 뽑히는지 테스트해보세요.
    test_video = r"D:\datasets\gait_eval_avi\videos\Female Synthetic Walker- Anger High intensity.avi"
    seq = extract_pose_sequence(test_video)
    print(f"총 {len(seq)} 프레임 추출 완료.")
    if len(seq) > 0:
        print(f"첫 프레임의 점 개수: {len(seq[0])}개")
        print(f"첫 프레임 좌표들: {seq[0]}")