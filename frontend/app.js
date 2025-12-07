/**
 * 걸음걸이 감정 인식 프론트엔드 JavaScript
 * 
 * API 서버와 통신하여 감정을 예측하고 결과를 표시합니다.
 */

// API 서버 URL (환경에 따라 변경 필요)
const API_URL = 'http://localhost:8000';

// MediaPipe 33개 관절 → 17개 관절 매핑
const MEDIAPIPE_TO_17_JOINTS = [
    0,   // 0: nose
    11,  // 1: left_shoulder
    12,  // 2: right_shoulder
    13,  // 3: left_elbow
    14,  // 4: right_elbow
    15,  // 5: left_wrist
    16,  // 6: right_wrist
    23,  // 7: left_hip
    24,  // 8: right_hip
    25,  // 9: left_knee
    26,  // 10: right_knee
    27,  // 11: left_ankle
    28,  // 12: right_ankle
    5,   // 13: left_eye
    2,   // 14: right_eye
    7,   // 15: left_ear
    8    // 16: right_ear
];

// 프레임 버퍼 (최소 30프레임 = 약 1초)
let skeletonDataBuffer = [];
const MIN_FRAMES = 30;

// MediaPipe Pose 관련 변수
let pose = null;
let camera = null;
let isWebcamActive = false;

/**
 * 샘플 키포인트 데이터 생성
 * 실제로는 비디오 분석이나 센서로부터 얻어진 데이터를 사용합니다.
 */
function loadSampleData() {
    const sampleData = [
        {
            "nose": [320, 100],
            "left_shoulder": [280, 150],
            "right_shoulder": [360, 150],
            "left_elbow": [250, 200],
            "right_elbow": [390, 200],
            "left_wrist": [230, 250],
            "right_wrist": [410, 250],
            "left_hip": [290, 300],
            "right_hip": [350, 300],
            "left_knee": [285, 400],
            "right_knee": [355, 400],
            "left_ankle": [280, 500],
            "right_ankle": [360, 500]
        },
        {
            "nose": [325, 105],
            "left_shoulder": [285, 155],
            "right_shoulder": [365, 155],
            "left_elbow": [255, 205],
            "right_elbow": [395, 205],
            "left_wrist": [235, 255],
            "right_wrist": [415, 255],
            "left_hip": [295, 305],
            "right_hip": [355, 305],
            "left_knee": [290, 405],
            "right_knee": [360, 405],
            "left_ankle": [285, 505],
            "right_ankle": [365, 505]
        },
        {
            "nose": [330, 110],
            "left_shoulder": [290, 160],
            "right_shoulder": [370, 160],
            "left_elbow": [260, 210],
            "right_elbow": [400, 210],
            "left_wrist": [240, 260],
            "right_wrist": [420, 260],
            "left_hip": [300, 310],
            "right_hip": [360, 310],
            "left_knee": [295, 410],
            "right_knee": [365, 410],
            "left_ankle": [290, 510],
            "right_ankle": [370, 510]
        }
    ];
    
    document.getElementById('keypointsInput').value = JSON.stringify(sampleData, null, 2);
}

/**
 * 감정 아이콘 반환 - 6가지 감정 (대소문자 무관)
 */
function getEmotionIcon(emotion) {
    const emotionLower = emotion.toLowerCase();
    const icons = {
        'happy': '😊',
        'sad': '😢',
        'fear': '😨',
        'disgust': '🤢',
        'angry': '😠',
        'neutral': '😐'
    };
    return icons[emotionLower] || '😐';
}

/**
 * 감정 레이블 한글 변환 - 6가지 감정
 */
function getEmotionLabel(emotion) {
    const emotionLower = emotion.toLowerCase();
    const labels = {
        'happy': '행복',
        'sad': '슬픔',
        'fear': '공포',
        'disgust': '혐오',
        'angry': '분노',
        'neutral': '중립'
    };
    return labels[emotionLower] || emotion;
}

/**
 * 신뢰도 수준 한글 변환
 */
function getConfidenceLevelLabel(level) {
    const labels = {
        'high': '높음',
        'medium': '보통',
        'low': '낮음'
    };
    return labels[level] || level;
}

/**
 * 결과 표시
 */
function displayResult(data) {
    const resultSection = document.getElementById('resultSection');
    
    // 감정 결과
    let html = `
        <div class="emotion-result">
            <div class="emotion-icon">${getEmotionIcon(data.emotion)}</div>
            <div class="emotion-label">${getEmotionLabel(data.emotion)}</div>
            <div class="confidence">
                신뢰도: ${(data.confidence * 100).toFixed(1)}%
                <span class="confidence-level ${data.confidence_level}">
                    ${getConfidenceLevelLabel(data.confidence_level)}
                </span>
            </div>
        </div>
    `;
    
    // 확률 분포
    if (data.probabilities) {
        html += `
            <div class="probabilities">
                <h3>감정별 확률 분포</h3>
        `;
        
        // 확률 내림차순 정렬
        const sortedProbs = Object.entries(data.probabilities)
            .sort((a, b) => b[1] - a[1]);
        
        sortedProbs.forEach(([emotion, prob]) => {
            const percentage = (prob * 100).toFixed(1);
            html += `
                <div class="prob-bar">
                    <div class="prob-label">
                        <span>${getEmotionIcon(emotion)} ${getEmotionLabel(emotion)}</span>
                        <span>${percentage}%</span>
                    </div>
                    <div class="prob-bar-container">
                        <div class="prob-bar-fill" style="width: ${percentage}%">
                            ${percentage}%
                        </div>
                    </div>
                </div>
            `;
        });
        
        html += `</div>`;
    }
    
    // 경고 메시지
    if (data.warning) {
        html += `<div class="warning">${data.warning}</div>`;
    }
    
    resultSection.innerHTML = html;
    resultSection.classList.add('show');
}

/**
 * 오류 표시
 */
function displayError(message) {
    const resultSection = document.getElementById('resultSection');
    resultSection.innerHTML = `<div class="error">${message}</div>`;
    resultSection.classList.add('show');
}

/**
 * 로딩 표시
 */
function showLoading() {
    const resultSection = document.getElementById('resultSection');
    resultSection.innerHTML = '<div class="loading">감정을 분석하고 있습니다</div>';
    resultSection.classList.add('show');
}

/**
 * 감정 예측 API 호출
 */
async function predictEmotion() {
    const input = document.getElementById('keypointsInput').value.trim();
    const predictBtn = document.getElementById('predictBtn');
    
    // 입력 검증
    if (!input) {
        displayError('키포인트 데이터를 입력해주세요.');
        return;
    }
    
    // JSON 파싱 검증
    let keypoints;
    try {
        keypoints = JSON.parse(input);
    } catch (e) {
        displayError('올바른 JSON 형식이 아닙니다. 형식을 확인해주세요.');
        return;
    }
    
    // 배열 검증
    if (!Array.isArray(keypoints) || keypoints.length < 2) {
        displayError('최소 2개 이상의 프레임 데이터가 필요합니다.');
        return;
    }
    
    // 버튼 비활성화 및 로딩 표시
    predictBtn.disabled = true;
    showLoading();
    
    try {
        // API 호출
        const response = await fetch(`${API_URL}/predict_emotion`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ keypoints: keypoints })
        });
        
        // 응답 처리
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '서버 오류가 발생했습니다.');
        }
        
        const data = await response.json();
        displayResult(data);
        
    } catch (error) {
        console.error('Error:', error);
        displayError(`오류가 발생했습니다: ${error.message}`);
    } finally {
        // 버튼 다시 활성화
        predictBtn.disabled = false;
    }
}

/**
 * Enter 키로 예측 실행
 */
document.addEventListener('DOMContentLoaded', function() {
    const textarea = document.getElementById('keypointsInput');
    textarea.addEventListener('keydown', function(e) {
        // Ctrl + Enter로 예측 실행
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            predictEmotion();
        }
    });
});

/**
 * API 서버 연결 테스트
 */
async function testConnection() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            console.log('✅ API 서버 연결 성공');
        } else {
            console.warn('⚠️ API 서버 응답 이상');
        }
    } catch (error) {
        console.error('❌ API 서버 연결 실패:', error.message);
        console.log('API URL을 확인하세요:', API_URL);
    }
}

// 페이지 로드 시 연결 테스트
testConnection();

/**
 * MediaPipe 결과를 서버 형식으로 변환
 */
function convertToServerFormat(poseLandmarks) {
    const skeleton_data = [];
    
    for (const mpIndex of MEDIAPIPE_TO_17_JOINTS) {
        const landmark = poseLandmarks[mpIndex];
        if (landmark) {
            skeleton_data.push(`${landmark.x},${landmark.y},${landmark.z}`);
        } else {
            skeleton_data.push("0.0,0.0,0.0");
        }
    }
    
    return skeleton_data;
}

/**
 * 웹캠 시작
 */
async function startWebcam() {
    if (isWebcamActive) {
        console.log('웹캠이 이미 실행 중입니다.');
        return;
    }

    try {
        // MediaPipe Pose 초기화
        if (!pose) {
            pose = new Pose({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
                }
            });
            
            pose.setOptions({
                modelComplexity: 1,
                smoothLandmarks: true,
                enableSegmentation: false,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });
            
            pose.onResults(onPoseResults);
        }

        // 웹캠 스트림 가져오기
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        
        const videoElement = document.getElementById('webcam');
        videoElement.srcObject = stream;
        
        // 카메라 초기화
        camera = new Camera(videoElement, {
            onFrame: async () => {
                await pose.send({ image: videoElement });
            },
            width: 640,
            height: 480
        });
        
        await camera.start();
        
        // UI 업데이트
        isWebcamActive = true;
        skeletonDataBuffer = [];
        document.getElementById('videoContainer').style.display = 'block';
        document.getElementById('webcamStatus').textContent = '🟢 웹캠 실행 중 - 프레임 수집: 0';
        document.getElementById('webcamStatus').className = 'webcam-status active';
        document.getElementById('startWebcamBtn').disabled = true;
        document.getElementById('stopWebcamBtn').disabled = false;
        document.getElementById('analyzeWebcamBtn').disabled = false;
        
        console.log('✅ 웹캠 시작 성공');
    } catch (error) {
        console.error('❌ 웹캠 시작 실패:', error);
        displayError(`웹캠 시작 실패: ${error.message}`);
    }
}

/**
 * 웹캠 중지
 */
function stopWebcam() {
    if (camera) {
        camera.stop();
        camera = null;
    }
    
    const videoElement = document.getElementById('webcam');
    if (videoElement.srcObject) {
        videoElement.srcObject.getTracks().forEach(track => track.stop());
        videoElement.srcObject = null;
    }
    
    isWebcamActive = false;
    document.getElementById('videoContainer').style.display = 'none';
    document.getElementById('webcamStatus').textContent = '웹캠이 꺼져 있습니다';
    document.getElementById('webcamStatus').className = 'webcam-status';
    document.getElementById('startWebcamBtn').disabled = false;
    document.getElementById('stopWebcamBtn').disabled = true;
    document.getElementById('analyzeWebcamBtn').disabled = true;
    
    console.log('웹캠 중지');
}

/**
 * MediaPipe Pose 결과 처리
 */
function onPoseResults(results) {
    if (!results.poseLandmarks) {
        return;
    }
    
    // 캔버스에 포즈 그리기
    const canvasElement = document.getElementById('output_canvas');
    const videoElement = document.getElementById('webcam');
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
    
    const canvasCtx = canvasElement.getContext('2d');
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // 포즈 그리기
    drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {
        color: '#00FF00',
        lineWidth: 4
    });
    drawLandmarks(canvasCtx, results.poseLandmarks, {
        color: '#FF0000',
        lineWidth: 2
    });
    
    canvasCtx.restore();
    
    // 스켈레톤 데이터를 버퍼에 추가
    const skeleton_data = convertToServerFormat(results.poseLandmarks);
    skeletonDataBuffer.push(skeleton_data);
    
    // 버퍼 크기 제한 (최대 300프레임 = 약 10초)
    if (skeletonDataBuffer.length > 300) {
        skeletonDataBuffer.shift();
    }
    
    // 상태 업데이트
    const status = document.getElementById('webcamStatus');
    if (skeletonDataBuffer.length >= MIN_FRAMES) {
        status.textContent = `🔴 수집 완료 - 프레임: ${skeletonDataBuffer.length}개 (분석 가능)`;
        status.className = 'webcam-status recording';
    } else {
        status.textContent = `🟡 프레임 수집 중: ${skeletonDataBuffer.length}/${MIN_FRAMES}`;
        status.className = 'webcam-status active';
    }
}

/**
 * 웹캠에서 감정 분석
 */
async function analyzeFromWebcam() {
    if (skeletonDataBuffer.length < MIN_FRAMES) {
        displayError(`최소 ${MIN_FRAMES}개 프레임이 필요합니다. 현재: ${skeletonDataBuffer.length}개`);
        return;
    }
    
    // 모든 프레임의 skeleton_data를 하나의 배열로 합침
    const allSkeletonData = skeletonDataBuffer.flat();
    
    console.log(`분석 시작: ${skeletonDataBuffer.length}개 프레임, ${allSkeletonData.length}개 좌표`);
    
    try {
        showLoading();
        
        const response = await fetch(`${API_URL}/predict_emotion`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                skeleton_data: allSkeletonData,
                n_joints: 17
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '서버 오류');
        }
        
        const data = await response.json();
        displayResult(data);
        
        console.log('✅ 분석 완료:', data);
        
    } catch (error) {
        console.error('❌ 분석 실패:', error);
        displayError(`오류: ${error.message}`);
    }
}
