/**
 * 걸음걸이 감정 인식 프론트엔드 JavaScript
 * (백엔드 요구에 맞게 keypoints 구조 자동 변환 지원)
 */

const API_URL = 'http://localhost:8000';

const MEDIAPIPE_TO_17_JOINTS = [
    0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 5, 2, 7, 8
];

let skeletonDataBuffer = [];
const MIN_FRAMES = 30;

let pose = null;
let camera = null;
let isWebcamActive = false;

/**
 * 샘플 키포인트 데이터 (딕셔너리 배열)
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
        }
    ];
    document.getElementById('keypointsInput').value = JSON.stringify(sampleData, null, 2);
}

/**
 * 다양한 입력 형식(keypoints: 딕셔너리배열 또는 좌표배열) → 서버 요구 [ [x, y, z] ... ] 형태로 변환
 * - (딕셔너리 값이 [x, y]처럼 z없어도 지원)
 */
function parseKeypointsForServer(origKeypoints) {
    if (
        Array.isArray(origKeypoints) &&
        typeof origKeypoints[0] === "object" &&
        origKeypoints[0] !== null &&
        !Array.isArray(origKeypoints[0])
    ) {
        // [ {nose: [...], ...}, ... ] 형태
        let out = [];
        for (const frame of origKeypoints) {
            for (const key in frame) {
                let kp = frame[key];
                // [x, y] → [x, y, 0.0] 보정 (백엔드는 z포함 3차원 좌표를 기대할 수도 있음)
                if (Array.isArray(kp) && kp.length === 2) {
                    out.push([kp[0], kp[1], 0.0]);
                } else if (Array.isArray(kp) && kp.length === 3) {
                    out.push(kp);
                }
            }
        }
        return out;
    }
    // 한 번 더: 좌표배열이고 [x, y]만 있는 경우, 전부 z = 0.0을 추가해줌
    if (
        Array.isArray(origKeypoints) &&
        Array.isArray(origKeypoints[0]) &&
        origKeypoints[0].length === 2
    ) {
        return origKeypoints.map(kp => [kp[0], kp[1], 0.0]);
    }
    // [ [x, y, z], ... ] 혹은 비슷한 형태면 그대로 반환
    return origKeypoints;
}

/**
 * 감정 예측 결과 출력 등(기존 코드 동일)
 */
function getEmotionIcon(emotion) {
    const icons = { happy:'😊', sad:'😢', fear:'😨', disgust:'🤢', angry:'😠', neutral:'😐' };
    return icons[emotion?.toLowerCase()] || '😐';
}
function getEmotionLabel(emotion) {
    const labels = { happy:'행복', sad:'슬픔', fear:'공포', disgust:'혐오', angry:'분노', neutral:'중립' };
    return labels[emotion?.toLowerCase()] || emotion;
}
function getConfidenceLevelLabel(level) {
    const labels = { high:'높음', medium:'보통', low:'낮음' };
    return labels[level] || level;
}
function displayResult(data) {
    const resultSection = document.getElementById('resultSection');
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
    if (data.probabilities) {
        html += `<div class="probabilities"><h3>감정별 확률 분포</h3>`;
        Object.entries(data.probabilities)
            .sort((a, b) => b[1] - a[1]).forEach(([emotion, prob]) => {
            const percentage = (prob * 100).toFixed(1);
            html += `
            <div class="prob-bar">
                <div class="prob-label">
                    <span>${getEmotionIcon(emotion)} ${getEmotionLabel(emotion)}</span>
                    <span>${percentage}%</span>
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill" style="width: ${percentage}%">${percentage}%</div>
                </div>
            </div>
            `;
        });
        html += `</div>`;
    }
    if (data.warning) html += `<div class="warning">${data.warning}</div>`;
    resultSection.innerHTML = html;
    resultSection.classList.add('show');
}
function displayError(message) {
    const resultSection = document.getElementById('resultSection');
    resultSection.innerHTML = `<div class="error">${message}</div>`;
    resultSection.classList.add('show');
}
function showLoading() {
    const resultSection = document.getElementById('resultSection');
    resultSection.innerHTML = '<div class="loading">감정을 분석하고 있습니다</div>';
    resultSection.classList.add('show');
}

/**
 * 감정 예측 API 호출 (textarea)
 */
async function predictEmotion() {
    const input = document.getElementById('keypointsInput').value.trim();
    const predictBtn = document.getElementById('predictBtn');
    if (!input) {
        displayError('키포인트 데이터를 입력해주세요.'); return;
    }
    let keypoints;
    try {
        keypoints = JSON.parse(input);
    } catch (e) {
        displayError('올바른 JSON 형식이 아닙니다. 형식을 확인해주세요.'); return;
    }
    keypoints = parseKeypointsForServer(keypoints);
    if (!Array.isArray(keypoints) || keypoints.length < 2) {
        displayError('최소 2개 이상의 좌표 배열이 필요합니다.'); return;
    }
    predictBtn.disabled = true;
    showLoading();
    try {
        const response = await fetch(`${API_URL}/predict_emotion`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ keypoints })
        });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '서버 오류가 발생했습니다.');
        }
        displayResult(await response.json());
    } catch (error) {
        console.error('Error:', error);
        displayError(`오류가 발생했습니다: ${error.message}`);
    } finally {
        predictBtn.disabled = false;
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const textarea = document.getElementById('keypointsInput');
    textarea.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault(); predictEmotion();
        }
    });
});
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
testConnection();

/**
 * skeleton_data 변환 및 웹캠 지원 부분(원본 유지)
 */
function convertToServerFormat(poseLandmarks) {
    if (!poseLandmarks || !Array.isArray(poseLandmarks)) {
        console.warn('Invalid poseLandmarks:', poseLandmarks); return null;
    }
    if (poseLandmarks.length < 33) {
        console.warn(`Not enough landmarks. Expected 33, got ${poseLandmarks.length}`); return null;
    }
    const skeleton_data = [];
    for (const mpIndex of MEDIAPIPE_TO_17_JOINTS) {
        const landmark = poseLandmarks[mpIndex];
        if (landmark && typeof landmark.x === 'number' && typeof landmark.y === 'number' && typeof landmark.z === 'number') {
            skeleton_data.push(`${landmark.x},${landmark.y},${landmark.z}`);
        } else {
            skeleton_data.push("0.0,0.0,0.0");
        }
    }
    return skeleton_data;
}
async function startWebcam() {
    if (isWebcamActive) { console.log('웹캠이 이미 실행 중입니다.'); return; }
    try {
        if (!pose) {
            pose = new Pose({
                locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
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
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        const videoElement = document.getElementById('webcam');
        videoElement.srcObject = stream;
        camera = new Camera(videoElement, {
            onFrame: async () => { await pose.send({ image: videoElement }); },
            width: 640,
            height: 480
        });
        await camera.start();
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
function stopWebcam() {
    if (camera) { camera.stop(); camera = null; }
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
function onPoseResults(results) {
    if (!results || !results.poseLandmarks || !Array.isArray(results.poseLandmarks)) {
        console.warn('Invalid pose results:', results); return;
    }
    const canvasElement = document.getElementById('output_canvas');
    const videoElement = document.getElementById('webcam');
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
    const canvasCtx = canvasElement.getContext('2d');
    canvasCtx.save(); canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 4 });
    drawLandmarks(canvasCtx, results.poseLandmarks, { color: '#FF0000', lineWidth: 2 });
    canvasCtx.restore();
    const skeleton_data = convertToServerFormat(results.poseLandmarks);
    if (skeleton_data) skeletonDataBuffer.push(skeleton_data);
    else { console.warn('Failed to convert pose landmarks to skeleton data'); return; }
    if (skeletonDataBuffer.length > 300) skeletonDataBuffer.shift();
    const status = document.getElementById('webcamStatus');
    if (skeletonDataBuffer.length >= MIN_FRAMES) {
        status.textContent = `🔴 수집 완료 - 프레임: ${skeletonDataBuffer.length}개 (분석 가능)`;
        status.className = 'webcam-status recording';
    } else {
        status.textContent = `🟡 프레임 수집 중: ${skeletonDataBuffer.length}/${MIN_FRAMES}`;
        status.className = 'webcam-status active';
    }
}
async function analyzeFromWebcam() {
    if (skeletonDataBuffer.length < MIN_FRAMES) {
        displayError(`최소 ${MIN_FRAMES}개 프레임이 필요합니다. 현재: ${skeletonDataBuffer.length}개`); return;
    }
    const allSkeletonData = skeletonDataBuffer.flat();
    console.log(`분석 시작: ${skeletonDataBuffer.length}개 프레임, ${allSkeletonData.length}개 좌표`);
    try {
        showLoading();
        const response = await fetch(`${API_URL}/predict_emotion`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ skeleton_data: allSkeletonData, n_joints: 17 })
        });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '서버 오류');
        }
        displayResult(await response.json());
        console.log('✅ 분석 완료');
    } catch (error) {
        console.error('❌ 분석 실패:', error);
        displayError(`오류: ${error.message}`);
    }
}
