/**
 * 걸음걸이 감정 인식 프론트엔드 JavaScript (최신 개선 버전)
 * 다양한 keypoints 구조를 서버 요구대로 자동 변환 + 예외·안내 강화
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
 * 사용자 샘플 데이터: 최소 2프레임, 관절별 [x, y]만 있어도 자동 보완
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
 * 입력 keypoints 구조(딕셔너리배열, [x, y], [x, y, z]) → [[x,y,z], ...]로 일반화 변환
 */
function parseKeypointsForServer(origKeypoints) {
    // 1. [{joint:[x,y]}...] → [[x,y,z], ...] : frame 단위로 joint 하나씩 평탄화
    if (
        Array.isArray(origKeypoints) &&
        typeof origKeypoints[0] === "object" &&
        origKeypoints[0] !== null &&
        !Array.isArray(origKeypoints[0])
    ) {
        let out = [];
        for (const frame of origKeypoints) {
            for (const key in frame) {
                let kp = frame[key];
                if (Array.isArray(kp)) {
                    if (kp.length === 2) out.push([kp[0], kp[1], 0.0]);
                    else if (kp.length === 3) out.push(kp);
                }
            }
        }
        return out;
    }
    // 2. [[x, y], ... ] → [[x, y, 0.0], ... ]
    if (
        Array.isArray(origKeypoints) &&
        Array.isArray(origKeypoints[0]) &&
        origKeypoints[0].length === 2
    ) {
        return origKeypoints.map(kp => [kp[0], kp[1], 0.0]);
    }
    // 3. 이미 [[x,y,z], ...] 인 경우 → pass
    if (
        Array.isArray(origKeypoints) &&
        Array.isArray(origKeypoints[0]) &&
        origKeypoints[0].length === 3
    ) {
        return origKeypoints;
    }
    // 4. 오류 (지원X)
    return null;
}

// 감정 결과/에러 안내 등 출력 유틸리티
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

function updateModelStatus(model) {
    const statusEl = document.getElementById('modelStatus');
    if (!statusEl) return;

    statusEl.className = 'model-status neutral';
    if (!model || typeof model !== 'object') {
        statusEl.textContent = '모델 상태 정보가 아직 제공되지 않았습니다.';
        return;
    }

    const mode = typeof model.mode === 'string' ? model.mode : 'unknown';
    const source = model.source || 'unknown';

    if (mode === 'trained') {
        statusEl.className = 'model-status trained';
        statusEl.textContent = `✅ 현재 모델: trained (source: ${source})`;
        return;
    }

    if (mode === 'fallback') {
        statusEl.className = 'model-status fallback';
        statusEl.textContent = `⚠️ 현재 모델: fallback/demo (source: ${source})`;
        return;
    }

    statusEl.textContent = `현재 모델 모드를 확인할 수 없습니다 (source: ${source})`;
}

/**
 * 입력 검증/변환 및 감정 예측 API 호출 (textarea)
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
        displayError('올바른 JSON 형식이 아닙니다. 샘플 버튼을 눌러 예시를 참고하세요.'); return;
    }
    const parsedKeypoints = parseKeypointsForServer(keypoints);
    if (!Array.isArray(parsedKeypoints) || parsedKeypoints.length < 2) {
        displayError('최소 2프레임 이상의 데이터가 필요합니다. 샘플 데이터를 확인하세요.'); return;
    }
    predictBtn.disabled = true;
    showLoading();
    try {
        const response = await fetch(`${API_URL}/predict_emotion`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ keypoints: parsedKeypoints })
        });
        if (!response.ok) {
            const errorData = await response.json();
            let msg = '서버 오류가 발생했습니다.';
            if (errorData && errorData.detail) {
                msg = typeof errorData.detail === 'string'
                    ? errorData.detail
                    : JSON.stringify(errorData.detail);
            }
            throw new Error(msg);
        }
        const data = await response.json();
        displayResult(data);
        updateModelStatus(data.model);
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
            const health = await response.json();
            updateModelStatus(health.model);
        } else {
            console.warn('⚠️ API 서버 응답 이상');
            updateModelStatus(null);
        }
    } catch (error) {
        console.error('❌ API 서버 연결 실패:', error.message);
        console.log('API URL을 확인하세요:', API_URL);
        updateModelStatus(null);
    }
}
testConnection();

/**
 * skeleton_data 변환 및 웹캠 분석 루틴(원본 유지)
 */

// 17개만 골라내지 말고 전체를 보내도록 수정(테스트용)
const ALL_33_JOINTS = Array.from({length:33}, (_, i) => i)

function convertToServerFormat(poseLandmarks) {
    if (!poseLandmarks || !Array.isArray(poseLandmarks)) {
        console.warn('Invalid poseLandmarks:', poseLandmarks); return null;
    }
    if (poseLandmarks.length < 33) {
        console.warn(`Not enough landmarks. Expected 33, got ${poseLandmarks.length}`); return null;
    }
    const skeleton_data = [];
    for (const mpIndex of ALL_33_JOINTS) { // 17개 대신 33개 사용
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
    // 1. 버퍼에 쌓인 모든 관절 데이터를 하나로 펼침
    let allSkeletonData = skeletonDataBuffer.flat();

    // 2. 수집 단계에서 33개 관절을 사용하므로 서버에도 같은 개수로 전송
    const numJoints = 33;
    const remainder = allSkeletonData.length % numJoints;
    if (remainder !== 0) {
        console.warn(`데이터 불일치 발생: ${remainder}개의 관절 데이터를 제외합니다.`);
        allSkeletonData = allSkeletonData.slice(0, allSkeletonData.length - remainder);
    }

    // 3. 서버가 원하는 ["x,y,z", "x,y,z"] 형식으로 변환
    const formattedData = allSkeletonData.map(joint => {
        return Array.isArray(joint) ? joint.join(',') : joint;
    });
    

    try {
        // Use the collected skeleton data for analysis
        const response = await fetch(`${API_URL}/predict_emotion`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ skeleton_data: formattedData, n_joints: numJoints })
        });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '서버 오류');
        }
        const data = await response.json();
        displayResult(data);
        updateModelStatus(data.model);
        console.log('✅ 분석 완료');
    } catch (error) {
        console.error('❌ 분석 실패:', error);
        displayError(`오류: ${error.message}`);
    }
}
