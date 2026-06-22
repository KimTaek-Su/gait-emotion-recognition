# Web Deployment Guide

## Goal

Deploy the FastAPI API and browser demo as one web service that stays available at a single public URL.

## What is already prepared

- The API serves the frontend directly from `/`.
- The browser demo calls the same host by default, so no separate frontend deployment is required.
- The Docker image honors the cloud-provided `PORT` variable.
- `render.yaml` is included for a Render deployment.

## Recommended target

Use a Docker-based host that supports public web services.

- Render Free for test deployment
- Render Starter or above for no idle spin-down
- Any VPS or container host that can run the provided Dockerfile

Free-tier services usually sleep or stop after inactivity, so they do not meet a strict permanent-availability requirement.

## Render deployment steps

1. Push this repository to GitHub.
2. Confirm that `models/deployment/gait_emotion_api_model.joblib` is present in the repository or fetched during build.
3. In Render, create a new Blueprint deployment from the repository.
4. Keep the generated web service settings from `render.yaml`.
5. Wait for the first build to finish.
6. Open the service URL and verify:
   - `/` returns the web UI
   - `/health` returns the health payload
   - `/docs` shows FastAPI Swagger

## Local container verification

```bash
docker-compose up --build
```

Then open `http://localhost:8000/`.

## Operational notes

- The deployed service depends on `models/deployment/gait_emotion_api_model.joblib`.
- The browser webcam flow uses MediaPipe JavaScript from CDN.
- Unknown large artifacts are excluded from the Docker build context through `.dockerignore` to keep cloud builds smaller and faster.
- The Blueprint can provision a Render Postgres instance and inject `DATABASE_URL` into the web service for prediction log storage.

## Prediction log storage

The API can persist one structured log row per `POST /predict_emotion` request.

- The default Blueprint now creates `gait-emotion-recognition-db` and wires its private-network connection string to `DATABASE_URL`.
- On first use, the app creates a `prediction_logs` table automatically.
- Each row stores request metadata, frame/joint counts, prediction result, confidence, latency, and error details.
- Only a preview of incoming keypoints or skeleton data is stored, not the full raw sequence.

If you already deployed the web service before this change, sync the updated Blueprint in Render so the database resource and `DATABASE_URL` binding are created.

## Local helper scripts

Use these wrappers on Windows to avoid repeating environment setup by hand.

- `scripts/start_local_web.cmd`
   - Bootstraps `.venv312` if needed.
   - Installs dependencies from `requirements.txt` if core server packages are missing.
   - Starts `uvicorn src.main:app` on `127.0.0.1:8000`.
- `scripts/start_public_tunnel.cmd`
   - Opens a temporary HTTPS tunnel through `localhost.run`.
   - Useful for trying the app from a phone or another machine without deploying.
- `scripts/render_preflight.cmd`
   - Checks `render.yaml`, `Dockerfile`, `requirements.txt`, the deployment model artifact, and deploy-scope Git state.

## Deploy-scope change summary

The current deployment-oriented changes are:

- Local startup wrapper for Windows PowerShell execution-policy environments.
- Temporary public tunnel wrapper for external device testing.
- Render preflight checker for deployment readiness.
- Webcam startup hardening in the frontend:
   - avoid duplicate camera initialization
   - surface clearer permission and secure-origin errors
   - cache-bust `app.js` so updated browser logic loads reliably

## Post-deploy verification checklist

Run this checklist after the first Render deployment or after any deploy that touches frontend, model loading, or server startup.

1. Open `/health` and confirm the response includes `status: healthy`.
2. Open `/` and confirm the main UI loads without broken scripts or blank sections.
3. Open `/docs` and confirm the FastAPI Swagger page renders.
4. In the web UI, click `샘플 데이터 로드`, then `감정 분석`, and confirm a result card appears instead of an error block.
5. Confirm the prediction response shows confidence and probability bars.
6. Test webcam startup on HTTPS:
    - allow camera permission in the browser
    - confirm the UI transitions from `웹캠이 꺼져 있습니다` to frame collection status
7. If webcam startup fails, verify the browser now shows an inline error instead of only a modal camera failure alert.
8. If prediction fails, inspect service logs first for:
    - missing `gait_emotion_api_model.joblib`
    - dependency import failure
    - request validation errors from `/predict_emotion`

## Quick operator flow

For a safe deployment cycle:

1. Run `scripts/render_preflight.cmd`.
2. Commit and push the deploy-scope changes.
3. Create or update the Render Blueprint deployment.
4. Run the post-deploy verification checklist above.