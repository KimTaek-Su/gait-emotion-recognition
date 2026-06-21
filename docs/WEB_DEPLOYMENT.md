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