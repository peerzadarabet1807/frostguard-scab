# syntax=docker/dockerfile:1.7
# FrostGuard-Scab API: FastAPI + ONNX Runtime + the trained YOLO11n detector.
# Used by docker-compose (port 8000) and deployed to Render (render.yaml); same image.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FROSTGUARD_CACHE_DIR=/tmp/frostguard-cache \
    PORT=8000 \
    WEB_CONCURRENCY=1

WORKDIR /app

# libgomp1: OpenMP runtime for onnxruntime's CPU kernels; curl for the health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt .
RUN pip install -r requirements-runtime.txt

COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY scripts/download_model.py scripts/

# Bake in the trained detector from the model-v1 release (SHA-256 verified; skipped if the
# build context already has it). Without network the API falls back to the heuristic graph.
RUN python scripts/download_model.py \
    || echo "WARNING: trained model unavailable - the API will serve the fallback detector"

# Non-root user (uid 1000, also the Hugging Face Spaces convention).
RUN useradd --create-home --uid 1000 frostguard \
    && mkdir -p /tmp/frostguard-cache \
    && chmod 1777 /tmp/frostguard-cache
USER frostguard

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["sh", "-c", "exec uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY} --proxy-headers"]
