# syntax=docker/dockerfile:1.7
# One slim image, two roles (see docker-compose.yml):
#   api       -> uvicorn src.api.app:app            (port 8000)
#   dashboard -> streamlit run demo/app_streamlit.py (port 8501)

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FROSTGUARD_CACHE_DIR=/tmp/frostguard-cache

WORKDIR /app

# libgomp1: OpenMP runtime used by onnxruntime's CPU kernels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt .
RUN pip install -r requirements-runtime.txt

COPY src/ src/
COPY demo/ demo/
COPY data/ data/
COPY models/ models/
COPY .streamlit/ .streamlit/

RUN useradd --create-home --uid 10001 frostguard \
    && mkdir -p /tmp/frostguard-cache \
    && chown -R frostguard /tmp/frostguard-cache
USER frostguard

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
