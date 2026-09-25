"""FrostGuard-Scab REST API.

Run locally::

    uvicorn src.api.app:app --reload

Interactive docs at http://localhost:8000/docs.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from src import __version__
from src.api.schemas import (
    DetectionResponse,
    HealthResponse,
    RiskRequest,
    RiskResponse,
    ZoneOut,
)
from src.engine.mills_physics import FrostScabPhysicsEngine
from src.engine.weather_fetcher import (
    DEFAULT_TIMEZONE,
    KASHMIR_ORCHARD_ZONES,
    WeatherFetcher,
    WeatherResult,
)
from src.vision.detector import ScabVisionDetector

logger = logging.getLogger("frostguard.api")

MAX_UPLOAD_BYTES = int(os.getenv("FROSTGUARD_MAX_UPLOAD_MB", "10")) * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 50_000_000  # guard against decompression bombs


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the heavy singletons once per process."""
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    app.state.offline = os.getenv("FROSTGUARD_OFFLINE", "0").lower() in {"1", "true", "yes"}
    app.state.engine = FrostScabPhysicsEngine()
    app.state.fetcher = WeatherFetcher(cache_dir=os.getenv("FROSTGUARD_CACHE_DIR", ".cache"))
    app.state.detector = ScabVisionDetector()
    app.state.detector.warmup()
    logger.info("FrostGuard API ready (vision backend: %s)", app.state.detector.backend)
    yield


app = FastAPI(
    title="FrostGuard-Scab API",
    version=__version__,
    summary="Physics-informed apple scab & frost early warning for Kashmir orchards.",
    description=(
        "Revised Mills infection-period modelling on Open-Meteo weather plus YOLO11n (ONNX Runtime) "
        "lesion detection. See the project README for the agronomic background."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("FROSTGUARD_CORS_ORIGINS", "*").split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _records_to_frame(request: RiskRequest) -> pd.DataFrame:
    assert request.weather is not None
    frame = pd.DataFrame([r.model_dump() for r in request.weather])
    # Let the normaliser derive whichever humidity variable was omitted entirely.
    return frame.dropna(axis=1, how="all")


def _resolve_location(request: RiskRequest) -> tuple[str | None, float | None, float | None]:
    if request.zone:
        zone = KASHMIR_ORCHARD_ZONES[request.zone.lower()]
        return zone.name, zone.latitude, zone.longitude
    return None, request.latitude, request.longitude


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> dict[str, Any]:
    detector: ScabVisionDetector = request.app.state.detector
    return {
        "status": "ok",
        "version": __version__,
        "vision_backend": detector.backend,
        "model": Path(detector.model_path).name,
        "time": datetime.now().astimezone(),
    }


@app.get("/api/v1/zones", response_model=list[ZoneOut], tags=["risk"])
def list_zones() -> list[dict[str, Any]]:
    """Kashmir orchard zones known to the service."""
    return [vars(zone) for zone in KASHMIR_ORCHARD_ZONES.values()]


@app.post("/api/v1/predict-risk", response_model=RiskResponse, tags=["risk"])
def predict_risk(payload: RiskRequest, request: Request) -> dict[str, Any]:
    """Run the Revised Mills scab model (and frost check) over a 48 h window.

    * **Location mode** (`zone` or `latitude`/`longitude`): fetches `past_hours` of recent
      weather plus `horizon_hours` of forecast from Open-Meteo, falling back to simulated
      weather if the API is unreachable (reported in `warnings`).
    * **Upload mode** (`weather`): scores the supplied hourly records as-is.
    """
    engine: FrostScabPhysicsEngine = request.app.state.engine
    fetcher: WeatherFetcher = request.app.state.fetcher
    warnings: list[str] = []
    name, lat, lon = _resolve_location(payload)

    if payload.weather is not None:
        weather = WeatherResult(_records_to_frame(payload), "uploaded")
        as_of = payload.as_of
    else:
        assert lat is not None and lon is not None
        weather = fetcher.fetch_with_fallback(
            latitude=lat,
            longitude=lon,
            past_hours=payload.past_hours,
            # +2 h buffer: Open-Meteo counts from its own "current hour", which can trail
            # our floored as_of (e.g. IST's +05:30 offset); the engine trims to the horizon.
            forecast_hours=payload.horizon_hours + 2,
            mock_scenario=payload.mock_scenario,
            use_mock=payload.use_mock or request.app.state.offline,
        )
        if weather.error:
            warnings.append(f"Open-Meteo unavailable ({weather.error}); using simulated weather.")
        elif weather.source == "mock" and not payload.use_mock:
            warnings.append(f"Offline mode (FROSTGUARD_OFFLINE=1): simulated '{payload.mock_scenario}' weather.")
        elif weather.source == "mock":
            warnings.append(f"Simulated '{payload.mock_scenario}' weather requested (use_mock=true).")
        as_of = payload.as_of or pd.Timestamp.now(tz=DEFAULT_TIMEZONE).floor("h")

    try:
        result = engine.assess(
            weather.data, as_of=as_of, horizon_hours=payload.horizon_hours, bud_break=payload.bud_break
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    if len(result.timeline) < payload.horizon_hours:
        warnings.append(
            f"Only {len(result.timeline)} of {payload.horizon_hours} requested hours available after as_of."
        )
    body = result.to_dict()
    body.update(
        location={"name": name, "latitude": lat, "longitude": lon},
        weather_source=weather.source,
        horizon_hours=payload.horizon_hours,
        warnings=warnings,
    )
    return body


@app.post("/api/v1/detect-lesion", response_model=DetectionResponse, tags=["vision"])
def detect_lesion(
    request: Request,
    file: Annotated[UploadFile, File(description="Leaf photo (JPEG/PNG/WebP)")],
) -> dict[str, Any]:
    """Detect apple scab lesions on a leaf photo and return boxes in image pixels."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(415, f"expected an image, got {file.content_type}")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"image larger than {MAX_UPLOAD_BYTES // 2**20} MB")
    if not data:
        raise HTTPException(400, "empty upload")

    detector: ScabVisionDetector = request.app.state.detector
    try:
        result = detector.detect(data)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(400, f"could not decode image: {exc}") from exc

    warnings: list[str] = []
    if detector.is_fallback:
        warnings.append(
            "models/scab_detector.onnx not found: using the colour-heuristic fallback detector. "
            "Train YOLO11n with notebooks/train_yolo_colab.ipynb for field-grade accuracy."
        )
    body = result.to_dict()
    confidences = [d.confidence for d in result.detections]
    body.update(
        filename=file.filename,
        scab_detected=bool(result.detections),
        max_confidence=max(confidences) if confidences else None,
        model=Path(result.model_path).name,
        warnings=warnings,
    )
    return body
