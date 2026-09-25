"""Pydantic request/response models for the FrostGuard REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.engine.weather_fetcher import KASHMIR_ORCHARD_ZONES

MockScenarioName = Literal["scab_outbreak", "frost", "dry", "spring_mixed"]


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------
class WeatherRecord(BaseModel):
    """One hour of weather. Provide relative humidity, dew point, or both."""

    time: datetime
    temperature_2m: float = Field(..., ge=-60, le=60, description="Air temperature at 2 m (°C)")
    relative_humidity_2m: float | None = Field(None, ge=0, le=100, description="Relative humidity (%)")
    precipitation: float | None = Field(0.0, ge=0, description="Precipitation in the hour (mm)")
    dew_point_2m: float | None = Field(None, ge=-80, le=60, description="Dew point at 2 m (°C)")


class RiskRequest(BaseModel):
    """Either a location (``zone`` or ``latitude``+``longitude``) or uploaded ``weather``."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"zone": "shopian", "horizon_hours": 48},
                {"latitude": 33.716, "longitude": 74.831, "use_mock": True, "mock_scenario": "scab_outbreak"},
                {
                    "weather": [
                        {"time": "2024-04-12T00:00:00+05:30", "temperature_2m": 11.2, "relative_humidity_2m": 96, "precipitation": 1.4},
                        {"time": "2024-04-12T01:00:00+05:30", "temperature_2m": 10.9, "relative_humidity_2m": 97, "precipitation": 0.8},
                    ]
                },
            ]
        }
    )

    zone: str | None = Field(None, description=f"One of {sorted(KASHMIR_ORCHARD_ZONES)}")
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    weather: list[WeatherRecord] | None = Field(None, min_length=1, max_length=24 * 400)
    horizon_hours: int = Field(48, ge=1, le=168, description="Length of the returned forward timeline")
    past_hours: int = Field(24, ge=0, le=168, description="Look-back fetched for live forecasts")
    as_of: datetime | None = Field(None, description="Decision time; defaults to now (live) or the first uploaded hour")
    bud_break: bool | None = Field(None, description="Force phenology; null infers it from the date")
    use_mock: bool = Field(False, description="Skip Open-Meteo and use simulated weather")
    mock_scenario: MockScenarioName = "spring_mixed"

    @model_validator(mode="after")
    def _check_source(self) -> RiskRequest:
        if self.zone is not None and self.zone.lower() not in KASHMIR_ORCHARD_ZONES:
            raise ValueError(f"unknown zone {self.zone!r}; choose from {sorted(KASHMIR_ORCHARD_ZONES)}")
        has_coords = self.latitude is not None and self.longitude is not None
        if self.weather is None and self.zone is None and not has_coords:
            raise ValueError("provide `weather` records, a `zone`, or both `latitude` and `longitude`")
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("`latitude` and `longitude` must be given together")
        return self


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------
class Location(BaseModel):
    name: str | None
    latitude: float | None
    longitude: float | None


class TimelinePoint(BaseModel):
    time: datetime
    temperature_c: float | None
    relative_humidity: float | None
    precipitation_mm: float | None
    leaf_wet: bool
    wet_hours: int
    risk_ratio: float = Field(..., ge=0, le=1, description="Fraction of the Mills infection requirement met")
    infection: bool
    frost_risk: bool


class FungicideWindow(BaseModel):
    action: Literal["none", "monitor", "protectant", "curative"]
    window_hours: float | None
    deadline: datetime | None
    message: str


class FrostRisk(BaseModel):
    alert: bool
    frost_hours: int
    min_temperature_c: float | None
    first_frost_time: datetime | None
    in_bud_break: bool
    threshold_c: float


class InfectionEvent(BaseModel):
    event_id: int
    start: datetime
    end: datetime
    wet_hours: int
    mean_temperature_c: float
    required_hours: float | None
    risk_ratio: float
    infected: bool
    infection_time: datetime | None


class RiskResponse(BaseModel):
    location: Location
    weather_source: Literal["open-meteo", "mock", "uploaded"]
    as_of: datetime
    horizon_hours: int
    infection_probability: float = Field(..., ge=0, le=1)
    current_risk: float
    peak_risk: float
    risk_level: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]
    fungicide: FungicideWindow
    frost: FrostRisk
    alerts: list[str]
    infection_events: list[InfectionEvent]
    timeline: list[TimelinePoint]
    warnings: list[str] = []


class LesionBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = Field(..., ge=0, le=1)
    class_id: int
    label: str


class DetectionResponse(BaseModel):
    filename: str | None
    scab_detected: bool
    lesion_count: int
    max_confidence: float | None
    detections: list[LesionBox]
    image_width: int
    image_height: int
    inference_ms: float
    backend: Literal["trained", "fallback"]
    model: str
    warnings: list[str] = []


class ZoneOut(BaseModel):
    key: str
    name: str
    latitude: float
    longitude: float
    elevation_m: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    vision_backend: Literal["trained", "fallback"]
    model: str
    time: datetime
