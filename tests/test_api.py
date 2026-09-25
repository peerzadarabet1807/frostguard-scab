"""HTTP-level tests for the FastAPI service (no network: live fetches are stubbed)."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.app import app
from src.engine.weather_fetcher import generate_mock_weather
from src.vision.synthetic import render_healthy_leaf, render_synthetic_leaf


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    mp = pytest.MonkeyPatch()
    mp.setenv("FROSTGUARD_MODEL_PATH", str(tmp_path_factory.mktemp("m") / "absent.onnx"))
    mp.setenv("FROSTGUARD_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))
    with TestClient(app) as test_client:
        yield test_client
    mp.undo()


@pytest.fixture
def offline(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every live Open-Meteo call fail so the fallback path is exercised."""

    def boom(*_: object, **__: object) -> pd.DataFrame:
        raise ConnectionError("offline in tests")

    monkeypatch.setattr(client.app.state.fetcher, "fetch_forecast", boom)


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _weather_payload(scenario: str, hours: int = 72) -> list[dict[str, object]]:
    df = generate_mock_weather(start="2024-04-12 00:00", hours=hours, scenario=scenario)
    df["time"] = df["time"].map(pd.Timestamp.isoformat)
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------
def test_health(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["vision_backend"] == "fallback"
    assert body["model"].endswith(".onnx")


def test_zones(client: TestClient) -> None:
    res = client.get("/api/v1/zones")
    assert res.status_code == 200
    assert {z["key"] for z in res.json()} == {"shopian", "sopore", "pulwama", "baramulla"}


def test_openapi_lists_endpoints(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/api/v1/predict-risk", "/api/v1/detect-lesion"} <= paths.keys()


# ---------------------------------------------------------------------------
# /api/v1/predict-risk
# ---------------------------------------------------------------------------
class TestPredictRisk:
    def test_uploaded_weather_outbreak(self, client: TestClient) -> None:
        res = client.post("/api/v1/predict-risk", json={"weather": _weather_payload("scab_outbreak")})
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["weather_source"] == "uploaded"
        assert len(body["timeline"]) == 48
        assert body["risk_level"] == "CRITICAL"
        assert body["infection_probability"] == pytest.approx(1.0)
        assert body["fungicide"]["action"] == "protectant"
        assert body["fungicide"]["window_hours"] > 0
        assert any(e["infected"] for e in body["infection_events"])
        assert body["alerts"]

    def test_uploaded_weather_with_as_of_gives_curative_window(self, client: TestClient) -> None:
        payload = {"weather": _weather_payload("scab_outbreak", hours=96), "as_of": "2024-04-13T12:00:00+05:30"}
        body = client.post("/api/v1/predict-risk", json=payload).json()
        assert body["fungicide"]["action"] == "curative"
        assert 0 < body["fungicide"]["window_hours"] <= 72
        assert body["alerts"][0].startswith("CRITICAL RISK: Apply curative fungicide within")

    def test_dry_weather_is_low_risk(self, client: TestClient) -> None:
        body = client.post("/api/v1/predict-risk", json={"weather": _weather_payload("dry")}).json()
        assert body["risk_level"] == "LOW"
        assert body["fungicide"]["action"] == "none"

    def test_frost_alert(self, client: TestClient) -> None:
        body = client.post("/api/v1/predict-risk", json={"weather": _weather_payload("frost")}).json()
        assert body["frost"]["alert"] is True
        assert body["frost"]["min_temperature_c"] < -2.0

    def test_dew_point_only_records(self, client: TestClient) -> None:
        records = [
            {"time": f"2024-04-12T{h:02d}:00:00+05:30", "temperature_2m": 12.0, "dew_point_2m": 12.0}
            for h in range(10)
        ]
        body = client.post("/api/v1/predict-risk", json={"weather": records, "horizon_hours": 10}).json()
        assert all(p["relative_humidity"] == pytest.approx(100.0) for p in body["timeline"])
        assert body["timeline"][-1]["infection"] is True  # 10 h wet at 12 °C >= 9 h

    def test_zone_with_mock_weather(self, client: TestClient) -> None:
        res = client.post("/api/v1/predict-risk", json={"zone": "Sopore", "use_mock": True, "mock_scenario": "scab_outbreak"})
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["location"]["name"] == "Sopore"
        assert body["weather_source"] == "mock"
        assert len(body["timeline"]) == 48
        assert body["warnings"]

    @pytest.mark.usefixtures("offline")
    def test_coordinates_fall_back_when_open_meteo_is_down(self, client: TestClient) -> None:
        res = client.post("/api/v1/predict-risk", json={"latitude": 33.716, "longitude": 74.831})
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["weather_source"] == "mock"
        assert "Open-Meteo unavailable" in body["warnings"][0]
        assert body["location"] == {"name": None, "latitude": 33.716, "longitude": 74.831}

    def test_live_forecast_path(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        now = pd.Timestamp.now(tz="Asia/Kolkata").floor("h")
        fake = generate_mock_weather(start=now - pd.Timedelta(hours=24), hours=72, scenario="spring_mixed")
        monkeypatch.setattr(client.app.state.fetcher, "fetch_forecast", lambda *a, **k: fake)
        body = client.post("/api/v1/predict-risk", json={"zone": "shopian"}).json()
        assert body["weather_source"] == "open-meteo"
        assert body["warnings"] == []
        assert pd.Timestamp(body["timeline"][0]["time"]) == now

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"latitude": 33.7},
            {"zone": "srinagar-downtown"},
            {"latitude": 123.0, "longitude": 74.8},
            {"zone": "shopian", "horizon_hours": 0},
            {"weather": []},
            {"weather": [{"time": "2024-04-12T00:00:00", "temperature_2m": 10.0, "relative_humidity_2m": 140}]},
        ],
    )
    def test_invalid_requests_are_rejected(self, client: TestClient, payload: dict[str, object]) -> None:
        assert client.post("/api/v1/predict-risk", json=payload).status_code == 422

    def test_weather_without_humidity_signal_is_422(self, client: TestClient) -> None:
        records = [{"time": "2024-04-12T00:00:00", "temperature_2m": 10.0}]
        res = client.post("/api/v1/predict-risk", json={"weather": records})
        assert res.status_code == 422
        assert "relative_humidity_2m" in res.json()["detail"]


# ---------------------------------------------------------------------------
# /api/v1/detect-lesion
# ---------------------------------------------------------------------------
class TestDetectLesion:
    def test_detects_lesions(self, client: TestClient) -> None:
        image, truth = render_synthetic_leaf(seed=3)
        res = client.post("/api/v1/detect-lesion", files={"file": ("leaf.png", _png_bytes(image), "image/png")})
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["scab_detected"] is True
        assert body["lesion_count"] == len(truth)
        assert body["backend"] == "fallback"
        assert body["warnings"]
        assert body["image_width"] == 800 and body["image_height"] == 600
        box = body["detections"][0]
        assert box["label"] == "apple_scab" and 0 < box["confidence"] <= 1
        assert 0 <= box["x1"] < box["x2"] <= 800

    def test_healthy_leaf(self, client: TestClient) -> None:
        files = {"file": ("healthy.png", _png_bytes(render_healthy_leaf()), "image/png")}
        body = client.post("/api/v1/detect-lesion", files=files).json()
        assert body["scab_detected"] is False
        assert body["lesion_count"] == 0
        assert body["max_confidence"] is None

    def test_sample_jpeg_from_repo(self, client: TestClient) -> None:
        sample = Path(__file__).resolve().parents[1] / "data" / "sample_leaf_scab.jpg"
        res = client.post("/api/v1/detect-lesion", files={"file": (sample.name, sample.read_bytes(), "image/jpeg")})
        assert res.json()["scab_detected"] is True

    def test_rejects_non_images(self, client: TestClient) -> None:
        res = client.post("/api/v1/detect-lesion", files={"file": ("notes.txt", b"hello", "text/plain")})
        assert res.status_code == 415

    def test_rejects_corrupt_image(self, client: TestClient) -> None:
        res = client.post("/api/v1/detect-lesion", files={"file": ("x.jpg", b"\xff\xd8garbage", "image/jpeg")})
        assert res.status_code == 400

    def test_rejects_empty_upload(self, client: TestClient) -> None:
        res = client.post("/api/v1/detect-lesion", files={"file": ("x.jpg", b"", "image/jpeg")})
        assert res.status_code == 400

    def test_missing_file_is_422(self, client: TestClient) -> None:
        assert client.post("/api/v1/detect-lesion").status_code == 422
