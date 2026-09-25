"""Headless smoke tests for the Streamlit dashboard (embedded API, offline weather)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from demo.charts import DARK, LIGHT, draw_detections, risk_chart
from src.engine.mills_physics import FrostScabPhysicsEngine
from src.engine.weather_fetcher import generate_mock_weather
from src.vision.synthetic import render_synthetic_leaf

APP = str(Path(__file__).resolve().parents[1] / "demo" / "app_streamlit.py")


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> Iterator[AppTest]:
    monkeypatch.setenv("FROSTGUARD_API_URL", "http://127.0.0.1:9")  # unreachable -> embedded API
    monkeypatch.setenv("FROSTGUARD_OFFLINE", "1")
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    yield at


def _assert_clean(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def test_default_view_renders(app: AppTest) -> None:
    _assert_clean(app)
    labels = [m.label for m in app.metric]
    assert labels == ["Infection probability", "Fungicide window", "Current risk", "Minimum temperature"]
    assert any("Embedded API" in c.value for c in app.sidebar.caption)


@pytest.mark.parametrize(
    ("scenario", "expected_alert"),
    [("scab_outbreak", "CRITICAL RISK"), ("frost", "FROST WARNING")],
)
def test_simulated_scenarios_raise_alerts(app: AppTest, scenario: str, expected_alert: str) -> None:
    app.sidebar.radio[0].set_value("Simulated scenario").run()
    app.sidebar.selectbox[0].set_value(scenario).run()
    _assert_clean(app)
    assert any(e.value.startswith(expected_alert) for e in app.error)


def test_historical_replay(app: AppTest) -> None:
    app.sidebar.radio[0].set_value("Historical replay").run()
    _assert_clean(app)
    assert any("ERA5 replay" in c.value for c in app.caption)


class TestRiskChart:
    @pytest.fixture
    def timeline(self) -> pd.DataFrame:
        weather = generate_mock_weather(start="2024-04-12", hours=72, scenario="scab_outbreak")
        return pd.DataFrame(FrostScabPhysicsEngine().assess(weather).to_dict()["timeline"])

    def test_four_single_measure_panels(self, timeline: pd.DataFrame) -> None:
        spec = risk_chart(timeline, LIGHT).to_dict()
        assert len(spec["vconcat"]) == 4
        for panel in spec["vconcat"]:  # one y-scale per panel: never a dual axis
            assert panel.get("resolve", {}).get("scale", {}).get("y") != "independent"

    def test_shared_hover_selection(self, timeline: pd.DataFrame) -> None:
        spec = risk_chart(timeline, DARK).to_dict()
        assert [p["name"] for p in spec.get("params", [])] == ["hover"]

    def test_rain_bars_have_explicit_baseline(self, timeline: pd.DataFrame) -> None:
        rain_layers = risk_chart(timeline, LIGHT).to_dict()["vconcat"][3]["layer"]
        rect = next(layer for layer in rain_layers if layer.get("mark", {}).get("type") == "rect")
        assert rect["encoding"]["y2"] == {"datum": 0}

    def test_dry_window_keeps_a_sane_rain_axis(self) -> None:
        weather = generate_mock_weather(start="2024-04-12", hours=72, scenario="dry")
        dry = pd.DataFrame(FrostScabPhysicsEngine().assess(weather).to_dict()["timeline"])
        rain_layers = risk_chart(dry, LIGHT).to_dict()["vconcat"][3]["layer"]
        rect = next(layer for layer in rain_layers if layer.get("mark", {}).get("type") == "rect")
        assert rect["encoding"]["y"]["scale"]["domain"] == [0, 1.0]


def test_draw_detections_keeps_size() -> None:
    image, _ = render_synthetic_leaf(seed=1)
    dets = [{"x1": 10.0, "y1": 40.0, "x2": 60.0, "y2": 90.0, "confidence": 0.9, "label": "apple_scab"}]
    out = draw_detections(image, dets, LIGHT)
    assert out.size == image.size and out.mode == "RGB"
    assert out.getpixel((10, 60)) != image.convert("RGB").getpixel((10, 60))  # box outline drawn


def test_zone_switch_and_leaf_scan(app: AppTest) -> None:
    app.selectbox(key="zone").set_value("baramulla").run()
    app.radio(key="leaf_sample").set_value("Scabbed leaf (synthetic)").run()
    _assert_clean(app)
    verdicts = [e.value for e in app.error] + [w.value for w in app.warning]
    assert any("scab lesions" in v for v in verdicts)
    assert not any(v.startswith("Detection failed") for v in verdicts)
