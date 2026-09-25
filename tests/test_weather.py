"""Tests for weather ingestion, normalisation and the offline mock generator."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.engine.psychrometrics import dew_point_magnus, relative_humidity_from_dew_point
from src.engine.weather_fetcher import (
    KASHMIR_ORCHARD_ZONES,
    MOCK_SCENARIOS,
    SHOPIAN,
    WEATHER_COLUMNS,
    WeatherFetcher,
    generate_mock_weather,
    normalize_weather_frame,
)


class TestPsychrometrics:
    def test_saturated_air_dew_point_equals_temperature(self) -> None:
        assert dew_point_magnus(20.0, 100.0) == pytest.approx(20.0, abs=1e-6)

    def test_known_value(self) -> None:
        # 20 °C at 50 % RH -> ~9.3 °C dew point.
        assert dew_point_magnus(20.0, 50.0) == pytest.approx(9.26, abs=0.05)

    def test_round_trip(self) -> None:
        temps = np.array([-5.0, 0.0, 8.0, 15.0, 25.0])
        rh = np.array([40.0, 65.0, 90.0, 99.0, 55.0])
        td = dew_point_magnus(temps, rh)
        np.testing.assert_allclose(relative_humidity_from_dew_point(temps, td), rh, rtol=1e-6)


class TestMockWeather:
    @pytest.mark.parametrize("scenario", sorted(MOCK_SCENARIOS))
    def test_shape_and_ranges(self, scenario: str) -> None:
        df = generate_mock_weather(start="2024-04-10", hours=72, scenario=scenario)
        assert list(df.columns) == list(WEATHER_COLUMNS)
        assert len(df) == 72
        assert (df["time"].diff().dropna() == pd.Timedelta(hours=1)).all()
        assert df["relative_humidity_2m"].between(0, 100).all()
        assert (df["precipitation"] >= 0).all()
        assert (df["dew_point_2m"] <= df["temperature_2m"] + 1e-6).all()
        assert str(df["time"].dt.tz) == "Asia/Kolkata"

    def test_deterministic_for_seed(self) -> None:
        a = generate_mock_weather(start="2024-04-10", hours=48, seed=3)
        b = generate_mock_weather(start="2024-04-10", hours=48, seed=3)
        c = generate_mock_weather(start="2024-04-10", hours=48, seed=4)
        pd.testing.assert_frame_equal(a, b)
        assert not a.equals(c)

    def test_rain_saturates_air(self) -> None:
        df = generate_mock_weather(start="2024-04-10", hours=48, scenario="scab_outbreak")
        raining = df["precipitation"] > 0
        assert raining.sum() > 0
        assert (df.loc[raining, "relative_humidity_2m"] >= 92.0).all()

    def test_diurnal_cycle(self) -> None:
        df = generate_mock_weather(start="2024-04-10", hours=24 * 5, scenario="dry", seed=1)
        by_hour = df.groupby(df["time"].dt.hour)["temperature_2m"].mean()
        assert by_hour.loc[13:16].mean() > by_hour.loc[1:5].mean() + 8

    def test_invalid_arguments(self) -> None:
        with pytest.raises(ValueError):
            generate_mock_weather(scenario="monsoon")
        with pytest.raises(ValueError):
            generate_mock_weather(hours=0)


class TestNormalisation:
    def test_sorts_dedupes_and_fills_gaps(self) -> None:
        raw = pd.DataFrame(
            {
                "time": ["2024-04-10 02:00", "2024-04-10 00:00", "2024-04-10 00:00", "2024-04-10 04:00"],
                "temperature_2m": [12.0, 10.0, 10.0, 14.0],
                "relative_humidity_2m": [80.0, 90.0, 90.0, 70.0],
                "precipitation": [0.0, 0.2, 0.2, 0.0],
            }
        )
        df = normalize_weather_frame(raw)
        assert len(df) == 5
        assert df["temperature_2m"].tolist() == pytest.approx([10.0, 11.0, 12.0, 13.0, 14.0])
        assert df["precipitation"].iloc[1] == 0.0
        assert "dew_point_2m" in df.columns

    def test_derives_humidity_from_dew_point(self) -> None:
        raw = pd.DataFrame(
            {"time": pd.date_range("2024-04-10", periods=3, freq="h"), "temperature_2m": [10.0] * 3, "dew_point_2m": [10.0] * 3}
        )
        df = normalize_weather_frame(raw)
        assert df["relative_humidity_2m"].tolist() == pytest.approx([100.0] * 3)
        assert df["precipitation"].tolist() == [0.0] * 3

    def test_converts_timezone(self) -> None:
        raw = pd.DataFrame(
            {
                "time": pd.date_range("2024-04-10 00:00", periods=2, freq="h", tz="UTC"),
                "temperature_2m": [5.0, 6.0],
                "relative_humidity_2m": [80.0, 85.0],
            }
        )
        df = normalize_weather_frame(raw)
        assert df["time"].iloc[0].hour == 5 and df["time"].iloc[0].minute == 0  # UTC+05:30 floored

    @pytest.mark.parametrize("missing", ["time", "temperature_2m"])
    def test_missing_required_columns(self, missing: str) -> None:
        raw = pd.DataFrame({"time": ["2024-04-10"], "temperature_2m": [5.0], "relative_humidity_2m": [80.0]})
        with pytest.raises(ValueError):
            normalize_weather_frame(raw.drop(columns=[missing]))

    def test_needs_some_humidity_signal(self) -> None:
        raw = pd.DataFrame({"time": ["2024-04-10"], "temperature_2m": [5.0]})
        with pytest.raises(ValueError):
            normalize_weather_frame(raw)


class TestFetcherFallback:
    def test_zones_cover_kashmir_valley(self) -> None:
        assert set(KASHMIR_ORCHARD_ZONES) == {"shopian", "sopore", "pulwama", "baramulla"}
        assert (SHOPIAN.latitude, SHOPIAN.longitude) == (33.716, 74.831)

    def test_use_mock_skips_network(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fetcher = WeatherFetcher()

        def boom(*_: object, **__: object) -> pd.DataFrame:
            raise AssertionError("network must not be called")

        monkeypatch.setattr(fetcher, "fetch_forecast", boom)
        result = fetcher.fetch_with_fallback(use_mock=True, past_hours=12, forecast_hours=48)
        assert result.source == "mock"
        assert len(result.data) == 60
        assert result.error is None

    def test_falls_back_when_api_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fetcher = WeatherFetcher()

        def offline(*_: object, **__: object) -> pd.DataFrame:
            raise ConnectionError("no route to host")

        monkeypatch.setattr(fetcher, "fetch_forecast", offline)
        result = fetcher.fetch_with_fallback()
        assert result.source == "mock"
        assert result.error is not None and "ConnectionError" in result.error
        assert len(result.data) == 72

    def test_async_wrapper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fetcher = WeatherFetcher()
        expected = generate_mock_weather(start="2024-04-10", hours=4)
        monkeypatch.setattr(fetcher, "fetch_forecast", lambda *a, **k: expected)
        result = asyncio.run(fetcher.afetch_forecast())
        pd.testing.assert_frame_equal(result, expected)


@pytest.mark.network
@pytest.mark.skipif(not os.getenv("FROSTGUARD_NETWORK_TESTS"), reason="set FROSTGUARD_NETWORK_TESTS=1 to hit Open-Meteo")
def test_live_open_meteo_archive(tmp_path: Path) -> None:
    df = WeatherFetcher(cache_dir=tmp_path).fetch_historical(start_date="2024-04-01", end_date="2024-04-02")
    assert len(df) == 48
    assert df["temperature_2m"].notna().all()
