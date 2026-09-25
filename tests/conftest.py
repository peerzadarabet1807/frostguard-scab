"""Shared pytest fixtures and weather-building helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd
import pytest

WeatherFactory = Callable[..., pd.DataFrame]


def build_weather(
    temps: Sequence[float] | float,
    wet: Sequence[bool] | None = None,
    hours: int | None = None,
    start: str = "2024-04-10 00:00",
    precip: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Hourly weather where ``wet[i]`` maps to RH 95 % (wet) or 60 % (dry)."""
    if isinstance(temps, (int, float)):
        if hours is None:
            hours = len(wet) if wet is not None else 24
        temps = [float(temps)] * hours
    n = len(temps)
    wet = list(wet) if wet is not None else [True] * n
    assert len(wet) == n, "temps and wet must be the same length"
    return pd.DataFrame(
        {
            "time": pd.date_range(start, periods=n, freq="h", tz="Asia/Kolkata"),
            "temperature_2m": list(temps),
            "relative_humidity_2m": [95.0 if w else 60.0 for w in wet],
            "precipitation": list(precip) if precip is not None else [0.0] * n,
        }
    )


@pytest.fixture
def make_weather() -> WeatherFactory:
    return build_weather
