"""Historical weather datasets that can be replayed through the risk engine.

A replay scores a real past window as if it were a live forecast: ``past_hours`` of
look-back before the decision time ``as_of`` plus ``horizon_hours`` after it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.engine.weather_fetcher import load_weather_csv

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class ReplayDataset:
    key: str
    zone: str
    title: str
    source: str
    path: Path


REPLAY_DATASETS: dict[str, ReplayDataset] = {
    "shopian_spring_2024": ReplayDataset(
        key="shopian_spring_2024",
        zone="shopian",
        title="Shopian, spring 2024 (1 Apr – 15 May)",
        source="ERA5 reanalysis via the Open-Meteo archive API",
        path=DATA_DIR / "shopian_spring_2024_hourly.csv",
    )
}


@lru_cache(maxsize=len(REPLAY_DATASETS))
def load_replay(key: str) -> pd.DataFrame:
    """Normalised hourly weather for a replay dataset (cached per process)."""
    if key not in REPLAY_DATASETS:
        raise KeyError(f"unknown replay dataset {key!r}; choose from {sorted(REPLAY_DATASETS)}")
    return load_weather_csv(REPLAY_DATASETS[key].path)


def replay_bounds(key: str, past_hours: int, horizon_hours: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Earliest and latest ``as_of`` that still leave a full look-back and horizon."""
    times = load_replay(key)["time"]
    return times.iloc[0] + pd.Timedelta(hours=past_hours), times.iloc[-1] - pd.Timedelta(hours=horizon_hours - 1)


def replay_window(key: str, as_of: pd.Timestamp, past_hours: int, horizon_hours: int) -> pd.DataFrame:
    """Slice ``[as_of - past_hours, as_of + horizon_hours)`` out of a replay dataset."""
    df = load_replay(key)
    tz = df["time"].dt.tz
    as_of = (as_of.tz_localize(tz) if as_of.tz is None else as_of.tz_convert(tz)).floor("h")
    start, end = replay_bounds(key, past_hours, horizon_hours)
    if not start <= as_of <= end:
        raise ValueError(
            f"as_of {as_of.isoformat()} is outside the replay range {start.isoformat()} – {end.isoformat()}"
        )
    mask = (df["time"] >= as_of - pd.Timedelta(hours=past_hours)) & (df["time"] < as_of + pd.Timedelta(hours=horizon_hours))
    return df.loc[mask].reset_index(drop=True)
