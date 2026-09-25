"""Hourly weather ingestion from Open-Meteo with a deterministic offline fallback.

Every public function returns a *normalised* :class:`pandas.DataFrame` with the columns
listed in :data:`WEATHER_COLUMNS`, one row per hour, and a timezone-aware ``time``
column in the orchard's local timezone (``Asia/Kolkata`` for Kashmir).

Network access is optional: :meth:`WeatherFetcher.fetch_with_fallback` transparently
switches to :func:`generate_mock_weather` when the API is unreachable, so the risk
engine, API and dashboard all keep working offline.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.engine.psychrometrics import dew_point_magnus, relative_humidity_from_dew_point

logger = logging.getLogger(__name__)

HOURLY_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "dew_point_2m",
)
WEATHER_COLUMNS: tuple[str, ...] = ("time", *HOURLY_VARIABLES)
DEFAULT_TIMEZONE = "Asia/Kolkata"


@dataclass(frozen=True)
class OrchardZone:
    """A named apple-growing zone in the Kashmir valley."""

    key: str
    name: str
    latitude: float
    longitude: float
    elevation_m: int


KASHMIR_ORCHARD_ZONES: dict[str, OrchardZone] = {
    "shopian": OrchardZone("shopian", "Shopian", 33.716, 74.831, 2146),
    "sopore": OrchardZone("sopore", "Sopore", 34.300, 74.470, 1580),
    "pulwama": OrchardZone("pulwama", "Pulwama", 33.874, 74.899, 1630),
    "baramulla": OrchardZone("baramulla", "Baramulla", 34.198, 74.364, 1590),
}
SHOPIAN = KASHMIR_ORCHARD_ZONES["shopian"]

WeatherSource = Literal["open-meteo", "mock", "uploaded"]


@dataclass(frozen=True)
class WeatherResult:
    """Weather frame plus provenance, so callers can flag simulated data in the UI."""

    data: pd.DataFrame
    source: WeatherSource
    error: str | None = None


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def normalize_weather_frame(df: pd.DataFrame, timezone: str = DEFAULT_TIMEZONE) -> pd.DataFrame:
    """Validate and normalise an hourly weather frame.

    * requires ``time``, ``temperature_2m`` and at least one of
      ``relative_humidity_2m`` / ``dew_point_2m``;
    * back-fills the missing humidity variable via the Magnus formula;
    * treats missing precipitation as 0 mm;
    * sorts by time, drops duplicate timestamps and regularises to a 1-hour grid
      (short gaps up to 3 h are interpolated, longer gaps stay NaN => "dry").
    """
    if "time" not in df.columns or "temperature_2m" not in df.columns:
        raise ValueError("weather data needs at least 'time' and 'temperature_2m' columns")
    has_rh = "relative_humidity_2m" in df.columns
    has_td = "dew_point_2m" in df.columns
    if not (has_rh or has_td):
        raise ValueError("weather data needs 'relative_humidity_2m' or 'dew_point_2m'")

    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    if out["time"].dt.tz is None:
        out["time"] = out["time"].dt.tz_localize(timezone)
    else:
        out["time"] = out["time"].dt.tz_convert(timezone)

    for col in HOURLY_VARIABLES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    if not has_rh:
        out["relative_humidity_2m"] = relative_humidity_from_dew_point(
            out["temperature_2m"].to_numpy(), out["dew_point_2m"].to_numpy()
        )
    if not has_td:
        out["dew_point_2m"] = dew_point_magnus(
            out["temperature_2m"].to_numpy(), out["relative_humidity_2m"].to_numpy()
        )
    if "precipitation" not in out.columns:
        out["precipitation"] = 0.0

    out = (
        out[list(WEATHER_COLUMNS)]
        .sort_values("time")
        .drop_duplicates(subset="time", keep="last")
        .set_index("time")
    )
    out.index = out.index.floor("h")
    out = out[~out.index.duplicated(keep="last")]
    if len(out) > 1:
        out = out.asfreq("h")
        smooth = ["temperature_2m", "relative_humidity_2m", "dew_point_2m"]
        out[smooth] = out[smooth].interpolate(limit=3, limit_area="inside")
        out["precipitation"] = out["precipitation"].fillna(0.0)
    return out.reset_index()


# ---------------------------------------------------------------------------
# Synthetic weather
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MockScenario:
    """Parameters of a synthetic spring weather regime."""

    mean_temp_c: float
    diurnal_amplitude_c: float
    mean_rh: float
    rain_windows: tuple[tuple[int, int], ...]  # (start_hour, duration_hours)
    rain_rate_mm: float = 1.2


MOCK_SCENARIOS: dict[str, MockScenario] = {
    # Mild, showery spell: a textbook primary-infection period.
    "scab_outbreak": MockScenario(13.0, 4.0, 78.0, ((18, 26), (62, 10))),
    # Clear, cold radiative nights at bud break.
    "frost": MockScenario(2.5, 7.0, 58.0, ()),
    # Warm, dry, low-risk anticyclone.
    "dry": MockScenario(17.0, 8.0, 42.0, ()),
    # Typical Kashmir April: cool days, one moderate shower.
    "spring_mixed": MockScenario(10.0, 6.0, 68.0, ((30, 8),)),
}


def generate_mock_weather(
    start: datetime | pd.Timestamp | None = None,
    hours: int = 96,
    scenario: str = "spring_mixed",
    seed: int = 42,
    timezone: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Generate deterministic, physically consistent hourly weather.

    Temperature follows a sinusoidal diurnal cycle (min ~03:00, max ~15:00) with AR(1)
    noise; relative humidity is anti-correlated with temperature and saturates during
    rain; dew point is derived from both via the Magnus formula.
    """
    if scenario not in MOCK_SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {sorted(MOCK_SCENARIOS)}")
    if hours <= 0:
        raise ValueError("hours must be positive")
    cfg = MOCK_SCENARIOS[scenario]
    rng = np.random.default_rng(seed)

    if start is None:
        start = pd.Timestamp.now(tz=timezone).floor("h")
    start_ts = pd.Timestamp(start)
    start_ts = start_ts.tz_localize(timezone) if start_ts.tz is None else start_ts.tz_convert(timezone)
    times = pd.date_range(start_ts.floor("h"), periods=hours, freq="h")

    local_hour = times.hour.to_numpy(dtype=float)
    amplitude = np.full(hours, cfg.diurnal_amplitude_c)
    raining = np.zeros(hours, dtype=bool)
    for rain_start, duration in cfg.rain_windows:
        raining[rain_start : rain_start + duration] = True
    amplitude[raining] *= 0.35  # cloud cover damps the diurnal cycle

    noise = np.zeros(hours)
    for i in range(1, hours):  # AR(1) weather "memory"
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0.0, 0.35)
    temperature = cfg.mean_temp_c + amplitude * np.sin(2 * np.pi * (local_hour - 9.0) / 24.0) + noise

    rh = cfg.mean_rh - 3.2 * (temperature - cfg.mean_temp_c) + rng.normal(0.0, 2.5, hours)
    precipitation = np.zeros(hours)
    precipitation[raining] = rng.gamma(shape=2.0, scale=cfg.rain_rate_mm / 2.0, size=int(raining.sum()))
    rh[raining] = rng.uniform(92.0, 99.0, int(raining.sum()))
    rh = np.clip(rh, 12.0, 100.0)

    return pd.DataFrame(
        {
            "time": times,
            "temperature_2m": np.round(temperature, 2),
            "relative_humidity_2m": np.round(rh, 1),
            "precipitation": np.round(precipitation, 2),
            "dew_point_2m": np.round(dew_point_magnus(temperature, rh), 2),
        }
    )


# ---------------------------------------------------------------------------
# Open-Meteo client
# ---------------------------------------------------------------------------
class WeatherFetcher:
    """Thin, cached, retrying wrapper around the ``openmeteo-requests`` client."""

    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(
        self,
        cache_dir: str | Path = ".cache",
        cache_ttl_seconds: int = 3600,
        retries: int = 3,
        timeout_seconds: float = 10.0,
        timezone: str = DEFAULT_TIMEZONE,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_ttl_seconds = cache_ttl_seconds
        self.retries = retries
        self.timeout_seconds = timeout_seconds
        self.timezone = timezone
        self._client: Any | None = None

    # -- client -------------------------------------------------------------
    def _get_client(self) -> Any:
        """Lazily build the client so importing this module never needs the network stack."""
        if self._client is None:
            import openmeteo_requests
            import requests_cache
            from retry_requests import retry

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            session = requests_cache.CachedSession(
                str(self.cache_dir / "openmeteo"), expire_after=self.cache_ttl_seconds
            )
            session = retry(session, retries=self.retries, backoff_factor=0.3)
            self._client = openmeteo_requests.Client(session=session)
        return self._client

    def _request(self, url: str, params: dict[str, Any]) -> pd.DataFrame:
        params = {**params, "hourly": list(HOURLY_VARIABLES), "timezone": self.timezone}
        responses = self._get_client().weather_api(url, params=params, timeout=self.timeout_seconds)
        hourly = responses[0].Hourly()
        times = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
        frame = pd.DataFrame({"time": times})
        for idx, name in enumerate(HOURLY_VARIABLES):
            frame[name] = hourly.Variables(idx).ValuesAsNumpy()
        return normalize_weather_frame(frame, self.timezone)

    # -- synchronous API ----------------------------------------------------
    def fetch_historical(
        self,
        latitude: float = SHOPIAN.latitude,
        longitude: float = SHOPIAN.longitude,
        start_date: date | str = "2024-04-01",
        end_date: date | str = "2024-04-30",
    ) -> pd.DataFrame:
        """Hourly reanalysis (ERA5 / ERA5-Land) for a past date range."""
        return self._request(
            self.ARCHIVE_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": str(start_date),
                "end_date": str(end_date),
            },
        )

    def fetch_forecast(
        self,
        latitude: float = SHOPIAN.latitude,
        longitude: float = SHOPIAN.longitude,
        past_hours: int = 24,
        forecast_hours: int = 48,
    ) -> pd.DataFrame:
        """Recent observations-plus-forecast window around *now*."""
        return self._request(
            self.FORECAST_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "past_hours": past_hours,
                "forecast_hours": forecast_hours,
            },
        )

    # -- asynchronous API ---------------------------------------------------
    async def afetch_historical(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Async variant of :meth:`fetch_historical` (runs in a worker thread)."""
        return await asyncio.to_thread(self.fetch_historical, *args, **kwargs)

    async def afetch_forecast(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Async variant of :meth:`fetch_forecast` (runs in a worker thread)."""
        return await asyncio.to_thread(self.fetch_forecast, *args, **kwargs)

    # -- resilient entry point ----------------------------------------------
    def fetch_with_fallback(
        self,
        latitude: float = SHOPIAN.latitude,
        longitude: float = SHOPIAN.longitude,
        past_hours: int = 24,
        forecast_hours: int = 48,
        mock_scenario: str = "spring_mixed",
        use_mock: bool = False,
    ) -> WeatherResult:
        """Forecast window from Open-Meteo, or a synthetic one if that fails."""
        if not use_mock:
            try:
                data = self.fetch_forecast(latitude, longitude, past_hours, forecast_hours)
                if data.empty:
                    raise RuntimeError("Open-Meteo returned no hourly rows")
                return WeatherResult(data=data, source="open-meteo")
            except Exception as exc:  # noqa: BLE001 - any failure => degrade gracefully
                logger.warning("Open-Meteo unavailable (%s); using mock weather", exc)
                error: str | None = f"{type(exc).__name__}: {exc}"
        else:
            error = None
        start = pd.Timestamp.now(tz=self.timezone).floor("h") - timedelta(hours=past_hours)
        seed = int(abs(latitude * 1000) + abs(longitude * 1000)) % 10_000
        data = generate_mock_weather(
            start=start,
            hours=past_hours + forecast_hours,
            scenario=mock_scenario,
            seed=seed,
            timezone=self.timezone,
        )
        return WeatherResult(data=data, source="mock", error=error)


def load_weather_csv(path: str | Path, timezone: str = DEFAULT_TIMEZONE) -> pd.DataFrame:
    """Read a CSV with :data:`WEATHER_COLUMNS` and normalise it."""
    return normalize_weather_frame(pd.read_csv(path), timezone)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    logging.basicConfig(level=logging.INFO)
    result = WeatherFetcher().fetch_with_fallback()
    print(f"source={result.source} rows={len(result.data)}")
    print(result.data.head(12).to_string(index=False))
