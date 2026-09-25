"""Physics-informed apple scab (*Venturia inaequalis*) and spring-frost risk engine.

Model summary
-------------
1. **Leaf wetness** — an hour is *wet* when relative humidity >= 90 % or
   precipitation > 0.1 mm.
2. **Wetting events** — consecutive wet hours form an event. Short dry interruptions
   pause the clock (no progress) but do not end the event; **8 consecutive dry hours**
   end it and reset the accumulated wetness (Revised Mills / MacHardy & Gadoury 1989).
3. **Revised Mills table** — each temperature band needs a minimum number of wet hours
   for ascospores to infect:

   ============  =========
   T (°C)        hours
   ============  =========
   [4, 6]        28
   (6, 8]        14
   (8, 10]       11
   (10, 12]      9
   (12, 15]      7
   (15, 24]      6
   ============  =========

   Outside 4-24 °C no infection progress is made.
4. **Infection risk ratio** — every wet hour at temperature *T* contributes
   ``1 / required_hours(T)``; an event's ratio is the running sum clipped to [0, 1].
   Under constant temperature this is exactly ``wet_hours / required_hours`` (classic
   Mills), but it stays physically consistent when temperature drifts through bands.
   A ratio of 1.0 means a Mills infection period has been completed.
5. **Curative window** — post-infection ("kick-back") fungicides such as DMIs act for a
   limited time counted from the *start* of the infection's wetting event
   (default 72 h).
6. **Frost** — during the bud-break window any hour colder than -2.0 °C is flagged as
   frost-injury risk for green tip → petal-fall tissue.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.engine.weather_fetcher import normalize_weather_frame

_INFECTION_EPS = 1e-9

# ---------------------------------------------------------------------------
# Revised Mills table
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MillsBand:
    """One temperature band of the Revised Mills table (upper bound inclusive)."""

    lower_c: float
    upper_c: float
    required_hours: float
    lower_inclusive: bool = False

    def contains(self, temperature_c: float) -> bool:
        above = temperature_c >= self.lower_c if self.lower_inclusive else temperature_c > self.lower_c
        return above and temperature_c <= self.upper_c


REVISED_MILLS_TABLE: tuple[MillsBand, ...] = (
    MillsBand(4.0, 6.0, 28.0, lower_inclusive=True),
    MillsBand(6.0, 8.0, 14.0),
    MillsBand(8.0, 10.0, 11.0),
    MillsBand(10.0, 12.0, 9.0),
    MillsBand(12.0, 15.0, 7.0),
    MillsBand(15.0, 24.0, 6.0),
)


def required_wetness_hours(temperature_c: float) -> float:
    """Wet hours needed for infection at ``temperature_c``; ``math.inf`` outside 4-24 °C."""
    if temperature_c is None or math.isnan(temperature_c):
        return math.inf
    for band in REVISED_MILLS_TABLE:
        if band.contains(temperature_c):
            return band.required_hours
    return math.inf


def required_wetness_hours_array(temperatures_c: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Vectorised :func:`required_wetness_hours` (NaN => ``inf``)."""
    t = np.asarray(temperatures_c, dtype=float)
    conditions = [
        ((t > b.lower_c) | ((t == b.lower_c) & b.lower_inclusive)) & (t <= b.upper_c)
        for b in REVISED_MILLS_TABLE
    ]
    return np.select(conditions, [b.required_hours for b in REVISED_MILLS_TABLE], default=np.inf)


# ---------------------------------------------------------------------------
# Configuration & results
# ---------------------------------------------------------------------------
class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"  # Mills infection period completed


class FungicideAction(str, Enum):
    NONE = "none"
    MONITOR = "monitor"
    PROTECTANT = "protectant"
    CURATIVE = "curative"


@dataclass(frozen=True)
class EngineConfig:
    """Tunable agronomic thresholds."""

    rh_wet_threshold: float = 90.0  # %, inclusive
    precip_wet_threshold_mm: float = 0.1  # mm/h, strictly greater
    dry_reset_hours: int = 8
    curative_window_hours: float = 72.0
    frost_threshold_c: float = -2.0  # strictly below
    # Green tip -> petal fall for Kashmir valley orchards (month, day), inclusive.
    bud_break_start: tuple[int, int] = (3, 15)
    bud_break_end: tuple[int, int] = (5, 20)
    moderate_threshold: float = 0.3
    high_threshold: float = 0.7


@dataclass
class WettingEvent:
    """A continuous leaf-wetness episode (short dry breaks included)."""

    event_id: int
    start: pd.Timestamp
    end: pd.Timestamp
    wet_hours: int
    mean_temperature_c: float
    risk_ratio: float
    infection_time: pd.Timestamp | None = None

    @property
    def infected(self) -> bool:
        return self.infection_time is not None

    @property
    def required_hours(self) -> float:
        """Classic-Mills requirement at the event's mean wet-hour temperature."""
        return required_wetness_hours(self.mean_temperature_c)

    def curative_deadline(self, curative_window_hours: float) -> pd.Timestamp:
        return self.start + pd.Timedelta(hours=curative_window_hours)

    def to_dict(self) -> dict[str, Any]:
        req = self.required_hours
        return {
            "event_id": self.event_id,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "wet_hours": self.wet_hours,
            "mean_temperature_c": round(self.mean_temperature_c, 2),
            "required_hours": None if math.isinf(req) else req,
            "risk_ratio": round(self.risk_ratio, 4),
            "infected": self.infected,
            "infection_time": self.infection_time.isoformat() if self.infection_time is not None else None,
        }


@dataclass(frozen=True)
class FungicideAdvice:
    action: FungicideAction
    window_hours: float | None
    deadline: pd.Timestamp | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "window_hours": None if self.window_hours is None else round(self.window_hours, 1),
            "deadline": None if self.deadline is None else self.deadline.isoformat(),
            "message": self.message,
        }


@dataclass(frozen=True)
class FrostAssessment:
    alert: bool
    frost_hours: int
    min_temperature_c: float | None
    first_frost_time: pd.Timestamp | None
    in_bud_break: bool
    threshold_c: float

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["first_frost_time"] = None if self.first_frost_time is None else self.first_frost_time.isoformat()
        out["min_temperature_c"] = None if self.min_temperature_c is None else round(self.min_temperature_c, 2)
        return out


@dataclass
class RiskAssessment:
    """Everything a caller needs to render a 48 h risk dashboard."""

    as_of: pd.Timestamp
    timeline: pd.DataFrame  # rows from ``as_of`` onwards (the forecast horizon)
    full_timeline: pd.DataFrame  # every analysed hour, including look-back
    events: list[WettingEvent]
    current_risk: float
    peak_risk: float
    infection_probability: float
    risk_level: RiskLevel
    fungicide: FungicideAdvice
    frost: FrostAssessment
    alerts: list[str] = field(default_factory=list)

    def to_dict(self, include_timeline: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "as_of": self.as_of.isoformat(),
            "current_risk": round(self.current_risk, 4),
            "peak_risk": round(self.peak_risk, 4),
            "infection_probability": round(self.infection_probability, 4),
            "risk_level": self.risk_level.value,
            "fungicide": self.fungicide.to_dict(),
            "frost": self.frost.to_dict(),
            "infection_events": [e.to_dict() for e in self.events if e.risk_ratio > 0],
            "alerts": list(self.alerts),
        }
        if include_timeline:
            out["timeline"] = timeline_records(self.timeline)
        return out


def timeline_records(timeline: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON-friendly rows of an engine timeline."""
    records: list[dict[str, Any]] = []
    for row in timeline.itertuples(index=False):
        records.append(
            {
                "time": row.time.isoformat(),
                "temperature_c": _round_or_none(row.temperature_2m, 2),
                "relative_humidity": _round_or_none(row.relative_humidity_2m, 1),
                "precipitation_mm": _round_or_none(row.precipitation, 2),
                "leaf_wet": bool(row.leaf_wet),
                "wet_hours": int(row.wet_hours),
                "risk_ratio": round(float(row.risk_ratio), 4),
                "infection": bool(row.infection),
                "frost_risk": bool(row.frost_risk),
            }
        )
    return records


def _round_or_none(value: float, ndigits: int) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), ndigits)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class FrostScabPhysicsEngine:
    """Hour-by-hour state machine implementing the Revised Mills model plus frost logic."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    # -- building blocks ----------------------------------------------------
    def leaf_wetness(self, relative_humidity: npt.ArrayLike, precipitation: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """Boolean wetness per hour (NaNs count as dry)."""
        rh = np.nan_to_num(np.asarray(relative_humidity, dtype=float), nan=0.0)
        rain = np.nan_to_num(np.asarray(precipitation, dtype=float), nan=0.0)
        return (rh >= self.config.rh_wet_threshold) | (rain > self.config.precip_wet_threshold_mm)

    def in_bud_break_window(self, times: pd.Series) -> npt.NDArray[np.bool_]:
        """Phenology mask from calendar dates (inclusive window, no year wrap)."""
        key = times.dt.month.to_numpy() * 100 + times.dt.day.to_numpy()
        start = self.config.bud_break_start[0] * 100 + self.config.bud_break_start[1]
        end = self.config.bud_break_end[0] * 100 + self.config.bud_break_end[1]
        return (key >= start) & (key <= end)

    # -- core simulation ----------------------------------------------------
    def compute_timeline(
        self, weather: pd.DataFrame, bud_break: bool | None = None
    ) -> tuple[pd.DataFrame, list[WettingEvent]]:
        """Run the wetness/infection state machine over hourly ``weather``.

        Args:
            weather: frame with ``time``, ``temperature_2m`` and humidity/precip columns.
            bud_break: force the phenology flag; ``None`` infers it from the date.

        Returns:
            ``(timeline, events)`` — per-hour state and the list of wetting events.
        """
        df = normalize_weather_frame(weather)
        n = len(df)
        temps = df["temperature_2m"].to_numpy(dtype=float)
        wet = self.leaf_wetness(df["relative_humidity_2m"], df["precipitation"])
        rates = 1.0 / required_wetness_hours_array(temps)  # inf -> 0 progress
        times = list(df["time"])

        wet_hours_col = np.zeros(n, dtype=np.int32)
        dry_streak_col = np.zeros(n, dtype=np.int32)
        event_col = np.full(n, -1, dtype=np.int32)
        risk_col = np.zeros(n, dtype=float)
        infection_col = np.zeros(n, dtype=bool)

        events: list[WettingEvent] = []
        active: WettingEvent | None = None
        progress = 0.0
        temp_sum = 0.0
        dry_streak = 0
        reset_after = self.config.dry_reset_hours

        for i in range(n):
            if wet[i]:
                if active is None:
                    active = WettingEvent(len(events), times[i], times[i], 0, math.nan, 0.0)
                    events.append(active)
                    progress, temp_sum = 0.0, 0.0
                dry_streak = 0
                active.wet_hours += 1
                active.end = times[i]
                if not math.isnan(temps[i]):
                    temp_sum += temps[i]
                progress += rates[i]
                if progress >= 1.0 - _INFECTION_EPS:  # absorb float drift from summing 1/28 etc.
                    progress = max(progress, 1.0)
                    if active.infection_time is None:
                        active.infection_time = times[i]
                active.mean_temperature_c = temp_sum / active.wet_hours
                active.risk_ratio = min(progress, 1.0)
            elif active is not None:
                dry_streak += 1
                if dry_streak >= reset_after:  # wetness clock reset
                    dry_streak_col[i] = dry_streak
                    active, progress, dry_streak = None, 0.0, 0
                    continue

            if active is not None:
                wet_hours_col[i] = active.wet_hours
                event_col[i] = active.event_id
                risk_col[i] = min(progress, 1.0)
                infection_col[i] = active.infected
            dry_streak_col[i] = dry_streak

        in_bud_break = np.full(n, bool(bud_break)) if bud_break is not None else self.in_bud_break_window(df["time"])
        frost = in_bud_break & (np.nan_to_num(temps, nan=np.inf) < self.config.frost_threshold_c)

        timeline = df.assign(
            leaf_wet=wet,
            wet_hours=wet_hours_col,
            dry_streak=dry_streak_col,
            event_id=event_col,
            required_hours=required_wetness_hours_array(temps),
            risk_ratio=risk_col,
            infection=infection_col,
            in_bud_break=in_bud_break,
            frost_risk=frost,
        )
        return timeline, events

    # -- decision layer -----------------------------------------------------
    def classify(self, risk: float) -> RiskLevel:
        if risk >= 1.0:
            return RiskLevel.CRITICAL
        if risk >= self.config.high_threshold:
            return RiskLevel.HIGH
        if risk >= self.config.moderate_threshold:
            return RiskLevel.MODERATE
        return RiskLevel.LOW

    def fungicide_advice(
        self, events: list[WettingEvent], as_of: pd.Timestamp, peak_risk: float
    ) -> FungicideAdvice:
        """Recommend a spray timing relative to ``as_of``."""
        window = self.config.curative_window_hours
        open_infections = [
            e for e in events if e.infected and e.curative_deadline(window) > as_of
        ]
        if open_infections:
            # Most urgent first: the curative window that closes soonest.
            event = min(open_infections, key=lambda e: e.curative_deadline(window))
            deadline = event.curative_deadline(window)
            assert event.infection_time is not None
            if event.infection_time <= as_of:
                hours = (deadline - as_of).total_seconds() / 3600.0
                return FungicideAdvice(
                    FungicideAction.CURATIVE,
                    hours,
                    deadline,
                    f"Mills infection period completed at {event.infection_time:%d %b %H:%M}. "
                    f"Apply a curative (kick-back) fungicide within {hours:.0f} hours.",
                )
            hours = (event.infection_time - as_of).total_seconds() / 3600.0
            return FungicideAdvice(
                FungicideAction.PROTECTANT,
                hours,
                event.infection_time,
                f"Infection period forecast to complete in {hours:.0f} hours "
                f"({event.infection_time:%d %b %H:%M}). Apply a protectant fungicide before then; "
                f"curative back-up possible until {deadline:%d %b %H:%M}.",
            )
        if peak_risk >= self.config.moderate_threshold:
            return FungicideAdvice(
                FungicideAction.MONITOR,
                None,
                None,
                f"Wetting is building scab risk ({peak_risk:.0%} of a Mills period). "
                "Keep protectant cover current and re-check the forecast.",
            )
        return FungicideAdvice(FungicideAction.NONE, None, None, "No scab infection period expected. No spray needed.")

    def assess(
        self,
        weather: pd.DataFrame,
        as_of: pd.Timestamp | str | None = None,
        horizon_hours: int | None = 48,
        bud_break: bool | None = None,
    ) -> RiskAssessment:
        """Full risk assessment.

        Args:
            weather: hourly weather; may include look-back hours before ``as_of`` so that
                wetting events already in progress are accounted for.
            as_of: the decision time ("now"). Defaults to the first timestamp.
            horizon_hours: length of the returned forward timeline (``None`` = all).
            bud_break: force the phenology flag (``None`` = infer from date).
        """
        full, events = self.compute_timeline(weather, bud_break=bud_break)
        if full.empty:
            raise ValueError("weather data is empty")
        tz = full["time"].dt.tz
        if as_of is None:
            as_of_ts = full["time"].iloc[0]
        else:
            as_of_ts = pd.Timestamp(as_of)
            as_of_ts = as_of_ts.tz_localize(tz) if as_of_ts.tz is None else as_of_ts.tz_convert(tz)
        as_of_ts = as_of_ts.floor("h")

        horizon = full[full["time"] >= as_of_ts]
        if horizon_hours is not None:
            horizon = horizon.head(horizon_hours)
        past = full[full["time"] <= as_of_ts]
        current_risk = float(past["risk_ratio"].iloc[-1]) if not past.empty else 0.0
        horizon_peak = float(horizon["risk_ratio"].max()) if not horizon.empty else 0.0
        peak_risk = max(current_risk, horizon_peak)

        fungicide = self.fungicide_advice(events, as_of_ts, peak_risk)
        if fungicide.action is FungicideAction.CURATIVE:
            peak_risk = 1.0  # an infection is on the books until its window closes

        frost_rows = horizon[horizon["frost_risk"]]
        frost = FrostAssessment(
            alert=not frost_rows.empty,
            frost_hours=int(len(frost_rows)),
            min_temperature_c=float(horizon["temperature_2m"].min()) if not horizon.empty else None,
            first_frost_time=frost_rows["time"].iloc[0] if not frost_rows.empty else None,
            in_bud_break=bool(horizon["in_bud_break"].any()) if not horizon.empty else False,
            threshold_c=self.config.frost_threshold_c,
        )

        level = self.classify(peak_risk)
        alerts = self._build_alerts(level, fungicide, frost)
        return RiskAssessment(
            as_of=as_of_ts,
            timeline=horizon.reset_index(drop=True),
            full_timeline=full,
            events=events,
            current_risk=current_risk,
            peak_risk=peak_risk,
            infection_probability=min(peak_risk, 1.0),
            risk_level=level,
            fungicide=fungicide,
            frost=frost,
            alerts=alerts,
        )

    @staticmethod
    def _build_alerts(level: RiskLevel, fungicide: FungicideAdvice, frost: FrostAssessment) -> list[str]:
        alerts: list[str] = []
        if fungicide.action is FungicideAction.CURATIVE:
            alerts.append(
                f"{level.value} RISK: Apply curative fungicide within {fungicide.window_hours:.0f} hours"
            )
        elif fungicide.action is FungicideAction.PROTECTANT:
            alerts.append(
                f"{level.value} RISK: Scab infection expected in {fungicide.window_hours:.0f} hours "
                "- apply protectant fungicide now"
            )
        elif fungicide.action is FungicideAction.MONITOR:
            alerts.append(f"{level.value} RISK: Leaf wetness building - monitor and maintain protectant cover")
        if frost.alert and frost.first_frost_time is not None:
            alerts.append(
                f"FROST WARNING: {frost.frost_hours} h below {frost.threshold_c:.1f} °C during bud break "
                f"(min {frost.min_temperature_c:.1f} °C from {frost.first_frost_time:%d %b %H:%M}) "
                "- activate frost protection"
            )
        return alerts
