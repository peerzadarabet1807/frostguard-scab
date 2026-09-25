"""Tests for the Revised Mills apple scab / frost physics engine."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
import pytest

from src.engine.mills_physics import (
    EngineConfig,
    FrostScabPhysicsEngine,
    FungicideAction,
    RiskLevel,
    required_wetness_hours,
    required_wetness_hours_array,
)
from src.engine.weather_fetcher import MOCK_SCENARIOS, generate_mock_weather
from tests.conftest import WeatherFactory

# (temperature, expected hours) incl. every band edge.
MILLS_CASES = [
    (4.0, 28.0),
    (5.0, 28.0),
    (6.0, 28.0),
    (6.01, 14.0),
    (8.0, 14.0),
    (8.5, 11.0),
    (10.0, 11.0),
    (10.5, 9.0),
    (12.0, 9.0),
    (12.1, 7.0),
    (15.0, 7.0),
    (15.1, 6.0),
    (20.0, 6.0),
    (24.0, 6.0),
]
BAND_MIDPOINTS = [(5.0, 28), (7.0, 14), (9.0, 11), (11.0, 9), (13.5, 7), (20.0, 6)]


@pytest.fixture
def engine() -> FrostScabPhysicsEngine:
    return FrostScabPhysicsEngine()


# ---------------------------------------------------------------------------
# Revised Mills table
# ---------------------------------------------------------------------------
class TestMillsTable:
    @pytest.mark.parametrize(("temp", "hours"), MILLS_CASES)
    def test_band_lookup(self, temp: float, hours: float) -> None:
        assert required_wetness_hours(temp) == hours

    @pytest.mark.parametrize("temp", [-10.0, 0.0, 3.99, 24.01, 30.0, math.nan])
    def test_outside_infective_range_is_infinite(self, temp: float) -> None:
        assert math.isinf(required_wetness_hours(temp))

    def test_vectorised_matches_scalar(self) -> None:
        temps = np.linspace(-5, 30, 701)
        vector = required_wetness_hours_array(temps)
        scalar = np.array([required_wetness_hours(t) for t in temps])
        np.testing.assert_array_equal(vector, scalar)

    def test_requirement_decreases_with_warmth(self) -> None:
        hours = [required_wetness_hours(t) for t, _ in BAND_MIDPOINTS]
        assert hours == sorted(hours, reverse=True)


# ---------------------------------------------------------------------------
# Leaf wetness
# ---------------------------------------------------------------------------
class TestLeafWetness:
    @pytest.mark.parametrize(
        ("rh", "rain", "expected"),
        [
            (90.0, 0.0, True),  # RH threshold is inclusive
            (89.9, 0.0, False),
            (100.0, 0.0, True),
            (50.0, 0.1, False),  # precipitation threshold is strict
            (50.0, 0.11, True),
            (50.0, 2.0, True),
            (np.nan, np.nan, False),  # missing data => dry
        ],
    )
    def test_thresholds(self, engine: FrostScabPhysicsEngine, rh: float, rain: float, expected: bool) -> None:
        assert bool(engine.leaf_wetness([rh], [rain])[0]) is expected

    def test_rain_alone_wets_leaves(self, engine: FrostScabPhysicsEngine) -> None:
        weather = pd.DataFrame(
            {
                "time": pd.date_range("2024-04-10", periods=6, freq="h"),
                "temperature_2m": [11.0] * 6,
                "relative_humidity_2m": [70.0] * 6,
                "precipitation": [1.5] * 6,
            }
        )
        timeline, _ = engine.compute_timeline(weather)
        assert timeline["leaf_wet"].all()
        assert timeline["wet_hours"].iloc[-1] == 6


# ---------------------------------------------------------------------------
# Infection accumulation
# ---------------------------------------------------------------------------
class TestInfectionPhysics:
    @pytest.mark.parametrize(("temp", "required"), BAND_MIDPOINTS)
    def test_infection_exactly_at_required_hours(
        self, engine: FrostScabPhysicsEngine, make_weather: WeatherFactory, temp: float, required: int
    ) -> None:
        timeline, events = engine.compute_timeline(make_weather(temp, hours=required))
        assert timeline["risk_ratio"].iloc[-2] < 1.0
        assert not timeline["infection"].iloc[-2]
        assert timeline["risk_ratio"].iloc[-1] == pytest.approx(1.0)
        assert timeline["infection"].iloc[-1]
        assert events[0].infected
        assert events[0].infection_time == timeline["time"].iloc[-1]

    @pytest.mark.parametrize(("temp", "required"), BAND_MIDPOINTS)
    def test_one_hour_short_is_not_infection(
        self, engine: FrostScabPhysicsEngine, make_weather: WeatherFactory, temp: float, required: int
    ) -> None:
        timeline, events = engine.compute_timeline(make_weather(temp, hours=required - 1))
        assert timeline["risk_ratio"].iloc[-1] == pytest.approx((required - 1) / required)
        assert not events[0].infected

    def test_ratio_is_wet_hours_over_requirement(
        self, engine: FrostScabPhysicsEngine, make_weather: WeatherFactory
    ) -> None:
        timeline, _ = engine.compute_timeline(make_weather(11.0, hours=3))
        assert timeline["risk_ratio"].tolist() == pytest.approx([1 / 9, 2 / 9, 3 / 9])

    def test_hourly_summation_across_bands(self, engine: FrostScabPhysicsEngine, make_weather: WeatherFactory) -> None:
        # 3 h @ 20 °C (3/6) + 3 h @ 13 °C (3/7) = 0.929 -> one more 13 °C hour infects.
        timeline, _ = engine.compute_timeline(make_weather([20.0] * 3 + [13.0] * 4))
        assert timeline["risk_ratio"].iloc[5] == pytest.approx(0.5 + 3 / 7)
        assert not timeline["infection"].iloc[5]
        assert timeline["infection"].iloc[6]

    def test_ratio_is_bounded_and_monotonic_while_wet(
        self, engine: FrostScabPhysicsEngine, make_weather: WeatherFactory
    ) -> None:
        timeline, events = engine.compute_timeline(make_weather(20.0, hours=40))
        risk = timeline["risk_ratio"].to_numpy()
        assert np.all(np.diff(risk) >= 0)
        assert risk.min() >= 0.0 and risk.max() == pytest.approx(1.0)
        assert len(events) == 1 and events[0].wet_hours == 40

    @pytest.mark.parametrize("temp", [-3.0, 2.0, 3.9, 25.0, 32.0])
    def test_no_progress_outside_infective_range(
        self, engine: FrostScabPhysicsEngine, make_weather: WeatherFactory, temp: float
    ) -> None:
        timeline, events = engine.compute_timeline(make_weather(temp, hours=48))
        assert timeline["risk_ratio"].max() == 0.0
        assert timeline["wet_hours"].iloc[-1] == 48  # still wet, just not infective
        assert not events[0].infected

    def test_dry_weather_has_zero_risk(self, engine: FrostScabPhysicsEngine, make_weather: WeatherFactory) -> None:
        timeline, events = engine.compute_timeline(make_weather(15.0, wet=[False] * 24))
        assert timeline["risk_ratio"].max() == 0.0
        assert events == []

    def test_mean_temperature_uses_wet_hours_only(
        self, engine: FrostScabPhysicsEngine, make_weather: WeatherFactory
    ) -> None:
        temps = [10.0, 12.0, 30.0, 14.0]
        _, events = engine.compute_timeline(make_weather(temps, wet=[True, True, False, True]))
        assert events[0].mean_temperature_c == pytest.approx(12.0)
        assert events[0].required_hours == 9.0


# ---------------------------------------------------------------------------
# Dry-period reset
# ---------------------------------------------------------------------------
class TestDryPeriodReset:
    def test_eight_dry_hours_reset_the_clock(
        self, engine: FrostScabPhysicsEngine, make_weather: WeatherFactory
    ) -> None:
        wet = [True] * 5 + [False] * 8 + [True] * 6
        timeline, events = engine.compute_timeline(make_weather(11.0, wet=wet))
        assert timeline["risk_ratio"].iloc[12] == 0.0  # reset on the 8th dry hour
        assert timeline["wet_hours"].iloc[12] == 0
        assert timeline["risk_ratio"].iloc[-1] == pytest.approx(6 / 9)
        assert len(events) == 2
        assert [e.wet_hours for e in events] == [5, 6]
        assert not any(e.infected for e in events)

    def test_seven_dry_hours_do_not_reset(self, engine: FrostScabPhysicsEngine, make_weather: WeatherFactory) -> None:
        wet = [True] * 5 + [False] * 7 + [True] * 4
        timeline, events = engine.compute_timeline(make_weather(11.0, wet=wet))
        assert len(events) == 1
        assert events[0].wet_hours == 9
        assert events[0].infected  # 5 + 4 = 9 wet hours = requirement at 11 °C
        assert timeline["infection"].iloc[-1]

    def test_dry_interruption_pauses_accumulation(
        self, engine: FrostScabPhysicsEngine, make_weather: WeatherFactory
    ) -> None:
        wet = [True] * 3 + [False] * 5
        timeline, _ = engine.compute_timeline(make_weather(11.0, wet=wet))
        paused = timeline["risk_ratio"].iloc[3:].to_numpy()
        assert np.allclose(paused, 3 / 9)
        assert timeline["dry_streak"].iloc[3:].tolist() == [1, 2, 3, 4, 5]

    def test_custom_reset_length(self, make_weather: WeatherFactory) -> None:
        engine = FrostScabPhysicsEngine(EngineConfig(dry_reset_hours=4))
        wet = [True] * 3 + [False] * 4 + [True] * 2
        _, events = engine.compute_timeline(make_weather(11.0, wet=wet))
        assert len(events) == 2

    def test_infection_is_recorded_before_reset(
        self, engine: FrostScabPhysicsEngine, make_weather: WeatherFactory
    ) -> None:
        wet = [True] * 7 + [False] * 10
        timeline, events = engine.compute_timeline(make_weather(13.0, wet=wet))
        assert events[0].infected
        assert timeline["risk_ratio"].iloc[-1] == 0.0


# ---------------------------------------------------------------------------
# Frost
# ---------------------------------------------------------------------------
class TestFrost:
    def test_frost_during_bud_break(self, engine: FrostScabPhysicsEngine, make_weather: WeatherFactory) -> None:
        timeline, _ = engine.compute_timeline(make_weather([-3.0, -1.0, 2.0], wet=[False] * 3, start="2024-04-10"))
        assert timeline["frost_risk"].tolist() == [True, False, False]

    def test_threshold_is_strict(self, engine: FrostScabPhysicsEngine, make_weather: WeatherFactory) -> None:
        timeline, _ = engine.compute_timeline(make_weather([-2.0, -2.01], wet=[False] * 2, start="2024-04-10"))
        assert timeline["frost_risk"].tolist() == [False, True]

    def test_no_frost_alert_outside_bud_break(
        self, engine: FrostScabPhysicsEngine, make_weather: WeatherFactory
    ) -> None:
        timeline, _ = engine.compute_timeline(make_weather(-8.0, wet=[False] * 6, start="2024-01-15"))
        assert not timeline["frost_risk"].any()
        assert not timeline["in_bud_break"].any()

    def test_bud_break_override(self, engine: FrostScabPhysicsEngine, make_weather: WeatherFactory) -> None:
        timeline, _ = engine.compute_timeline(make_weather(-8.0, wet=[False] * 6, start="2024-01-15"), bud_break=True)
        assert timeline["frost_risk"].all()

    def test_assessment_raises_frost_alert(self, engine: FrostScabPhysicsEngine) -> None:
        weather = generate_mock_weather(start="2024-04-05 00:00", hours=72, scenario="frost")
        result = engine.assess(weather)
        assert result.frost.alert
        assert result.frost.min_temperature_c is not None and result.frost.min_temperature_c < -2.0
        assert any(alert.startswith("FROST WARNING") for alert in result.alerts)


# ---------------------------------------------------------------------------
# Assessment, fungicide windows & risk index boundaries
# ---------------------------------------------------------------------------
class TestAssessment:
    @pytest.mark.parametrize(
        ("risk", "level"),
        [
            (0.0, RiskLevel.LOW),
            (0.299, RiskLevel.LOW),
            (0.3, RiskLevel.MODERATE),
            (0.699, RiskLevel.MODERATE),
            (0.7, RiskLevel.HIGH),
            (0.999, RiskLevel.HIGH),
            (1.0, RiskLevel.CRITICAL),
        ],
    )
    def test_risk_level_boundaries(self, engine: FrostScabPhysicsEngine, risk: float, level: RiskLevel) -> None:
        assert engine.classify(risk) is level

    def _infection_weather(self, make_weather: WeatherFactory) -> pd.DataFrame:
        # 9 wet hours at 11 °C (infection completes at hour 8), then 3 dry days.
        return make_weather(11.0, wet=[True] * 9 + [False] * 91)

    def test_protectant_window_before_infection(
        self, engine: FrostScabPhysicsEngine, make_weather: WeatherFactory
    ) -> None:
        result = engine.assess(self._infection_weather(make_weather))
        assert result.fungicide.action is FungicideAction.PROTECTANT
        assert result.fungicide.window_hours == pytest.approx(8.0)
        assert result.risk_level is RiskLevel.CRITICAL
        assert result.infection_probability == pytest.approx(1.0)

    def test_curative_window_after_infection(
        self, engine: FrostScabPhysicsEngine, make_weather: WeatherFactory
    ) -> None:
        weather = self._infection_weather(make_weather)
        as_of = weather["time"].iloc[12]
        result = engine.assess(weather, as_of=as_of)
        assert result.fungicide.action is FungicideAction.CURATIVE
        assert result.fungicide.window_hours == pytest.approx(72.0 - 12.0)
        assert result.fungicide.deadline == weather["time"].iloc[0] + pd.Timedelta(hours=72)
        assert result.risk_level is RiskLevel.CRITICAL
        assert result.alerts[0] == "CRITICAL RISK: Apply curative fungicide within 60 hours"

    def test_curative_window_closes(self, engine: FrostScabPhysicsEngine, make_weather: WeatherFactory) -> None:
        weather = self._infection_weather(make_weather)
        result = engine.assess(weather, as_of=weather["time"].iloc[80])
        assert result.fungicide.action is FungicideAction.NONE
        assert result.risk_level is RiskLevel.LOW

    def test_monitor_when_risk_is_building(
        self, engine: FrostScabPhysicsEngine, make_weather: WeatherFactory
    ) -> None:
        result = engine.assess(make_weather(11.0, wet=[True] * 5 + [False] * 20))
        assert result.fungicide.action is FungicideAction.MONITOR
        assert result.risk_level is RiskLevel.MODERATE
        assert result.peak_risk == pytest.approx(5 / 9)

    def test_lookback_captures_event_in_progress(
        self, engine: FrostScabPhysicsEngine, make_weather: WeatherFactory
    ) -> None:
        weather = make_weather(11.0, hours=60)
        result = engine.assess(weather, as_of=weather["time"].iloc[4], horizon_hours=48)
        assert result.current_risk == pytest.approx(5 / 9)
        assert len(result.timeline) == 48
        assert result.timeline["time"].iloc[0] == weather["time"].iloc[4]

    def test_horizon_length(self, engine: FrostScabPhysicsEngine) -> None:
        weather = generate_mock_weather(start="2024-04-12", hours=96, scenario="spring_mixed")
        result = engine.assess(weather, horizon_hours=48)
        assert len(result.timeline) == 48
        assert len(result.full_timeline) == 96

    @pytest.mark.parametrize("scenario", sorted(MOCK_SCENARIOS))
    def test_risk_index_bounded_for_all_scenarios(self, engine: FrostScabPhysicsEngine, scenario: str) -> None:
        weather = generate_mock_weather(start="2024-04-12", hours=120, scenario=scenario, seed=7)
        result = engine.assess(weather, horizon_hours=None)
        risk = result.full_timeline["risk_ratio"]
        assert risk.between(0.0, 1.0).all()
        assert 0.0 <= result.infection_probability <= 1.0

    def test_scenarios_rank_sensibly(self, engine: FrostScabPhysicsEngine) -> None:
        outbreak = engine.assess(generate_mock_weather(start="2024-04-12", hours=72, scenario="scab_outbreak"))
        dry = engine.assess(generate_mock_weather(start="2024-04-12", hours=72, scenario="dry"))
        assert outbreak.risk_level is RiskLevel.CRITICAL
        assert dry.risk_level is RiskLevel.LOW
        assert dry.fungicide.action is FungicideAction.NONE

    def test_to_dict_is_json_serialisable(self, engine: FrostScabPhysicsEngine) -> None:
        result = engine.assess(generate_mock_weather(start="2024-04-12", hours=72, scenario="scab_outbreak"))
        payload = json.loads(json.dumps(result.to_dict()))
        assert len(payload["timeline"]) == 48
        assert {"risk_level", "fungicide", "frost", "infection_events", "alerts"} <= payload.keys()

    def test_empty_weather_is_rejected(self, engine: FrostScabPhysicsEngine) -> None:
        empty = pd.DataFrame(columns=["time", "temperature_2m", "relative_humidity_2m", "precipitation"])
        with pytest.raises(ValueError):
            engine.assess(empty)


def test_engine_throughput(engine: FrostScabPhysicsEngine) -> None:
    """Sanity floor for the >10k hours/s benchmark target (lenient for slow CI)."""
    weather = generate_mock_weather(start="2020-03-01", hours=20_000, scenario="scab_outbreak")
    engine.compute_timeline(weather)  # warm-up
    t0 = time.perf_counter()
    engine.compute_timeline(weather)
    elapsed = time.perf_counter() - t0
    assert 20_000 / elapsed > 10_000
