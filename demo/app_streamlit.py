"""FrostGuard-Scab interactive dashboard.

Run from the repository root::

    streamlit run demo/app_streamlit.py

The dashboard talks to the FastAPI service at ``FROSTGUARD_API_URL`` (default
``http://localhost:8000``). If that is not reachable it mounts the very same FastAPI app
in-process, so it works as a single command with identical request/response contracts.
"""

from __future__ import annotations

import io
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402
import pandas as pd  # noqa: E402
import pydeck as pdk  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from demo.charts import DARK, LIGHT, Palette, draw_detections, risk_chart  # noqa: E402
from src.engine.weather_fetcher import KASHMIR_ORCHARD_ZONES, MOCK_SCENARIOS, load_weather_csv  # noqa: E402

API_URL = os.getenv("FROSTGUARD_API_URL", "http://localhost:8000").rstrip("/")
DATA_DIR = PROJECT_ROOT / "data"
REPLAY_CSV = DATA_DIR / "shopian_spring_2024_hourly.csv"
HORIZON_HOURS = 48

st.set_page_config(page_title="FrostGuard-Scab", page_icon="🍎", layout="wide")


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
def palette() -> Palette:
    try:
        return DARK if st.context.theme.type == "dark" else LIGHT
    except AttributeError:  # older Streamlit without st.context.theme
        return LIGHT


# ---------------------------------------------------------------------------
# Backend: remote FastAPI service, or the same app mounted in-process
# ---------------------------------------------------------------------------
class ApiBackend:
    def __init__(self) -> None:
        self.remote = self._reachable(API_URL)
        if self.remote:
            self.client: Any = httpx.Client(base_url=API_URL, timeout=60.0)
            self.label = f"FastAPI service at {API_URL}"
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from fastapi.testclient import TestClient

                from src.api.app import app

            self.client = TestClient(app)
            self.client.__enter__()  # run the lifespan (loads engine + detector once)
            self.label = "Embedded API (in-process)"

    @staticmethod
    def _reachable(url: str) -> bool:
        try:
            return httpx.get(f"{url}/health", timeout=1.5).status_code == 200
        except httpx.HTTPError:
            return False

    def health(self) -> dict[str, Any]:
        return self.client.get("/health").json()

    def predict_risk(self, payload: dict[str, Any]) -> dict[str, Any]:
        res = self.client.post("/api/v1/predict-risk", json=payload)
        res.raise_for_status()
        return res.json()

    def detect(self, image_bytes: bytes, filename: str) -> dict[str, Any]:
        mime = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
        res = self.client.post("/api/v1/detect-lesion", files={"file": (filename, image_bytes, mime)})
        res.raise_for_status()
        return res.json()


@st.cache_resource(show_spinner="Starting FrostGuard engine…")
def get_backend() -> ApiBackend:
    return ApiBackend()


@st.cache_data(ttl=600, show_spinner="Scoring infection risk…")
def cached_risk(payload_json: str) -> dict[str, Any]:
    return get_backend().predict_risk(json.loads(payload_json))


@st.cache_data(show_spinner=False)
def replay_weather() -> pd.DataFrame:
    return load_weather_csv(REPLAY_CSV)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
LEVEL_STYLE = {
    "LOW": ("green", ":material/check_circle:"),
    "MODERATE": ("orange", ":material/warning:"),
    "HIGH": ("red", ":material/error:"),
    "CRITICAL": ("red", ":material/crisis_alert:"),
}


def zone_map(selected: str) -> None:
    """Clickable map of the orchard zones; a click selects the zone."""
    pal = palette()
    points = pd.DataFrame(
        [
            {
                "key": z.key,
                "name": z.name,
                "lat": z.latitude,
                "lon": z.longitude,
                "elev": z.elevation_m,
                "color": _hex_rgba(pal.risk if z.key == selected else pal.muted, 235 if z.key == selected else 170),
                "radius": 3200 if z.key == selected else 2200,
            }
            for z in KASHMIR_ORCHARD_ZONES.values()
        ]
    )
    deck = pdk.Deck(
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                id="zones",
                data=points,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius="radius",
                stroked=True,
                get_line_color=[255, 255, 255, 220],
                line_width_min_pixels=2,
                pickable=True,
            ),
            pdk.Layer(
                "TextLayer",
                id="labels",
                data=points,
                get_position="[lon, lat]",
                get_text="name",
                get_size=14,
                get_pixel_offset=[0, -22],
                get_color=list(pal.map_label),
                font_weight=600,
            ),
        ],
        initial_view_state=pdk.ViewState(latitude=34.01, longitude=74.63, zoom=7.9),
        map_style=pdk.map_styles.CARTO_LIGHT if pal is LIGHT else pdk.map_styles.CARTO_DARK,
        tooltip={"text": "{name}\n{elev} m a.s.l.\nClick to select"},
    )
    try:
        st.pydeck_chart(deck, key="zone_map", on_select="rerun", selection_mode="single-object", height=330)
    except TypeError:  # Streamlit without pydeck selections
        st.pydeck_chart(deck, height=330)


def _hex_rgba(hex_color: str, alpha: int) -> list[int]:
    h = hex_color.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)] + [alpha]


def sync_zone_from_map() -> None:
    """Apply a map click to the zone selector *before* that widget is created."""
    state = st.session_state.get("zone_map")
    try:
        objects = state["selection"]["objects"].get("zones", []) if state else []
    except (KeyError, TypeError, AttributeError):
        objects = []
    if objects:
        picked = objects[0].get("key")
        if picked in KASHMIR_ORCHARD_ZONES and picked != st.session_state.get("_last_map_pick"):
            st.session_state["_last_map_pick"] = picked
            st.session_state["zone"] = picked


def show_alerts(result: dict[str, Any]) -> None:
    alerts = result["alerts"]
    if not alerts:
        st.success("LOW RISK: No scab infection period expected in the next 48 h. No spray needed.", icon="✅")
    for alert in alerts:
        if alert.startswith("FROST"):
            st.error(alert, icon="❄️")
        elif alert.startswith(("CRITICAL", "HIGH")):
            st.error(alert, icon="🚨")
        else:
            st.warning(alert, icon="⚠️")
    for warning in result.get("warnings", []):
        st.caption(f"ℹ️ {warning}")


def show_metrics(result: dict[str, Any]) -> None:
    fungicide, frost = result["fungicide"], result["frost"]
    level = result["risk_level"]
    color, icon = LEVEL_STYLE[level]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Infection probability", f"{result['infection_probability']:.0%}", help="Peak share of the Revised Mills wetness requirement met")
        st.badge(level, icon=icon, color=color)
    with c2:
        window = fungicide["window_hours"]
        st.metric(
            "Fungicide window",
            f"{window:.0f} h" if window is not None else "—",
            help="Hours left to act, counted from the decision time",
        )
        st.caption(fungicide["action"].capitalize())
    with c3:
        st.metric("Current risk", f"{result['current_risk']:.0%}", help="Mills requirement met right now in the ongoing wetting event")
        st.caption(f"{sum(p['leaf_wet'] for p in result['timeline'])} wet hours ahead")
    with c4:
        min_t = frost["min_temperature_c"]
        st.metric("Minimum temperature", f"{min_t:.1f} °C" if min_t is not None else "—")
        st.caption(
            f"❄️ {frost['frost_hours']} h below {frost['threshold_c']:.0f} °C"
            if frost["alert"]
            else ("Bud-break window: frost-sensitive" if frost["in_bud_break"] else "Outside bud-break window")
        )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
backend = get_backend()
health = backend.health()

with st.sidebar:
    st.title("🍎 FrostGuard-Scab")
    st.caption("Apple scab & frost early warning for Kashmir orchards")
    source = st.radio(
        "Weather source",
        ["Live forecast", "Historical replay", "Simulated scenario"],
        help="Live: Open-Meteo forecast. Replay: real ERA5 hours for Shopian, spring 2024. Simulated: offline scenarios.",
    )
    scenario = "spring_mixed"
    replay_as_of: pd.Timestamp | None = None
    if source == "Simulated scenario":
        scenario = st.selectbox(
            "Scenario", sorted(MOCK_SCENARIOS), index=sorted(MOCK_SCENARIOS).index("scab_outbreak"),
            format_func=lambda s: s.replace("_", " ").title(),
        )
    elif source == "Historical replay":
        replay = replay_weather()
        first, last = replay["time"].iloc[24], replay["time"].iloc[-HORIZON_HOURS]
        day = st.date_input("Decision date", value=pd.Timestamp("2024-04-26").date(), min_value=first.date(), max_value=last.date())
        hour = st.slider("Decision hour", 0, 23, 6)
        replay_as_of = pd.Timestamp(day).tz_localize(replay["time"].dt.tz) + pd.Timedelta(hours=hour)
        replay_as_of = min(max(replay_as_of, first), last)
        st.caption("Replay uses Shopian ERA5 reanalysis (1 Apr – 15 May 2024).")
    bud = st.selectbox("Bud-break stage", ["Auto (from date)", "Yes", "No"], help="Frost alerts only fire during bud break")
    st.divider()
    st.caption(f"**Backend:** {backend.label}")
    backend_kind = "trained YOLO11n" if health["vision_backend"] == "trained" else "fallback heuristic"
    st.caption(f"**Vision model:** `{health['model']}` ({backend_kind})")
    st.caption(f"[API docs]({API_URL}/docs)" if backend.remote else "Start `uvicorn src.api.app:app` for the REST API")

st.title("Orchard scab & frost risk")
st.caption("Revised Mills infection periods from hourly leaf wetness and temperature, plus lesion confirmation from leaf photos.")

# --- Zone selection ----------------------------------------------------------
st.session_state.setdefault("zone", "shopian")
sync_zone_from_map()
map_col, info_col = st.columns([1.35, 1], gap="large")
with info_col:
    zone_key = st.selectbox(
        "Orchard zone",
        list(KASHMIR_ORCHARD_ZONES),
        key="zone",
        format_func=lambda k: KASHMIR_ORCHARD_ZONES[k].name,
        help="Or click a zone on the map",
    )
    zone = KASHMIR_ORCHARD_ZONES[zone_key]
    st.markdown(f"**{zone.name}** · {zone.latitude:.3f}° N, {zone.longitude:.3f}° E · {zone.elevation_m} m a.s.l.")
    if source == "Historical replay" and zone_key != "shopian":
        st.info("Historical replay data is for Shopian; using Shopian weather.", icon="ℹ️")
with map_col:
    zone_map(zone_key)

# --- Risk ---------------------------------------------------------------------
payload: dict[str, Any] = {"zone": zone_key, "horizon_hours": HORIZON_HOURS}
if bud != "Auto (from date)":
    payload["bud_break"] = bud == "Yes"
if source == "Simulated scenario":
    payload.update(use_mock=True, mock_scenario=scenario)
    payload.setdefault("bud_break", True)  # the simulated regimes are spring (bud-break) weather
elif source == "Historical replay" and replay_as_of is not None:
    window = replay[(replay["time"] >= replay_as_of - pd.Timedelta(hours=24)) & (replay["time"] < replay_as_of + pd.Timedelta(hours=HORIZON_HOURS))]
    records = window.assign(time=window["time"].map(pd.Timestamp.isoformat)).to_dict(orient="records")
    payload.update(zone="shopian", weather=records, as_of=replay_as_of.isoformat())

try:
    result = cached_risk(json.dumps(payload, sort_keys=True, default=str))
except httpx.HTTPError as exc:
    st.error(f"Risk service error: {exc}")
    st.stop()

with info_col:
    source_label = {"open-meteo": "Open-Meteo live forecast", "mock": "simulated weather", "uploaded": "ERA5 replay"}[result["weather_source"]]
    st.caption(f"Decision time **{pd.Timestamp(result['as_of']):%d %b %Y, %H:%M}** · {source_label}")
    show_alerts(result)

st.subheader("Next 48 hours")
show_metrics(result)
st.caption(result["fungicide"]["message"])
timeline = pd.DataFrame(result["timeline"])
st.altair_chart(risk_chart(timeline, palette()), width="stretch")

with st.expander("Table view: hourly timeline"):
    table = timeline.assign(time=pd.to_datetime(timeline["time"]).dt.strftime("%d %b %H:%M"), risk_pct=(timeline["risk_ratio"] * 100).round(1))
    st.dataframe(
        table[["time", "temperature_c", "relative_humidity", "precipitation_mm", "leaf_wet", "wet_hours", "risk_pct", "infection", "frost_risk"]],
        hide_index=True,
        width="stretch",
        column_config={
            "time": "Hour",
            "temperature_c": st.column_config.NumberColumn("Temp (°C)", format="%.1f"),
            "relative_humidity": st.column_config.NumberColumn("RH (%)", format="%.0f"),
            "precipitation_mm": st.column_config.NumberColumn("Rain (mm)", format="%.1f"),
            "leaf_wet": "Leaf wet",
            "wet_hours": "Wet h",
            "risk_pct": st.column_config.ProgressColumn("Risk (%)", min_value=0, max_value=100, format="%.0f"),
            "infection": "Infection",
            "frost_risk": "Frost",
        },
    )
events = [e for e in result["infection_events"] if e["risk_ratio"] >= 0.3]
with st.expander(f"Wetting events ({len(events)} significant)"):
    if events:
        ev = pd.DataFrame(events)
        for col in ("start", "end", "infection_time"):
            ev[col] = pd.to_datetime(ev[col]).dt.strftime("%d %b %H:%M")
        ev["risk_pct"] = (ev["risk_ratio"] * 100).round(0)
        st.dataframe(
            ev[["start", "end", "wet_hours", "mean_temperature_c", "required_hours", "risk_pct", "infected", "infection_time"]],
            hide_index=True,
            width="stretch",
        )
    else:
        st.write("No wetting event reached 30 % of a Mills infection period.")

# --- Leaf scan ----------------------------------------------------------------
st.divider()
st.subheader("Leaf scan: confirm lesions")
up_col, out_col = st.columns([1, 1.6], gap="large")
with up_col:
    upload = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png", "webp"])
    sample = st.radio("…or try a sample", ["None", "Scabbed leaf (synthetic)", "Healthy leaf (synthetic)"], key="leaf_sample")
    if health["vision_backend"] == "fallback":
        st.info(
            "Running the colour-heuristic fallback detector. Train YOLO11n with "
            "`notebooks/train_yolo_colab.ipynb` and drop it in `models/scab_detector.onnx` for field photos.",
            icon="🧪",
        )

image_bytes: bytes | None = None
filename = "leaf.jpg"
if upload is not None:
    image_bytes, filename = upload.getvalue(), upload.name
elif sample != "None":
    path = DATA_DIR / ("sample_leaf_scab.jpg" if sample.startswith("Scabbed") else "sample_leaf_healthy.jpg")
    image_bytes, filename = path.read_bytes(), path.name

with out_col:
    if image_bytes is None:
        st.caption("Upload a photo or pick a sample to run lesion detection.")
    else:
        try:
            detection = backend.detect(image_bytes, filename)
        except httpx.HTTPStatusError as exc:
            st.error(f"Detection failed: {exc.response.text}")
        else:
            annotated = draw_detections(Image.open(io.BytesIO(image_bytes)), detection["detections"], palette())
            st.image(annotated, caption=f"{detection['lesion_count']} lesion(s) · {detection['inference_ms']:.1f} ms · {detection['backend']} model", width="stretch")
            level, action = result["risk_level"], result["fungicide"]["action"]
            window = result["fungicide"]["window_hours"]
            if detection["scab_detected"] and action == "curative":
                st.error(
                    "CONFIRMED: visible scab lesions and a completed Mills infection period. Apply a curative "
                    f"fungicide within {window:.0f} hours and remove heavily infected leaves.",
                    icon="🚨",
                )
            elif detection["scab_detected"] and action == "protectant":
                st.error(
                    f"CONFIRMED: visible scab lesions and an infection period forecast in {window:.0f} hours. "
                    "Apply a protectant fungicide before then; the lesions will shed secondary conidia in the rain.",
                    icon="🚨",
                )
            elif detection["scab_detected"] and level == "HIGH":
                st.error(
                    "Visible scab lesions and wetness close to a Mills infection period. Renew protectant "
                    "cover now; lesions will shed secondary conidia at the next rain.",
                    icon="🚨",
                )
            elif detection["scab_detected"]:
                st.warning(
                    "Scab lesions present: they will release secondary conidia at the next wetting. "
                    "Tighten the protectant schedule before forecast rain.",
                    icon="⚠️",
                )
            elif level in ("HIGH", "CRITICAL"):
                st.info(
                    "No visible lesions yet. Scab takes 9–17 days to show symptoms after infection, "
                    "so follow the fungicide window above.",
                    icon="🔎",
                )
            else:
                st.success("No lesions detected and low weather risk.", icon="✅")
            if detection["detections"]:
                st.dataframe(pd.DataFrame(detection["detections"]), hide_index=True, width="stretch")
