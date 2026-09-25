"""Chart and image-annotation builders for the dashboard (pure functions, no Streamlit)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# The crosshair deliberately shares one hover selection across all four panels.
warnings.filterwarnings("ignore", message="Automatically deduplicated selection parameter")


@dataclass(frozen=True)
class Palette:
    """Validated reference palette: categorical slots 1-3, sequential blue, chart chrome."""

    risk: str
    temperature: str
    humidity: str
    rain: str
    wet_band: str
    rule: str
    muted: str
    lesion_box: str
    map_label: tuple[int, int, int, int]
    wet_opacity: float


LIGHT = Palette("#2a78d6", "#eb6834", "#1baf7a", "#5598e7", "#cde2fb", "#c3c2b7", "#898781", "#eda100", (40, 40, 40, 255), 0.55)
DARK = Palette("#3987e5", "#d95926", "#199e70", "#3987e5", "#184f95", "#383835", "#898781", "#c98500", (235, 235, 235, 255), 0.32)


def risk_chart(timeline: pd.DataFrame, pal: Palette = LIGHT) -> alt.VConcatChart:
    """Small multiples on one shared time axis: risk, temperature, humidity, rain.

    One measure per panel (no dual axes); a single hover selection drives a crosshair
    and tooltip across all four panels.
    """
    df = timeline.copy()
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)  # render in orchard-local time
    df["time_end"] = df["time"] + pd.Timedelta(hours=1)
    df["risk_pct"] = (df["risk_ratio"] * 100).round(1)
    df["wetness"] = np.where(df["leaf_wet"], "Wet", "Dry")

    hover = alt.selection_point(
        name="hover", fields=["time"], nearest=True, on="pointerover", clear="pointerout", empty=False
    )
    x_axis = alt.X("time:T", title=None, axis=alt.Axis(format="%d %b %H:%M", labelAngle=0, tickCount=8, grid=False))
    tooltip = [
        alt.Tooltip("time:T", title="Hour", format="%a %d %b %H:%M"),
        alt.Tooltip("risk_pct:Q", title="Infection risk (%)"),
        alt.Tooltip("temperature_c:Q", title="Temperature (°C)"),
        alt.Tooltip("relative_humidity:Q", title="Humidity (%)"),
        alt.Tooltip("precipitation_mm:Q", title="Rain (mm)"),
        alt.Tooltip("wetness:N", title="Leaf"),
        alt.Tooltip("wet_hours:Q", title="Wet hours in event"),
    ]
    base = alt.Chart(df).encode(x=x_axis)

    def crosshair() -> alt.LayerChart:
        catcher = base.mark_rule(strokeWidth=12, opacity=0).encode(tooltip=tooltip).add_params(hover)
        rule = base.mark_rule(color=pal.muted, strokeWidth=1).encode(
            opacity=alt.condition(hover, alt.value(0.9), alt.value(0))
        )
        return alt.layer(catcher, rule)

    def threshold(values: list[float], labels: list[str] | None = None) -> alt.LayerChart | alt.Chart:
        ref = pd.DataFrame({"y": values, "label": labels or [""] * len(values)})
        rules = alt.Chart(ref).mark_rule(color=pal.rule, strokeWidth=1).encode(y="y:Q")
        if labels is None:
            return rules
        text = alt.Chart(ref).mark_text(
            align="right", baseline="bottom", dx=-4, dy=-3, color=pal.muted, fontSize=11
        ).encode(y="y:Q", x=alt.value("width"), text="label:N")
        return alt.layer(rules, text)

    wet = alt.Chart(df[df["leaf_wet"]]).mark_rect(color=pal.wet_band, opacity=pal.wet_opacity).encode(x="time:T", x2="time_end:T")
    risk_line = base.mark_line(color=pal.risk, strokeWidth=2).encode(
        y=alt.Y(
            "risk_pct:Q",
            title="Infection risk (%)",
            scale=alt.Scale(domain=[0, 105]),
            axis=alt.Axis(values=[0, 30, 70, 100]),
        )
    )
    risk_dot = base.mark_point(color=pal.risk, filled=True, size=70).encode(
        y="risk_pct:Q", opacity=alt.condition(hover, alt.value(1), alt.value(0))
    )
    risk_panel = alt.layer(
        wet, threshold([30, 70, 100], ["Moderate", "High", "Infection period"]), risk_line, risk_dot, crosshair()
    ).properties(
        width="container",
        height=230,
        title=alt.Title("Scab infection risk (% of the Revised Mills requirement)", subtitle="Shaded hours: leaf wet"),
    )

    temp_layers: list[Any] = [
        base.mark_line(color=pal.temperature, strokeWidth=2).encode(
            y=alt.Y("temperature_c:Q", title="°C", scale=alt.Scale(zero=False))
        )
    ]
    temp_title: Any = "Air temperature"
    if df["temperature_c"].min() <= 2:
        temp_layers.insert(0, threshold([-2.0]))
        temp_title = alt.Title("Air temperature", subtitle="Rule: frost-injury threshold during bud break (−2 °C)")
    temp_panel = alt.layer(*temp_layers, crosshair()).properties(width="container", height=120, title=temp_title)

    rh_panel = alt.layer(
        threshold([90]),
        base.mark_line(color=pal.humidity, strokeWidth=2).encode(
            y=alt.Y("relative_humidity:Q", title="%", scale=alt.Scale(domain=[0, 100]))
        ),
        crosshair(),
    ).properties(
        width="container",
        height=120,
        title=alt.Title("Relative humidity", subtitle="Rule: leaf-wetness threshold (90 %)"),
    )

    # Explicit y2 baseline: with x/x2 ranges Vega-Lite would otherwise orient the bars horizontally.
    rain_max = float(df["precipitation_mm"].max(skipna=True) or 0.0)
    rain_bars = (
        alt.Chart(df[df["precipitation_mm"] > 0])
        .mark_rect(color=pal.rain, cornerRadiusTopLeft=2, cornerRadiusTopRight=2, xOffset=1, x2Offset=-1)
        .encode(
            x=x_axis,
            x2="time_end:T",
            y=alt.Y("precipitation_mm:Q", title="mm", scale=alt.Scale(domain=[0, max(1.0, rain_max)]), axis=alt.Axis(tickCount=3)),
            y2=alt.datum(0),
        )
    )
    rain_title: Any = "Precipitation" if rain_max > 0 else alt.Title("Precipitation", subtitle="No rain in this window")
    rain_panel = alt.layer(rain_bars, crosshair()).properties(width="container", height=90, title=rain_title)

    return (
        alt.vconcat(risk_panel, temp_panel, rh_panel, rain_panel, spacing=14)
        .resolve_scale(x="shared")
        .configure_title(anchor="start", fontSize=13, subtitleFontSize=11, subtitleColor=pal.muted)
        .configure_axis(
            labelColor=pal.muted,
            titleColor=pal.muted,
            domainColor=pal.rule,
            tickColor=pal.rule,
            gridColor=pal.rule,
            gridOpacity=0.5,
        )
        .configure_view(strokeWidth=0)
    )


def draw_detections(image: Image.Image, detections: list[dict[str, Any]], pal: Palette = LIGHT) -> Image.Image:
    """Render lesion boxes with a legible label chip above each one."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    width = max(2, out.width // 280)
    font = ImageFont.load_default(size=max(12, out.width // 50))
    for det in detections:
        draw.rectangle((det["x1"], det["y1"], det["x2"], det["y2"]), outline=pal.lesion_box, width=width)
        text = f"{det['label']} {det['confidence']:.2f}"
        tx, ty = det["x1"], max(0, det["y1"] - font.size - 6)
        bbox = draw.textbbox((tx + 3, ty + 2), text, font=font)
        draw.rectangle((tx, ty, bbox[2] + 3, bbox[3] + 3), fill=(20, 20, 20))
        draw.text((tx + 3, ty + 2), text, fill=(255, 255, 255), font=font)
    return out
