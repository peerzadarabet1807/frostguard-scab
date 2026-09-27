import { useMemo } from "react";
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatDayHour, formatTemp, formatTick, istHour } from "../lib/format";
import type { ChartPalette } from "../lib/theme";
import type { TimelinePoint } from "../types";

const HOUR_MS = 3_600_000;
const Y_WIDTH = 44;

interface Row {
  t: number;
  risk: number;
  temp: number | null;
  rh: number | null;
  rain: number;
  wet: boolean;
  wetHours: number;
  infection: boolean;
  frost: boolean;
}

interface TooltipProps {
  active?: boolean;
  payload?: { payload: Row }[];
  palette: ChartPalette;
}

function HourTooltip({ active, payload, palette }: TooltipProps) {
  const row = active ? payload?.[0]?.payload : undefined;
  if (!row) return null;
  const items: [string, string, string][] = [
    [palette.risk, "Infection risk", `${row.risk.toFixed(0)}%`],
    [palette.temperature, "Temperature", formatTemp(row.temp)],
    [palette.humidity, "Humidity", row.rh === null ? "—" : `${row.rh.toFixed(0)}%`],
    [palette.rain, "Rain", `${row.rain.toFixed(1)} mm`],
  ];
  return (
    <div className="chart-tooltip">
      <strong>{formatDayHour(row.t)}</strong>
      {items.map(([color, label, value]) => (
        <div key={label} className="tt-row">
          <span className="swatch" style={{ background: color }} />
          <span>{label}</span>
          <span className="tt-val">{value}</span>
        </div>
      ))}
      <div className="tt-foot">
        Leaf {row.wet ? `wet · ${row.wetHours} h in event` : "dry"}
        {row.infection && " · infection period met"}
        {row.frost && " · frost risk"}
      </div>
    </div>
  );
}

function wetRuns(rows: Row[]): { x1: number; x2: number }[] {
  const runs: { x1: number; x2: number }[] = [];
  for (const r of rows) {
    const last = runs.at(-1);
    if (!r.wet) continue;
    if (last && last.x2 === r.t) last.x2 = r.t + HOUR_MS;
    else runs.push({ x1: r.t, x2: r.t + HOUR_MS });
  }
  return runs;
}

interface Props {
  timeline: TimelinePoint[];
  palette: ChartPalette;
}

export function RiskCharts({ timeline, palette }: Props) {
  const rows = useMemo<Row[]>(
    () =>
      timeline.map((p) => ({
        t: Date.parse(p.time),
        risk: Math.round(p.risk_ratio * 1000) / 10,
        temp: p.temperature_c,
        rh: p.relative_humidity,
        rain: p.precipitation_mm ?? 0,
        wet: p.leaf_wet,
        wetHours: p.wet_hours,
        infection: p.infection,
        frost: p.frost_risk,
      })),
    [timeline],
  );
  if (rows.length === 0) return <p className="note">No hourly data in this window.</p>;

  const domain: [number, number] = [rows[0]!.t, rows.at(-1)!.t + HOUR_MS];
  const ticks = rows.filter((r) => istHour(new Date(r.t).toISOString()) % 6 === 0).map((r) => r.t);
  const runs = wetRuns(rows);
  const temps = rows.map((r) => r.temp).filter((v): v is number => v !== null);
  const minTemp = Math.min(...temps);
  const rainMax = Math.max(0, ...rows.map((r) => r.rain));
  const tick = { fill: palette.muted, fontSize: 11 };
  const cursor = { stroke: palette.muted, strokeWidth: 1 };

  const xAxis = (
    <XAxis
      dataKey="t"
      type="number"
      scale="time"
      domain={domain}
      ticks={ticks}
      tickFormatter={formatTick}
      stroke={palette.axis}
      tick={tick}
      tickLine={false}
      allowDataOverflow
    />
  );
  const grid = <CartesianGrid vertical={false} stroke={palette.grid} />;
  const chartProps = { data: rows, syncId: "timeline", margin: { top: 10, right: 12, bottom: 0, left: 0 } };
  const label = (value: string) => ({ value, position: "insideTopRight" as const, fill: palette.muted, fontSize: 11 });

  return (
    <div className="charts">
      <figure className="chart-panel" aria-label="Scab infection risk over the next 48 hours">
        <figcaption>
          <strong>Scab infection risk</strong>
          <span>% of the Revised Mills wetness requirement met · shaded hours: leaf wet</span>
        </figcaption>
        <ResponsiveContainer width="100%" height={230}>
          <ComposedChart {...chartProps}>
            {grid}
            {runs.map((r) => (
              <ReferenceArea key={r.x1} x1={r.x1} x2={r.x2} fill={palette.wetBand} fillOpacity={palette.wetOpacity} ifOverflow="hidden" />
            ))}
            {xAxis}
            <YAxis domain={[0, 105]} ticks={[0, 30, 70, 100]} width={Y_WIDTH} stroke={palette.axis} tick={tick} tickLine={false} unit="%" />
            <ReferenceLine y={30} stroke={palette.axis} label={label("Moderate")} />
            <ReferenceLine y={70} stroke={palette.axis} label={label("High")} />
            <ReferenceLine y={100} stroke={palette.axis} label={label("Infection period")} />
            <Tooltip content={<HourTooltip palette={palette} />} cursor={cursor} isAnimationActive={false} />
            <Line
              dataKey="risk"
              type="linear"
              stroke={palette.risk}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, strokeWidth: 2 }}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </figure>

      <figure className="chart-panel" aria-label="Air temperature">
        <figcaption>
          <strong>Air temperature</strong>
          <span>°C{minTemp <= 2 ? " · rule: frost-injury threshold during bud break (−2 °C)" : ""}</span>
        </figcaption>
        <ResponsiveContainer width="100%" height={130}>
          <ComposedChart {...chartProps}>
            {grid}
            {xAxis}
            <YAxis width={Y_WIDTH} stroke={palette.axis} tick={tick} tickLine={false} domain={["auto", "auto"]} tickCount={4} />
            {minTemp <= 2 && <ReferenceLine y={-2} stroke={palette.axis} />}
            <Tooltip content={() => null} cursor={cursor} isAnimationActive={false} />
            <Line dataKey="temp" type="monotone" stroke={palette.temperature} strokeWidth={2} dot={false} activeDot={{ r: 3 }} isAnimationActive={false} connectNulls />
          </ComposedChart>
        </ResponsiveContainer>
      </figure>

      <figure className="chart-panel" aria-label="Relative humidity">
        <figcaption>
          <strong>Relative humidity</strong>
          <span>% · rule: leaf-wetness threshold (90 %)</span>
        </figcaption>
        <ResponsiveContainer width="100%" height={130}>
          <ComposedChart {...chartProps}>
            {grid}
            {xAxis}
            <YAxis width={Y_WIDTH} domain={[0, 100]} ticks={[0, 50, 100]} stroke={palette.axis} tick={tick} tickLine={false} />
            <ReferenceLine y={90} stroke={palette.axis} />
            <Tooltip content={() => null} cursor={cursor} isAnimationActive={false} />
            <Line dataKey="rh" type="monotone" stroke={palette.humidity} strokeWidth={2} dot={false} activeDot={{ r: 3 }} isAnimationActive={false} connectNulls />
          </ComposedChart>
        </ResponsiveContainer>
      </figure>

      <figure className="chart-panel" aria-label="Precipitation">
        <figcaption>
          <strong>Precipitation</strong>
          <span>mm per hour{rainMax === 0 ? " · no rain in this window" : ""}</span>
        </figcaption>
        <ResponsiveContainer width="100%" height={110}>
          <ComposedChart {...chartProps} barCategoryGap={1}>
            {grid}
            {xAxis}
            <YAxis width={Y_WIDTH} domain={[0, Math.max(1, Math.ceil(rainMax))]} stroke={palette.axis} tick={tick} tickLine={false} tickCount={3} allowDecimals={false} />
            <Tooltip content={() => null} cursor={{ fill: palette.grid, fillOpacity: 0.5 }} isAnimationActive={false} />
            <Bar dataKey="rain" fill={palette.rain} radius={[2, 2, 0, 0]} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </figure>
    </div>
  );
}
