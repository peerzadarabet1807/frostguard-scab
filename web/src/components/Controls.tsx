import { lazy, Suspense } from "react";
import { istDate, istHour, istIso } from "../lib/format";
import type { ChartPalette, ThemeName } from "../lib/theme";
import type { ReplayDataset, Scenario, Zone } from "../types";

// Leaflet is ~150 kB; load it only once the backend is up and the controls render.
const ZoneMap = lazy(() => import("./ZoneMap").then((m) => ({ default: m.ZoneMap })));

export type SourceMode = "live" | "replay" | "scenario";
export type BudBreak = "auto" | "yes" | "no";

export interface ControlState {
  zone: string;
  source: SourceMode;
  scenario: string;
  replayDate: string; // YYYY-MM-DD, orchard-local
  replayHour: number;
  budBreak: BudBreak;
}

interface Props {
  state: ControlState;
  onChange: (patch: Partial<ControlState>) => void;
  zones: Zone[];
  scenarios: Scenario[];
  dataset: ReplayDataset | undefined;
  theme: ThemeName;
  palette: ChartPalette;
}

const SOURCES: { key: SourceMode; label: string; hint: string }[] = [
  { key: "live", label: "Live forecast", hint: "Open-Meteo, next 48 h" },
  { key: "replay", label: "Replay 2024", hint: "Real ERA5 hours" },
  { key: "scenario", label: "Scenario", hint: "Simulated weather" },
];

export function Controls({ state, onChange, zones, scenarios, dataset, theme, palette }: Props) {
  const zone = zones.find((z) => z.key === state.zone);
  const replayZoneMismatch = state.source === "replay" && dataset && dataset.zone !== state.zone;
  const scenario = scenarios.find((s) => s.key === state.scenario);

  return (
    <aside className="controls">
      <section className="card">
        <h2 className="card-title">Orchard zone</h2>
        <Suspense fallback={<div className="zone-map" aria-hidden="true" />}>
          <ZoneMap zones={zones} selected={state.zone} onSelect={(zone) => onChange({ zone })} theme={theme} palette={palette} />
        </Suspense>
        <div className="chip-row" role="radiogroup" aria-label="Orchard zone">
          {zones.map((z) => (
            <button
              key={z.key}
              type="button"
              role="radio"
              aria-checked={z.key === state.zone}
              className={`chip ${z.key === state.zone ? "chip-active" : ""}`}
              onClick={() => onChange({ zone: z.key })}
            >
              {z.name}
            </button>
          ))}
        </div>
        {zone && (
          <p className="meta-line">
            {zone.latitude.toFixed(3)}° N, {zone.longitude.toFixed(3)}° E · {zone.elevation_m.toLocaleString()} m a.s.l.
          </p>
        )}
      </section>

      <section className="card">
        <h2 className="card-title">Weather</h2>
        <div className="segmented" role="radiogroup" aria-label="Weather source">
          {SOURCES.map((s) => (
            <button
              key={s.key}
              type="button"
              role="radio"
              aria-checked={state.source === s.key}
              className={state.source === s.key ? "active" : ""}
              onClick={() => onChange({ source: s.key })}
            >
              <span>{s.label}</span>
              <small>{s.hint}</small>
            </button>
          ))}
        </div>

        {state.source === "scenario" && (
          <label className="field">
            <span>Scenario</span>
            <select value={state.scenario} onChange={(e) => onChange({ scenario: e.target.value })}>
              {scenarios.map((s) => (
                <option key={s.key} value={s.key}>
                  {s.label}
                </option>
              ))}
            </select>
            {scenario && <small className="hint">{scenario.description}</small>}
          </label>
        )}

        {state.source === "replay" && dataset && (
          <>
            <div className="field-row">
              <label className="field">
                <span>Decision date</span>
                <input
                  type="date"
                  value={state.replayDate}
                  min={istDate(dataset.start)}
                  max={istDate(dataset.end)}
                  onChange={(e) => e.target.value && onChange({ replayDate: e.target.value })}
                />
              </label>
              <label className="field">
                <span>Hour · {String(state.replayHour).padStart(2, "0")}:00</span>
                <input
                  type="range"
                  min={0}
                  max={23}
                  value={state.replayHour}
                  onChange={(e) => onChange({ replayHour: Number(e.target.value) })}
                />
              </label>
            </div>
            <small className="hint">
              {dataset.title}. {dataset.source}.
              {replayZoneMismatch && " Replay data exists for Shopian only, so Shopian weather is used."}
            </small>
          </>
        )}

        <label className="field">
          <span>Bud-break stage</span>
          <select value={state.budBreak} onChange={(e) => onChange({ budBreak: e.target.value as BudBreak })}>
            <option value="auto">Auto (15 Mar – 20 May)</option>
            <option value="yes">Yes: frost-sensitive tissue</option>
            <option value="no">No</option>
          </select>
          <small className="hint">Frost alerts fire only during bud break.</small>
        </label>
      </section>
    </aside>
  );
}

/** Clamp a replay decision time into the dataset's valid range. */
export function clampReplay(dataset: ReplayDataset, date: string, hour: number): { date: string; hour: number } {
  const iso = istIso(date, hour);
  if (Date.parse(iso) < Date.parse(dataset.start)) return { date: istDate(dataset.start), hour: istHour(dataset.start) };
  if (Date.parse(iso) > Date.parse(dataset.end)) return { date: istDate(dataset.end), hour: istHour(dataset.end) };
  return { date, hour };
}
