import { lazy, Suspense, useMemo, useState } from "react";
import { BackendGate } from "./components/BackendGate";
import { Controls, clampReplay, type ControlState } from "./components/Controls";
import { DataTables } from "./components/DataTables";
import { Footer } from "./components/Footer";
import { Header } from "./components/Header";
import { LeafScan } from "./components/LeafScan";
import { ModelPanel } from "./components/ModelPanel";
import { RiskSummary } from "./components/RiskSummary";
import { API_URL_STORAGE_KEY, ApiClient, resolveApiUrl, safeStorage } from "./lib/api";
import { istIso } from "./lib/format";
import { useBackend, useLiveWeather, useMeta, useRisk } from "./lib/hooks";
import { PALETTES, useTheme } from "./lib/theme";
import type { RiskRequest, WeatherRecord } from "./types";

// Recharts is the largest dependency; split it out of the first paint.
const RiskCharts = lazy(() => import("./components/RiskCharts").then((m) => ({ default: m.RiskCharts })));

const INITIAL: ControlState = {
  zone: "shopian",
  source: "live",
  scenario: "scab_outbreak",
  // A real 2024 infection period: forecast to complete 19 h after this decision time.
  replayDate: "2024-04-26",
  replayHour: 6,
  budBreak: "auto",
};

interface LiveWeather {
  records: WeatherRecord[] | null;
  asOf: string | null;
  loading: boolean;
}

function buildRequest(state: ControlState, live: LiveWeather): RiskRequest | null {
  const req: RiskRequest = { zone: state.zone, horizon_hours: 48 };
  if (state.source === "live") {
    if (live.loading) return null; // wait for the browser's Open-Meteo fetch
    if (live.records && live.asOf) {
      req.weather = live.records;
      req.as_of = live.asOf;
    }
    // Otherwise the API fetches the forecast itself (and falls back to simulated weather).
  }
  if (state.budBreak !== "auto") req.bud_break = state.budBreak === "yes";
  if (state.source === "scenario") {
    req.use_mock = true;
    req.mock_scenario = state.scenario;
    req.bud_break ??= true; // simulated regimes are spring weather
  }
  if (state.source === "replay") {
    req.replay = "shopian_spring_2024";
    req.as_of = istIso(state.replayDate, state.replayHour);
    delete req.zone; // replay data is Shopian's
  }
  return req;
}

export default function App() {
  const [apiUrl, setApiUrl] = useState(() => resolveApiUrl());
  const client = useMemo(() => new ApiClient(apiUrl), [apiUrl]);
  const { theme, toggle } = useTheme();
  const palette = PALETTES[theme];

  const backend = useBackend(client);
  const online = backend.status === "online";
  const { meta, error: metaError } = useMeta(client, online);
  const [controls, setControls] = useState<ControlState>(INITIAL);
  const dataset = meta?.datasets[0];

  const zone = meta?.zones.find((z) => z.key === (controls.source === "replay" ? dataset?.zone : controls.zone));
  const live = useLiveWeather(zone, online && controls.source === "live");
  const { records: liveRecords, asOf: liveAsOf, loading: liveLoading } = live;
  const request = useMemo(
    () => (online && meta ? buildRequest(controls, { records: liveRecords, asOf: liveAsOf, loading: liveLoading }) : null),
    [online, meta, controls, liveRecords, liveAsOf, liveLoading],
  );
  const risk = useRisk(client, request);
  const liveViaBrowser = controls.source === "live" && liveRecords !== null;

  const onApiUrlChange = (url: string | null) => {
    const storage = safeStorage();
    try {
      if (url) storage?.setItem(API_URL_STORAGE_KEY, url);
      else storage?.removeItem(API_URL_STORAGE_KEY);
    } catch {
      /* storage blocked: the URL still applies for this session */
    }
    const params = new URLSearchParams(window.location.search);
    params.delete("api");
    window.history.replaceState(null, "", `${window.location.pathname}${params.size ? `?${params}` : ""}`);
    setApiUrl(url ? url.replace(/\/+$/, "") : resolveApiUrl(""));
  };

  const onControlsChange = (patch: Partial<ControlState>) =>
    setControls((prev) => {
      const next = { ...prev, ...patch };
      if (dataset && next.source === "replay") {
        const { date, hour } = clampReplay(dataset, next.replayDate, next.replayHour);
        next.replayDate = date;
        next.replayHour = hour;
      }
      return next;
    });

  return (
    <>
      <Header
        status={backend.status}
        health={backend.health}
        apiUrl={apiUrl}
        onApiUrlChange={onApiUrlChange}
        theme={theme}
        onToggleTheme={toggle}
      />
      <main className="page">
        <BackendGate
          status={backend.status}
          apiUrl={apiUrl}
          attempt={backend.attempt}
          maxAttempts={backend.maxAttempts}
          onRetry={backend.retry}
        />
        {metaError && <p className="alert sev-critical">Could not load reference data: {metaError}</p>}

        {online && meta && (
          <>
            <div className="layout">
              <Controls
                state={controls}
                onChange={onControlsChange}
                zones={meta.zones}
                scenarios={meta.scenarios}
                dataset={dataset}
                theme={theme}
                palette={palette}
              />
              <div className="main-col">
                {risk.error && (
                  <p className="alert sev-critical" role="alert">
                    <span className="alert-icon" aria-hidden="true">✕</span>
                    <span>Risk service error: {risk.error}</span>
                  </p>
                )}
                {risk.data ? (
                  <>
                    <RiskSummary
                      risk={risk.data}
                      zone={zone}
                      loading={risk.loading}
                      liveViaBrowser={liveViaBrowser}
                      liveError={controls.source === "live" ? live.error : null}
                    />
                    <section className={`card ${risk.loading ? "is-stale" : ""}`}>
                      <Suspense fallback={<div className="charts-placeholder" aria-hidden="true" />}>
                        <RiskCharts timeline={risk.data.timeline} palette={palette} />
                      </Suspense>
                    </section>
                  </>
                ) : (
                  !risk.error && (
                    <section className="card gate" role="status">
                      <div className="spinner" aria-hidden="true" />
                      <h2>Scoring infection risk…</h2>
                    </section>
                  )
                )}
              </div>
            </div>

            {risk.data && <DataTables timeline={risk.data.timeline} events={risk.data.infection_events} />}

            <div className="layout-bottom">
              <LeafScan client={client} risk={risk.data} palette={palette} enabled={online} />
              <ModelPanel model={meta.model} />
            </div>
          </>
        )}
      </main>
      <Footer apiUrl={apiUrl} />
    </>
  );
}
