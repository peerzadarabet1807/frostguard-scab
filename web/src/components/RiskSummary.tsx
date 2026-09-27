import { formatFull, formatTemp, pct } from "../lib/format";
import { LEVEL_META, classifyAlerts } from "../lib/risk";
import type { RiskResponse, Zone } from "../types";

const SOURCE_LABEL: Record<RiskResponse["weather_source"], string> = {
  "open-meteo": "Open-Meteo live forecast",
  mock: "Simulated weather",
  uploaded: "Uploaded weather",
  replay: "ERA5 replay",
};

const ACTION_LABEL: Record<RiskResponse["fungicide"]["action"], string> = {
  none: "No spray needed",
  monitor: "Monitor",
  protectant: "Protectant spray",
  curative: "Curative spray",
};

interface Props {
  risk: RiskResponse;
  zone: Zone | undefined;
  loading: boolean;
  /** The browser fetched the forecast from Open-Meteo and the API scored it. */
  liveViaBrowser?: boolean;
  /** Why the browser couldn't fetch the forecast (the API fetched it instead). */
  liveError?: string | null;
}

export function RiskSummary({ risk, zone, loading, liveViaBrowser = false, liveError = null }: Props) {
  const source = liveViaBrowser ? "Open-Meteo live forecast" : SOURCE_LABEL[risk.weather_source];
  const level = LEVEL_META[risk.risk_level];
  const blurb =
    risk.risk_level !== "CRITICAL"
      ? level.blurb
      : risk.fungicide.action === "protectant"
        ? "Mills infection period forecast"
        : "Mills infection period completed";
  const alerts = classifyAlerts(risk);
  const wetHours = risk.timeline.filter((p) => p.leaf_wet).length;
  const window = risk.fungicide.window_hours;

  return (
    <section className={`card summary ${loading ? "is-stale" : ""}`} aria-busy={loading}>
      <div className="summary-head">
        <div>
          <p className="eyebrow">
            {zone?.name ?? risk.location.name ?? "Custom location"} · {source}
          </p>
          <h1>Next {risk.horizon_hours} hours</h1>
          <p className="meta-line">Decision time {formatFull(risk.as_of)} IST</p>
        </div>
        <div className={`level-badge sev-${level.severity}`} role="status" aria-label={`Risk level ${level.label}`}>
          <span className="level-icon" aria-hidden="true">
            {level.icon}
          </span>
          <span>
            <strong>{level.label} risk</strong>
            <small>{blurb}</small>
          </span>
        </div>
      </div>

      <ul className="alerts">
        {alerts.map((a) => (
          <li key={a.text} className={`alert sev-${a.severity}`}>
            <span className="alert-icon" aria-hidden="true">
              {a.kind === "frost" ? "❄" : LEVEL_META[risk.risk_level].icon}
            </span>
            <span>{a.text}</span>
          </li>
        ))}
      </ul>

      <div className="tiles">
        <div className="tile">
          <span className="tile-label">Infection probability</span>
          <span className="tile-value">{pct(risk.infection_probability)}</span>
          <span className="tile-sub">Peak share of the Mills wetness requirement</span>
        </div>
        <div className="tile">
          <span className="tile-label">Fungicide window</span>
          <span className="tile-value">{window === null ? "—" : `${Math.round(window)} h`}</span>
          <span className="tile-sub">{ACTION_LABEL[risk.fungicide.action]}</span>
        </div>
        <div className="tile">
          <span className="tile-label">Current risk</span>
          <span className="tile-value">{pct(risk.current_risk)}</span>
          <span className="tile-sub">{wetHours} wet hours ahead</span>
        </div>
        <div className="tile">
          <span className="tile-label">Minimum temperature</span>
          <span className="tile-value">{formatTemp(risk.frost.min_temperature_c)}</span>
          <span className="tile-sub">
            {risk.frost.alert
              ? `${risk.frost.frost_hours} h below ${formatTemp(risk.frost.threshold_c)}`
              : risk.frost.in_bud_break
                ? "Bud break: frost-sensitive"
                : "Outside bud break"}
          </span>
        </div>
      </div>

      <p className="advice">{risk.fungicide.message}</p>
      {liveError && <p className="note">ℹ Couldn't fetch the forecast from this browser ({liveError}); the API fetched it instead.</p>}
      {risk.warnings.map((w) => (
        <p key={w} className="note">
          ℹ {w}
        </p>
      ))}
    </section>
  );
}
