import { useState } from "react";
import { formatDayHour, formatTemp } from "../lib/format";
import type { InfectionEvent, TimelinePoint } from "../types";

interface Props {
  timeline: TimelinePoint[];
  events: InfectionEvent[];
}

export function DataTables({ timeline, events }: Props) {
  const significant = events.filter((e) => e.risk_ratio >= 0.3);
  const [tab, setTab] = useState<"hourly" | "events">("hourly");

  return (
    <section className="card">
      <div className="tabs" role="tablist" aria-label="Data tables">
        <button role="tab" type="button" aria-selected={tab === "hourly"} className={tab === "hourly" ? "active" : ""} onClick={() => setTab("hourly")}>
          Hourly timeline <span className="count">{timeline.length}</span>
        </button>
        <button role="tab" type="button" aria-selected={tab === "events"} className={tab === "events" ? "active" : ""} onClick={() => setTab("events")}>
          Wetting events <span className="count">{significant.length}</span>
        </button>
      </div>

      {tab === "hourly" ? (
        <div className="table-wrap" role="tabpanel">
          <table>
            <thead>
              <tr>
                <th>Hour (IST)</th>
                <th className="num">Temp</th>
                <th className="num">RH</th>
                <th className="num">Rain</th>
                <th>Leaf</th>
                <th className="num">Wet h</th>
                <th>Infection risk</th>
                <th>Flags</th>
              </tr>
            </thead>
            <tbody>
              {timeline.map((p) => (
                <tr key={p.time}>
                  <td>{formatDayHour(p.time)}</td>
                  <td className="num">{formatTemp(p.temperature_c)}</td>
                  <td className="num">{p.relative_humidity === null ? "—" : `${p.relative_humidity.toFixed(0)}%`}</td>
                  <td className="num">{(p.precipitation_mm ?? 0).toFixed(1)} mm</td>
                  <td>{p.leaf_wet ? "Wet" : "Dry"}</td>
                  <td className="num">{p.wet_hours}</td>
                  <td>
                    <span className="bar-cell">
                      <span className="bar-fill" style={{ width: `${Math.min(100, p.risk_ratio * 100)}%` }} />
                      <span className="bar-text">{(p.risk_ratio * 100).toFixed(0)}%</span>
                    </span>
                  </td>
                  <td>{[p.infection && "infection", p.frost_risk && "frost"].filter(Boolean).join(", ") || ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : significant.length === 0 ? (
        <p className="note" role="tabpanel">
          No wetting event reached 30% of a Mills infection period in this window.
        </p>
      ) : (
        <div className="table-wrap" role="tabpanel">
          <table>
            <thead>
              <tr>
                <th>Wetting started</th>
                <th>Ended</th>
                <th className="num">Wet h</th>
                <th className="num">Mean temp</th>
                <th className="num">Needed h</th>
                <th className="num">Risk</th>
                <th>Infection</th>
              </tr>
            </thead>
            <tbody>
              {significant.map((e) => (
                <tr key={e.event_id}>
                  <td>{formatDayHour(e.start)}</td>
                  <td>{formatDayHour(e.end)}</td>
                  <td className="num">{e.wet_hours}</td>
                  <td className="num">{formatTemp(e.mean_temperature_c)}</td>
                  <td className="num">{e.required_hours ?? "—"}</td>
                  <td className="num">{(e.risk_ratio * 100).toFixed(0)}%</td>
                  <td>{e.infection_time ? `Met at ${formatDayHour(e.infection_time)}` : "Not met"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
