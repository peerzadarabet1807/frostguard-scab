import { useCallback, useEffect, useRef, useState } from "react";
import type { ApiClient } from "../lib/api";
import { leafVerdict } from "../lib/risk";
import { SAMPLE_ATTRIBUTION, SAMPLE_LEAVES } from "../lib/samples";
import type { ChartPalette } from "../lib/theme";
import type { DetectionResponse, RiskResponse } from "../types";

interface Props {
  client: ApiClient;
  risk: RiskResponse | null;
  palette: ChartPalette;
  enabled: boolean;
}

interface Selected {
  url: string;
  name: string;
  revoke: boolean;
}

const MAX_BYTES = 10 * 1024 * 1024;

export function LeafScan({ client, risk, palette, enabled }: Props) {
  const [selected, setSelected] = useState<Selected | null>(null);
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => () => {
    if (selected?.revoke) URL.revokeObjectURL(selected.url);
  }, [selected]);

  const run = useCallback(
    async (blob: Blob, name: string, url: string, revoke: boolean) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setSelected({ url, name, revoke });
      setResult(null);
      setError(null);
      setBusy(true);
      try {
        setResult(await client.detectLesion(blob, name, controller.signal));
      } catch (err) {
        if (!controller.signal.aborted) setError((err as Error).message);
      } finally {
        if (abortRef.current === controller) setBusy(false);
      }
    },
    [client],
  );

  const onFile = (file: File | undefined) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) return setError("Please choose an image file (JPEG, PNG or WebP).");
    if (file.size > MAX_BYTES) return setError("Images must be under 10 MB.");
    void run(file, file.name, URL.createObjectURL(file), true);
  };

  const onSample = async (url: string, id: string) => {
    const blob = await (await fetch(url)).blob();
    void run(blob, `${id}.jpg`, url, false);
  };

  const verdict = result ? leafVerdict(result, risk) : null;

  return (
    <section className="card leaf-scan" id="leaf-scan">
      <h2 className="card-title">Leaf scan · confirm lesions</h2>
      <div className="leaf-grid">
        <div>
          <div
            className={`dropzone ${dragging ? "dragging" : ""}`}
            role="button"
            tabIndex={0}
            aria-disabled={!enabled}
            onClick={() => enabled && inputRef.current?.click()}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && enabled && inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              if (enabled) onFile(e.dataTransfer.files[0]);
            }}
          >
            <span className="drop-icon" aria-hidden="true">⤒</span>
            <strong>Drop a leaf photo or click to upload</strong>
            <small>JPEG, PNG or WebP · up to 10 MB · one leaf on a plain background works best</small>
            <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" hidden onChange={(e) => onFile(e.target.files?.[0])} />
          </div>

          <p className="field-label">Or try a held-out sample</p>
          <div className="samples">
            {SAMPLE_LEAVES.map((s) => (
              <button
                key={s.id}
                type="button"
                className={`sample ${selected?.url === s.url ? "active" : ""}`}
                onClick={() => void onSample(s.url, s.id)}
                disabled={!enabled}
                title={s.label}
              >
                <img src={s.url} alt={s.label} loading="lazy" />
                <span>{s.healthy ? "Healthy" : `Scab ${s.id.split("_").pop()}`}</span>
              </button>
            ))}
          </div>
          <p className="fine-print">{SAMPLE_ATTRIBUTION}</p>
        </div>

        <div className="scan-result" aria-live="polite">
          {!selected && <div className="scan-empty">Detections appear here, drawn over the photo.</div>}
          {selected && (
            <figure className={`scan-figure ${busy ? "is-stale" : ""}`}>
              <div className="scan-image">
                <img src={selected.url} alt={`Uploaded leaf: ${selected.name}`} />
                {result && (
                  <svg viewBox={`0 0 ${result.image_width} ${result.image_height}`} preserveAspectRatio="none" aria-hidden="true">
                    {result.detections.map((d, i) => {
                      // Numbered markers keep overlapping lesions legible; the table lists label + confidence.
                      const stroke = Math.max(1.5, result.image_width / 200);
                      const r = Math.max(7, result.image_width / 34);
                      const cx = Math.max(r, Math.min(result.image_width - r, d.x1));
                      const cy = Math.max(r, Math.min(result.image_height - r, d.y1));
                      return (
                        <g key={i}>
                          <rect x={d.x1} y={d.y1} width={d.x2 - d.x1} height={d.y2 - d.y1} fill="none" stroke={palette.lesion} strokeWidth={stroke} rx={stroke} />
                          <circle cx={cx} cy={cy} r={r} fill="rgba(11,11,11,0.82)" stroke={palette.lesion} strokeWidth={stroke * 0.8} />
                          <text x={cx} y={cy} fontSize={r * 1.15} fill="#fff" fontWeight={700} textAnchor="middle" dominantBaseline="central" fontFamily="system-ui, sans-serif">
                            {i + 1}
                          </text>
                        </g>
                      );
                    })}
                  </svg>
                )}
              </div>
              <figcaption>
                {busy
                  ? "Running the detector…"
                  : result
                    ? `${result.lesion_count} lesion${result.lesion_count === 1 ? "" : "s"} · ${result.inference_ms.toFixed(0)} ms inference · ${result.backend === "trained" ? "trained YOLO11n" : "fallback heuristic"}`
                    : selected.name}
              </figcaption>
            </figure>
          )}

          {error && (
            <p className="alert sev-critical" role="alert">
              <span className="alert-icon" aria-hidden="true">✕</span>
              <span>Detection failed: {error}</span>
            </p>
          )}

          {verdict && (
            <div className={`verdict sev-${verdict.severity}`}>
              <strong>{verdict.title}</strong>
              <span>{verdict.body}</span>
            </div>
          )}

          {result?.warnings.map((w) => (
            <p key={w} className="note">
              ℹ {w}
            </p>
          ))}

          {result && result.detections.length > 0 && (
            <div className="table-wrap compact">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Label</th>
                    <th className="num">Confidence</th>
                    <th className="num">Box (x1, y1, x2, y2) px</th>
                  </tr>
                </thead>
                <tbody>
                  {result.detections.map((d, i) => (
                    <tr key={i}>
                      <td>{i + 1}</td>
                      <td>{d.label}</td>
                      <td className="num">{d.confidence.toFixed(2)}</td>
                      <td className="num">
                        {[d.x1, d.y1, d.x2, d.y2].map((v) => v.toFixed(0)).join(", ")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
