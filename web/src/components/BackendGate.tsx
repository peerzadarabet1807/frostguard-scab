import type { BackendStatus } from "../lib/hooks";

interface Props {
  status: BackendStatus;
  apiUrl: string;
  attempt: number;
  maxAttempts: number;
  onRetry: () => void;
}

/** Shown until the API answers. Free Hugging Face Spaces sleep when idle and take ~1 min to wake. */
export function BackendGate({ status, apiUrl, attempt, maxAttempts, onRetry }: Props) {
  if (status === "online") return null;
  const isHosted = apiUrl.includes(".hf.space");

  if (status === "offline") {
    return (
      <section className="card gate gate-offline" role="alert">
        <h2>Can't reach the backend</h2>
        <p>
          The FrostGuard API at <code>{apiUrl}</code> didn't respond after {maxAttempts} attempts.
        </p>
        <p>To run it locally instead:</p>
        <pre>
          <code>{"uvicorn src.api.app:app --port 8000\n# then open this page with ?api=http://localhost:8000"}</code>
        </pre>
        <button type="button" className="btn btn-primary" onClick={onRetry}>
          Try again
        </button>
      </section>
    );
  }

  return (
    <section className="card gate" role="status" aria-live="polite">
      <div className="spinner" aria-hidden="true" />
      <div>
        <h2>{status === "connecting" ? "Connecting to the backend…" : "Waking up the backend…"}</h2>
        <p>
          {isHosted
            ? "The model runs on a free Hugging Face Space, which sleeps when idle. The first visit after a quiet spell takes up to a minute."
            : `Waiting for the API at ${apiUrl}.`}
        </p>
        {status === "waking" && (
          <div className="progress" aria-label={`Attempt ${attempt} of ${maxAttempts}`}>
            <div style={{ width: `${Math.min(100, (attempt / maxAttempts) * 100)}%` }} />
          </div>
        )}
      </div>
    </section>
  );
}
