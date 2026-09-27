import { useCallback, useEffect, useRef, useState } from "react";
import type { ApiClient } from "./api";
import type { Health, ModelInfo, ReplayDataset, RiskRequest, RiskResponse, Scenario, Zone } from "../types";

export type BackendStatus = "connecting" | "waking" | "online" | "offline";

const WAKE_RETRY_MS = 5_000;
const WAKE_MAX_ATTEMPTS = 36; // ~3 minutes: long enough for a sleeping free-tier Space to boot

interface ProbeState {
  key: string; // which backend URL + retry generation this result belongs to
  status: BackendStatus;
  health: Health | null;
  attempt: number;
}

/** Poll /health until the backend answers; a sleeping Hugging Face Space takes a while to wake. */
export function useBackend(client: ApiClient) {
  const [generation, setGeneration] = useState(0);
  const [probe, setProbe] = useState<ProbeState | null>(null);
  const key = `${client.baseUrl}#${generation}`;

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const run = async (n: number) => {
      try {
        const health = await client.health(15_000);
        if (!cancelled) setProbe({ key, status: "online", health, attempt: n });
      } catch {
        if (cancelled) return;
        const done = n >= WAKE_MAX_ATTEMPTS;
        setProbe({ key, status: done ? "offline" : "waking", health: null, attempt: n });
        if (!done) timer = setTimeout(() => void run(n + 1), WAKE_RETRY_MS);
      }
    };
    void run(1);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [client, key]);

  // A result for a previous URL or retry generation means we're connecting afresh.
  const current = probe?.key === key ? probe : null;
  const retry = useCallback(() => setGeneration((g) => g + 1), []);
  return {
    status: current?.status ?? ("connecting" as BackendStatus),
    health: current?.health ?? null,
    attempt: current?.attempt ?? 0,
    maxAttempts: WAKE_MAX_ATTEMPTS,
    retry,
  };
}

export interface Meta {
  zones: Zone[];
  scenarios: Scenario[];
  datasets: ReplayDataset[];
  model: ModelInfo | null;
}

export function useMeta(client: ApiClient, enabled: boolean) {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    Promise.all([client.zones(), client.scenarios(), client.datasets(), client.model().catch(() => null)])
      .then(([zones, scenarios, datasets, model]) => {
        if (!cancelled) setMeta({ zones, scenarios, datasets, model });
      })
      .catch((err: Error) => !cancelled && setError(err.message));
    return () => {
      cancelled = true;
    };
  }, [client, enabled]);
  return { meta, error };
}

/** Fetch a risk assessment whenever the request changes, keeping the last result while reloading. */
export function useRisk(client: ApiClient, request: RiskRequest | null) {
  const [data, setData] = useState<RiskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const key = request ? JSON.stringify(request) : null;
  const latest = useRef(0);

  useEffect(() => {
    if (!key) return;
    const id = ++latest.current;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      setLoading(true);
      client
        .predictRisk(JSON.parse(key) as RiskRequest, controller.signal)
        .then((res) => {
          if (id !== latest.current) return;
          setData(res);
          setError(null);
        })
        .catch((err: Error) => {
          if (id === latest.current && !controller.signal.aborted) setError(err.message);
        })
        .finally(() => id === latest.current && setLoading(false));
    }, 200);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [client, key]);

  return { data, loading, error };
}
