import type {
  DetectionResponse,
  Health,
  ModelInfo,
  ReplayDataset,
  RiskRequest,
  RiskResponse,
  Scenario,
  Zone,
} from "../types";

export const API_URL_STORAGE_KEY = "frostguard.apiUrl";
const DEFAULT_API_URL = "http://localhost:8000";

function normalise(url: string): string {
  return url.trim().replace(/\/+$/, "");
}

/**
 * Where the backend lives, in priority order:
 * `?api=` query parameter → saved setting → build-time `VITE_API_URL` → localhost.
 */
export function resolveApiUrl(
  search: string = typeof window === "undefined" ? "" : window.location.search,
  storage: Pick<Storage, "getItem"> | null = safeStorage(),
  buildDefault: string | undefined = import.meta.env.VITE_API_URL,
): string {
  const fromQuery = new URLSearchParams(search).get("api");
  if (fromQuery) return normalise(fromQuery);
  const saved = storage?.getItem(API_URL_STORAGE_KEY);
  if (saved) return normalise(saved);
  return normalise(buildDefault || DEFAULT_API_URL);
}

export function safeStorage(): Storage | null {
  try {
    return typeof window === "undefined" ? null : window.localStorage;
  } catch {
    return null;
  }
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number | null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export class ApiClient {
  constructor(
    readonly baseUrl: string,
    private readonly timeoutMs = 60_000,
  ) {}

  private async request<T>(path: string, init: RequestInit = {}, timeoutMs = this.timeoutMs): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const outer = init.signal;
    outer?.addEventListener("abort", () => controller.abort(), { once: true });
    try {
      const res = await fetch(`${this.baseUrl}${path}`, { ...init, signal: controller.signal });
      if (!res.ok) {
        let detail = res.statusText;
        try {
          const body = await res.json();
          detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
        } catch {
          /* non-JSON error body */
        }
        throw new ApiError(detail || `HTTP ${res.status}`, res.status);
      }
      return (await res.json()) as T;
    } catch (err) {
      if (err instanceof ApiError) throw err;
      if (outer?.aborted) throw err;
      const reason = controller.signal.aborted ? "timed out" : "is unreachable";
      throw new ApiError(`Backend at ${this.baseUrl} ${reason}`, null);
    } finally {
      clearTimeout(timer);
    }
  }

  health(timeoutMs = 10_000): Promise<Health> {
    return this.request("/health", {}, timeoutMs);
  }

  zones(): Promise<Zone[]> {
    return this.request("/api/v1/zones");
  }

  scenarios(): Promise<Scenario[]> {
    return this.request("/api/v1/scenarios");
  }

  datasets(): Promise<ReplayDataset[]> {
    return this.request("/api/v1/datasets");
  }

  model(): Promise<ModelInfo> {
    return this.request("/api/v1/model");
  }

  predictRisk(body: RiskRequest, signal?: AbortSignal): Promise<RiskResponse> {
    return this.request("/api/v1/predict-risk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
  }

  detectLesion(file: Blob, filename: string, signal?: AbortSignal): Promise<DetectionResponse> {
    const form = new FormData();
    form.append("file", file, filename);
    return this.request("/api/v1/detect-lesion", { method: "POST", body: form, signal });
  }
}
