import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiClient, ApiError, resolveApiUrl } from "./api";

const storage = (value: string | null) => ({ getItem: () => value });

describe("resolveApiUrl", () => {
  it("prefers the ?api= query parameter", () => {
    expect(resolveApiUrl("?api=http://127.0.0.1:9000/", storage("https://saved.example"), "https://build.example")).toBe(
      "http://127.0.0.1:9000",
    );
  });

  it("falls back to the saved setting, then the build default, then localhost", () => {
    expect(resolveApiUrl("", storage("https://saved.example/"), "https://build.example")).toBe("https://saved.example");
    expect(resolveApiUrl("", storage(null), "https://build.example")).toBe("https://build.example");
    expect(resolveApiUrl("", null, undefined)).toBe("http://localhost:8000");
  });
});

describe("ApiClient", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("surfaces FastAPI error details", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ detail: "as_of is outside the replay range" }), { status: 422 })),
    );
    const err = await new ApiClient("https://api.example").predictRisk({ replay: "x" }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(422);
    expect((err as ApiError).message).toContain("outside the replay range");
  });

  it("reports unreachable backends clearly", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject(new TypeError("Failed to fetch"))));
    const err = await new ApiClient("https://down.example").health().catch((e: unknown) => e);
    expect((err as ApiError).message).toBe("Backend at https://down.example is unreachable");
    expect((err as ApiError).status).toBeNull();
  });

  it("posts risk requests as JSON", async () => {
    const fetchMock = vi.fn(async () => new Response("{}", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    await new ApiClient("https://api.example").predictRisk({ zone: "shopian" });
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("https://api.example/api/v1/predict-risk");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({ zone: "shopian" });
  });
});
