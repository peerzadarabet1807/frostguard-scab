import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchForecast, forecastUrl, toWeatherRecords } from "./openmeteo";
import { istHourFloor } from "./format";

const payload = {
  hourly: {
    time: [1714091400, 1714095000, 1714098600], // 26 Apr 2024 06:00, 07:00, 08:00 IST
    temperature_2m: [9.4, 13.0, null],
    relative_humidity_2m: [53, 57, 60],
    precipitation: [0, null, 0],
    dew_point_2m: [0.3, 4.7, null],
  },
};

describe("Open-Meteo browser fetch", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("asks for the API's four variables on IST hours", () => {
    const url = new URL(forecastUrl(33.716, 74.831));
    expect(url.searchParams.get("hourly")).toBe("temperature_2m,relative_humidity_2m,precipitation,dew_point_2m");
    expect(url.searchParams.get("timezone")).toBe("Asia/Kolkata");
    expect(url.searchParams.get("past_hours")).toBe("24");
  });

  it("converts columns to records and drops hours without temperature", () => {
    const records = toWeatherRecords(payload);
    expect(records).toHaveLength(2);
    expect(records[0]).toEqual({
      time: "2024-04-26T00:30:00.000Z",
      temperature_2m: 9.4,
      relative_humidity_2m: 53,
      precipitation: 0,
      dew_point_2m: 0.3,
    });
    expect(records[1]?.precipitation).toBe(0); // missing rain treated as none
  });

  it("surfaces Open-Meteo's own error reason", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ error: true, reason: "Daily API request limit exceeded." }), { status: 429 })),
    );
    await expect(fetchForecast(33.7, 74.8)).rejects.toThrow("Daily API request limit exceeded.");
  });
});

describe("istHourFloor", () => {
  it("floors to the IST hour, which falls on :30 UTC", () => {
    expect(istHourFloor(Date.parse("2024-04-26T01:10:00Z"))).toBe("2024-04-26T00:30:00.000Z"); // 06:40 IST -> 06:00
    expect(istHourFloor(Date.parse("2024-04-26T01:30:00Z"))).toBe("2024-04-26T01:30:00.000Z"); // exactly 07:00 IST
  });
});
