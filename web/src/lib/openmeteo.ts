import type { WeatherRecord } from "../types";

// Live forecasts are fetched by the browser, not the API: Open-Meteo's free tier rate-limits per IP,
// and free hosting platforms share outbound IPs across many apps. The records are then scored by the
// backend exactly like uploaded weather, so the Mills model still runs server-side.
const FORECAST_URL = "https://api.open-meteo.com/v1/forecast";
const HOURLY = ["temperature_2m", "relative_humidity_2m", "precipitation", "dew_point_2m"] as const;

interface ForecastPayload {
  hourly?: {
    time: number[];
    temperature_2m: (number | null)[];
    relative_humidity_2m: (number | null)[];
    precipitation: (number | null)[];
    dew_point_2m: (number | null)[];
  };
  error?: boolean;
  reason?: string;
}

export function forecastUrl(latitude: number, longitude: number, pastHours = 24, forecastHours = 50): string {
  const params = new URLSearchParams({
    latitude: latitude.toFixed(4),
    longitude: longitude.toFixed(4),
    hourly: HOURLY.join(","),
    timezone: "Asia/Kolkata",
    timeformat: "unixtime",
    past_hours: String(pastHours),
    forecast_hours: String(forecastHours),
  });
  return `${FORECAST_URL}?${params}`;
}

/** Convert Open-Meteo's column arrays into API weather records, dropping hours without temperature. */
export function toWeatherRecords(payload: ForecastPayload): WeatherRecord[] {
  const h = payload.hourly;
  if (!h) throw new Error(payload.reason ?? "Open-Meteo returned no hourly data");
  const records: WeatherRecord[] = [];
  h.time.forEach((t, i) => {
    const temperature = h.temperature_2m[i];
    if (temperature === null || temperature === undefined) return;
    records.push({
      time: new Date(t * 1000).toISOString(),
      temperature_2m: temperature,
      relative_humidity_2m: h.relative_humidity_2m[i] ?? null,
      precipitation: h.precipitation[i] ?? 0,
      dew_point_2m: h.dew_point_2m[i] ?? null,
    });
  });
  if (records.length === 0) throw new Error("Open-Meteo returned an empty forecast");
  return records;
}

export async function fetchForecast(latitude: number, longitude: number, signal?: AbortSignal): Promise<WeatherRecord[]> {
  const res = await fetch(forecastUrl(latitude, longitude), { signal });
  const payload = (await res.json().catch(() => ({}))) as ForecastPayload;
  if (!res.ok || payload.error) throw new Error(payload.reason ?? `Open-Meteo HTTP ${res.status}`);
  return toWeatherRecords(payload);
}
