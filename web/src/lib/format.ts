// Orchard-local formatting. All Kashmir zones share IST (UTC+05:30).
export const ORCHARD_TZ = "Asia/Kolkata";

const hourFmt = new Intl.DateTimeFormat("en-GB", { timeZone: ORCHARD_TZ, hour: "2-digit", minute: "2-digit", hour12: false });
const dayHourFmt = new Intl.DateTimeFormat("en-GB", {
  timeZone: ORCHARD_TZ,
  weekday: "short",
  day: "numeric",
  month: "short",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});
const dayFmt = new Intl.DateTimeFormat("en-GB", { timeZone: ORCHARD_TZ, day: "numeric", month: "short" });
const fullFmt = new Intl.DateTimeFormat("en-GB", {
  timeZone: ORCHARD_TZ,
  day: "numeric",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});

export const formatHour = (iso: string | number): string => hourFmt.format(new Date(iso));
export const formatDayHour = (iso: string | number): string => dayHourFmt.format(new Date(iso)).replace(",", "");
export const formatDay = (iso: string | number): string => dayFmt.format(new Date(iso));
export const formatFull = (iso: string | number): string => fullFmt.format(new Date(iso));

/** Axis tick: the date at midnight, the hour otherwise. */
export function formatTick(ms: number): string {
  return formatHour(ms) === "00:00" ? formatDay(ms) : formatHour(ms);
}

export const pct = (x: number, digits = 0): string => `${(x * 100).toFixed(digits)}%`;

export function formatTemp(c: number | null | undefined): string {
  if (c === null || c === undefined || Number.isNaN(c)) return "—";
  return `${c.toFixed(1).replace("-", "−")} °C`;
}

/** ISO string for a date (YYYY-MM-DD) and hour in orchard-local time. */
export function istIso(date: string, hour: number): string {
  return `${date}T${String(hour).padStart(2, "0")}:00:00+05:30`;
}

/** YYYY-MM-DD of an ISO timestamp in orchard-local time. */
export function istDate(iso: string): string {
  const parts = new Intl.DateTimeFormat("en-CA", { timeZone: ORCHARD_TZ, year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date(iso));
  return parts;
}

export function istHour(iso: string): number {
  return Number(new Intl.DateTimeFormat("en-GB", { timeZone: ORCHARD_TZ, hour: "2-digit", hour12: false }).format(new Date(iso))) % 24;
}
