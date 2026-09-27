import { useEffect, useState } from "react";
import { safeStorage } from "./api";

export type ThemeName = "light" | "dark";

/** Chart palette: validated reference slots (see the data-viz notes in the README). */
export interface ChartPalette {
  risk: string;
  temperature: string;
  humidity: string;
  rain: string;
  wetBand: string;
  wetOpacity: number;
  grid: string;
  axis: string;
  muted: string;
  lesion: string;
  zone: string;
  zoneSelected: string;
}

export const PALETTES: Record<ThemeName, ChartPalette> = {
  light: {
    risk: "#2a78d6",
    temperature: "#eb6834",
    humidity: "#1baf7a",
    rain: "#5598e7",
    wetBand: "#cde2fb",
    wetOpacity: 0.6,
    grid: "#e1e0d9",
    axis: "#c3c2b7",
    muted: "#898781",
    lesion: "#eda100",
    zone: "#898781",
    zoneSelected: "#2a78d6",
  },
  dark: {
    risk: "#3987e5",
    temperature: "#d95926",
    humidity: "#199e70",
    rain: "#3987e5",
    wetBand: "#184f95",
    wetOpacity: 0.4,
    grid: "#2c2c2a",
    axis: "#383835",
    muted: "#898781",
    lesion: "#fab219",
    zone: "#898781",
    zoneSelected: "#3987e5",
  },
};

const THEME_KEY = "frostguard.theme";

function systemTheme(): ThemeName {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/** Resolved theme: an explicit choice wins, otherwise follow the OS (live). */
export function useTheme(): { theme: ThemeName; toggle: () => void } {
  const [explicit, setExplicit] = useState<ThemeName | null>(() => {
    const saved = safeStorage()?.getItem(THEME_KEY);
    return saved === "light" || saved === "dark" ? saved : null;
  });
  const [system, setSystem] = useState<ThemeName>(systemTheme);

  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    if (!mq) return;
    const onChange = () => setSystem(mq.matches ? "dark" : "light");
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  const theme = explicit ?? system;
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  const toggle = () => {
    const next: ThemeName = theme === "dark" ? "light" : "dark";
    setExplicit(next);
    try {
      safeStorage()?.setItem(THEME_KEY, next);
    } catch {
      /* storage blocked: theme still applies for this session */
    }
  };
  return { theme, toggle };
}
