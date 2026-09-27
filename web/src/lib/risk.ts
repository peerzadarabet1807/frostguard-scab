import type { DetectionResponse, RiskLevel, RiskResponse } from "../types";

export type Severity = "good" | "warning" | "serious" | "critical" | "info";

export const LEVEL_META: Record<RiskLevel, { severity: Severity; icon: string; label: string; blurb: string }> = {
  LOW: { severity: "good", icon: "✓", label: "Low", blurb: "No infection period expected" },
  MODERATE: { severity: "warning", icon: "!", label: "Moderate", blurb: "Leaf wetness is building" },
  HIGH: { severity: "serious", icon: "▲", label: "High", blurb: "Close to a Mills infection period" },
  CRITICAL: { severity: "critical", icon: "✕", label: "Critical", blurb: "Mills infection period met" },
};

export interface Alert {
  severity: Severity;
  kind: "scab" | "frost" | "info";
  text: string;
}

/** Classify the engine's alert strings (and warnings) for display. */
export function classifyAlerts(risk: RiskResponse): Alert[] {
  const alerts: Alert[] = risk.alerts.map((text) => {
    if (text.startsWith("FROST")) return { severity: "critical", kind: "frost", text };
    const level = text.split(" ")[0] as RiskLevel;
    return { severity: LEVEL_META[level]?.severity ?? "warning", kind: "scab", text };
  });
  if (alerts.length === 0) {
    alerts.push({
      severity: "good",
      kind: "scab",
      text: `LOW RISK: No scab infection period expected in the next ${risk.horizon_hours} h. No spray needed.`,
    });
  }
  return alerts;
}

export interface Verdict {
  severity: Severity;
  title: string;
  body: string;
}

/** Combine what the leaf photo shows with what the weather says. */
export function leafVerdict(detection: DetectionResponse, risk: RiskResponse | null): Verdict {
  const action = risk?.fungicide.action ?? "none";
  const window = risk?.fungicide.window_hours;
  const level = risk?.risk_level ?? "LOW";
  const n = detection.lesion_count;
  const lesions = `${n} scab lesion${n === 1 ? "" : "s"} detected`;

  if (detection.scab_detected && action === "curative" && window !== null && window !== undefined) {
    return {
      severity: "critical",
      title: `Confirmed: ${lesions} and a completed infection period`,
      body: `Apply a curative fungicide within ${Math.round(window)} hours and remove heavily infected leaves.`,
    };
  }
  if (detection.scab_detected && action === "protectant" && window !== null && window !== undefined) {
    return {
      severity: "critical",
      title: `Confirmed: ${lesions}, infection period forecast in ${Math.round(window)} h`,
      body: "Apply a protectant fungicide before then; existing lesions will shed secondary conidia in the rain.",
    };
  }
  if (detection.scab_detected && level === "HIGH") {
    return {
      severity: "serious",
      title: `${lesions}; wetness close to an infection period`,
      body: "Renew protectant cover now; lesions will shed secondary conidia at the next rain.",
    };
  }
  if (detection.scab_detected) {
    return {
      severity: "warning",
      title: lesions,
      body: "Lesions release secondary spores at the next wetting. Tighten the protectant schedule before forecast rain.",
    };
  }
  if (level === "HIGH" || level === "CRITICAL") {
    return {
      severity: "info",
      title: "No visible lesions yet",
      body: "Scab takes 9–17 days to show symptoms after infection, so follow the fungicide window above.",
    };
  }
  return { severity: "good", title: "No lesions detected", body: "The leaf looks clean and weather risk is low." };
}
