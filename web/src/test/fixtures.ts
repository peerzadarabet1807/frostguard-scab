import type { DetectionResponse, RiskResponse, TimelinePoint } from "../types";

export function timeline(n = 48, start = "2024-04-26T06:00:00+05:30"): TimelinePoint[] {
  const t0 = Date.parse(start);
  return Array.from({ length: n }, (_, i) => ({
    time: new Date(t0 + i * 3_600_000).toISOString(),
    temperature_c: 9 + (i % 6),
    relative_humidity: i > 10 ? 96 : 60,
    precipitation_mm: i > 10 ? 1.2 : 0,
    leaf_wet: i > 10,
    wet_hours: Math.max(0, i - 10),
    risk_ratio: Math.min(1, Math.max(0, (i - 10) / 11)),
    infection: i >= 21,
    frost_risk: false,
  }));
}

export function riskResponse(overrides: Partial<RiskResponse> = {}): RiskResponse {
  return {
    location: { name: "Shopian", latitude: 33.716, longitude: 74.831 },
    weather_source: "replay",
    as_of: "2024-04-26T06:00:00+05:30",
    horizon_hours: 48,
    infection_probability: 1,
    current_risk: 0,
    peak_risk: 1,
    risk_level: "CRITICAL",
    fungicide: {
      action: "protectant",
      window_hours: 19,
      deadline: "2024-04-27T01:00:00+05:30",
      message: "Infection period forecast to complete in 19 hours.",
    },
    frost: { alert: false, frost_hours: 0, min_temperature_c: 5.5, first_frost_time: null, in_bud_break: true, threshold_c: -2 },
    alerts: ["CRITICAL RISK: Scab infection expected in 19 hours - apply protectant fungicide now"],
    infection_events: [],
    timeline: timeline(),
    warnings: [],
    ...overrides,
  };
}

export function detection(n: number): DetectionResponse {
  return {
    filename: "leaf.jpg",
    scab_detected: n > 0,
    lesion_count: n,
    max_confidence: n > 0 ? 0.59 : null,
    detections: Array.from({ length: n }, (_, i) => ({
      x1: 10 * i,
      y1: 10,
      x2: 10 * i + 20,
      y2: 40,
      confidence: 0.5,
      class_id: 0,
      label: "Apple - Scab",
    })),
    image_width: 256,
    image_height: 256,
    inference_ms: 55,
    backend: "trained",
    model: "scab_detector.onnx",
    warnings: [],
  };
}
