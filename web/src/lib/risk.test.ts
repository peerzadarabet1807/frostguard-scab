import { describe, expect, it } from "vitest";
import { detection, riskResponse } from "../test/fixtures";
import { classifyAlerts, leafVerdict } from "./risk";

describe("classifyAlerts", () => {
  it("maps engine alert prefixes to severities", () => {
    const alerts = classifyAlerts(
      riskResponse({ alerts: ["CRITICAL RISK: x", "FROST WARNING: 3 h below -2.0 °C"], risk_level: "CRITICAL" }),
    );
    expect(alerts.map((a) => [a.severity, a.kind])).toEqual([
      ["critical", "scab"],
      ["critical", "frost"],
    ]);
  });

  it("adds an all-clear when the engine raised nothing", () => {
    const [alert] = classifyAlerts(riskResponse({ alerts: [], risk_level: "LOW" }));
    expect(alert?.severity).toBe("good");
    expect(alert?.text).toMatch(/^LOW RISK/);
  });
});

describe("leafVerdict", () => {
  it("confirms a forecast infection with a protectant window", () => {
    const v = leafVerdict(detection(3), riskResponse());
    expect(v.severity).toBe("critical");
    expect(v.title).toBe("Confirmed: 3 scab lesions detected, infection period forecast in 19 h");
  });

  it("uses the curative window after a completed infection", () => {
    const risk = riskResponse({ fungicide: { action: "curative", window_hours: 60, deadline: null, message: "" } });
    expect(leafVerdict(detection(1), risk).body).toContain("within 60 hours");
  });

  it("warns about lesions when weather risk is low", () => {
    const risk = riskResponse({ risk_level: "LOW", fungicide: { action: "none", window_hours: null, deadline: null, message: "" } });
    expect(leafVerdict(detection(1), risk)).toMatchObject({ severity: "warning", title: "1 scab lesion detected" });
  });

  it("explains the incubation lag for clean leaves in high risk", () => {
    const v = leafVerdict(detection(0), riskResponse());
    expect(v.severity).toBe("info");
    expect(v.body).toContain("9–17 days");
  });

  it("is all-clear for a clean leaf in low risk", () => {
    const risk = riskResponse({ risk_level: "LOW", fungicide: { action: "none", window_hours: null, deadline: null, message: "" } });
    expect(leafVerdict(detection(0), risk).severity).toBe("good");
  });
});
