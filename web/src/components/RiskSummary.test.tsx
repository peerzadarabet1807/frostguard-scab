import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { riskResponse } from "../test/fixtures";
import { RiskSummary } from "./RiskSummary";

const zone = { key: "shopian", name: "Shopian", latitude: 33.716, longitude: 74.831, elevation_m: 2146 };

describe("RiskSummary", () => {
  it("shows the level with an icon and label, not colour alone", () => {
    render(<RiskSummary risk={riskResponse()} zone={zone} loading={false} />);
    const badge = screen.getByRole("status", { name: "Risk level Critical" });
    expect(badge).toHaveTextContent("Critical risk");
    expect(badge).toHaveTextContent("Mills infection period forecast"); // protectant: not yet completed
    expect(screen.getByText(/Scab infection expected in 19 hours/)).toBeInTheDocument();
  });

  it("renders the four stat tiles", () => {
    render(<RiskSummary risk={riskResponse()} zone={zone} loading={false} />);
    expect(screen.getByText("100%")).toBeInTheDocument();
    expect(screen.getByText("19 h")).toBeInTheDocument();
    expect(screen.getByText("Protectant spray")).toBeInTheDocument();
    expect(screen.getByText("5.5 °C")).toBeInTheDocument();
    expect(screen.getByText(/Shopian · ERA5 replay/)).toBeInTheDocument();
  });

  it("labels browser-fetched forecasts as live Open-Meteo data", () => {
    const risk = riskResponse({ weather_source: "uploaded" });
    render(<RiskSummary risk={risk} zone={zone} loading={false} liveViaBrowser />);
    expect(screen.getByText(/Shopian · Open-Meteo live forecast/)).toBeInTheDocument();
  });

  it("explains when the browser could not fetch the forecast", () => {
    render(<RiskSummary risk={riskResponse()} zone={zone} loading={false} liveError="blocked" />);
    expect(screen.getByText(/the API fetched it instead/)).toBeInTheDocument();
  });

  it("marks stale content while reloading", () => {
    const { container } = render(<RiskSummary risk={riskResponse()} zone={zone} loading />);
    expect(container.querySelector(".summary")).toHaveClass("is-stale");
  });
});
