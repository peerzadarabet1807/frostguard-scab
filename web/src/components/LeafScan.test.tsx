import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ApiClient } from "../lib/api";
import { INPUT_CHECK_COPY } from "../lib/inputCheck";
import { detection, riskResponse } from "../test/fixtures";
import type { DetectionResponse } from "../types";
import { LeafScan } from "./LeafScan";

const palette = { lesion: "#f00" } as never;

function rejected(): DetectionResponse {
  return {
    ...detection(0),
    analysed: false,
    scab_detected: null,
    input_check: { score: 0.4, threshold: 0.86, passed: false, skipped: false, model: "m", version: "v", inference_ms: 40 },
    warnings: ["This doesn't look like an apple leaf, so the scab detector was not run. Repeat with skip_input_check=true to analyse it anyway."],
  };
}

function uncheckedResult(): DetectionResponse {
  return {
    ...detection(2),
    input_check: { score: 0.4, threshold: 0.86, passed: false, skipped: true, model: "m", version: "v", inference_ms: 40 },
  };
}

afterEach(() => vi.restoreAllMocks());

async function upload(detectLesion: ReturnType<typeof vi.fn>) {
  vi.stubGlobal("URL", { ...URL, createObjectURL: () => "blob:x", revokeObjectURL: () => undefined });
  const client = { detectLesion } as unknown as ApiClient;
  const { container } = render(<LeafScan client={client} risk={riskResponse()} palette={palette} enabled />);
  const input = container.querySelector('input[type="file"]') as HTMLInputElement;
  fireEvent.change(input, { target: { files: [new File(["x"], "sky.jpg", { type: "image/jpeg" })] } });
}

describe("LeafScan input check", () => {
  it("shows the notice instead of a verdict, focuses it, and offers three actions", async () => {
    await upload(vi.fn(async () => rejected()));
    const heading = await screen.findByRole("heading", { name: INPUT_CHECK_COPY.heading });
    await waitFor(() => expect(heading).toHaveFocus());
    expect(screen.getByText(INPUT_CHECK_COPY.why)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Try another photo" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Photo tips" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Analyse anyway" })).toBeInTheDocument();
    expect(screen.queryByText(/No scab lesions found/)).toBeNull();
    expect(screen.queryByText(/skip_input_check/)).toBeNull(); // the API's raw warning is not repeated
  });

  it("Analyse anyway sends the photo again with the check skipped and labels the result", async () => {
    const detect = vi.fn(async (_b: Blob, _n: string, _s: AbortSignal, skip: boolean) => (skip ? uncheckedResult() : rejected()));
    await upload(detect);
    fireEvent.click(await screen.findByRole("button", { name: "Analyse anyway" }));
    expect(await screen.findByTestId("unchecked-label")).toHaveTextContent(INPUT_CHECK_COPY.uncheckedLabel);
    expect(detect).toHaveBeenLastCalledWith(expect.any(Blob), "sky.jpg", expect.anything(), true);
    expect(screen.queryByTestId("input-check-notice")).toBeNull();
  });

  it("shows results from an API older than the input check (no analysed field) as before", async () => {
    const older = { ...detection(2) } as Partial<DetectionResponse>;
    delete older.analysed;
    delete older.input_check;
    await upload(vi.fn(async () => older as DetectionResponse));
    expect(await screen.findByText(/2 scab lesions detected/)).toBeInTheDocument();
    expect(screen.queryByTestId("input-check-notice")).toBeNull();
  });

  it("a leaf without lesions is never called healthy", async () => {
    await upload(vi.fn(async () => detection(0)));
    expect(await screen.findByText("No scab lesions found.")).toBeInTheDocument();
    expect(screen.getByText(/This model only looks for scab; it cannot see other diseases\./)).toBeInTheDocument();
    expect(screen.queryByText(/clean|healthy leaf/i)).toBeNull();
  });
});
