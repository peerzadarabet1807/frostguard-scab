import { describe, expect, it } from "vitest";
import { formatTemp, formatTick, istDate, istHour, istIso } from "./format";

describe("orchard-local time helpers", () => {
  it("builds IST ISO strings", () => {
    expect(istIso("2024-04-26", 6)).toBe("2024-04-26T06:00:00+05:30");
  });

  it("reads the IST date and hour back regardless of the viewer's timezone", () => {
    expect(istDate("2024-04-25T19:30:00Z")).toBe("2024-04-26"); // 01:00 IST next day
    expect(istHour("2024-04-25T19:30:00Z")).toBe(1);
  });

  it("labels midnight ticks with the date and others with the hour", () => {
    expect(formatTick(Date.parse("2024-04-27T00:00:00+05:30"))).toBe("27 Apr");
    expect(formatTick(Date.parse("2024-04-27T06:00:00+05:30"))).toBe("06:00");
  });
});

describe("formatTemp", () => {
  it("uses a true minus sign and handles gaps", () => {
    expect(formatTemp(-4.53)).toBe("−4.5 °C");
    expect(formatTemp(null)).toBe("—");
  });
});
