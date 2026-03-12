import { MAX_FINAL_LENGTH_SEC } from "./clamp-range";

export function validateTrimRange(params: {
  startSec: number;
  endSec: number;
  sourceDurationSec: number;
}) {
  const { startSec, endSec, sourceDurationSec } = params;

  if (!Number.isFinite(startSec) || !Number.isFinite(endSec)) {
    throw new Error("Invalid trim range");
  }

  if (endSec <= startSec) {
    throw new Error("End time must be greater than start time");
  }

  if (startSec < 0) {
    throw new Error("Start time cannot be negative");
  }

  if (endSec > sourceDurationSec) {
    throw new Error("End time exceeds source duration");
  }

  if (endSec - startSec > MAX_FINAL_LENGTH_SEC) {
    throw new Error(`Trim duration exceeds the maximum allowed length of ${MAX_FINAL_LENGTH_SEC} seconds`);
  }
}
