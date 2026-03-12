export const MAX_FINAL_LENGTH_SEC = 179;

export type TrimRange = {
  startSec: number;
  endSec: number;
};

export function clampTrimRange(
  startSec: number,
  endSec: number,
  sourceDurationSec: number
): TrimRange {
  let start = Math.max(0, startSec);
  let end = Math.min(sourceDurationSec, endSec);

  if (end < start) {
    [start, end] = [end, start];
  }

  if (end - start > MAX_FINAL_LENGTH_SEC) {
    end = start + MAX_FINAL_LENGTH_SEC;
  }

  if (end > sourceDurationSec) {
    end = sourceDurationSec;
    start = Math.max(0, end - MAX_FINAL_LENGTH_SEC);
  }

  return { startSec: start, endSec: end };
}
