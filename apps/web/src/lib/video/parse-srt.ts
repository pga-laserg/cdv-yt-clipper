export type TranscriptSegment = {
  start: number;
  end: number;
  text: string;
  id?: string;
};

export function parseSrtToSegments(srt: string): TranscriptSegment[] {
  const blocks = srt.trim().split(/\r?\n\r?\n+/);
  const segments: TranscriptSegment[] = [];

  for (let i = 0; i < blocks.length; i++) {
    const block = blocks[i];
    const lines = block.split(/\r?\n/).filter(Boolean);
    if (lines.length < 3) continue;

    const timeLine = lines[1];
    const match = timeLine.match(
      /(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2}),(\d{3})/
    );
    if (!match) continue;

    const start =
      Number(match[1]) * 3600 +
      Number(match[2]) * 60 +
      Number(match[3]) +
      Number(match[4]) / 1000;
    const end =
      Number(match[5]) * 3600 +
      Number(match[6]) * 60 +
      Number(match[7]) +
      Number(match[8]) / 1000;

    const text = lines.slice(2).join(" ").trim();
    if (!text) continue;

    segments.push({ start, end, text, id: `seg-${i}` });
  }

  return segments;
}
