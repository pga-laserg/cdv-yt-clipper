import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import { OpenAI } from 'openai';
import { findHighlights } from './pipeline/highlights';

dotenv.config({ path: path.resolve(__dirname, '../../../.env') });

interface Segment {
    start: number;
    end: number;
    text: string;
}

interface BoundaryResult {
    speaker_start_sec: number;
    speaker_end_sec: number;
    confidence: number;
    notes?: string;
}

const openaiKey = process.env.OPENAI_API_KEY;
if (!openaiKey) {
    throw new Error('OPENAI_API_KEY is required for test-openai-analysis.');
}
const openai = new OpenAI({ apiKey: openaiKey });

function cleanText(text: string): string {
    return text
        .replace(/\s+/g, ' ')
        .replace(/^\s*\[[^\]]+\]\s*$/g, '')
        .trim();
}

function chunkArray<T>(input: T[], size: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < input.length; i += size) chunks.push(input.slice(i, i + size));
    return chunks;
}

async function correctChunk(chunk: Segment[], model = 'gpt-4o-mini'): Promise<string[]> {
    const indexed = chunk.map((s, i) => `${i}|||${cleanText(s.text)}`);
    const prompt = `
You are correcting transcript text for readability.
Rules:
- Keep original language.
- Fix punctuation, capitalization, and obvious grammar/OCR errors.
- Preserve meaning.
- Do not add new facts.
- Return EXACTLY one corrected line per input line, in the same order.
- Output JSON object: {"lines":["..."]}.

Input lines:
${indexed.join('\n')}
`;

    const response = await openai.chat.completions.create({
        model,
        messages: [{ role: 'user', content: prompt }],
        response_format: { type: 'json_object' }
    });

    const content = response.choices[0]?.message?.content;
    if (!content) throw new Error('OpenAI returned empty content for correction chunk.');
    const parsed = JSON.parse(content) as { lines?: string[] };
    if (!Array.isArray(parsed.lines) || parsed.lines.length !== chunk.length) {
        console.warn(
            `Correction response length mismatch for chunk (expected ${chunk.length}, got ${parsed.lines?.length ?? 0}). Using original lines for this chunk.`
        );
        return chunk.map((s) => cleanText(s.text));
    }
    return parsed.lines.map((line) => cleanText(String(line)));
}

async function correctTranscript(segments: Segment[]): Promise<Segment[]> {
    const chunks = chunkArray(segments, 120);
    const corrected: Segment[] = [];

    for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        console.log(`Correcting transcript chunk ${i + 1}/${chunks.length}...`);
        const lines = await correctChunk(chunk);
        for (let j = 0; j < chunk.length; j++) {
            corrected.push({
                start: chunk[j].start,
                end: chunk[j].end,
                text: lines[j]
            });
        }
    }

    return corrected;
}

function toParagraphMarkdown(segments: Segment[]): string {
    const paragraphs: string[] = [];
    let current: Segment[] = [];

    for (const seg of segments) {
        const text = cleanText(seg.text);
        if (!text) continue;

        const prev = current.length > 0 ? current[current.length - 1] : null;
        const gap = prev ? seg.start - prev.end : 0;
        const shouldBreak = !prev || gap > 1.2 || /[.!?]["')\]]?$/.test(prev.text);

        if (shouldBreak && current.length > 0) {
            paragraphs.push(current.map((s) => s.text).join(' '));
            current = [];
        }
        current.push({ ...seg, text });
    }
    if (current.length > 0) paragraphs.push(current.map((s) => s.text).join(' '));

    return paragraphs
        .filter(Boolean)
        .map((p, idx) => `### Paragraph ${idx + 1}\n\n${p}`)
        .join('\n\n');
}

function buildTimelineBlocks(segments: Segment[], blockSeconds = 30): Array<{ start: number; end: number; text: string }> {
    if (segments.length === 0) return [];

    const start = Math.floor(segments[0].start);
    const end = Math.ceil(segments[segments.length - 1].end);
    const blocks: Array<{ start: number; end: number; text: string }> = [];

    for (let t = start; t < end; t += blockSeconds) {
        const bStart = t;
        const bEnd = Math.min(t + blockSeconds, end);
        const text = segments
            .filter((s) => s.start < bEnd && s.end > bStart)
            .map((s) => s.text)
            .join(' ')
            .trim();
        blocks.push({ start: bStart, end: bEnd, text });
    }

    return blocks;
}

async function detectSermonSpeakerBounds(segments: Segment[], model = 'gpt-4o-mini'): Promise<BoundaryResult> {
    const blocks = buildTimelineBlocks(segments, 30);
    const timeline = blocks
        .map((b) => `[${b.start}-${b.end}] ${b.text.slice(0, 220)}`)
        .join('\n');

    const prompt = `
You are analyzing a church service transcript timeline.
Find the MAIN SERMON SPEAKER section (not songs, children stories, offering, announcements, prayers by others).
Return the first second where this main speaker starts the sermon message and the last second where this speaker ends the sermon.

Rules:
- If there are intros by other people before the sermon speaker, exclude them.
- If someone else speaks after the sermon speaker closes, exclude that too.
- Return JSON object:
{
  "speaker_start_sec": number,
  "speaker_end_sec": number,
  "confidence": number,
  "notes": "short explanation"
}

Timeline blocks:
${timeline}
`;

    const response = await openai.chat.completions.create({
        model,
        messages: [{ role: 'user', content: prompt }],
        response_format: { type: 'json_object' }
    });

    const content = response.choices[0]?.message?.content;
    if (!content) throw new Error('OpenAI returned empty content for boundary detection.');
    const parsed = JSON.parse(content) as Partial<BoundaryResult>;

    const speaker_start_sec = Number(parsed.speaker_start_sec);
    const speaker_end_sec = Number(parsed.speaker_end_sec);
    const confidence = Number(parsed.confidence ?? 0.5);

    if (!Number.isFinite(speaker_start_sec) || !Number.isFinite(speaker_end_sec) || speaker_end_sec <= speaker_start_sec) {
        throw new Error(`Invalid boundary response: ${content}`);
    }

    return {
        speaker_start_sec,
        speaker_end_sec,
        confidence: Math.max(0, Math.min(1, confidence)),
        notes: parsed.notes ? String(parsed.notes) : ''
    };
}

function applyBoundaryPadding(raw: BoundaryResult, allSegments: Segment[]) {
    const sorted = [...allSegments].sort((a, b) => a.start - b.start);
    const prev = [...sorted].reverse().find((s) => s.end <= raw.speaker_start_sec);
    const next = sorted.find((s) => s.start >= raw.speaker_end_sec);

    const gapBefore = Math.max(0, raw.speaker_start_sec - (prev?.end ?? 0));
    const prePad = Math.min(10, gapBefore);
    const paddedStart = Math.max(0, raw.speaker_start_sec - prePad);

    const gapAfter = next ? Math.max(0, next.start - raw.speaker_end_sec) : 10;
    const postPad = Math.min(10, gapAfter);
    const paddedEnd = next
        ? Math.min(next.start, raw.speaker_end_sec + postPad)
        : raw.speaker_end_sec + postPad;

    return {
        ...raw,
        padded_start_sec: paddedStart,
        padded_end_sec: paddedEnd,
        applied_pre_pad_sec: prePad,
        applied_post_pad_sec: postPad,
        next_speech_start_sec: next?.start ?? null
    };
}

async function main() {
    const inputArg = process.argv[2] || 'apps/test_data/ingest_test/transcript.json';
    const transcriptPath = path.resolve(process.cwd(), inputArg);
    if (!fs.existsSync(transcriptPath)) {
        throw new Error(`Transcript not found: ${transcriptPath}`);
    }

    const outDir = path.dirname(transcriptPath);
    const raw = JSON.parse(fs.readFileSync(transcriptPath, 'utf8')) as Segment[];
    const segments = raw
        .filter((s) => Number.isFinite(s.start) && Number.isFinite(s.end) && typeof s.text === 'string')
        .map((s) => ({ start: Number(s.start), end: Number(s.end), text: cleanText(s.text) }));

    console.log(`Loaded ${segments.length} transcript segments from ${transcriptPath}`);

    const corrected = await correctTranscript(segments);
    const correctedPath = path.join(outDir, 'transcript.corrected.json');
    fs.writeFileSync(correctedPath, JSON.stringify(corrected, null, 2));
    console.log(`Wrote corrected segments: ${correctedPath}`);

    const paragraphMd = toParagraphMarkdown(corrected);
    const paragraphPath = path.join(outDir, 'transcript.paragraphs.md');
    fs.writeFileSync(paragraphPath, paragraphMd);
    console.log(`Wrote paragraph markdown: ${paragraphPath}`);

    const rawBoundary = await detectSermonSpeakerBounds(corrected);
    const paddedBoundary = applyBoundaryPadding(rawBoundary, corrected);
    const boundaryPath = path.join(outDir, 'sermon.boundaries.openai.json');
    fs.writeFileSync(boundaryPath, JSON.stringify({ raw: rawBoundary, padded: paddedBoundary }, null, 2));
    console.log(`Wrote sermon boundaries: ${boundaryPath}`);

    const sermonTranscript = corrected.filter(
        (s) => s.start >= paddedBoundary.padded_start_sec && s.end <= paddedBoundary.padded_end_sec
    );
    const highlights = await findHighlights(sermonTranscript, 'openai');
    const highlightPath = path.join(outDir, 'highlights.openai.json');
    fs.writeFileSync(highlightPath, JSON.stringify(highlights, null, 2));
    console.log(`Wrote highlights: ${highlightPath}`);
}

main().catch((error) => {
    console.error('test-openai-analysis failed:', error);
    process.exit(1);
});
