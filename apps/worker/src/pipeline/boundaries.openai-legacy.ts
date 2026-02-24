import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { OpenAI } from 'openai';

export interface SermonBoundaries {
    start: number;
    end: number;
}

interface Segment {
    start: number;
    end: number;
    text: string;
}

interface SpeakerTurn {
    start: number;
    end: number;
    speaker: string;
}

interface BoundaryResult {
    speaker_start_sec: number;
    speaker_end_sec: number;
    confidence: number;
    notes?: string;
    start_candidates?: Array<{ sec: number; confidence: number; note?: string }>;
    end_candidates?: Array<{ sec: number; confidence: number; note?: string }>;
}

interface FindBoundariesOptions {
    workDir?: string;
    audioPath?: string;
}

interface CorrectionOptions {
    workDir?: string;
}

function getOpenAIClient(): OpenAI | null {
    const key = process.env.OPENAI_API_KEY;
    if (!key) return null;
    return new OpenAI({ apiKey: key });
}

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

async function correctChunk(openai: OpenAI, chunk: Segment[], model = 'gpt-4o-mini'): Promise<string[]> {
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
    if (!content) return chunk.map((s) => cleanText(s.text));
    const parsed = JSON.parse(content) as { lines?: string[] };
    if (!Array.isArray(parsed.lines) || parsed.lines.length !== chunk.length) {
        return chunk.map((s) => cleanText(s.text));
    }
    return parsed.lines.map((line) => cleanText(String(line)));
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

export async function ensureCorrectedTranscript(
    transcript: Segment[],
    options: CorrectionOptions = {}
): Promise<Segment[]> {
    const normalized = transcript
        .filter((s) => Number.isFinite(s.start) && Number.isFinite(s.end))
        .map((s) => ({ start: Number(s.start), end: Number(s.end), text: cleanText(String(s.text ?? '')) }));

    if (!options.workDir) return normalized;

    const correctedPath = path.join(options.workDir, 'transcript.corrected.json');
    if (fs.existsSync(correctedPath)) {
        try {
            const raw = JSON.parse(fs.readFileSync(correctedPath, 'utf8'));
            if (Array.isArray(raw) && raw.length > 0) return raw as Segment[];
        } catch {
            // Ignore and regenerate.
        }
    }

    const openai = getOpenAIClient();
    if (!openai) {
        fs.writeFileSync(correctedPath, JSON.stringify(normalized, null, 2));
        fs.writeFileSync(path.join(options.workDir, 'transcript.paragraphs.md'), toParagraphMarkdown(normalized));
        return normalized;
    }

    const model = process.env.ANALYZE_OPENAI_MODEL || 'gpt-4o-mini';
    const corrected: Segment[] = [];
    const chunks = chunkArray(normalized, 120);
    for (const chunk of chunks) {
        const lines = await correctChunk(openai, chunk, model);
        for (let i = 0; i < chunk.length; i++) {
            corrected.push({ start: chunk[i].start, end: chunk[i].end, text: lines[i] });
        }
    }

    fs.writeFileSync(correctedPath, JSON.stringify(corrected, null, 2));
    fs.writeFileSync(path.join(options.workDir, 'transcript.paragraphs.md'), toParagraphMarkdown(corrected));
    return corrected;
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

async function detectSermonSpeakerBounds(segments: Segment[]): Promise<BoundaryResult | null> {
    const openai = getOpenAIClient();
    if (!openai) return null;
    const model = process.env.ANALYZE_OPENAI_MODEL || 'gpt-4o-mini';
    const blocks = buildTimelineBlocks(segments, 30);
    const timeline = blocks.map((b) => `[${b.start}-${b.end}] ${b.text.slice(0, 220)}`).join('\n');

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
  "notes": "short explanation",
  "start_candidates": [
    {"sec": number, "confidence": number, "note": "primary"},
    {"sec": number, "confidence": number, "note": "alt-1"},
    {"sec": number, "confidence": number, "note": "alt-2"}
  ],
  "end_candidates": [
    {"sec": number, "confidence": number, "note": "primary"},
    {"sec": number, "confidence": number, "note": "alt-1"},
    {"sec": number, "confidence": number, "note": "alt-2"}
  ]
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
    if (!content) return null;
    const parsed = JSON.parse(content) as Partial<BoundaryResult>;
    const start = Number(parsed.speaker_start_sec);
    const end = Number(parsed.speaker_end_sec);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return null;

    const confidence = Number(parsed.confidence ?? 0.5);
    const normalizeCandidates = (
        input: unknown,
        fallbackSec: number
    ): Array<{ sec: number; confidence: number; note?: string }> => {
        const items = Array.isArray(input) ? input : [];
        const normalized = items
            .map((item) => {
                if (!item || typeof item !== 'object') return null;
                const obj = item as Record<string, unknown>;
                const sec = Number(obj.sec);
                const conf = Number(obj.confidence);
                if (!Number.isFinite(sec)) return null;
                return {
                    sec,
                    confidence: Number.isFinite(conf) ? Math.max(0, Math.min(1, conf)) : confidence,
                    note: typeof obj.note === 'string' ? obj.note : undefined
                };
            })
            .filter(Boolean) as Array<{ sec: number; confidence: number; note?: string }>;

        normalized.push({ sec: fallbackSec, confidence: Math.max(0, Math.min(1, confidence)), note: 'fallback-primary' });
        const seen = new Set<number>();
        return normalized
            .sort((a, b) => b.confidence - a.confidence)
            .filter((c) => {
                const key = Math.round(c.sec * 100) / 100;
                if (seen.has(key)) return false;
                seen.add(key);
                return true;
            })
            .slice(0, 3);
    };

    return {
        speaker_start_sec: start,
        speaker_end_sec: end,
        confidence: Math.max(0, Math.min(1, confidence)),
        notes: parsed.notes ? String(parsed.notes) : '',
        start_candidates: normalizeCandidates(parsed.start_candidates, start),
        end_candidates: normalizeCandidates(parsed.end_candidates, end)
    };
}

function parseGuess(boundary: BoundaryResult): {
    start: number;
    end: number;
    startCandidates: number[];
    endCandidates: number[];
} {
    const start = Number(boundary.speaker_start_sec);
    const end = Number(boundary.speaker_end_sec);
    const normalize = (items: Array<{ sec: number }> | undefined, fallback: number): number[] => {
        const values = Array.isArray(items) ? items.map((x) => Number(x.sec)).filter(Number.isFinite) : [];
        values.unshift(fallback);
        const out: number[] = [];
        const seen = new Set<number>();
        for (const v of values) {
            const key = Math.round(v * 100) / 100;
            if (seen.has(key)) continue;
            seen.add(key);
            out.push(v);
        }
        return out.slice(0, 3);
    };
    return { start, end, startCandidates: normalize(boundary.start_candidates, start), endCandidates: normalize(boundary.end_candidates, end) };
}

function sanitizeCandidates(
    primary: number,
    candidates: number[],
    minAllowed: number,
    maxAllowed: number,
    maxDelta: number
): number[] {
    const out: number[] = [];
    const seen = new Set<number>();
    const push = (v: number) => {
        const key = Math.round(v * 100) / 100;
        if (seen.has(key)) return;
        seen.add(key);
        out.push(v);
    };

    push(primary);
    for (const c of candidates) {
        if (!Number.isFinite(c)) continue;
        if (c < minAllowed || c > maxAllowed) continue;
        if (Math.abs(c - primary) > maxDelta) continue;
        push(c);
    }
    return out;
}

function runProcess(cmd: string, args: string[]): Promise<{ stdout: string; stderr: string }> {
    return new Promise((resolve, reject) => {
        const proc = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'] });
        let stdout = '';
        let stderr = '';
        proc.stdout.on('data', (d) => {
            stdout += d.toString();
        });
        proc.stderr.on('data', (d) => {
            stderr += d.toString();
        });
        proc.on('error', (err) => reject(err));
        proc.on('close', (code) => {
            if (code !== 0) return reject(new Error(`${cmd} exited with code ${code}. stderr tail: ${stderr.slice(-1200)}`));
            resolve({ stdout, stderr });
        });
    });
}

function overlap(aStart: number, aEnd: number, bStart: number, bEnd: number): number {
    return Math.max(0, Math.min(aEnd, bEnd) - Math.max(aStart, bStart));
}

function isNonSpeech(text?: string): boolean {
    const t = (text ?? '').toLowerCase().trim();
    if (!t) return true;
    if (t === 'music' || t === 'música' || t === 'piano' || t === 'silence' || t === 'silencio') return true;
    if (/^\[.*\]$/.test(t) || /^\(.*\)$/.test(t)) return true;
    if (/^(music|música|piano|instrumental|aplausos|applause|silence|silencio)\b/.test(t)) return true;
    return false;
}

async function extractChunk(audioPath: string, outPath: string, startSec: number, endSec: number): Promise<void> {
    const duration = Math.max(0.01, endSec - startSec);
    await runProcess('ffmpeg', ['-y', '-i', audioPath, '-ss', String(startSec), '-t', String(duration), '-acodec', 'pcm_s16le', outPath]);
}

async function runDiarize(chunkPath: string): Promise<SpeakerTurn[]> {
    const pythonScriptCandidates = [
        path.resolve(__dirname, 'python/diarize.py'),
        path.resolve(__dirname, '../../src/pipeline/python/diarize.py')
    ];
    const pythonScript = pythonScriptCandidates.find((p) => fs.existsSync(p)) ?? pythonScriptCandidates[0];
    const venv311 = path.resolve(__dirname, '../../venv311/bin/python3');
    const workerVenv = path.resolve(__dirname, '../../venv/bin/python3');
    const pythonBin = process.env.DIARIZATION_PYTHON_BIN
        ? path.resolve(process.env.DIARIZATION_PYTHON_BIN)
        : (fs.existsSync(venv311) ? venv311 : workerVenv);
    const token =
        process.env.PYANNOTE_ACCESS_TOKEN ||
        process.env.HUGGINGFACE_TOKEN ||
        process.env.HF_TOKEN ||
        process.env.HUGGINGFACE_ACCESS_TOKEN;
    const args = [pythonScript, chunkPath];
    if (token) args.push('--token', token);
    const { stdout } = await runProcess(pythonBin, args);
    const parsed = JSON.parse(stdout) as SpeakerTurn[];
    if (!Array.isArray(parsed)) throw new Error('Invalid diarization output (expected array).');
    return parsed;
}

function absoluteTurns(turns: SpeakerTurn[], offsetSec: number, speakerPrefix?: string): SpeakerTurn[] {
    return turns.map((t) => ({
        ...t,
        speaker: speakerPrefix ? `${speakerPrefix}:${t.speaker}` : t.speaker,
        start: t.start + offsetSec,
        end: t.end + offsetSec
    }));
}

function pickAnchorSpeaker(turns: SpeakerTurn[], anchorStart: number, anchorEnd: number): string | null {
    const scores = new Map<string, number>();
    for (const t of turns) {
        const ol = overlap(t.start, t.end, anchorStart, anchorEnd);
        if (ol <= 0) continue;
        scores.set(t.speaker, (scores.get(t.speaker) ?? 0) + ol);
    }
    if (scores.size === 0) return null;
    return [...scores.entries()].sort((a, b) => b[1] - a[1])[0][0];
}

function hasOtherSpeakerAround(turns: SpeakerTurn[], speaker: string, start: number, end: number, minOverlapSec = 0.35): boolean {
    for (const t of turns) {
        if (t.speaker === speaker) continue;
        if (overlap(t.start, t.end, start, end) >= minOverlapSec) return true;
    }
    return false;
}

function refineStart(turns: SpeakerTurn[], speaker: string, guessStart: number, gapJoinSec = 1.5, lookbackSec = 120): number {
    const same = turns
        .filter((t) => t.speaker === speaker && t.end >= guessStart - lookbackSec && t.start <= guessStart + 25)
        .sort((a, b) => a.start - b.start);
    if (same.length === 0) return guessStart;

    let idx = same.findIndex((t) => t.start <= guessStart && t.end >= guessStart);
    if (idx < 0) idx = same.findIndex((t) => t.start >= guessStart);
    if (idx < 0) idx = same.length - 1;
    let start = same[idx].start;

    for (let i = idx - 1; i >= 0; i--) {
        const gap = same[i + 1].start - same[i].end;
        if (gap > gapJoinSec) break;
        const bridgeStart = Math.min(same[i].end, same[i + 1].start) - 0.2;
        const bridgeEnd = Math.max(same[i].end, same[i + 1].start) + 0.2;
        if (hasOtherSpeakerAround(turns, speaker, bridgeStart, bridgeEnd)) break;
        start = same[i].start;
    }
    return start;
}

function refineEnd(turns: SpeakerTurn[], speaker: string, guessEnd: number, gapJoinSec = 1.5, lookaheadSec = 120): number {
    const same = turns
        .filter((t) => t.speaker === speaker && t.end >= guessEnd - 40 && t.start <= guessEnd + lookaheadSec)
        .sort((a, b) => a.start - b.start);
    if (same.length === 0) return guessEnd;

    let idx = same.findIndex((t) => t.start <= guessEnd && t.end >= guessEnd);
    if (idx < 0) idx = [...same].reverse().findIndex((t) => t.end <= guessEnd);
    if (idx < 0) idx = 0;
    else if (same.findIndex((t) => t.start <= guessEnd && t.end >= guessEnd) < 0) idx = same.length - 1 - idx;

    let end = same[idx].end;
    for (let i = idx + 1; i < same.length; i++) {
        const gap = same[i].start - same[i - 1].end;
        if (gap > gapJoinSec) break;
        const bridgeStart = Math.min(same[i - 1].end, same[i].start) - 0.2;
        const bridgeEnd = Math.max(same[i - 1].end, same[i].start) + 0.2;
        if (hasOtherSpeakerAround(turns, speaker, bridgeStart, bridgeEnd)) break;
        end = same[i].end;
    }
    return end;
}

function previousTurnBefore(turns: SpeakerTurn[], t: number): SpeakerTurn | null {
    for (let i = turns.length - 1; i >= 0; i--) {
        if (turns[i].end <= t + 0.001) return turns[i];
    }
    return null;
}

function nextTurnAfter(turns: SpeakerTurn[], t: number): SpeakerTurn | null {
    for (let i = 0; i < turns.length; i++) {
        if (turns[i].start >= t - 0.001) return turns[i];
    }
    return null;
}

function findLatestOtherToTargetStartTransition(turns: SpeakerTurn[], targetSpeaker: string, upperBound: number): number | null {
    let candidate: number | null = null;
    for (let i = 1; i < turns.length; i++) {
        const prev = turns[i - 1];
        const cur = turns[i];
        if (cur.start > upperBound + 15) break;
        if (prev.speaker !== targetSpeaker && cur.speaker === targetSpeaker) candidate = cur.start;
    }
    return candidate;
}

function findEarliestTargetToOtherEndTransition(turns: SpeakerTurn[], targetSpeaker: string, lowerBound: number): number | null {
    for (let i = 1; i < turns.length; i++) {
        const prev = turns[i - 1];
        const cur = turns[i];
        if (prev.end < lowerBound - 15) continue;
        if (prev.speaker === targetSpeaker && cur.speaker !== targetSpeaker) return prev.end;
    }
    return null;
}

function applyPadding(
    speakerStart: number,
    speakerEnd: number,
    segments: Segment[],
    turns: SpeakerTurn[],
    startSpeaker: string,
    endSpeaker: string
) {
    const speechSegments = segments.filter((s) => !isNonSpeech(s.text));
    const prevSpeech = [...speechSegments].reverse().find((s) => s.end <= speakerStart);
    const nextSpeech = speechSegments.find((s) => s.start >= speakerEnd);

    const prevOtherTurn = [...turns]
        .filter((t) => t.speaker !== startSpeaker && t.end <= speakerStart)
        .sort((a, b) => b.end - a.end)[0];
    const nextOtherTurn = turns
        .filter((t) => t.speaker !== endSpeaker && t.start >= speakerEnd)
        .sort((a, b) => a.start - b.start)[0];

    const prevBoundary = Math.max(prevSpeech?.end ?? 0, prevOtherTurn?.end ?? 0);
    const nextBoundary = Math.min(nextSpeech?.start ?? Number.POSITIVE_INFINITY, nextOtherTurn?.start ?? Number.POSITIVE_INFINITY);

    const gapBefore = Math.max(0, speakerStart - prevBoundary);
    const prePad = Math.min(10, gapBefore);
    const clipStart = Math.max(0, speakerStart - prePad);

    const gapAfter = Number.isFinite(nextBoundary) ? Math.max(0, nextBoundary - speakerEnd) : 10;
    const postPad = Math.min(10, gapAfter);
    const clipEnd = Number.isFinite(nextBoundary) ? Math.min(nextBoundary, speakerEnd + postPad) : speakerEnd + postPad;

    return {
        clip_start_sec: clipStart,
        clip_end_sec: clipEnd,
        applied_pre_pad_sec: prePad,
        applied_post_pad_sec: postPad,
        next_speech_start_sec: Number.isFinite(nextBoundary) ? nextBoundary : null,
        prev_other_speaker_end_sec: prevOtherTurn?.end ?? null,
        next_other_speaker_start_sec: nextOtherTurn?.start ?? null
    };
}

function fallbackHeuristicBoundaries(transcript: Segment[]): SermonBoundaries {
    const startKeywords = ['predicación', 'mensaje', 'abramos la palabra', 'lectura de hoy', 'bíblica', 'sermón'];
    const endKeywords = ['oremos', 'vamos a orar', 'despedida', 'anuncios', 'bendición', 'amén'];
    let start = 0;
    let end = transcript.length > 0 ? transcript[transcript.length - 1].end : 0;
    for (const segment of transcript) {
        const text = segment.text.toLowerCase();
        if (startKeywords.some((k) => text.includes(k))) {
            start = segment.start;
            break;
        }
    }
    for (let i = transcript.length - 1; i >= 0; i--) {
        const text = transcript[i].text.toLowerCase();
        if (endKeywords.some((k) => text.includes(k))) {
            end = transcript[i].end;
            break;
        }
    }
    if (start >= end) return { start: 0, end: transcript.length > 0 ? transcript[transcript.length - 1].end : 0 };
    return { start, end };
}

export async function findSermonBoundaries(
    transcript: Segment[],
    options: FindBoundariesOptions = {}
): Promise<SermonBoundaries> {
    console.log('Finding sermon boundaries (openai + targeted diarization)...');

    const normalized = transcript
        .filter((s) => Number.isFinite(s.start) && Number.isFinite(s.end))
        .map((s) => ({ start: Number(s.start), end: Number(s.end), text: cleanText(String(s.text ?? '')) }))
        .sort((a, b) => a.start - b.start);
    if (normalized.length === 0) return { start: 0, end: 0 };

    const workDir = options.workDir;
    const targetedPath = workDir ? path.join(workDir, 'sermon.boundaries.targeted-diarization.json') : '';
    if (targetedPath && fs.existsSync(targetedPath)) {
        try {
            const raw = JSON.parse(fs.readFileSync(targetedPath, 'utf8')) as any;
            const start = Number(raw?.final_clip_bounds?.clip_start_sec);
            const end = Number(raw?.final_clip_bounds?.clip_end_sec);
            if (Number.isFinite(start) && Number.isFinite(end) && end > start) return { start, end };
        } catch {
            // Ignore invalid cache.
        }
    }

    let boundary: BoundaryResult | null = null;
    if (workDir) {
        const openaiPath = path.join(workDir, 'sermon.boundaries.openai.json');
        if (fs.existsSync(openaiPath)) {
            try {
                const raw = JSON.parse(fs.readFileSync(openaiPath, 'utf8')) as any;
                boundary = raw?.raw ?? raw;
            } catch {
                // Ignore and regenerate.
            }
        }
    }

    if (!boundary) {
        boundary = await detectSermonSpeakerBounds(normalized);
    }

    if (!boundary) {
        return fallbackHeuristicBoundaries(normalized);
    }

    const guess = parseGuess(boundary);
    if (workDir) {
        fs.writeFileSync(path.join(workDir, 'sermon.boundaries.openai.json'), JSON.stringify({ raw: boundary }, null, 2));
    }

    if (!options.audioPath || !workDir) {
        return { start: guess.start, end: guess.end };
    }

    try {
        const audioEnd = normalized.length > 0 ? normalized[normalized.length - 1].end : guess.end + 1;
        const preScanSec = Number(process.env.TARGET_DIAR_PRE_SCAN_SEC ?? 120);
        const postScanSec = Number(process.env.TARGET_DIAR_POST_SCAN_SEC ?? 120);
        const anchorSec = Number(process.env.TARGET_DIAR_ANCHOR_SEC ?? 30);
        const maxConfirmGapSec = Number(process.env.TARGET_DIAR_MAX_CONFIRM_GAP_SEC ?? 30);

        const maxCandidateDeltaSec = Number(process.env.TARGET_DIAR_MAX_CANDIDATE_DELTA_SEC ?? 900);
        const startCandidates = sanitizeCandidates(
            guess.start,
            guess.startCandidates.length > 0 ? guess.startCandidates : [guess.start],
            0,
            guess.end - 30,
            maxCandidateDeltaSec
        );
        const endCandidates = sanitizeCandidates(
            guess.end,
            guess.endCandidates.length > 0 ? guess.endCandidates : [guess.end],
            guess.start + 30,
            audioEnd,
            maxCandidateDeltaSec
        );
        const earliestStartSeed = Math.min(...startCandidates);
        const latestStartSeed = Math.max(...startCandidates);
        const earliestEndSeed = Math.min(...endCandidates);
        const latestEndSeed = Math.max(...endCandidates);

        const preWindowStart = Math.max(0, earliestStartSeed - preScanSec);
        const preWindowEnd = Math.min(audioEnd, latestStartSeed + anchorSec);
        const postWindowStart = Math.max(0, earliestEndSeed - anchorSec);
        const postWindowEnd = Math.min(audioEnd, latestEndSeed + postScanSec);

        const tmpDir = path.join(workDir, 'processed');
        fs.mkdirSync(tmpDir, { recursive: true });
        const preChunk = path.join(tmpDir, 'diar.pre.wav');
        const postChunk = path.join(tmpDir, 'diar.post.wav');

        await extractChunk(options.audioPath, preChunk, preWindowStart, preWindowEnd);
        const preTurnsAbs = absoluteTurns(await runDiarize(preChunk), preWindowStart, 'pre');
        await extractChunk(options.audioPath, postChunk, postWindowStart, postWindowEnd);
        const postTurnsAbs = absoluteTurns(await runDiarize(postChunk), postWindowStart, 'post');

        const preAnchorStart = Math.max(preWindowStart, guess.start + 10);
        const preAnchorEnd = Math.min(preWindowEnd, guess.start + 70);
        const postAnchorStart = Math.max(postWindowStart, guess.end - 70);
        const postAnchorEnd = Math.min(postWindowEnd, guess.end - 10);

        const preSpeaker = pickAnchorSpeaker(preTurnsAbs, preAnchorStart, preAnchorEnd);
        const postSpeaker = pickAnchorSpeaker(postTurnsAbs, postAnchorStart, postAnchorEnd) ?? preSpeaker;
        if (!preSpeaker || !postSpeaker) throw new Error('Failed to resolve anchor speaker.');

        const startSpeaker = preSpeaker;
        const endSpeaker = postSpeaker;
        const allTurns = [...preTurnsAbs, ...postTurnsAbs].sort((a, b) => a.start - b.start);
        let refinedStart = refineStart(preTurnsAbs, startSpeaker, guess.start);
        let refinedEnd = refineEnd(postTurnsAbs, endSpeaker, guess.end);
        let startChosenBy = 'primary';
        let endChosenBy = 'primary';

        for (const c of startCandidates) {
            const candidateStart = refineStart(preTurnsAbs, startSpeaker, c);
            const prev = previousTurnBefore(preTurnsAbs, candidateStart);
            if (prev && prev.speaker !== startSpeaker && candidateStart - prev.end <= maxConfirmGapSec) {
                refinedStart = candidateStart;
                startChosenBy = `candidate:${c}`;
                break;
            }
        }
        if (startChosenBy === 'primary') {
            const fallbackStart = findLatestOtherToTargetStartTransition(preTurnsAbs, startSpeaker, refinedStart);
            if (fallbackStart != null) {
                refinedStart = refineStart(preTurnsAbs, startSpeaker, fallbackStart);
                startChosenBy = 'sequential-fallback';
            }
        }

        for (const c of endCandidates) {
            const candidateEnd = refineEnd(postTurnsAbs, endSpeaker, c);
            const next = nextTurnAfter(postTurnsAbs, candidateEnd);
            if (next && next.speaker !== endSpeaker && next.start - candidateEnd <= maxConfirmGapSec) {
                refinedEnd = candidateEnd;
                endChosenBy = `candidate:${c}`;
                break;
            }
        }
        if (endChosenBy === 'primary') {
            const fallbackEnd = findEarliestTargetToOtherEndTransition(postTurnsAbs, endSpeaker, refinedEnd);
            if (fallbackEnd != null) {
                refinedEnd = refineEnd(postTurnsAbs, endSpeaker, fallbackEnd);
                endChosenBy = 'sequential-fallback';
            }
        }

        const padded = applyPadding(refinedStart, refinedEnd, normalized, allTurns, startSpeaker, endSpeaker);
        const prevAtStart = previousTurnBefore(preTurnsAbs, refinedStart);
        const nextAtEnd = nextTurnAfter(postTurnsAbs, refinedEnd);

        const output = {
            guess: {
                start_sec: guess.start,
                end_sec: guess.end,
                start_candidates_sec: startCandidates,
                end_candidates_sec: endCandidates
            },
            selected_target_speaker: {
                start: startSpeaker,
                end: endSpeaker
            },
            boundary_confirmation: {
                start_confirmed_change_of_speaker: Boolean(prevAtStart && prevAtStart.speaker !== startSpeaker),
                end_confirmed_change_of_speaker: Boolean(nextAtEnd && nextAtEnd.speaker !== endSpeaker),
                start_chosen_by: startChosenBy,
                end_chosen_by: endChosenBy,
                start_previous_turn: prevAtStart,
                end_next_turn: nextAtEnd
            },
            targeted_windows: {
                pre: { start_sec: preWindowStart, end_sec: preWindowEnd, anchor_speaker: preSpeaker, turns: preTurnsAbs },
                post: { start_sec: postWindowStart, end_sec: postWindowEnd, anchor_speaker: postSpeaker, turns: postTurnsAbs }
            },
            refined_speaker_bounds: {
                start_sec: refinedStart,
                end_sec: refinedEnd
            },
            final_clip_bounds: padded
        };

        fs.writeFileSync(targetedPath, JSON.stringify(output, null, 2));
        return { start: padded.clip_start_sec, end: padded.clip_end_sec };
    } catch (error) {
        console.error('Targeted diarization refinement failed, using OpenAI guess:', error);
        return { start: guess.start, end: guess.end };
    }
}
