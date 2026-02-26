import fs from 'fs';
import path from 'path';
import { spawn, spawnSync } from 'child_process';
import { OpenAI } from 'openai';
import { v2 as speechV2 } from '@google-cloud/speech';
import { Storage } from '@google-cloud/storage';
import { emitLlmCueEvents } from './llm-cue-telemetry';

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

interface FaceBoundaryResult {
    start: number;
    end: number;
    start_candidates?: number[];
    end_candidates?: number[];
}

interface AudioEvent {
    label: string;
    start: number;
    end: number;
}

interface AudioEventPassResult {
    source: string;
    duration_sec: number;
    segments: AudioEvent[];
}

interface OcrEvent {
    start: number;
    end: number;
    text: string;
    type: 'speaker_name' | 'bible_verse' | 'sermon_title' | 'song_lyric' | 'on_screen_text';
    region: 'lower_third' | 'slide' | 'full';
    confidence: number;
    near_scene_cut?: boolean;
}

interface OcrEventPassResult {
    source: string;
    duration_sec: number;
    sample_sec: number;
    scene_cuts_sec: number[];
    segments: OcrEvent[];
}

interface SlideOcrEvent {
    start: number;
    end: number;
    text: string;
    type: 'speaker_name' | 'bible_verse' | 'sermon_title' | 'song_lyric' | 'on_screen_text';
    confidence: number;
    slide_id?: string;
}

interface SlideOcrPassResult {
    source: string;
    duration_sec: number;
    sample_sec: number;
    shot_boundaries_sec?: number[];
    events: SlideOcrEvent[];
    summary?: Record<string, any>;
}

type OcrPipeline = 'v1' | 'v2' | 'v3';

interface TranscriptBoundarySignal {
    sermonCueScore: number;
    prayerCueScore: number;
    worshipCueScore: number;
    announcementCueScore: number;
    closingCueScore: number;
    handoffCueScore: number;
    hasFelizSabado: boolean;
    sermonLikely: boolean;
    prayerDominant: boolean;
    worshipDominant: boolean;
    announcementDominant: boolean;
    closingLikely: boolean;
    handoffLikely: boolean;
}

interface FindBoundariesOptions {
    workDir?: string;
    audioPath?: string;
    videoPath?: string;
}

interface CorrectionOptions {
    workDir?: string;
}

interface LlmCoarseBounds {
    coarse_start: number;
    coarse_end: number;
    confidence?: number;
    reason?: string;
}

interface LlmRefinedBounds {
    start: number;
    end: number;
    confidence?: number;
    rationale?: string;
    start_evidence_line?: string;
    end_evidence_line?: string;
}

interface LlmEndStyleDecision {
    end_sec: number;
    include_following_song?: boolean;
    confidence?: number;
    reason?: string;
}

type BoundaryPipelineProfile = 'default' | 'light';

function getBoundaryPipelineProfile(): BoundaryPipelineProfile {
    const raw = String(process.env.BOUNDARY_PIPELINE_PROFILE ?? 'default').trim().toLowerCase();
    return raw === 'light' ? 'light' : 'default';
}

function normalizeOcrPipeline(raw: string | undefined | null): OcrPipeline {
    const v = String(raw ?? 'v1').trim().toLowerCase();
    if (v === 'v2' || v === '2' || v === 'ocr_v2') return 'v2';
    if (v === 'v3' || v === '3' || v === 'ocr_v3') return 'v3';
    return 'v1';
}

function getBoundaryOcrPipeline(): OcrPipeline {
    return normalizeOcrPipeline(process.env.BOUNDARY_OCR_PIPELINE ?? process.env.OCR_PIPELINE ?? 'v1');
}

function getOcrEventsFilename(pipeline: OcrPipeline): string {
    if (pipeline === 'v2') return 'ocr.events.v2.json';
    if (pipeline === 'v3') return 'ocr.events.v3.json';
    return 'ocr.events.json';
}

function profileNumber(
    profile: BoundaryPipelineProfile,
    envKey: string,
    defaults: { default: number; light: number }
): number {
    const raw = process.env[envKey];
    if (raw != null && raw !== '') return Number(raw);
    return profile === 'light' ? defaults.light : defaults.default;
}

function profileString(
    profile: BoundaryPipelineProfile,
    envKey: string,
    defaults: { default: string; light: string }
): string {
    const raw = process.env[envKey];
    if (raw != null && raw !== '') return raw;
    return profile === 'light' ? defaults.light : defaults.default;
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

function formatSec(sec: number): string {
    const s = Math.max(0, Math.floor(sec));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const ss = s % 60;
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
}

function trimForPrompt(text: string, maxLen = 160): string {
    const t = cleanText(text).replace(/\s+/g, ' ');
    if (t.length <= maxLen) return t;
    return `${t.slice(0, maxLen - 1)}…`;
}

function clamp(n: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, n));
}

function safeJsonParse<T>(raw: string, fallback: T): T {
    try {
        return JSON.parse(raw) as T;
    } catch {
        return fallback;
    }
}

function segmentsInRange(segments: Segment[], start: number, end: number): Segment[] {
    return segments.filter((s) => s.end >= start && s.start <= end);
}

function buildTranscriptBins(
    segments: Segment[],
    audioEvents: AudioEvent[] | null,
    durationSec: number,
    binSec: number,
    textMaxChars = 180
): Array<{ start: number; end: number; text: string; speech_ratio: number; music_ratio: number; noenergy_ratio: number }> {
    const bins: Array<{ start: number; end: number; text: string; speech_ratio: number; music_ratio: number; noenergy_ratio: number }> = [];
    for (let t = 0; t < durationSec; t += binSec) {
        const start = t;
        const end = Math.min(durationSec, t + binSec);
        const segs = segmentsInRange(segments, start, end).filter((s) => !isNonSpeech(s.text));
        const text = trimForPrompt(segs.map((s) => s.text).join(' '), textMaxChars);
        let speechRatio = 0;
        let musicRatio = 0;
        let noenergyRatio = 0;
        if (audioEvents && audioEvents.length > 0) {
            const durations = getWindowDurations(audioEvents, start, end);
            const total = Math.max(0.001, end - start);
            speechRatio = [...durations.entries()]
                .filter(([k]) => isSpeechLikeLabel(k))
                .reduce((sum, [, d]) => sum + d, 0) / total;
            musicRatio = (durations.get('music') ?? 0) / total;
            noenergyRatio = (durations.get('noenergy') ?? 0) / total;
        }
        bins.push({
            start,
            end,
            text,
            speech_ratio: Number(speechRatio.toFixed(3)),
            music_ratio: Number(musicRatio.toFixed(3)),
            noenergy_ratio: Number(noenergyRatio.toFixed(3))
        });
    }
    return bins;
}

function buildFocusedBins(
    allBins: Array<{ start: number; end: number; text: string; speech_ratio: number; music_ratio: number; noenergy_ratio: number }>,
    focusRanges: Array<{ start: number; end: number }>
) {
    if (!focusRanges.length) return allBins;
    return allBins.filter((b) =>
        focusRanges.some((r) => overlap(b.start, b.end, r.start, r.end) > 0)
    );
}

async function llmStage1CoarseBounds(
    openai: OpenAI,
    bins: Array<{ start: number; end: number; text: string; speech_ratio: number; music_ratio: number; noenergy_ratio: number }>,
    durationSec: number,
    localHint?: { start: number; end: number; startCandidates?: number[]; endCandidates?: number[] },
    workDir?: string
): Promise<LlmCoarseBounds | null> {
    const model = process.env.BOUNDARY_LLM_MODEL || process.env.ANALYZE_OPENAI_MODEL || 'gpt-4o-mini';
    const cachePath = workDir ? path.join(workDir, 'sermon.boundaries.openai.stage1.json') : '';
    if (cachePath && fs.existsSync(cachePath)) {
        const cached = safeJsonParse<LlmCoarseBounds | null>(fs.readFileSync(cachePath, 'utf8'), null);
        if (cached && Number.isFinite(cached.coarse_start) && Number.isFinite(cached.coarse_end)) return cached;
    }

    const binLines = bins
        .map((b, idx) => `${idx}|${formatSec(b.start)}-${formatSec(b.end)}|sp=${b.speech_ratio}|mu=${b.music_ratio}|sil=${b.noenergy_ratio}|${b.text || '[no_text]'}`)
        .join('\n');

    const prompt = [
        'Find the sermon body coarse boundaries from indexed bins.',
        'Return JSON only: {"coarse_start":number,"coarse_end":number,"confidence":number,"reason":"short"}',
        'Rules:',
        '- Sermon body is sustained expository preaching (not announcements, worship songs, offering logistics, greetings).',
        '- Prefer continuous speech-heavy bins with doctrinal/biblical content.',
        '- Keep bounds inside [0,duration].',
        `duration_sec=${durationSec.toFixed(3)}`,
        localHint
            ? `local_hint_start=${localHint.start.toFixed(3)} local_hint_end=${localHint.end.toFixed(3)}`
            : '',
        localHint?.startCandidates?.length
            ? `local_start_candidates=${localHint.startCandidates.slice(0, 6).map((v) => v.toFixed(3)).join(',')}`
            : '',
        localHint?.endCandidates?.length
            ? `local_end_candidates=${localHint.endCandidates.slice(0, 6).map((v) => v.toFixed(3)).join(',')}`
            : '',
        'Prefer boundaries near local hints unless transcript evidence strongly contradicts them.',
        'Bins:',
        binLines
    ].join('\n');

    const response = await openai.chat.completions.create({
        model,
        messages: [{ role: 'user', content: prompt }],
        response_format: { type: 'json_object' }
    });
    const content = response.choices[0]?.message?.content ?? '{}';
    const parsed = safeJsonParse<LlmCoarseBounds | null>(content, null);
    if (!parsed || !Number.isFinite(parsed.coarse_start) || !Number.isFinite(parsed.coarse_end)) return null;
    const llmMaxDrift = Number(process.env.BOUNDARY_LLM_STAGE1_MAX_DRIFT_SEC ?? 1500);
    const stage1Min = localHint ? clamp(localHint.start - llmMaxDrift, 0, durationSec) : 0;
    const stage1Max = localHint ? clamp(localHint.end + llmMaxDrift, 0, durationSec) : durationSec;
    parsed.coarse_start = clamp(parsed.coarse_start, stage1Min, stage1Max);
    parsed.coarse_end = clamp(parsed.coarse_end, stage1Min, stage1Max);
    if (parsed.coarse_end <= parsed.coarse_start + 30) return null;
    await emitLlmCueEvents(
        [
            {
                source_pass: 'boundary_stage1_coarse',
                model,
                section_type: 'sermon',
                cue_kind: 'boundary_coarse',
                cue_text: parsed.reason || 'coarse_bounds',
                cue_time_sec: parsed.coarse_start,
                confidence: parsed.confidence ?? null,
                metadata: { coarse_start: parsed.coarse_start, coarse_end: parsed.coarse_end }
            }
        ],
        workDir
    );
    if (cachePath) fs.writeFileSync(cachePath, JSON.stringify(parsed, null, 2));
    return parsed;
}

async function llmStage2RefineBounds(
    openai: OpenAI,
    transcript: Segment[],
    coarse: LlmCoarseBounds,
    durationSec: number,
    workDir?: string
): Promise<LlmRefinedBounds | null> {
    const model = process.env.BOUNDARY_LLM_MODEL || process.env.ANALYZE_OPENAI_MODEL || 'gpt-4o-mini';
    const cachePath = workDir ? path.join(workDir, 'sermon.boundaries.openai.stage2.json') : '';
    if (cachePath && fs.existsSync(cachePath)) {
        const cached = safeJsonParse<LlmRefinedBounds | null>(fs.readFileSync(cachePath, 'utf8'), null);
        if (cached && Number.isFinite(cached.start) && Number.isFinite(cached.end)) return cached;
    }

    const windowSec = Number(process.env.BOUNDARY_LLM_TIGHT_WINDOW_SEC ?? 360);
    const startWinStart = clamp(coarse.coarse_start - windowSec, 0, durationSec);
    const startWinEnd = clamp(coarse.coarse_start + windowSec, 0, durationSec);
    const endWinStart = clamp(coarse.coarse_end - windowSec, 0, durationSec);
    const endWinEnd = clamp(coarse.coarse_end + windowSec, 0, durationSec);
    const startSegs = segmentsInRange(transcript, startWinStart, startWinEnd).slice(0, 500);
    const endSegs = segmentsInRange(transcript, endWinStart, endWinEnd).slice(0, 500);

    const startLines = startSegs
        .map((s, i) => `S${i}|${formatSec(s.start)}-${formatSec(s.end)}|${trimForPrompt(s.text, 160)}`)
        .join('\n');
    const endLines = endSegs
        .map((s, i) => `E${i}|${formatSec(s.start)}-${formatSec(s.end)}|${trimForPrompt(s.text, 160)}`)
        .join('\n');

    const prompt = [
        'Refine sermon exact start/end from tight transcript windows.',
        'Return JSON only: {"start":number,"end":number,"confidence":number,"rationale":"short","start_evidence_line":"S#","end_evidence_line":"E#"}',
        'Rules:',
        '- Start: first clear transition into sustained sermon preaching.',
        '- End: final sermon statement/prayer before announcements/worship handoff.',
        '- Do not include unrelated post-sermon host speech.',
        `coarse_start=${coarse.coarse_start.toFixed(3)}`,
        `coarse_end=${coarse.coarse_end.toFixed(3)}`,
        `duration_sec=${durationSec.toFixed(3)}`,
        'START WINDOW:',
        startLines || '[empty]',
        'END WINDOW:',
        endLines || '[empty]'
    ].join('\n');

    const response = await openai.chat.completions.create({
        model,
        messages: [{ role: 'user', content: prompt }],
        response_format: { type: 'json_object' }
    });
    const content = response.choices[0]?.message?.content ?? '{}';
    const parsed = safeJsonParse<LlmRefinedBounds | null>(content, null);
    if (!parsed || !Number.isFinite(parsed.start) || !Number.isFinite(parsed.end)) return null;
    parsed.start = clamp(parsed.start, 0, durationSec);
    parsed.end = clamp(parsed.end, 0, durationSec);
    if (parsed.end <= parsed.start + 30) return null;
    await emitLlmCueEvents(
        [
            {
                source_pass: 'boundary_stage2_refine',
                model,
                section_type: 'sermon',
                cue_kind: 'boundary_refined',
                cue_text: parsed.rationale || 'refined_bounds',
                cue_time_sec: parsed.start,
                confidence: parsed.confidence ?? null,
                metadata: {
                    start: parsed.start,
                    end: parsed.end,
                    start_line: parsed.start_evidence_line ?? null,
                    end_line: parsed.end_evidence_line ?? null
                }
            }
        ],
        workDir
    );
    if (cachePath) fs.writeFileSync(cachePath, JSON.stringify(parsed, null, 2));
    return parsed;
}

function applyTranscriptOnlyPadding(speakerStart: number, speakerEnd: number, segments: Segment[]) {
    const speechSegments = segments.filter((s) => !isNonSpeech(s.text));
    const prevSpeech = [...speechSegments].reverse().find((s) => s.end <= speakerStart);
    const nextSpeech = speechSegments.find((s) => s.start >= speakerEnd);
    const prevBoundary = prevSpeech?.end ?? 0;
    const nextBoundary = nextSpeech?.start ?? Number.POSITIVE_INFINITY;
    const prePad = Math.min(10, Math.max(0, speakerStart - prevBoundary));
    const postPad = Number.isFinite(nextBoundary)
        ? Math.min(10, Math.max(0, nextBoundary - speakerEnd))
        : 10;
    const clipStart = Math.max(0, speakerStart - prePad);
    const clipEnd = Number.isFinite(nextBoundary)
        ? Math.min(nextBoundary, speakerEnd + postPad)
        : speakerEnd + postPad;
    return {
        clip_start_sec: clipStart,
        clip_end_sec: clipEnd,
        applied_pre_pad_sec: prePad,
        applied_post_pad_sec: postPad,
        next_speech_start_sec: Number.isFinite(nextBoundary) ? nextBoundary : null,
        prev_other_speaker_end_sec: null,
        next_other_speaker_start_sec: null
    };
}

function snapEndToSegmentEnd(segments: Segment[], t: number): number {
    const sorted = [...segments].sort((a, b) => a.end - b.end);
    let best = t;
    for (const s of sorted) {
        if (s.end <= t + 0.25) best = s.end;
        else break;
    }
    return best;
}

async function llmDecideEndingStyle(
    openai: OpenAI,
    transcript: Segment[],
    clipStart: number,
    candidateEnd: number,
    durationSec: number,
    workDir?: string
): Promise<LlmEndStyleDecision | null> {
    const model = process.env.BOUNDARY_LLM_MODEL || process.env.ANALYZE_OPENAI_MODEL || 'gpt-4o-mini';
    const cachePath = workDir ? path.join(workDir, 'sermon.boundaries.openai.end-style.json') : '';
    if (cachePath && fs.existsSync(cachePath)) {
        const cached = safeJsonParse<LlmEndStyleDecision | null>(fs.readFileSync(cachePath, 'utf8'), null);
        if (cached && Number.isFinite(cached.end_sec)) return cached;
    }

    const preSec = Number(process.env.BOUNDARY_LLM_END_STYLE_PRE_SEC ?? 240);
    const postSec = Number(process.env.BOUNDARY_LLM_END_STYLE_POST_SEC ?? 180);
    const winStart = clamp(candidateEnd - preSec, 0, durationSec);
    const winEnd = clamp(candidateEnd + postSec, 0, durationSec);
    const lines = segmentsInRange(transcript, winStart, winEnd)
        .slice(-350)
        .map((s, i) => `L${i}|${formatSec(s.start)}-${formatSec(s.end)}|${trimForPrompt(s.text, 180)}`)
        .join('\n');

    const prompt = [
        'Decide exact sermon clip ending from timeline lines.',
        'Return JSON only: {"end_sec":number,"include_following_song":boolean,"confidence":number,"reason":"short"}',
        'Goal:',
        '- Keep a coherent sermonic ending (closing prayer/amen can be included).',
        '- Exclude post-sermon social interactions and logistics (e.g., greetings, invitations to hug, announcements).',
        '- Include a final song ONLY if it is directly thematically connected and not followed by irrelevant host chatter.',
        `clip_start_sec=${clipStart.toFixed(3)}`,
        `candidate_end_sec=${candidateEnd.toFixed(3)}`,
        `window_start_sec=${winStart.toFixed(3)} window_end_sec=${winEnd.toFixed(3)}`,
        'Timeline:',
        lines || '[empty]'
    ].join('\n');

    const response = await openai.chat.completions.create({
        model,
        messages: [{ role: 'user', content: prompt }],
        response_format: { type: 'json_object' }
    });
    const content = response.choices[0]?.message?.content ?? '{}';
    const parsed = safeJsonParse<LlmEndStyleDecision | null>(content, null);
    if (!parsed || !Number.isFinite(parsed.end_sec)) return null;
    parsed.end_sec = clamp(parsed.end_sec, clipStart + 60, durationSec);
    await emitLlmCueEvents(
        [
            {
                source_pass: 'boundary_end_style',
                model,
                section_type: 'sermon',
                cue_kind: parsed.include_following_song ? 'include_following_song' : 'exclude_following_song',
                cue_text: parsed.reason || 'end_style_decision',
                cue_time_sec: parsed.end_sec,
                confidence: parsed.confidence ?? null,
                metadata: { include_following_song: parsed.include_following_song ?? null }
            }
        ],
        workDir
    );
    if (cachePath) fs.writeFileSync(cachePath, JSON.stringify(parsed, null, 2));
    return parsed;
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

function parseFaceGuess(boundary: FaceBoundaryResult): {
    start: number;
    end: number;
    startCandidates: number[];
    endCandidates: number[];
} {
    const start = Number(boundary.start);
    const end = Number(boundary.end);
    const startCandidates = Array.isArray(boundary.start_candidates)
        ? boundary.start_candidates.map((v) => Number(v)).filter(Number.isFinite)
        : [start];
    const endCandidates = Array.isArray(boundary.end_candidates)
        ? boundary.end_candidates.map((v) => Number(v)).filter(Number.isFinite)
        : [end];
    return {
        start,
        end,
        startCandidates: startCandidates.length > 0 ? startCandidates : [start],
        endCandidates: endCandidates.length > 0 ? endCandidates : [end]
    };
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

function runProcess(
    cmd: string,
    args: string[],
    options: { streamStdout?: boolean; streamStderr?: boolean; logPrefix?: string } = {}
): Promise<{ stdout: string; stderr: string }> {
    return new Promise((resolve, reject) => {
        const proc = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'] });
        let stdout = '';
        let stderr = '';
        const prefix = options.logPrefix ? `${options.logPrefix} ` : '';
        proc.stdout.on('data', (d) => {
            const chunk = d.toString();
            stdout += chunk;
            if (options.streamStdout) process.stdout.write(`${prefix}${chunk}`);
        });
        proc.stderr.on('data', (d) => {
            const chunk = d.toString();
            stderr += chunk;
            if (options.streamStderr) process.stderr.write(`${prefix}${chunk}`);
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

function isSpeechLikeLabel(label: string): boolean {
    const l = String(label ?? '').toLowerCase();
    return l === 'male' || l === 'female' || l === 'speech';
}

function getWindowDurations(events: AudioEvent[], start: number, end: number): Map<string, number> {
    const durations = new Map<string, number>();
    for (const e of events) {
        const ol = overlap(e.start, e.end, start, end);
        if (ol <= 0) continue;
        const key = String(e.label ?? '').toLowerCase();
        durations.set(key, (durations.get(key) ?? 0) + ol);
    }
    return durations;
}

function dominantLabel(durations: Map<string, number>, labels: string[]): string | null {
    let best: string | null = null;
    let bestDur = 0;
    for (const label of labels) {
        const d = durations.get(label) ?? 0;
        if (d > bestDur) {
            bestDur = d;
            best = label;
        }
    }
    return best;
}

function boundaryAudioSignal(events: AudioEvent[], t: number, windowSec = 12) {
    const beforeStart = Math.max(0, t - windowSec);
    const beforeEnd = t;
    const afterStart = t;
    const afterEnd = t + windowSec;
    const before = getWindowDurations(events, beforeStart, beforeEnd);
    const after = getWindowDurations(events, afterStart, afterEnd);
    const beforeTotal = Math.max(0.001, beforeEnd - beforeStart);
    const afterTotal = Math.max(0.001, afterEnd - afterStart);

    const beforeMusicRatio = (before.get('music') ?? 0) / beforeTotal;
    const afterMusicRatio = (after.get('music') ?? 0) / afterTotal;
    const beforeNoenergyRatio = (before.get('noenergy') ?? 0) / beforeTotal;
    const afterNoenergyRatio = (after.get('noenergy') ?? 0) / afterTotal;
    const beforeSpeechRatio = [...before.entries()]
        .filter(([k]) => isSpeechLikeLabel(k))
        .reduce((s, [, d]) => s + d, 0) / beforeTotal;
    const afterSpeechRatio = [...after.entries()]
        .filter(([k]) => isSpeechLikeLabel(k))
        .reduce((s, [, d]) => s + d, 0) / afterTotal;
    const beforeGender = dominantLabel(before, ['male', 'female']);
    const afterGender = dominantLabel(after, ['male', 'female']);
    const genderChange = Boolean(beforeGender && afterGender && beforeGender !== afterGender);
    const musicToSpeech = beforeMusicRatio >= 0.35 && afterSpeechRatio >= 0.45;
    const speechToMusic = beforeSpeechRatio >= 0.45 && afterMusicRatio >= 0.35;
    const pauseToSpeech = beforeNoenergyRatio >= 0.22 && afterSpeechRatio >= 0.45;
    const speechToPause = beforeSpeechRatio >= 0.45 && afterNoenergyRatio >= 0.22;

    return {
        beforeMusicRatio,
        afterMusicRatio,
        beforeNoenergyRatio,
        afterNoenergyRatio,
        beforeSpeechRatio,
        afterSpeechRatio,
        beforeGender,
        afterGender,
        genderChange,
        musicToSpeech,
        speechToMusic,
        pauseToSpeech,
        speechToPause
    };
}

function getOcrWindowDurations(events: OcrEvent[], start: number, end: number): Map<string, number> {
    const durations = new Map<string, number>();
    for (const e of events) {
        const ol = overlap(e.start, e.end, start, end);
        if (ol <= 0) continue;
        const key = String(e.type ?? 'on_screen_text').toLowerCase();
        durations.set(key, (durations.get(key) ?? 0) + ol);
    }
    return durations;
}

function boundaryOcrSignal(ocr: OcrEventPassResult, t: number, windowSec = 18) {
    const segments = Array.isArray(ocr?.segments) ? ocr.segments : [];
    if (segments.length === 0) return null;
    const beforeStart = Math.max(0, t - windowSec);
    const beforeEnd = t;
    const afterStart = t;
    const afterEnd = t + windowSec;
    const before = getOcrWindowDurations(segments, beforeStart, beforeEnd);
    const after = getOcrWindowDurations(segments, afterStart, afterEnd);
    const beforeTotal = Math.max(0.001, beforeEnd - beforeStart);
    const afterTotal = Math.max(0.001, afterEnd - afterStart);

    const beforeSermonCueRatio =
        ((before.get('sermon_title') ?? 0) + (before.get('bible_verse') ?? 0)) / beforeTotal;
    const afterSermonCueRatio =
        ((after.get('sermon_title') ?? 0) + (after.get('bible_verse') ?? 0)) / afterTotal;
    const beforeLyricRatio = (before.get('song_lyric') ?? 0) / beforeTotal;
    const afterLyricRatio = (after.get('song_lyric') ?? 0) / afterTotal;
    const beforeSpeakerNameRatio = (before.get('speaker_name') ?? 0) / beforeTotal;
    const afterSpeakerNameRatio = (after.get('speaker_name') ?? 0) / afterTotal;
    const lyricToSermon = beforeLyricRatio >= 0.12 && afterSermonCueRatio >= 0.08;
    const sermonToLyric = beforeSermonCueRatio >= 0.08 && afterLyricRatio >= 0.12;
    const lyricDominantNear = (beforeLyricRatio + afterLyricRatio) / 2 >= 0.2;
    const hasSpeakerNameNear = beforeSpeakerNameRatio >= 0.08 || afterSpeakerNameRatio >= 0.08;
    const hasSermonCueNear = beforeSermonCueRatio >= 0.08 || afterSermonCueRatio >= 0.08;
    const nearSceneCut = Array.isArray(ocr.scene_cuts_sec)
        ? ocr.scene_cuts_sec.some((c) => Math.abs(Number(c) - t) <= 3.0)
        : false;

    return {
        beforeSermonCueRatio,
        afterSermonCueRatio,
        beforeLyricRatio,
        afterLyricRatio,
        beforeSpeakerNameRatio,
        afterSpeakerNameRatio,
        lyricToSermon,
        sermonToLyric,
        lyricDominantNear,
        hasSpeakerNameNear,
        hasSermonCueNear,
        nearSceneCut
    };
}

function getSlideWindowDurations(events: SlideOcrEvent[], start: number, end: number): Map<string, number> {
    const durations = new Map<string, number>();
    for (const e of events) {
        const ol = overlap(e.start, e.end, start, end);
        if (ol <= 0) continue;
        const key = String(e.type ?? 'on_screen_text').toLowerCase();
        durations.set(key, (durations.get(key) ?? 0) + ol);
    }
    return durations;
}

function boundarySlideSignal(slide: SlideOcrPassResult, t: number, windowSec = 18) {
    const events = Array.isArray(slide?.events) ? slide.events : [];
    if (events.length === 0) return null;
    const beforeStart = Math.max(0, t - windowSec);
    const beforeEnd = t;
    const afterStart = t;
    const afterEnd = t + windowSec;
    const before = getSlideWindowDurations(events, beforeStart, beforeEnd);
    const after = getSlideWindowDurations(events, afterStart, afterEnd);
    const beforeTotal = Math.max(0.001, beforeEnd - beforeStart);
    const afterTotal = Math.max(0.001, afterEnd - afterStart);

    const beforeSermonCueRatio =
        ((before.get('sermon_title') ?? 0) + (before.get('bible_verse') ?? 0)) / beforeTotal;
    const afterSermonCueRatio =
        ((after.get('sermon_title') ?? 0) + (after.get('bible_verse') ?? 0)) / afterTotal;
    const beforeLyricRatio = (before.get('song_lyric') ?? 0) / beforeTotal;
    const afterLyricRatio = (after.get('song_lyric') ?? 0) / afterTotal;
    const lyricToSermon = beforeLyricRatio >= 0.12 && afterSermonCueRatio >= 0.08;
    const sermonToLyric = beforeSermonCueRatio >= 0.08 && afterLyricRatio >= 0.12;
    const lyricDominantNear = (beforeLyricRatio + afterLyricRatio) / 2 >= 0.2;
    const hasSermonCueNear = beforeSermonCueRatio >= 0.08 || afterSermonCueRatio >= 0.08;
    const hasSermonTitleNear = events.some((e) => e.type === 'sermon_title' && overlap(e.start, e.end, t - 15, t + 15) > 0);
    const hasBibleVerseNear = events.some((e) => e.type === 'bible_verse' && overlap(e.start, e.end, t - 15, t + 15) > 0);

    return {
        beforeSermonCueRatio,
        afterSermonCueRatio,
        beforeLyricRatio,
        afterLyricRatio,
        lyricToSermon,
        sermonToLyric,
        lyricDominantNear,
        hasSermonCueNear,
        hasSermonTitleNear,
        hasBibleVerseNear,
    };
}

function isNonSpeech(text?: string): boolean {
    const t = (text ?? '').toLowerCase().trim();
    if (!t) return true;
    if (t === 'music' || t === 'música' || t === 'piano' || t === 'silence' || t === 'silencio') return true;
    if (/^\[.*\]$/.test(t) || /^\(.*\)$/.test(t)) return true;
    if (/^(music|música|piano|instrumental|aplausos|applause|silence|silencio)\b/.test(t)) return true;
    return false;
}

function hasSermonHandoffCue(segments: Segment[], aroundSec: number, windowSec = 90): boolean {
    const start = Math.max(0, aroundSec - windowSec);
    const end = aroundSec + windowSec;
    const text = segmentsInRange(segments, start, end)
        .map((s) => cleanText(s.text).toLowerCase())
        .join(' ');
    if (!text) return false;
    const patterns = [
        /\b(vamos a cantar|cantemos|alabanza)\b/,
        /\b(invito al (equipo|grupo) de alabanza)\b/,
        /\b(pasen (aquí|adelante)|ponerse de pie)\b/,
        /\b(terminamos cantando|oración final)\b/,
        /\b(les queremos agradecer|gracias pastor)\b/
    ];
    return patterns.some((rx) => rx.test(text));
}

function collectSpeechText(segments: Segment[], start: number, end: number): string {
    return segmentsInRange(segments, start, end)
        .filter((s) => !isNonSpeech(s.text))
        .map((s) => cleanText(s.text).toLowerCase())
        .join(' ');
}

function scoreRegexMatches(text: string, patterns: RegExp[]): number {
    if (!text) return 0;
    let score = 0;
    for (const rx of patterns) {
        const m = text.match(rx);
        if (!m) continue;
        score += Array.isArray(m) ? Math.max(1, m.length) : 1;
    }
    return score;
}

function boundaryTranscriptSignal(segments: Segment[], t: number, kind: 'start' | 'end'): TranscriptBoundarySignal {
    const beforeSec = kind === 'start' ? 180 : 160;
    const afterSec = kind === 'start' ? 210 : 170;
    const beforeText = collectSpeechText(segments, Math.max(0, t - beforeSec), Math.max(0, t - 8));
    const afterText = collectSpeechText(segments, Math.max(0, t - 8), t + afterSec);
    const nearText = `${beforeText} ${afterText}`.trim();

    const sermonPatterns = [
        /\bfeliz s[áa]bado\b/g,
        /\b(predicaci[oó]n|serm[oó]n|mensaje)\b/g,
        /\b(tema|t[ií]tulo)\b/g,
        /\b(abr(a|an|amos)\s+(su|sus)?\s*biblias?|abra su biblia)\b/g,
        /\b(palabra de dios|evangelio|texto b[ií]blico|pasaje)\b/g,
        /\b(vamos a estudiar|estudiemos)\b/g
    ];
    const prayerPatterns = [
        /\b(oremos|oraci[oó]n|orar|inclinar el rostro|inclinen el rostro|pong[aá]monos de rodillas)\b/g,
        /\b(te lo pedimos|en el nombre de jes[uú]s|am[eé]n[,.\s]|am[eé]n$)\b/g,
        /\b(padre nuestro|se[ñn]or te damos gracias)\b/g
    ];
    const worshipPatterns = [
        /\b(cantemos|vamos a cantar|alabanza|adoraci[oó]n|himno|canci[oó]n|m[uú]sica)\b/g,
        /\b(grupo de alabanza|equipo de alabanza|coro)\b/g
    ];
    const announcementPatterns = [
        /\b(anuncios|anuncio|recordatorio|actividad|programaci[oó]n)\b/g,
        /\b(diezmos?|ofrendas?|ofertorio|potluck|convivio)\b/g,
        /\b(abrazo al que est[aá] al lado|saluda al que est[aá] a tu lado)\b/g
    ];
    const closingPatterns = [
        /\b(am[eé]n[,.\s]|am[eé]n$)\b/g,
        /\b(oraci[oó]n final|oramos|oremos)\b/g,
        /\b(bendici[oó]n|despedida|que dios les bendiga)\b/g
    ];
    const handoffPatterns = [
        /\b(vamos a cantar|cantemos|alabanza)\b/g,
        /\b(invito al (equipo|grupo) de alabanza)\b/g,
        /\b(p[óo]nganse de pie|ponerse de pie|pasen aqu[ií]|pasar aqu[ií])\b/g,
        /\b(les queremos agradecer|gracias pastor)\b/g
    ];

    const sermonCueScore =
        scoreRegexMatches(afterText, sermonPatterns) * 1.0 + scoreRegexMatches(beforeText, sermonPatterns) * 0.4;
    const prayerCueScore =
        scoreRegexMatches(afterText, prayerPatterns) * 0.95 + scoreRegexMatches(beforeText, prayerPatterns) * 0.6;
    const worshipCueScore =
        scoreRegexMatches(afterText, worshipPatterns) * 1.0 + scoreRegexMatches(beforeText, worshipPatterns) * 0.6;
    const announcementCueScore =
        scoreRegexMatches(afterText, announcementPatterns) * 1.0 + scoreRegexMatches(beforeText, announcementPatterns) * 0.5;
    const closingCueScore =
        scoreRegexMatches(afterText, closingPatterns) * 1.0 + scoreRegexMatches(beforeText, closingPatterns) * 0.4;
    const handoffCueScore =
        scoreRegexMatches(afterText, handoffPatterns) * 1.0 + scoreRegexMatches(beforeText, handoffPatterns) * 0.5;

    const hasFelizSabado = /\bfeliz s[áa]bado\b/.test(afterText);
    const prayerDominant = prayerCueScore >= 2.2 && sermonCueScore < 1.2;
    const worshipDominant = worshipCueScore >= 1.9 && sermonCueScore < 1.2;
    const announcementDominant = announcementCueScore >= 1.7 && sermonCueScore < 1.2;
    const sermonLikely =
        (sermonCueScore + (hasFelizSabado ? 1.2 : 0)) >= 1.8 &&
        !(prayerDominant || worshipDominant || announcementDominant);
    const closingLikely = closingCueScore >= 1.3 || (closingCueScore >= 0.8 && handoffCueScore >= 0.8);
    const handoffLikely = handoffCueScore >= 1.0 || hasSermonHandoffCue(segments, t, 95);

    return {
        sermonCueScore,
        prayerCueScore,
        worshipCueScore,
        announcementCueScore,
        closingCueScore,
        handoffCueScore,
        hasFelizSabado,
        sermonLikely,
        prayerDominant,
        worshipDominant,
        announcementDominant,
        closingLikely,
        handoffLikely
    };
}

function deriveTranscriptStartCandidates(
    segments: Segment[],
    guessStart: number,
    guessEnd: number,
    searchBackSec = 420,
    searchForwardSec = 1800,
    limit = 8
): number[] {
    const startMin = Math.max(0, guessStart - searchBackSec);
    const startMax = Math.max(startMin + 30, Math.min(guessEnd - 30, guessStart + searchForwardSec));
    const cuePatterns: Array<{ rx: RegExp; score: number }> = [
        { rx: /\bfeliz s[áa]bado\b/i, score: 2.4 },
        { rx: /\b(abr(a|an|amos)\s+(su|sus)?\s*biblias?|abra su biblia)\b/i, score: 2.0 },
        { rx: /\b(predicaci[oó]n|serm[oó]n|mensaje)\b/i, score: 1.8 },
        { rx: /\b(tema|t[ií]tulo)\b/i, score: 1.2 },
        { rx: /\b(vamos a estudiar|estudiemos|palabra de dios)\b/i, score: 1.2 }
    ];
    const rejectPatterns = [
        /\b(cantemos|vamos a cantar|alabanza|adoraci[oó]n|himno)\b/i,
        /\b(ofrendas?|diezmos?|anuncios?)\b/i
    ];
    const scored = segments
        .filter((s) => s.start >= startMin && s.start <= startMax && !isNonSpeech(s.text))
        .map((s) => {
            const text = cleanText(s.text).toLowerCase();
            let score = 0;
            for (const cue of cuePatterns) {
                if (cue.rx.test(text)) score += cue.score;
            }
            for (const rx of rejectPatterns) {
                if (rx.test(text)) score -= 1.4;
            }
            score += Math.max(0, 1 - Math.abs(s.start - guessStart) / 900);
            return { start: s.start, score };
        })
        .filter((v) => v.score >= 1.2)
        .sort((a, b) => b.score - a.score || a.start - b.start);
    const out: number[] = [];
    for (const row of scored) {
        if (out.some((v) => Math.abs(v - row.start) < 15)) continue;
        out.push(row.start);
        if (out.length >= limit) break;
    }
    return out;
}

function deriveTranscriptEndCandidates(
    segments: Segment[],
    guessStart: number,
    guessEnd: number,
    searchBackSec = 1600,
    searchForwardSec = 500,
    limit = 8
): number[] {
    const endMin = Math.max(guessStart + 60, guessEnd - searchBackSec);
    const endMax = guessEnd + searchForwardSec;
    const cuePatterns: Array<{ rx: RegExp; score: number }> = [
        { rx: /\b(am[eé]n[,.\s]|am[eé]n$)\b/i, score: 1.3 },
        { rx: /\b(oraci[oó]n final|oremos|oramos)\b/i, score: 1.6 },
        { rx: /\b(vamos a cantar|cantemos|equipo de alabanza)\b/i, score: 1.4 },
        { rx: /\b(les queremos agradecer|gracias pastor|que dios les bendiga)\b/i, score: 1.4 },
        { rx: /\b(ponerse de pie|p[óo]nganse de pie)\b/i, score: 1.0 }
    ];
    const scored = segments
        .filter((s) => s.end >= endMin && s.end <= endMax && !isNonSpeech(s.text))
        .map((s) => {
            const text = cleanText(s.text).toLowerCase();
            let score = 0;
            for (const cue of cuePatterns) {
                if (cue.rx.test(text)) score += cue.score;
            }
            score += Math.max(0, 1 - Math.abs(s.end - guessEnd) / 900);
            return { end: s.end, score };
        })
        .filter((v) => v.score >= 1.0)
        .sort((a, b) => b.score - a.score || a.end - b.end);
    const out: number[] = [];
    for (const row of scored) {
        if (out.some((v) => Math.abs(v - row.end) < 15)) continue;
        out.push(row.end);
        if (out.length >= limit) break;
    }
    return out;
}

async function extractChunk(audioPath: string, outPath: string, startSec: number, endSec: number): Promise<void> {
    const duration = Math.max(0.01, endSec - startSec);
    await runProcess('ffmpeg', ['-y', '-i', audioPath, '-ss', String(startSec), '-t', String(duration), '-acodec', 'pcm_s16le', outPath]);
}

function durationToSec(v: any): number {
    if (!v) return 0;
    const seconds = Number(v.seconds ?? 0);
    const nanos = Number(v.nanos ?? 0);
    const secPart = Number.isFinite(seconds) ? seconds : 0;
    const nanoPart = Number.isFinite(nanos) ? nanos / 1e9 : 0;
    return secPart + nanoPart;
}

function normalizeSpeakerLabel(v: unknown): string {
    if (typeof v === 'string' && v.trim()) {
        const clean = v.trim();
        if (/^SPEAKER_/i.test(clean)) return clean.toUpperCase();
        if (/^\d+$/.test(clean)) return `SPEAKER_${clean}`;
        return `SPEAKER_${clean.replace(/\s+/g, '_').toUpperCase()}`;
    }
    if (typeof v === 'number' && Number.isFinite(v)) return `SPEAKER_${String(v)}`;
    return 'SPEAKER_UNKNOWN';
}

function wordsToSpeakerTurns(words: Array<{ start: number; end: number; speaker: string }>, maxGapSec = 1.2): SpeakerTurn[] {
    if (words.length === 0) return [];
    const sorted = [...words].sort((a, b) => a.start - b.start);
    const out: SpeakerTurn[] = [];
    let cur: SpeakerTurn = { start: sorted[0].start, end: sorted[0].end, speaker: sorted[0].speaker };
    for (let i = 1; i < sorted.length; i++) {
        const w = sorted[i];
        const gap = Math.max(0, w.start - cur.end);
        if (w.speaker === cur.speaker && gap <= maxGapSec) {
            cur.end = Math.max(cur.end, w.end);
            continue;
        }
        out.push(cur);
        cur = { start: w.start, end: w.end, speaker: w.speaker };
    }
    out.push(cur);
    return out;
}

async function runDiarizeGoogleChirp(chunkPath: string): Promise<SpeakerTurn[]> {
    const projectId = process.env.GOOGLE_CLOUD_PROJECT || process.env.GCLOUD_PROJECT;
    const bucket = process.env.GOOGLE_STT_BUCKET;
    if (!projectId || !bucket) {
        throw new Error('GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT and GOOGLE_STT_BUCKET are required for google_chirp3 diarization backend.');
    }
    const location = process.env.GOOGLE_STT_LOCATION || 'us-central1';
    const language = process.env.GOOGLE_STT_LANGUAGE || 'es-US';
    const model = process.env.GOOGLE_STT_MODEL || 'chirp_3';
    const minSpeakerCount = Number(process.env.GOOGLE_STT_MIN_SPEAKERS ?? process.env.TARGET_DIAR_MIN_SPEAKERS ?? 2);
    const maxSpeakerCount = Number(process.env.GOOGLE_STT_MAX_SPEAKERS ?? process.env.TARGET_DIAR_MAX_SPEAKERS ?? 8);

    const storage = new Storage({ projectId });
    const runId = new Date().toISOString().replace(/[-:.TZ]/g, '').slice(0, 14);
    const object = `cdv-google-targeted/${runId}/${path.basename(chunkPath).replace(/\s+/g, '_')}`;
    const gcsUri = `gs://${bucket}/${object}`;
    console.log(`[targeted-diarization][google_chirp3] upload chunk -> ${gcsUri}`);
    await storage.bucket(bucket).upload(chunkPath, {
        destination: object,
        contentType: 'audio/wav'
    });

    const speechClient =
        location === 'global'
            ? new speechV2.SpeechClient({ projectId })
            : new speechV2.SpeechClient({ projectId, apiEndpoint: `${location}-speech.googleapis.com` });
    const recognizer = `projects/${projectId}/locations/${location}/recognizers/_`;
    const request: any = {
        recognizer,
        config: {
            autoDecodingConfig: {},
            model,
            languageCodes: [language],
            features: {
                enableWordTimeOffsets: true,
                enableAutomaticPunctuation: true,
                diarizationConfig: { minSpeakerCount, maxSpeakerCount }
            }
        },
        files: [{ uri: gcsUri }],
        recognitionOutputConfig: { inlineResponseConfig: {} },
        processingStrategy: 'DYNAMIC_BATCHING'
    };

    const t0 = Date.now();
    console.log(`[targeted-diarization][google_chirp3] submitting batchRecognize model=${model} language=${language} speakers=${minSpeakerCount}-${maxSpeakerCount}`);
    const [operation] = await speechClient.batchRecognize(request);
    console.log('[targeted-diarization][google_chirp3] operation submitted, waiting for completion...');
    const [response] = await operation.promise();
    console.log(`[targeted-diarization][google_chirp3] operation completed in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

    const resultEntry = (response as any)?.results?.[gcsUri];
    const transcript = resultEntry?.transcript ?? resultEntry?.inlineResult?.transcript ?? null;
    const speechResults = transcript?.results;
    const words: Array<{ start: number; end: number; speaker: string }> = [];
    if (Array.isArray(speechResults)) {
        for (const res of speechResults) {
            const alt = res?.alternatives?.[0];
            const altWords = alt?.words;
            if (!Array.isArray(altWords)) continue;
            for (const w of altWords) {
                const start = durationToSec(w?.startOffset);
                const end = durationToSec(w?.endOffset);
                if (!(end > start)) continue;
                words.push({
                    start,
                    end,
                    speaker: normalizeSpeakerLabel(w?.speakerLabel)
                });
            }
        }
    }

    if (String(process.env.GOOGLE_STT_KEEP_TEMP_FILES ?? 'false').toLowerCase() !== 'true') {
        try {
            await storage.bucket(bucket).file(object).delete({ ignoreNotFound: true } as any);
            console.log(`[targeted-diarization][google_chirp3] deleted temp object: ${gcsUri}`);
        } catch {
            // non-fatal
        }
    }

    if (words.length === 0) {
        throw new Error('Google Chirp 3 returned no diarized words for targeted chunk.');
    }
    console.log(`[targeted-diarization][google_chirp3] parsed diarized words=${words.length}`);
    return wordsToSpeakerTurns(words);
}

async function runDiarizeLocal(chunkPath: string): Promise<SpeakerTurn[]> {
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
    console.log(`[targeted-diarization][local] running pyannote on ${path.basename(chunkPath)}`);
    const t0 = Date.now();
    const { stdout } = await runProcess(pythonBin, args, { streamStderr: true, logPrefix: '[targeted-diarization][local]' });
    console.log(`[targeted-diarization][local] completed in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
    const parsed = JSON.parse(stdout) as SpeakerTurn[];
    if (!Array.isArray(parsed)) throw new Error('Invalid diarization output (expected array).');
    console.log(`[targeted-diarization][local] turns=${parsed.length}`);
    return parsed;
}

async function runDiarize(chunkPath: string): Promise<SpeakerTurn[]> {
    const backend = String(process.env.TARGET_DIAR_BACKEND ?? 'google_chirp3').toLowerCase();
    console.log(`[targeted-diarization] backend=${backend} chunk=${path.basename(chunkPath)}`);
    if (backend === 'google_chirp3') {
        try {
            console.log('Running targeted diarization backend: google_chirp3');
            return await runDiarizeGoogleChirp(chunkPath);
        } catch (error) {
            const allowFallback = String(process.env.TARGET_DIAR_GOOGLE_FALLBACK_LOCAL ?? 'true').toLowerCase() === 'true';
            if (!allowFallback) throw error;
            console.warn('google_chirp3 diarization failed, falling back to local pyannote:', error);
            return runDiarizeLocal(chunkPath);
        }
    }
    return runDiarizeLocal(chunkPath);
}

async function runFacePass(
    videoPath: string,
    outPath: string,
    profile: BoundaryPipelineProfile = 'default'
): Promise<FaceBoundaryResult> {
    const pythonScriptCandidates = [
        path.resolve(__dirname, 'python/face_sermon_bounds.py'),
        path.resolve(__dirname, '../../src/pipeline/python/face_sermon_bounds.py')
    ];
    const pythonScript = pythonScriptCandidates.find((p) => fs.existsSync(p)) ?? pythonScriptCandidates[0];
    const venv311 = path.resolve(__dirname, '../../venv311/bin/python3');
    const workerVenv = path.resolve(__dirname, '../../venv/bin/python3');
    const pythonBin = process.env.FACE_PYTHON_BIN
        ? path.resolve(process.env.FACE_PYTHON_BIN)
        : (fs.existsSync(venv311) ? venv311 : workerVenv);

    const stepSec = profileString(profile, 'FACE_SCAN_STEP_SEC', { default: '4.0', light: '8.0' });
    const fineStepSec = profileString(profile, 'FACE_SCAN_FINE_STEP_SEC', { default: '0.75', light: '1.0' });
    const detSize = profileString(profile, 'FACE_DET_SIZE', { default: '320', light: '256' });
    const maxFacesPerFrame = profileString(profile, 'FACE_MAX_FACES_PER_FRAME', { default: '1', light: '1' });
    const denseWindowSec = profileString(profile, 'FACE_DENSE_WINDOW_SEC', { default: '30', light: '20' });
    console.log(
        `[face-pass] profile=${profile} step=${stepSec}s fine_step=${fineStepSec}s det_size=${detSize} dense_window=${denseWindowSec}s`
    );
    const args = [
        pythonScript,
        videoPath,
        '--out',
        outPath,
        '--step-sec',
        stepSec,
        '--fine-step-sec',
        fineStepSec,
        '--det-size',
        detSize,
        '--max-faces-per-frame',
        maxFacesPerFrame,
        '--dense-window-sec',
        denseWindowSec
    ];
    await runProcess(pythonBin, args, { streamStderr: true });

    const raw = JSON.parse(fs.readFileSync(outPath, 'utf8')) as FaceBoundaryResult;
    if (!Number.isFinite(Number(raw?.start)) || !Number.isFinite(Number(raw?.end)) || Number(raw.end) <= Number(raw.start)) {
        throw new Error('Invalid face-pass boundary output');
    }
    return raw;
}

async function runAudioEventPass(audioPath: string, outPath: string): Promise<AudioEventPassResult> {
    const pythonScriptCandidates = [
        path.resolve(__dirname, 'python/audio_events.py'),
        path.resolve(__dirname, '../../src/pipeline/python/audio_events.py')
    ];
    const pythonScript = pythonScriptCandidates.find((p) => fs.existsSync(p)) ?? pythonScriptCandidates[0];
    const venv311 = path.resolve(__dirname, '../../venv311/bin/python3');
    const workerVenv = path.resolve(__dirname, '../../venv/bin/python3');
    const pythonBin = process.env.FACE_PYTHON_BIN
        ? path.resolve(process.env.FACE_PYTHON_BIN)
        : (fs.existsSync(venv311) ? venv311 : workerVenv);
    const stepSec = process.env.AUDIO_EVENT_STEP_SEC ?? '2.0';
    const args = [pythonScript, audioPath, '--out', outPath, '--step-sec', stepSec];
    await runProcess(pythonBin, args, { streamStderr: true, logPrefix: '[audio-events]' });
    const raw = JSON.parse(fs.readFileSync(outPath, 'utf8')) as AudioEventPassResult;
    if (!Array.isArray(raw?.segments)) throw new Error('Invalid audio-events output');
    return raw;
}

function pythonHasOcrModule(pythonBin: string): boolean {
    try {
        const probe = spawnSync(
            pythonBin,
            ['-c', 'import importlib.util as u; print(int(bool(u.find_spec("easyocr") or u.find_spec("pytesseract"))))'],
            { encoding: 'utf8' }
        );
        if (probe.status !== 0) return false;
        return String(probe.stdout ?? '').trim() === '1';
    } catch {
        return false;
    }
}

function resolveOcrPythonBin(): { pythonBin: string; hasOcrModule: boolean } {
    const venv311 = path.resolve(__dirname, '../../venv311/bin/python3');
    const workerVenv = path.resolve(__dirname, '../../venv/bin/python3');
    const explicitBins = [process.env.OCR_PYTHON_BIN, process.env.FACE_PYTHON_BIN]
        .map((v) => (v ? path.resolve(v) : ''))
        .filter(Boolean);
    const candidates = [...explicitBins, venv311, workerVenv, 'python3', 'python'];
    let fallback = explicitBins[0] || (fs.existsSync(venv311) ? venv311 : workerVenv);
    for (const c of candidates) {
        const exists = c === 'python3' || c === 'python' ? true : fs.existsSync(c);
        if (!exists) continue;
        if (!fallback) fallback = c;
        if (pythonHasOcrModule(c)) return { pythonBin: c, hasOcrModule: true };
    }
    return { pythonBin: fallback || 'python3', hasOcrModule: false };
}

function resolvePreferredOcrVideoPath(videoPath: string, outPath: string): string {
    const preferHq = String(process.env.OCR_PREFER_HQ_SOURCE ?? 'true').toLowerCase() === 'true';
    if (!preferHq) return videoPath;
    const workDir = path.dirname(outPath);
    const candidates = [
        path.join(workDir, 'source.mp4'),
        path.join(workDir, 'source.original.mp4'),
        path.join(workDir, 'source.original.mkv'),
        path.join(workDir, 'source.original.webm'),
        path.join(workDir, 'source.original.mov'),
    ];
    const providedLooksLight = /source\.light\./i.test(path.basename(videoPath));
    if (providedLooksLight) {
        const better = candidates.find((p) => fs.existsSync(p));
        if (better) return better;
    }
    if (fs.existsSync(videoPath)) return videoPath;
    const fallback = candidates.find((p) => fs.existsSync(p));
    return fallback ?? videoPath;
}

async function runOcrEventPass(
    videoPath: string,
    outPath: string,
    durationSec: number,
    pipeline: OcrPipeline
): Promise<OcrEventPassResult> {
    const pythonScriptCandidates =
        pipeline === 'v2'
            ? [path.resolve(__dirname, 'python/ocr_v2.py'), path.resolve(__dirname, '../../src/pipeline/python/ocr_v2.py')]
            : pipeline === 'v3'
              ? [path.resolve(__dirname, 'python/ocr_v3.py'), path.resolve(__dirname, '../../src/pipeline/python/ocr_v3.py')]
              : [path.resolve(__dirname, 'python/ocr_events.py'), path.resolve(__dirname, '../../src/pipeline/python/ocr_events.py')];
    const pythonScript = pythonScriptCandidates.find((p) => fs.existsSync(p)) ?? pythonScriptCandidates[0];
    const { pythonBin, hasOcrModule } = resolveOcrPythonBin();
    if (!hasOcrModule) {
        console.warn(
            `[ocr-events] No OCR Python modules found (easyocr/pytesseract). Running probe anyway with ${pythonBin}; source may become ocr-none.`
        );
    }
    const sampleSec =
        pipeline === 'v2'
            ? process.env.BOUNDARY_OCR_V2_SAMPLE_SEC ??
              process.env.OCR_V2_SAMPLE_SEC ??
              process.env.BOUNDARY_OCR_SAMPLE_SEC ??
              process.env.OCR_SAMPLE_SEC ??
              '1.0'
            : pipeline === 'v3'
              ? process.env.BOUNDARY_OCR_V3_SAMPLE_SEC ??
                process.env.OCR_V3_SAMPLE_SEC ??
                process.env.BOUNDARY_OCR_SAMPLE_SEC ??
                process.env.OCR_SAMPLE_SEC ??
                '1.0'
              : process.env.BOUNDARY_OCR_SAMPLE_SEC ?? process.env.OCR_SAMPLE_SEC ?? '2.5';
    const configuredMaxSec = Number(
        pipeline === 'v2'
            ? process.env.BOUNDARY_OCR_V2_MAX_DURATION_SEC ??
              process.env.OCR_V2_MAX_DURATION_SEC ??
              process.env.BOUNDARY_OCR_MAX_DURATION_SEC ??
              process.env.OCR_MAX_DURATION_SEC ??
              0
            : pipeline === 'v3'
              ? process.env.BOUNDARY_OCR_V3_MAX_DURATION_SEC ??
                process.env.OCR_V3_MAX_DURATION_SEC ??
                process.env.BOUNDARY_OCR_MAX_DURATION_SEC ??
                process.env.OCR_MAX_DURATION_SEC ??
                0
              : process.env.BOUNDARY_OCR_MAX_DURATION_SEC ?? process.env.OCR_MAX_DURATION_SEC ?? 0
    );
    const maxSec = configuredMaxSec > 0 ? Math.min(configuredMaxSec, durationSec) : durationSec;
    const selectedVideoPath = resolvePreferredOcrVideoPath(videoPath, outPath);
    const args = [
        pythonScript,
        selectedVideoPath,
        '--out',
        outPath,
        '--sample-sec',
        String(sampleSec),
        '--max-sec',
        String(maxSec)
    ];
    console.log(`[ocr-events] pipeline=${pipeline} video=${path.basename(selectedVideoPath)} sample=${sampleSec}s max_sec=${maxSec.toFixed(1)}`);
    const ocrLogPrefix = pipeline === 'v2' ? '[ocr-v2]' : pipeline === 'v3' ? '[ocr-v3]' : '[ocr-events]';
    await runProcess(pythonBin, args, { streamStderr: true, logPrefix: ocrLogPrefix });
    const raw = JSON.parse(fs.readFileSync(outPath, 'utf8')) as OcrEventPassResult;
    if (!Array.isArray(raw?.segments)) throw new Error('Invalid ocr-events output');
    return raw;
}

async function runSlideOcrPass(
    videoPath: string,
    outPath: string,
    durationSec: number
): Promise<SlideOcrPassResult> {
    const pythonScriptCandidates = [
        path.resolve(__dirname, 'python/slide_ocr_v2.py'),
        path.resolve(__dirname, '../../src/pipeline/python/slide_ocr_v2.py')
    ];
    const pythonScript = pythonScriptCandidates.find((p) => fs.existsSync(p)) ?? pythonScriptCandidates[0];
    const { pythonBin } = resolveOcrPythonBin();

    const workDir = path.dirname(outPath);
    const lowCandidates = [
        path.join(workDir, 'source.light.mp4'),
        videoPath,
        path.join(workDir, 'source.mp4'),
    ];
    const lowVideoPath = lowCandidates.find((p) => fs.existsSync(p)) ?? videoPath;
    const hqVideoPath = resolvePreferredOcrVideoPath(videoPath, outPath);
    const sampleSec = process.env.BOUNDARY_SLIDE_OCR_SAMPLE_SEC ?? process.env.SLIDE_OCR_SAMPLE_SEC ?? '1.0';
    const configuredMaxSec = Number(
        process.env.BOUNDARY_SLIDE_OCR_MAX_DURATION_SEC ??
        process.env.SLIDE_OCR_MAX_DURATION_SEC ??
        process.env.BOUNDARY_OCR_MAX_DURATION_SEC ??
        process.env.OCR_MAX_DURATION_SEC ??
        0
    );
    const maxSec = configuredMaxSec > 0 ? Math.min(configuredMaxSec, durationSec) : durationSec;
    const args = [
        pythonScript,
        lowVideoPath,
        '--hq-video',
        hqVideoPath,
        '--out',
        outPath,
        '--sample-sec',
        String(sampleSec),
        '--max-sec',
        String(maxSec),
    ];
    console.log(
        `[slide-ocr] low=${path.basename(lowVideoPath)} hq=${path.basename(hqVideoPath)} sample=${sampleSec}s max_sec=${maxSec.toFixed(1)}`
    );
    await runProcess(pythonBin, args, { streamStderr: true, logPrefix: '[slide-ocr]' });
    const raw = JSON.parse(fs.readFileSync(outPath, 'utf8')) as SlideOcrPassResult;
    if (!Array.isArray(raw?.events)) throw new Error('Invalid slide-ocr output');
    return raw;
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
    console.log('Finding sermon boundaries (face + targeted diarization)...');
    const profile = getBoundaryPipelineProfile();
    console.log(`[boundaries] pipeline_profile=${profile}`);

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

    let guess: { start: number; end: number; startCandidates: number[]; endCandidates: number[] } | null = null;
    if (workDir) {
        const facePath = path.join(workDir, 'sermon.boundaries.face-pass.json');
        try {
            const faceRaw = fs.existsSync(facePath)
                ? (JSON.parse(fs.readFileSync(facePath, 'utf8')) as FaceBoundaryResult)
                : (options.videoPath ? await runFacePass(options.videoPath, facePath, profile) : null);
            if (faceRaw) {
                guess = parseFaceGuess(faceRaw);
            }
        } catch (error) {
            console.error('Face-pass boundary seed failed:', error);
        }
    }

    if (!guess) {
        // Production path intentionally avoids OpenAI boundary inference.
        return fallbackHeuristicBoundaries(normalized);
    }

    if (!options.audioPath || !workDir) {
        return { start: guess.start, end: guess.end };
    }

    try {
        const audioEnd = normalized.length > 0 ? normalized[normalized.length - 1].end : guess.end + 1;
        const preScanSec = profileNumber(profile, 'TARGET_DIAR_PRE_SCAN_SEC', { default: 120, light: 60 });
        const postScanSec = profileNumber(profile, 'TARGET_DIAR_POST_SCAN_SEC', { default: 120, light: 60 });
        const anchorSec = profileNumber(profile, 'TARGET_DIAR_ANCHOR_SEC', { default: 30, light: 20 });
        const maxConfirmGapSec = profileNumber(profile, 'TARGET_DIAR_MAX_CONFIRM_GAP_SEC', { default: 30, light: 20 });
        console.log(
            `[targeted-diarization] profile=${profile} pre_scan=${preScanSec}s post_scan=${postScanSec}s anchor=${anchorSec}s confirm_gap=${maxConfirmGapSec}s`
        );

        const maxCandidateDeltaSec = Number(process.env.TARGET_DIAR_MAX_CANDIDATE_DELTA_SEC ?? 900);
        const transcriptStartCandidates = deriveTranscriptStartCandidates(
            normalized,
            guess.start,
            guess.end,
            Number(process.env.TARGET_DIAR_TRANSCRIPT_START_SEARCH_BACK_SEC ?? 420),
            Number(process.env.TARGET_DIAR_TRANSCRIPT_START_SEARCH_FWD_SEC ?? 1800),
            Number(process.env.TARGET_DIAR_TRANSCRIPT_START_CANDIDATE_LIMIT ?? 8)
        );
        const transcriptEndCandidates = deriveTranscriptEndCandidates(
            normalized,
            guess.start,
            guess.end,
            Number(process.env.TARGET_DIAR_TRANSCRIPT_END_SEARCH_BACK_SEC ?? 1600),
            Number(process.env.TARGET_DIAR_TRANSCRIPT_END_SEARCH_FWD_SEC ?? 500),
            Number(process.env.TARGET_DIAR_TRANSCRIPT_END_CANDIDATE_LIMIT ?? 8)
        );
        const startCandidates = sanitizeCandidates(
            guess.start,
            guess.startCandidates.length > 0
                ? [...guess.startCandidates, ...transcriptStartCandidates]
                : [guess.start, ...transcriptStartCandidates],
            0,
            guess.end - 30,
            maxCandidateDeltaSec
        );
        const endCandidates = sanitizeCandidates(
            guess.end,
            guess.endCandidates.length > 0 ? [...guess.endCandidates, ...transcriptEndCandidates] : [guess.end, ...transcriptEndCandidates],
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
        let audioEvents: AudioEventPassResult | null = null;
        const audioEventsPath = path.join(workDir, 'audio.events.json');
        try {
            audioEvents = fs.existsSync(audioEventsPath)
                ? (JSON.parse(fs.readFileSync(audioEventsPath, 'utf8')) as AudioEventPassResult)
                : await runAudioEventPass(options.audioPath, audioEventsPath);
        } catch (error) {
            console.warn('Audio-event pass unavailable, continuing without it:', error);
        }
        let ocrEvents: OcrEventPassResult | null = null;
        const ocrEnabled =
            String(process.env.BOUNDARY_ENABLE_OCR_SIGNALS ?? process.env.ANALYSIS_ENABLE_OCR_SIGNALS ?? 'true').toLowerCase() ===
            'true';
        const ocrPipeline = getBoundaryOcrPipeline();
        const ocrEventsPath = path.join(workDir, getOcrEventsFilename(ocrPipeline));
        if (ocrEnabled && options.videoPath) {
            try {
                ocrEvents = fs.existsSync(ocrEventsPath)
                    ? (JSON.parse(fs.readFileSync(ocrEventsPath, 'utf8')) as OcrEventPassResult)
                    : await runOcrEventPass(options.videoPath, ocrEventsPath, audioEnd, ocrPipeline);
                if (!Array.isArray(ocrEvents?.segments)) ocrEvents = null;
            } catch (error) {
                console.warn('OCR-event pass unavailable, continuing without it:', error);
            }
        }
        let slideOcrEvents: SlideOcrPassResult | null = null;
        const slideOcrEnabled =
            String(process.env.BOUNDARY_ENABLE_SLIDE_OCR_SIGNALS ?? process.env.SLIDE_OCR_ENABLED ?? 'false').toLowerCase() ===
            'true';
        const slideOcrEventsPath = path.join(workDir, 'slide.events.json');
        if (slideOcrEnabled && options.videoPath) {
            try {
                slideOcrEvents = fs.existsSync(slideOcrEventsPath)
                    ? (JSON.parse(fs.readFileSync(slideOcrEventsPath, 'utf8')) as SlideOcrPassResult)
                    : await runSlideOcrPass(options.videoPath, slideOcrEventsPath, audioEnd);
                if (!Array.isArray(slideOcrEvents?.events)) slideOcrEvents = null;
            } catch (error) {
                console.warn('Slide-OCR pass unavailable, continuing without it:', error);
            }
        }

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

        const ocrWindowSec = Number(process.env.BOUNDARY_OCR_WINDOW_SEC ?? 18);
        const startSeedList = [guess.start, ...startCandidates].filter((v, i, arr) => arr.indexOf(v) === i);
        const scoredStart = startSeedList.map((seed) => {
            let candidateStart = refineStart(preTurnsAbs, startSpeaker, seed);
            const seedTranscriptSignal = boundaryTranscriptSignal(normalized, seed, 'start');
            let transcriptSignal = boundaryTranscriptSignal(normalized, candidateStart, 'start');
            const draggedIntoPreSection =
                candidateStart < seed - 10 &&
                (transcriptSignal.prayerDominant || transcriptSignal.worshipDominant || transcriptSignal.announcementDominant) &&
                (seedTranscriptSignal.sermonLikely || seedTranscriptSignal.hasFelizSabado || seedTranscriptSignal.sermonCueScore >= 1.6);
            if (draggedIntoPreSection) {
                candidateStart = seed;
                transcriptSignal = seedTranscriptSignal;
            }
            const prev = previousTurnBefore(preTurnsAbs, candidateStart);
            const diarChange = Boolean(prev && prev.speaker !== startSpeaker && candidateStart - prev.end <= maxConfirmGapSec);
            const audio = audioEvents ? boundaryAudioSignal(audioEvents.segments, candidateStart) : null;
            const ocr = ocrEvents ? boundaryOcrSignal(ocrEvents, candidateStart, ocrWindowSec) : null;
            const slide = slideOcrEvents ? boundarySlideSignal(slideOcrEvents, candidateStart, ocrWindowSec) : null;
            const score =
                (diarChange ? 3 : 0) +
                ((audio?.genderChange && (audio.musicToSpeech || audio.pauseToSpeech)) ? 1.8 : 0) +
                (audio?.musicToSpeech ? 1.25 : 0) +
                (audio?.pauseToSpeech ? 1.5 : 0) +
                (ocr?.lyricToSermon ? 1.6 : 0) +
                (slide?.lyricToSermon ? 1.6 : 0) +
                ((ocr?.afterSermonCueRatio ?? 0) >= 0.1 ? 1.1 : 0) +
                ((slide?.afterSermonCueRatio ?? 0) >= 0.1 ? 1.1 : 0) +
                (slide?.hasSermonTitleNear ? 0.8 : 0) +
                (slide?.hasBibleVerseNear ? 0.55 : 0) +
                (ocr?.hasSpeakerNameNear ? 0.4 : 0) +
                (ocr?.nearSceneCut ? 0.25 : 0) -
                ((ocr && !ocr.lyricToSermon && ocr.afterLyricRatio >= 0.18) ? 1.2 : 0) -
                ((ocr && !ocr.lyricToSermon && ocr.lyricDominantNear) ? 0.6 : 0) -
                ((slide && !slide.lyricToSermon && slide.afterLyricRatio >= 0.18) ? 1.2 : 0) -
                ((slide && !slide.lyricToSermon && slide.lyricDominantNear) ? 0.6 : 0) -
                transcriptSignal.prayerCueScore * 0.45 -
                transcriptSignal.worshipCueScore * 0.4 -
                transcriptSignal.announcementCueScore * 0.35 +
                transcriptSignal.sermonCueScore * 1.2 +
                (transcriptSignal.hasFelizSabado ? 1.6 : 0) -
                (transcriptSignal.prayerDominant ? 2.2 : 0) -
                (transcriptSignal.worshipDominant ? 2.0 : 0) -
                (transcriptSignal.announcementDominant ? 1.4 : 0) -
                Math.abs(candidateStart - guess.start) / 90;
            return { seed, candidateStart, prev, diarChange, audio, ocr, slide, transcriptSignal, score };
        });
        scoredStart.sort((a, b) => b.score - a.score);
        if (scoredStart.length > 0) {
            refinedStart = scoredStart[0].candidateStart;
            startChosenBy = `scored-candidate:${scoredStart[0].seed}`;
        }
        const startCandidateStrong =
            Boolean(scoredStart[0]?.diarChange) ||
            Boolean(scoredStart[0]?.audio && (scoredStart[0].audio.musicToSpeech || scoredStart[0].audio.pauseToSpeech)) ||
            Boolean(scoredStart[0]?.transcriptSignal?.sermonLikely);
        if (!startCandidateStrong) {
            const fallbackStart = findLatestOtherToTargetStartTransition(preTurnsAbs, startSpeaker, refinedStart);
            if (fallbackStart != null) {
                const fallbackTranscript = boundaryTranscriptSignal(normalized, fallbackStart, 'start');
                const maxFallbackDriftSec = Number(process.env.TARGET_DIAR_START_FALLBACK_MAX_DRIFT_SEC ?? 360);
                const referenceStart = scoredStart[0]?.candidateStart ?? refinedStart;
                const fallbackDrift = Math.abs(fallbackStart - referenceStart);
                const fallbackAllowed =
                    fallbackDrift <= maxFallbackDriftSec &&
                    (fallbackTranscript.sermonLikely || fallbackTranscript.hasFelizSabado) &&
                    !(fallbackTranscript.prayerDominant || fallbackTranscript.worshipDominant || fallbackTranscript.announcementDominant);
                if (fallbackAllowed) {
                    refinedStart = refineStart(preTurnsAbs, startSpeaker, fallbackStart);
                    startChosenBy = 'sequential-fallback';
                } else {
                    startChosenBy = `scored-candidate:${scoredStart[0]?.seed}:fallback-rejected`;
                }
            }
        }

        const endSeedList = [guess.end, ...endCandidates].filter((v, i, arr) => arr.indexOf(v) === i);
        const scoredEnd = endSeedList.map((seed) => {
            const candidateEnd = refineEnd(postTurnsAbs, endSpeaker, seed);
            const next = nextTurnAfter(postTurnsAbs, candidateEnd);
            const diarChange = Boolean(next && next.speaker !== endSpeaker && next.start - candidateEnd <= maxConfirmGapSec);
            const audio = audioEvents ? boundaryAudioSignal(audioEvents.segments, candidateEnd) : null;
            const ocr = ocrEvents ? boundaryOcrSignal(ocrEvents, candidateEnd, ocrWindowSec) : null;
            const slide = slideOcrEvents ? boundarySlideSignal(slideOcrEvents, candidateEnd, ocrWindowSec) : null;
            const transcriptSignal = boundaryTranscriptSignal(normalized, candidateEnd, 'end');
            const handoffCue = hasSermonHandoffCue(normalized, candidateEnd, 110);
            const score =
                (diarChange ? 3 : 0) +
                ((audio?.genderChange && (audio.speechToMusic || audio.speechToPause)) ? 1.8 : 0) +
                (audio?.speechToMusic ? 1.25 : 0) +
                (audio?.speechToPause ? 1.0 : 0) +
                (handoffCue ? 1.8 : 0) +
                (ocr?.sermonToLyric ? 1.7 : 0) +
                (slide?.sermonToLyric ? 1.7 : 0) +
                ((ocr?.afterLyricRatio ?? 0) >= 0.16 ? 0.9 : 0) +
                ((slide?.afterLyricRatio ?? 0) >= 0.16 ? 0.9 : 0) +
                ((ocr?.afterSpeakerNameRatio ?? 0) >= 0.08 ? 0.45 : 0) +
                (ocr?.nearSceneCut ? 0.25 : 0) -
                ((ocr?.beforeLyricRatio ?? 0) >= 0.2 ? 1.2 : 0) -
                ((slide?.beforeLyricRatio ?? 0) >= 0.2 ? 1.2 : 0) -
                ((ocr?.afterSermonCueRatio ?? 0) >= 0.16 ? 0.8 : 0) -
                ((slide?.afterSermonCueRatio ?? 0) >= 0.16 ? 0.8 : 0) -
                transcriptSignal.closingCueScore * 0.9 +
                transcriptSignal.handoffCueScore * 1.0 -
                (transcriptSignal.sermonCueScore >= 1.6 ? 0.8 : 0) -
                Math.abs(candidateEnd - guess.end) / 90;
            return { seed, candidateEnd, next, diarChange, audio, ocr, slide, transcriptSignal, handoffCue, score };
        });
        scoredEnd.sort((a, b) => b.score - a.score);
        if (scoredEnd.length > 0) {
            refinedEnd = scoredEnd[0].candidateEnd;
            endChosenBy = `scored-candidate:${scoredEnd[0].seed}`;
        }
        const endCandidateStrong =
            Boolean(scoredEnd[0]?.diarChange) ||
            Boolean(scoredEnd[0]?.audio && (scoredEnd[0].audio.speechToMusic || scoredEnd[0].audio.speechToPause)) ||
            Boolean(scoredEnd[0]?.transcriptSignal?.closingLikely || scoredEnd[0]?.transcriptSignal?.handoffLikely);
        if (!endCandidateStrong) {
            const fallbackEnd = findEarliestTargetToOtherEndTransition(postTurnsAbs, endSpeaker, refinedEnd);
            if (fallbackEnd != null) {
                const fallbackTranscript = boundaryTranscriptSignal(normalized, fallbackEnd, 'end');
                const maxFallbackDriftSec = Number(process.env.TARGET_DIAR_END_FALLBACK_MAX_DRIFT_SEC ?? 420);
                const referenceEnd = scoredEnd[0]?.candidateEnd ?? refinedEnd;
                const fallbackDrift = Math.abs(fallbackEnd - referenceEnd);
                const fallbackAllowed =
                    fallbackDrift <= maxFallbackDriftSec &&
                    (fallbackTranscript.closingLikely || fallbackTranscript.handoffLikely);
                if (fallbackAllowed) {
                    refinedEnd = refineEnd(postTurnsAbs, endSpeaker, fallbackEnd);
                    endChosenBy = 'sequential-fallback';
                } else {
                    endChosenBy = `scored-candidate:${scoredEnd[0]?.seed}:fallback-rejected`;
                }
            }
        }

        const padded = applyPadding(refinedStart, refinedEnd, normalized, allTurns, startSpeaker, endSpeaker);
        const prevAtStart = previousTurnBefore(preTurnsAbs, refinedStart);
        const nextAtEnd = nextTurnAfter(postTurnsAbs, refinedEnd);
        const startAudioSignal = audioEvents ? boundaryAudioSignal(audioEvents.segments, refinedStart) : null;
        const endAudioSignal = audioEvents ? boundaryAudioSignal(audioEvents.segments, refinedEnd) : null;
        const startOcrSignal = ocrEvents ? boundaryOcrSignal(ocrEvents, refinedStart, ocrWindowSec) : null;
        const endOcrSignal = ocrEvents ? boundaryOcrSignal(ocrEvents, refinedEnd, ocrWindowSec) : null;
        const startSlideSignal = slideOcrEvents ? boundarySlideSignal(slideOcrEvents, refinedStart, ocrWindowSec) : null;
        const endSlideSignal = slideOcrEvents ? boundarySlideSignal(slideOcrEvents, refinedEnd, ocrWindowSec) : null;
        const startTranscriptSignal = boundaryTranscriptSignal(normalized, refinedStart, 'start');
        const endTranscriptSignal = boundaryTranscriptSignal(normalized, refinedEnd, 'end');

        const startDiarChange = Boolean(prevAtStart && prevAtStart.speaker !== startSpeaker && refinedStart - prevAtStart.end <= maxConfirmGapSec);
        const startAudioStrong = Boolean(startAudioSignal && (startAudioSignal.musicToSpeech || startAudioSignal.pauseToSpeech));
        const startGenderSupported = Boolean(
            startAudioSignal?.genderChange && (startAudioStrong || startTranscriptSignal.sermonLikely || startOcrSignal?.lyricToSermon)
        );
        const startOcrStrong = Boolean(
            startOcrSignal?.lyricToSermon ||
                (((startOcrSignal?.afterSermonCueRatio ?? 0) >= 0.12) && (startOcrSignal?.afterLyricRatio ?? 0) < 0.2)
        );
        const startSlideStrong = Boolean(
            startSlideSignal?.lyricToSermon ||
                (((startSlideSignal?.afterSermonCueRatio ?? 0) >= 0.12) && (startSlideSignal?.afterLyricRatio ?? 0) < 0.2) ||
                startSlideSignal?.hasSermonTitleNear
        );
        const startTranscriptStrong = Boolean(
            startTranscriptSignal.sermonLikely ||
                startTranscriptSignal.hasFelizSabado ||
                startTranscriptSignal.sermonCueScore >= 1.9
        );
        const startTranscriptBlocked = Boolean(
            startTranscriptSignal.prayerDominant || startTranscriptSignal.worshipDominant || startTranscriptSignal.announcementDominant
        );
        const startConfirmed = Boolean(
            !startTranscriptBlocked &&
                (startDiarChange || startAudioStrong || startGenderSupported || startOcrStrong || startSlideStrong || startTranscriptStrong)
        );

        const endDiarChange = Boolean(nextAtEnd && nextAtEnd.speaker !== endSpeaker && nextAtEnd.start - refinedEnd <= maxConfirmGapSec);
        const endAudioStrong = Boolean(endAudioSignal && (endAudioSignal.speechToMusic || endAudioSignal.speechToPause));
        const endGenderSupported = Boolean(
            endAudioSignal?.genderChange && (endAudioStrong || endTranscriptSignal.closingLikely || endTranscriptSignal.handoffLikely)
        );
        const endOcrStrong = Boolean(
            endOcrSignal?.sermonToLyric ||
                (((endOcrSignal?.afterLyricRatio ?? 0) >= 0.14) && ((endOcrSignal?.beforeLyricRatio ?? 0) < 0.2))
        );
        const endSlideStrong = Boolean(
            endSlideSignal?.sermonToLyric ||
                (((endSlideSignal?.afterLyricRatio ?? 0) >= 0.14) && ((endSlideSignal?.beforeLyricRatio ?? 0) < 0.2))
        );
        const endTranscriptStrong = Boolean(endTranscriptSignal.closingLikely || endTranscriptSignal.handoffLikely);
        const endConfirmed = Boolean(endDiarChange || endAudioStrong || endGenderSupported || endOcrStrong || endSlideStrong || endTranscriptStrong);

        let finalClipBounds: any = padded;
        let llmFallback: any = null;
        const singleAdjEnabled = String(process.env.BOUNDARY_ENABLE_SINGLE_LLM_ADJUDICATOR ?? 'true').toLowerCase() === 'true';
        const forceLlm = !singleAdjEnabled && String(process.env.BOUNDARY_FORCE_LLM_FALLBACK ?? 'false').toLowerCase() === 'true';
        const llmEnabled = !singleAdjEnabled && String(process.env.BOUNDARY_ENABLE_LLM_FALLBACK ?? 'true').toLowerCase() === 'true';
        const shouldUseLlmFallback = llmEnabled && (forceLlm || !startConfirmed || !endConfirmed);

        if (shouldUseLlmFallback) {
            const openai = getOpenAIClient();
            if (openai) {
                try {
                    const correctedPath = path.join(workDir, 'transcript.corrected.json');
                    const boundaryTranscript = fs.existsSync(correctedPath)
                        ? (safeJsonParse<Segment[]>(fs.readFileSync(correctedPath, 'utf8'), normalized) || normalized)
                        : normalized;
                    const binSec = Number(process.env.BOUNDARY_LLM_INDEX_BIN_SEC ?? 45);
                    const binsAll = buildTranscriptBins(
                        boundaryTranscript,
                        audioEvents?.segments ?? null,
                        audioEnd,
                        Math.max(20, binSec)
                    );
                    const focusSec = Number(process.env.BOUNDARY_LLM_FOCUS_RANGE_SEC ?? 900);
                    const focusRanges = [
                        { start: clamp(guess.start - focusSec, 0, audioEnd), end: clamp(guess.start + focusSec, 0, audioEnd) },
                        { start: clamp(guess.end - focusSec, 0, audioEnd), end: clamp(guess.end + focusSec, 0, audioEnd) },
                        ...startCandidates.map((v) => ({ start: clamp(v - focusSec / 2, 0, audioEnd), end: clamp(v + focusSec / 2, 0, audioEnd) })),
                        ...endCandidates.map((v) => ({ start: clamp(v - focusSec / 2, 0, audioEnd), end: clamp(v + focusSec / 2, 0, audioEnd) }))
                    ];
                    const bins = buildFocusedBins(binsAll, focusRanges);
                    const coarse = await llmStage1CoarseBounds(
                        openai,
                        bins,
                        audioEnd,
                        { start: guess.start, end: guess.end, startCandidates, endCandidates },
                        workDir
                    );
                    const refined = coarse
                        ? await llmStage2RefineBounds(openai, boundaryTranscript, coarse, audioEnd, workDir)
                        : null;
                    let applied = false;
                    let rejectedReason: string | null = null;
                    let driftFromLocal: number | null = null;
                    if (refined) {
                        const paddedFromLlm = applyTranscriptOnlyPadding(refined.start, refined.end, boundaryTranscript);
                        const maxDriftSec = Number(process.env.BOUNDARY_LLM_MAX_DRIFT_SEC ?? 1800);
                        driftFromLocal =
                            Math.abs(paddedFromLlm.clip_start_sec - padded.clip_start_sec) +
                            Math.abs(paddedFromLlm.clip_end_sec - padded.clip_end_sec);
                        if (driftFromLocal <= maxDriftSec || forceLlm) {
                            finalClipBounds = paddedFromLlm;
                            applied = true;
                        } else {
                            rejectedReason = `drift_exceeds_limit:${driftFromLocal.toFixed(2)}>${maxDriftSec}`;
                        }
                    }
                    llmFallback = {
                        invoked: true,
                        reason: forceLlm ? 'forced' : 'low_local_confidence',
                        coarse,
                        refined,
                        accepted: Boolean(refined),
                        applied,
                        rejected_reason: rejectedReason,
                        drift_from_local_sec: driftFromLocal,
                        applied_bounds: finalClipBounds
                    };
                } catch (error) {
                    llmFallback = { invoked: true, error: String(error) };
                }
            } else {
                llmFallback = { invoked: false, reason: 'missing_openai_api_key' };
            }
        } else {
            llmFallback = { invoked: false, reason: 'local_confidence_ok' };
        }

        let llmEndStyle: any = null;
        const endStyleEnabled =
            !singleAdjEnabled && String(process.env.BOUNDARY_ENABLE_LLM_END_STYLE ?? 'true').toLowerCase() === 'true';
        if (endStyleEnabled) {
            const openai = getOpenAIClient();
            if (openai) {
                try {
                    const correctedPath = path.join(workDir, 'transcript.corrected.json');
                    const boundaryTranscript = fs.existsSync(correctedPath)
                        ? (safeJsonParse<Segment[]>(fs.readFileSync(correctedPath, 'utf8'), normalized) || normalized)
                        : normalized;
                    const decision = await llmDecideEndingStyle(
                        openai,
                        boundaryTranscript,
                        finalClipBounds.clip_start_sec,
                        finalClipBounds.clip_end_sec,
                        audioEnd,
                        workDir
                    );
                    if (decision && Number.isFinite(decision.end_sec)) {
                        const snapped = snapEndToSegmentEnd(boundaryTranscript, decision.end_sec);
                        const oldEnd = finalClipBounds.clip_end_sec;
                        const endMaxBackwardSec = Number(process.env.BOUNDARY_LLM_END_STYLE_MAX_BACKWARD_SEC ?? 180);
                        const endMaxExtensionSec = Number(process.env.BOUNDARY_LLM_END_STYLE_MAX_EXTENSION_SEC ?? 15);
                        const hardMinByClipLength =
                            finalClipBounds.clip_start_sec +
                            Number(process.env.BOUNDARY_LLM_END_STYLE_MIN_CLIP_SEC ?? 120);
                        const endMinByDrift = Math.max(0, oldEnd - endMaxBackwardSec);
                        const endMin = Math.max(hardMinByClipLength, endMinByDrift);
                        const endMax = oldEnd + endMaxExtensionSec;
                        const bounded = clamp(snapped, endMin, endMax);
                        finalClipBounds.clip_end_sec = bounded;
                        finalClipBounds.applied_post_pad_sec = Math.max(0, bounded - refinedEnd);
                        const nextSpeech = normalized.find((s) => s.start >= bounded && !isNonSpeech(s.text));
                        finalClipBounds.next_speech_start_sec = nextSpeech?.start ?? null;
                        llmEndStyle = {
                            invoked: true,
                            decision,
                            snapped_end_sec: snapped,
                            previous_end_sec: oldEnd,
                            applied_end_sec: bounded
                        };
                    } else {
                        llmEndStyle = { invoked: true, decision: null };
                    }
                } catch (error) {
                    llmEndStyle = { invoked: true, error: String(error) };
                }
            } else {
                llmEndStyle = { invoked: false, reason: 'missing_openai_api_key' };
            }
        } else {
            llmEndStyle = { invoked: false, reason: 'disabled' };
        }

        const output = {
            guess: {
                start_sec: guess.start,
                end_sec: guess.end,
                start_candidates_sec: startCandidates,
                end_candidates_sec: endCandidates,
                transcript_start_candidates_sec: transcriptStartCandidates,
                transcript_end_candidates_sec: transcriptEndCandidates
            },
            selected_target_speaker: {
                start: startSpeaker,
                end: endSpeaker
            },
            boundary_confirmation: {
                start_confirmed_change_of_speaker: startConfirmed,
                end_confirmed_change_of_speaker: endConfirmed,
                start_chosen_by: startChosenBy,
                end_chosen_by: endChosenBy,
                start_previous_turn: prevAtStart,
                end_next_turn: nextAtEnd,
                start_audio_signal: startAudioSignal,
                end_audio_signal: endAudioSignal,
                start_ocr_signal: startOcrSignal,
                end_ocr_signal: endOcrSignal,
                start_slide_signal: startSlideSignal,
                end_slide_signal: endSlideSignal,
                start_transcript_signal: startTranscriptSignal,
                end_transcript_signal: endTranscriptSignal
            },
            targeted_windows: {
                pre: { start_sec: preWindowStart, end_sec: preWindowEnd, anchor_speaker: preSpeaker, turns: preTurnsAbs },
                post: { start_sec: postWindowStart, end_sec: postWindowEnd, anchor_speaker: postSpeaker, turns: postTurnsAbs }
            },
            audio_events: audioEvents
                ? {
                      source: audioEvents.source,
                      duration_sec: audioEvents.duration_sec,
                      total_segments: audioEvents.segments.length
                  }
                : null,
            ocr_events: ocrEvents
                ? {
                      source: ocrEvents.source,
                      duration_sec: ocrEvents.duration_sec,
                      sample_sec: ocrEvents.sample_sec,
                      total_segments: ocrEvents.segments.length,
                      scene_cuts: Array.isArray(ocrEvents.scene_cuts_sec) ? ocrEvents.scene_cuts_sec.length : 0
                  }
                : null,
            slide_ocr_events: slideOcrEvents
                ? {
                      source: slideOcrEvents.source,
                      duration_sec: slideOcrEvents.duration_sec,
                      sample_sec: slideOcrEvents.sample_sec,
                      total_events: slideOcrEvents.events.length,
                      shot_boundaries: Array.isArray(slideOcrEvents.shot_boundaries_sec)
                          ? slideOcrEvents.shot_boundaries_sec.length
                          : 0
                  }
                : null,
            candidate_scoring: {
                start: scoredStart,
                end: scoredEnd
            },
            refined_speaker_bounds: {
                start_sec: refinedStart,
                end_sec: refinedEnd
            },
            final_clip_bounds: finalClipBounds,
            llm_fallback: llmFallback,
            llm_end_style: llmEndStyle
        };

        fs.writeFileSync(targetedPath, JSON.stringify(output, null, 2));
        return { start: finalClipBounds.clip_start_sec, end: finalClipBounds.clip_end_sec };
    } catch (error) {
        console.error('Targeted diarization refinement failed, using face-pass guess:', error);
        return { start: guess.start, end: guess.end };
    }
}
