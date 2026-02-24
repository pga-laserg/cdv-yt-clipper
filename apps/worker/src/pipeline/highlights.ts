import { OpenAI } from 'openai';
import dotenv from 'dotenv';
import path from 'path';

dotenv.config({ path: path.resolve(__dirname, '../../../.env') });

const getOpenAIClient = () => {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) return null;
    return new OpenAI({ apiKey });
};

export interface HighlightClip {
    start: number;
    end: number;
    title: string;
    excerpt: string;
    hook: string;
    confidence: number;
    score: number;
    score_breakdown?: {
        model_confidence: number;
        duration_preference: number;
        ending_completeness: number;
    };
}

interface ParagraphLite {
    chapter_index?: number;
    chapter_title?: string;
    start: number;
    end: number;
    text: string;
}

interface ChapterLite {
    start: number;
    end: number;
    title: string;
    type?: string;
    confidence?: number;
}

interface HighlightContext {
    sermonStart: number;
    sermonEnd: number;
    paragraphs?: ParagraphLite[] | null;
    chapters?: ChapterLite[] | null;
}

interface HighlightRequest {
    count: number;
    minDurationSec: number;
    maxDurationSec: number;
    usedRanges: Array<{ start: number; end: number }>;
}

export async function findHighlights(
    transcript: { start: number; end: number; text: string }[],
    strategy: 'openai' | 'deepseek' | 'cohere' = 'openai',
    context?: Partial<HighlightContext>
): Promise<HighlightClip[]> {
    console.log(`Finding highlights using strategy: ${strategy}`);
    if (strategy !== 'openai') {
        throw new Error(`Highlight strategy "${strategy}" is disabled. Use "openai".`);
    }
    return findHighlightsOpenAI(transcript, context);
}

function clamp(n: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, n));
}

function cleanText(text: string): string {
    return String(text ?? '').replace(/\s+/g, ' ').trim();
}

function trimForPrompt(text: string, max = 220): string {
    const t = cleanText(text);
    if (t.length <= max) return t;
    return `${t.slice(0, max - 1)}…`;
}

function overlaps(a: { start: number; end: number }, b: { start: number; end: number }, pad = 2): boolean {
    return a.start < b.end + pad && b.start < a.end + pad;
}

function normalizeClip(
    raw: any,
    sermonStart: number,
    sermonEnd: number,
    minDurationSec: number,
    maxDurationSec: number
): HighlightClip | null {
    const startRaw = Number(raw?.start);
    const endRaw = Number(raw?.end);
    if (!Number.isFinite(startRaw) || !Number.isFinite(endRaw)) return null;
    let start = clamp(startRaw, sermonStart, sermonEnd);
    let end = clamp(endRaw, sermonStart, sermonEnd);
    if (end <= start) return null;
    if (end - start > maxDurationSec) end = start + maxDurationSec;
    if (end - start < minDurationSec) return null;
    start = Number(start.toFixed(2));
    end = Number(end.toFixed(2));
    return {
        start,
        end,
        title: cleanText(raw?.title || 'Sermon Highlight'),
        excerpt: cleanText(raw?.excerpt || ''),
        hook: cleanText(raw?.hook || ''),
        confidence: clamp(Number(raw?.confidence ?? raw?.score ?? 0.55), 0, 1),
        score: clamp(Number(raw?.score ?? raw?.confidence ?? 0.55), 0, 1)
    };
}

function buildTranscriptLines(transcript: { start: number; end: number; text: string }[], maxLines = 180): string {
    if (transcript.length <= maxLines) {
        return transcript.map((s) => `[${s.start.toFixed(2)}-${s.end.toFixed(2)}] ${trimForPrompt(s.text, 180)}`).join('\n');
    }
    const step = Math.max(1, Math.floor(transcript.length / maxLines));
    const sampled = transcript.filter((_, idx) => idx % step === 0).slice(0, maxLines);
    return sampled.map((s) => `[${s.start.toFixed(2)}-${s.end.toFixed(2)}] ${trimForPrompt(s.text, 180)}`).join('\n');
}

function buildParagraphLines(paragraphs: ParagraphLite[], maxLines = 140): string {
    if (!paragraphs.length) return '';
    const list = paragraphs.length <= maxLines
        ? paragraphs
        : paragraphs.filter((_, idx) => idx % Math.max(1, Math.floor(paragraphs.length / maxLines)) === 0).slice(0, maxLines);
    return list
        .map((p) => {
            const chapter = cleanText(p.chapter_title ?? '');
            const chapterPart = chapter ? `${chapter}: ` : '';
            return `[${Number(p.start).toFixed(2)}-${Number(p.end).toFixed(2)}] ${chapterPart}${trimForPrompt(p.text, 180)}`;
        })
        .join('\n');
}

function parseHighlights(content: string): any[] {
    try {
        const parsed = JSON.parse(content);
        if (Array.isArray(parsed)) return parsed;
        if (Array.isArray(parsed?.highlights)) return parsed.highlights;
        return [];
    } catch {
        return [];
    }
}

async function askHighlights(
    openai: OpenAI,
    model: string,
    payload: {
        sermonStart: number;
        sermonEnd: number;
        minDurationSec: number;
        maxDurationSec: number;
        preferredDurationSec: number;
        count: number;
        paragraphLines: string;
        transcriptLines: string;
        chapterLines: string;
        usedRanges: Array<{ start: number; end: number }>;
    }
): Promise<any[]> {
    const used = payload.usedRanges
        .map((r, idx) => `${idx + 1}. ${r.start.toFixed(2)}-${r.end.toFixed(2)}`)
        .join('\n');
    const prompt = [
        'Select social-ready sermon highlights from this sermon-only payload.',
        'Output JSON only in this schema:',
        '{"highlights":[{"start":number,"end":number,"title":string,"excerpt":string,"hook":string,"confidence":number,"score":number,"why_end_complete":"short"}]}',
        `Constraints:`,
        `- return exactly ${payload.count} highlight(s)`,
        `- each duration must be between ${payload.minDurationSec}s and ${payload.maxDurationSec}s`,
        `- prefer durations close to ${payload.preferredDurationSec}s`,
        '- each clip must be catchy and complete thought (prioritize complete endings over intros)',
        '- avoid clipping before the core idea lands; ending completeness is critical',
        '- avoid overlap with already selected ranges',
        '- spread clips across different sermon moments',
        '- clips must stay within sermon bounds',
        `Sermon bounds: ${payload.sermonStart.toFixed(2)}-${payload.sermonEnd.toFixed(2)}`,
        used ? `Already selected ranges (avoid overlap):\n${used}` : 'Already selected ranges: none',
        '',
        'Chapter hints (within sermon):',
        payload.chapterLines || '[none]',
        '',
        'Paragraphs (preferred source):',
        payload.paragraphLines || '[none]',
        '',
        'Transcript fallback:',
        payload.transcriptLines
    ].join('\n');

    const response = await openai.chat.completions.create({
        model,
        messages: [{ role: 'user', content: prompt }],
        response_format: { type: 'json_object' }
    });
    const content = response.choices[0]?.message?.content ?? '{}';
    return parseHighlights(content);
}

function endingCompletenessScore(
    clip: { start: number; end: number },
    transcript: { start: number; end: number; text: string }[]
): number {
    const tail = transcript
        .filter((s) => s.end > clip.start && s.start < clip.end)
        .slice(-2)
        .map((s) => cleanText(s.text))
        .join(' ')
        .trim();
    if (!tail) return 0.4;
    if (/[.!?]["')\]]?\s*$/.test(tail)) return 1;
    if (/\b(am[ée]n|amen|gracias|oramos|oremos)\b/i.test(tail)) return 0.9;
    if (/[,;:]\s*$/.test(tail)) return 0.45;
    return 0.65;
}

function durationPreferenceScore(duration: number, preferred: number, min: number, max: number): number {
    if (duration < min || duration > max) return 0;
    const span = Math.max(1, max - min);
    const d = Math.abs(duration - preferred);
    return clamp(1 - d / (span / 2), 0, 1);
}

function scoreClip(
    clip: HighlightClip,
    transcript: { start: number; end: number; text: string }[],
    minDurationSec: number,
    maxDurationSec: number,
    preferredDurationSec: number
): HighlightClip {
    const duration = clip.end - clip.start;
    const conf = clamp(clip.confidence, 0, 1);
    const durScore = durationPreferenceScore(duration, preferredDurationSec, minDurationSec, maxDurationSec);
    const endScore = endingCompletenessScore(clip, transcript);
    const score = clamp(conf * 0.35 + durScore * 0.25 + endScore * 0.4, 0, 1);
    return {
        ...clip,
        score: Number(score.toFixed(3)),
        score_breakdown: {
            model_confidence: Number(conf.toFixed(3)),
            duration_preference: Number(durScore.toFixed(3)),
            ending_completeness: Number(endScore.toFixed(3))
        }
    };
}

async function findHighlightsOpenAI(
    transcript: { start: number; end: number; text: string }[],
    context?: Partial<HighlightContext>
): Promise<HighlightClip[]> {
    const openai = getOpenAIClient();
    if (!openai) {
        console.warn('OpenAI API key missing, skipping AI highlights.');
        return [];
    }
    if (!transcript.length) return [];

    const sermonStart = Number(context?.sermonStart ?? transcript[0].start);
    const sermonEnd = Number(context?.sermonEnd ?? transcript[transcript.length - 1].end);
    const minDurationSec = Number(process.env.HIGHLIGHTS_MIN_DURATION_SEC ?? 25);
    const maxDurationSec = Number(process.env.HIGHLIGHTS_MAX_DURATION_SEC ?? 105);
    const preferredDurationSec = Number(process.env.HIGHLIGHTS_PREFERRED_DURATION_SEC ?? 60);
    const targetCount = Number(process.env.HIGHLIGHTS_TARGET_COUNT ?? 10);
    const model = process.env.HIGHLIGHTS_OPENAI_MODEL || process.env.ANALYZE_OPENAI_MODEL || 'gpt-5-mini';

    const paragraphsAll = Array.isArray(context?.paragraphs) ? context!.paragraphs! : [];
    const paragraphs = paragraphsAll
        .filter((p) => Number.isFinite(Number(p.start)) && Number.isFinite(Number(p.end)))
        .filter((p) => Number(p.end) >= sermonStart && Number(p.start) <= sermonEnd)
        .map((p) => ({
            ...p,
            start: Number(p.start),
            end: Number(p.end),
            text: cleanText(p.text)
        }));

    const chaptersAll = Array.isArray(context?.chapters) ? context!.chapters! : [];
    const chapters = chaptersAll
        .filter((c) => Number.isFinite(Number(c.start)) && Number.isFinite(Number(c.end)))
        .filter((c) => Number(c.end) >= sermonStart && Number(c.start) <= sermonEnd)
        .map((c) => ({ ...c, start: Number(c.start), end: Number(c.end), title: cleanText(c.title) }));
    const chapterLines = chapters
        .map((c) => `[${c.start.toFixed(2)}-${c.end.toFixed(2)}] ${c.type ?? 'other'} | ${c.title}`)
        .join('\n');

    const paragraphLines = buildParagraphLines(paragraphs);
    const transcriptLines = buildTranscriptLines(transcript);

    const accepted: HighlightClip[] = [];
    let usedRanges: Array<{ start: number; end: number }> = [];

    const pass1 = await askHighlights(openai, model, {
        sermonStart,
        sermonEnd,
        minDurationSec,
        maxDurationSec,
        preferredDurationSec,
        count: targetCount,
        paragraphLines,
        transcriptLines,
        chapterLines,
        usedRanges
    });
    for (const raw of pass1) {
        const clip = normalizeClip(raw, sermonStart, sermonEnd, minDurationSec, maxDurationSec);
        if (!clip) continue;
        if (accepted.some((x) => overlaps(x, clip))) continue;
        accepted.push(scoreClip(clip, transcript, minDurationSec, maxDurationSec, preferredDurationSec));
        usedRanges.push({ start: clip.start, end: clip.end });
        if (accepted.length >= targetCount) break;
    }

    const remaining = Math.max(0, targetCount - accepted.length);
    if (remaining > 0) {
        const pass2 = await askHighlights(openai, model, {
            sermonStart,
            sermonEnd,
            minDurationSec,
            maxDurationSec,
            preferredDurationSec,
            count: remaining,
            paragraphLines,
            transcriptLines,
            chapterLines,
            usedRanges
        });
        for (const raw of pass2) {
            const clip = normalizeClip(raw, sermonStart, sermonEnd, minDurationSec, maxDurationSec);
            if (!clip) continue;
            if (accepted.some((x) => overlaps(x, clip))) continue;
            accepted.push(scoreClip(clip, transcript, minDurationSec, maxDurationSec, preferredDurationSec));
            usedRanges.push({ start: clip.start, end: clip.end });
            if (accepted.length >= targetCount) break;
        }
    }

    return accepted
        .sort((a, b) => b.score - a.score || a.start - b.start)
        .slice(0, targetCount);
}
