import { loadWorkerEnv } from '../lib/load-env';
import { generateStructuredOutput, LLMProvider } from '../lib/llm';

loadWorkerEnv();

export type HookType = 'question' | 'promise' | 'scripture' | 'story' | 'contrast' | 'none';

/** A single B-roll cue point identified by the LLM within a clip. */
export interface BrollCue {
    /** Seconds from clip start (0 = clip.start) where the B-roll cutaway would fit */
    offset_sec: number;
    /** 2–3 searchable stock-footage keywords, e.g. ["open bible", "church congregation"] */
    keywords: string[];
    /** Short human-readable description of the suggested visual */
    description: string;
}

export interface HighlightClip {
    start: number;
    end: number;
    title: string;
    excerpt: string;
    hook: string;
    confidence: number;
    score: number;
    hook_type?: HookType;
    score_breakdown?: {
        /** 0–1: How strongly the opening line grabs attention (LLM-rated) */
        hook_strength: number;
        /** 0–1: Theological depth, conviction, or emotional resonance (LLM-rated) */
        spiritual_impact: number;
        /** 0–1: "I'm sending this to someone" potential (LLM-rated) */
        shareability: number;
        /** 0–1: How cleanly the clip ends on a complete thought (local heuristic) */
        ending_completeness: number;
        /** 0–1: Model's self-assessed coherence and correct bounding (metadata only, not in composite) */
        model_confidence: number;
    };
    /** B-roll cue points (only present when HIGHLIGHTS_BROLL_CUES_ENABLED=true) */
    broll_cues?: BrollCue[];
}

export type HighlightSelectionProfile = 'standard' | 'dense' | 'arc';
type HighlightSelectionProfileInput =
    | HighlightSelectionProfile
    | 'dynamic'
    | 'chopped'
    | 'skimmed'
    | 'structured'
    | 'narrative';

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
    profile?: HighlightSelectionProfileInput | null;
    /** When true, ask the LLM to identify B-roll cue moments per clip. Defaults to HIGHLIGHTS_BROLL_CUES_ENABLED env var. */
    includeBrollCues?: boolean;
}

interface HighlightSelectionConfig {
    profile: HighlightSelectionProfile;
    minDurationSec: number;
    maxDurationSec: number;
    preferredDurationSec: number;
    targetCount: number;
    scoreWeights: {
        hookStrength: number;
        spiritualImpact: number;
        shareability: number;
        endingCompleteness: number;
    };
    promptGuidance: string[];
}

export async function findHighlights(
    transcript: { start: number; end: number; text: string }[],
    strategy: 'openai' | 'deepseek' | 'cohere' = 'openai',
    context?: Partial<HighlightContext>
): Promise<HighlightClip[]> {
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

function parseHookType(raw: unknown): HookType {
    const value = String(raw ?? '').trim().toLowerCase();
    const valid: HookType[] = ['question', 'promise', 'scripture', 'story', 'contrast', 'none'];
    return (valid.includes(value as HookType) ? value : 'none') as HookType;
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

    // Parse new 4-subscore fields. If the LLM returns the legacy flat `score`
    // float without subscores, use it as spiritual_impact and default the rest.
    const hasNewSubscores =
        Number.isFinite(Number(raw?.hook_strength)) ||
        Number.isFinite(Number(raw?.spiritual_impact)) ||
        Number.isFinite(Number(raw?.shareability));

    const hook_strength = hasNewSubscores
        ? clamp(Number(raw?.hook_strength ?? 0.5), 0, 1)
        : 0.5;
    const spiritual_impact = hasNewSubscores
        ? clamp(Number(raw?.spiritual_impact ?? 0.5), 0, 1)
        : clamp(Number(raw?.score ?? raw?.confidence ?? 0.5), 0, 1);
    const shareability = hasNewSubscores
        ? clamp(Number(raw?.shareability ?? 0.5), 0, 1)
        : 0.5;
    const confidence = clamp(Number(raw?.confidence ?? 0.55), 0, 1);
    const hook_type = parseHookType(raw?.hook_type);

    // Parse optional B-roll cues array
    const broll_cues = parseBrollCues(raw?.broll_cues, start);

    return {
        start,
        end,
        title: cleanText(raw?.title || 'Sermon Highlight'),
        excerpt: cleanText(raw?.excerpt || ''),
        hook: cleanText(raw?.hook || ''),
        confidence,
        score: 0, // will be computed by scoreClip
        hook_type,
        score_breakdown: {
            hook_strength,
            spiritual_impact,
            shareability,
            ending_completeness: 0, // filled by scoreClip
            model_confidence: confidence
        },
        ...(broll_cues.length ? { broll_cues } : {})
    };
}

/** Parse and sanitize the LLM's raw broll_cues array. Returns [] if missing or malformed. */
function parseBrollCues(raw: unknown, clipStart: number): BrollCue[] {
    if (!Array.isArray(raw) || !raw.length) return [];
    const out: BrollCue[] = [];
    for (const item of raw) {
        if (!item || typeof item !== 'object') continue;
        const offset = Number((item as any).offset_sec ?? (item as any).at_seconds ?? 0);
        if (!Number.isFinite(offset) || offset < 0) continue;
        const rawKw = (item as any).keywords;
        const keywords: string[] = Array.isArray(rawKw)
            ? rawKw.map((k: unknown) => cleanText(String(k ?? ''))).filter(Boolean).slice(0, 4)
            : [];
        if (!keywords.length) continue;
        const description = cleanText(String((item as any).description ?? '')).slice(0, 120);
        out.push({ offset_sec: Number(offset.toFixed(2)), keywords, description });
        if (out.length >= 4) break; // cap at 4 cues per clip
    }
    return out;
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

function resolveHighlightProfile(raw: unknown): HighlightSelectionProfile {
    const value = String(raw ?? process.env.HIGHLIGHTS_PROFILE ?? 'standard').trim().toLowerCase();
    if (!value || value === 'standard') return 'standard';
    if (value === 'dense' || value === 'dynamic' || value === 'chopped') return 'dense';
    if (value === 'arc' || value === 'skimmed' || value === 'structured' || value === 'narrative') return 'arc';
    console.warn(`Unknown highlight profile "${value}", defaulting to "standard".`);
    return 'standard';
}

function getHighlightSelectionConfig(context?: Partial<HighlightContext>): HighlightSelectionConfig {
    const profile = resolveHighlightProfile(context?.profile);
    const minDurationSec = Number(process.env.HIGHLIGHTS_MIN_DURATION_SEC ?? 30);
    const maxDurationSec = Number(process.env.HIGHLIGHTS_MAX_DURATION_SEC ?? 180);
    const preferredDurationSec = Number(process.env.HIGHLIGHTS_PREFERRED_DURATION_SEC ?? 90);
    const targetCount = Number(process.env.HIGHLIGHTS_TARGET_COUNT ?? 10);

    const base: HighlightSelectionConfig = {
        profile,
        minDurationSec,
        maxDurationSec,
        preferredDurationSec,
        targetCount,
        scoreWeights: {
            hookStrength: 0.20,
            spiritualImpact: 0.30,
            shareability: 0.15,
            endingCompleteness: 0.35
        },
        promptGuidance: []
    };

    switch (profile) {
        case 'dense':
            base.promptGuidance = [
                '- prefer clips with high idea density and minimal runway before the core point lands',
                '- use spoken seconds efficiently; avoid long setup, repetition, and obvious filler when a denser window is available',
                '- until word-level trims exist, choose naturally dense windows that still sound clean without micro-cuts'
            ];
            break;
        case 'arc':
            base.promptGuidance = [
                '- prefer clips with a clear internal arc: intro, development, and conclusion',
                '- for sermons, prefer endings that land in reflection, invitation, conviction, or practical action',
                '- if forced to choose, pick structural completeness over raw punchiness'
            ];
            break;
        case 'standard':
        default:
            base.promptGuidance = [
                '- prefer clips that are complete, strong, and easy to understand without surrounding context'
            ];
            break;
    }

    return base;
}

async function askHighlights(
    provider: LLMProvider,
    model: string,
    payload: {
        profile: HighlightSelectionProfile;
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
        promptGuidance: string[];
        includeBrollCues: boolean;
    }
): Promise<any[]> {
    const used = payload.usedRanges
        .map((r, idx) => `${idx + 1}. ${r.start.toFixed(2)}-${r.end.toFixed(2)}`)
        .join('\n');

    const brollSchemaLine = payload.includeBrollCues
        ? '  "broll_cues":[{"offset_sec":number,"keywords":[string],"description":string}],'
        : '';

    const prompt = [
        'Select social-ready sermon highlights from this sermon-only payload.',
        'Output JSON only in this schema:',
        '{"highlights":[{',
        '  "start":number, "end":number,',
        '  "title":string, "excerpt":string, "hook":string,',
        '  "confidence":number,',
        '  "hook_type":"question"|"promise"|"scripture"|"story"|"contrast"|"none",',
        '  "hook_strength":number,',
        '  "spiritual_impact":number,',
        '  "shareability":number,',
        brollSchemaLine,
        '  "why_end_complete":"short"',
        '}]}',
        '',
        'SERMON VIRALITY SCORING — fill these three subscores per clip (0..1 each):',
        '- "hook_strength": How strongly the opening line grabs attention.',
        '  0.9-1.0 = bold claim, unexpected question, or striking scripture. 0.5 = decent opener. 0.0 = flat or mid-sentence start.',
        '- "spiritual_impact": Theological depth, spiritual conviction, or emotional resonance with a faith audience.',
        '  0.9-1.0 = transformative moment, clear application, or deeply moving. 0.5 = solid content. 0.0 = factual/logistical filler.',
        '- "shareability": Would someone immediately send this to a friend or family member?',
        '  0.9-1.0 = universal insight or memorable line anyone can relate to. 0.5 = meaningful but niche. 0.0 = internal church logistics.',
        '',
        'HOOK TYPE — classify the opening move of each clip:',
        '  "question": Opens with a question that creates curiosity.',
        '  "promise": Makes a bold claim or gives a promise ("If you do X, Y will happen").',
        '  "scripture": Opens directly with a Bible verse or reference.',
        '  "story": Starts with a personal anecdote or narrative.',
        '  "contrast": Uses before/after, problem/solution, or paradox framing.',
        '  "none": No discernible hook pattern.',
        '',
        ...(payload.includeBrollCues ? [
            'B-ROLL CUES — for each clip, identify 2-3 moments where stock footage would enhance the visual:',
            '- "offset_sec": seconds from clip.start (0 = very beginning of the clip)',
            '- "keywords": 2-3 short, searchable stock-footage search terms (e.g. ["open bible", "person praying"])',
            '  Use concrete, visual nouns. Prefer universal imagery over brand-specific references.',
            '  Good examples: "church congregation", "sunrise over city", "family at dinner table", "hands clasped in prayer"',
            '  Avoid abstract or non-visual terms like "faith" or "hope" alone.',
            '- "description": one short phrase describing the suggested visual cutaway',
            '- Identify moments when the speaker references a concrete object, place, person, or concept that benefits from illustration.',
            '- Prefer emotional peaks, metaphor explanations, or scripture context moments.',
            '',
        ] : []),
        'CLIP SELECTION CONSTRAINTS:',
        `- return exactly ${payload.count} highlight(s)`,
        '- select the best overall ideas, not the best ideas that merely happen to be near the preferred duration',
        '- prioritize clips whose central idea is strong, memorable, emotionally resonant, and likely to hold attention',
        '- prioritize complete ideas over compact runtimes; a stronger 120s clip can beat a weaker 60s clip',
        `- each duration must be between ${payload.minDurationSec}s and ${payload.maxDurationSec}s`,
        `- duration near ${payload.preferredDurationSec}s is only a weak preference, not a hard target`,
        '- do not sacrifice a complete idea just to land near the preferred duration',
        '- each clip must be a catchy, complete thought (prioritize complete endings over intros)',
        '- avoid clipping before the core idea lands; ending completeness is critical',
        '- avoid overlap with already selected ranges',
        '- spread clips across different sermon moments',
        '- clips must stay within sermon bounds',
        `- selection profile: ${payload.profile}`,
        '- "confidence" must mean your confidence that the selected clip is coherent, complete, and correctly bounded',
        `Sermon bounds: ${payload.sermonStart.toFixed(2)}-${payload.sermonEnd.toFixed(2)}`,
        used ? `Already selected ranges (avoid overlap):\n${used}` : 'Already selected ranges: none',
        payload.promptGuidance.length ? `Profile guidance:\n${payload.promptGuidance.join('\n')}` : '',
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

    const result = await generateStructuredOutput<{ highlights: any[] }>(provider, model, {
        prompt
    });
    return result?.highlights || [];
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
    if (/[.!?]["')]?\s*$/.test(tail)) return 1;
    if (/\b(amé?n|gracias|oramos|oremos)\b/i.test(tail)) return 0.9;
    if (/[,;:]\s*$/.test(tail)) return 0.45;
    return 0.65;
}

function scoreClip(
    clip: HighlightClip,
    transcript: { start: number; end: number; text: string }[],
    config: HighlightSelectionConfig
): HighlightClip {
    const { hookStrength, spiritualImpact, shareability, endingCompleteness } = config.scoreWeights;
    const sb = clip.score_breakdown!;

    const endScore = endingCompletenessScore(clip, transcript);
    const score = clamp(
        sb.hook_strength * hookStrength +
        sb.spiritual_impact * spiritualImpact +
        sb.shareability * shareability +
        endScore * endingCompleteness,
        0,
        1
    );

    return {
        ...clip,
        score: Number(score.toFixed(3)),
        score_breakdown: {
            hook_strength: Number(sb.hook_strength.toFixed(3)),
            spiritual_impact: Number(sb.spiritual_impact.toFixed(3)),
            shareability: Number(sb.shareability.toFixed(3)),
            ending_completeness: Number(endScore.toFixed(3)),
            model_confidence: Number(sb.model_confidence.toFixed(3))
        }
    };
}

async function findHighlightsOpenAI(
    transcript: { start: number; end: number; text: string }[],
    context?: Partial<HighlightContext>
): Promise<HighlightClip[]> {
    const provider = (process.env.HIGHLIGHTS_LLM_PROVIDER || 'openai').toLowerCase() as LLMProvider;
    
    // Resolve model based on provider
    let model = '';
    if (provider === 'google') {
        model = process.env.HIGHLIGHTS_GOOGLE_MODEL || 'gemini-2.0-flash';
    } else if (provider === 'anthropic') {
        model = process.env.HIGHLIGHTS_ANTHROPIC_MODEL || 'claude-3-5-sonnet-20241022';
    } else {
        model = process.env.HIGHLIGHTS_OPENAI_MODEL || process.env.ANALYZE_OPENAI_MODEL || 'gpt-4o-mini';
    }
    
    if (!transcript.length) return [];

    const selection = getHighlightSelectionConfig(context);
    const sermonStart = Number(context?.sermonStart ?? transcript[0].start);
    const sermonEnd = Number(context?.sermonEnd ?? transcript[transcript.length - 1].end);
    const { minDurationSec, maxDurationSec, preferredDurationSec, targetCount } = selection;
    
    const includeBrollCues =
        context?.includeBrollCues ??
        String(process.env.HIGHLIGHTS_BROLL_CUES_ENABLED ?? 'false').toLowerCase() === 'true';
    
    console.log(
        `Finding highlights using strategy: ${provider} model=${model} profile=${selection.profile} range=${minDurationSec}-${maxDurationSec}s preferred=${preferredDurationSec}s broll=${includeBrollCues}`
    );

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

    const sharedPayload = {
        profile: selection.profile,
        sermonStart,
        sermonEnd,
        minDurationSec,
        maxDurationSec,
        preferredDurationSec,
        paragraphLines,
        transcriptLines,
        chapterLines,
        promptGuidance: selection.promptGuidance,
        includeBrollCues
    };

    const pass1 = await askHighlights(provider, model, {
        ...sharedPayload,
        count: targetCount,
        usedRanges
    });
    for (const raw of pass1) {
        const clip = normalizeClip(raw, sermonStart, sermonEnd, minDurationSec, maxDurationSec);
        if (!clip) continue;
        if (accepted.some((x) => overlaps(x, clip))) continue;
        accepted.push(scoreClip(clip, transcript, selection));
        usedRanges.push({ start: clip.start, end: clip.end });
        if (accepted.length >= targetCount) break;
    }

    const remaining = Math.max(0, targetCount - accepted.length);
    if (remaining > 0) {
        const pass2 = await askHighlights(provider, model, {
            ...sharedPayload,
            count: remaining,
            usedRanges
        });
        for (const raw of pass2) {
            const clip = normalizeClip(raw, sermonStart, sermonEnd, minDurationSec, maxDurationSec);
            if (!clip) continue;
            if (accepted.some((x) => overlaps(x, clip))) continue;
            accepted.push(scoreClip(clip, transcript, selection));
            usedRanges.push({ start: clip.start, end: clip.end });
            if (accepted.length >= targetCount) break;
        }
    }

    return accepted
        .sort((a, b) => b.score - a.score || a.start - b.start)
        .slice(0, targetCount);
}
