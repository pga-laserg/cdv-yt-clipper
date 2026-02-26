import fs from 'fs';
import path from 'path';
import { OpenAI } from 'openai';

interface Segment {
    start: number;
    end: number;
    text: string;
}

interface Boundaries {
    start: number;
    end: number;
}

interface AdjudicationResult {
    start: number;
    end: number;
    confidence: number;
    decision: 'keep_local' | 'adjust';
    reason: string;
    rejected?: string;
}

interface ChapterLite {
    start: number;
    end: number;
    title?: string;
    type?: string;
    confidence?: number;
}

interface ClosingArbiterDecision {
    end_sec: number;
    include_until_reason?: string;
    confidence?: number;
    reason?: string;
}

function cleanText(text: string): string {
    return String(text ?? '').replace(/\s+/g, ' ').trim();
}

function isPrayerLikeText(text: string): boolean {
    const t = cleanText(text).toLowerCase();
    if (!t) return false;
    return /(oremos|oraci[oó]n|inclinemos|inclinar.*rostro|arrodill|te lo pedimos|en el nombre de (tu hijo )?cristo jes[uú]s|\bam[eé]n\b|padre (nuestro|que est[aá]s))/i.test(
        t
    );
}

function isWorshipLikeText(text: string): boolean {
    const t = cleanText(text).toLowerCase();
    if (!t) return false;
    return /(vamos a cantar|cantemos|equipo de alabanza|alabemos|worship|himno|coro|reproclamamos|en lo alto est[aá]s|tu nombre)/i.test(
        t
    );
}

function isLikelySermonKickoffText(text: string): boolean {
    const t = cleanText(text).toLowerCase();
    if (!t) return false;
    return /(buenos d[ií]as[, ]+feliz s[áa]bado|feliz s[áa]bado(?:\s+nuevamente|\s+familia|\s+iglesia|\s+hermanos?)?|abran? (sus )?biblias|mensaje de hoy|tema de hoy|hoy quiero|quiero compartir|la biblia|vers[íi]culo|cap[íi]tulo|quiero preguntar)/i.test(
        t
    );
}

function refineChapterStartFromTranscript(chapter: ChapterLite, transcript: Segment[]): number {
    const lookaheadMaxSec = Number(process.env.BOUNDARY_ADJ_SERMON_START_LOOKAHEAD_SEC ?? 900);
    const lookaheadCheckSec = Number(process.env.BOUNDARY_ADJ_SERMON_START_WINDOW_SEC ?? 160);
    const minSustainedSegments = Number(process.env.BOUNDARY_ADJ_SERMON_START_MIN_SEGMENTS ?? 4);

    const rangeEnd = Math.min(Number(chapter.end), Number(chapter.start) + lookaheadMaxSec);
    const inRange = transcript
        .filter((s) => Number.isFinite(s.start) && Number.isFinite(s.end))
        .filter((s) => s.end >= Number(chapter.start) && s.start <= rangeEnd)
        .sort((a, b) => a.start - b.start);
    if (inRange.length === 0) return Number(chapter.start);

    for (const seg of inRange) {
        const text = cleanText(seg.text);
        if (!text || isPrayerLikeText(text) || isWorshipLikeText(text)) continue;

        const window = inRange.filter((s) => s.start >= seg.start && s.start <= seg.start + lookaheadCheckSec);
        const nonLiturgical = window.filter((s) => {
            const st = cleanText(s.text);
            return Boolean(st) && !isPrayerLikeText(st) && !isWorshipLikeText(st);
        });
        const hasKickoffNearby = window.some((s) => isLikelySermonKickoffText(s.text));
        if (isLikelySermonKickoffText(text) || (hasKickoffNearby && nonLiturgical.length >= minSustainedSegments)) {
            return seg.start;
        }
    }

    return Number(chapter.start);
}

function clamp(n: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, n));
}

function safeParse<T>(raw: string, fallback: T): T {
    try {
        return JSON.parse(raw) as T;
    } catch {
        return fallback;
    }
}

function getOpenAIClient(): OpenAI | null {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) return null;
    return new OpenAI({ apiKey });
}

function snapToSegmentStart(segments: Segment[], value: number): number {
    if (segments.length === 0) return value;
    let best = segments[0].start;
    let bestD = Math.abs(best - value);
    for (const s of segments) {
        const d = Math.abs(s.start - value);
        if (d < bestD) {
            bestD = d;
            best = s.start;
        }
    }
    return best;
}

function snapToSegmentEnd(segments: Segment[], value: number): number {
    if (segments.length === 0) return value;
    let best = segments[0].end;
    let bestD = Math.abs(best - value);
    for (const s of segments) {
        const d = Math.abs(s.end - value);
        if (d < bestD) {
            bestD = d;
            best = s.end;
        }
    }
    return best;
}

function excerptParagraphsNearBounds(
    paragraphs: Array<{ start: number; end: number; text: string; chapter_title?: string }>,
    bounds: Boundaries,
    aroundSec = 220
): string {
    const nearStart = paragraphs
        .filter((p) => p.end >= bounds.start - aroundSec && p.start <= bounds.start + aroundSec)
        .slice(0, 10);
    const nearEnd = paragraphs
        .filter((p) => p.end >= bounds.end - aroundSec && p.start <= bounds.end + aroundSec)
        .slice(-10);

    const fmt = (p: { start: number; end: number; text: string; chapter_title?: string }) =>
        `[${p.start.toFixed(2)}-${p.end.toFixed(2)}] ${p.chapter_title ? `${p.chapter_title}: ` : ''}${cleanText(p.text).slice(0, 240)}`;

    return [
        'Near start:',
        ...nearStart.map(fmt),
        '',
        'Near end:',
        ...nearEnd.map(fmt)
    ].join('\n');
}

function deriveChapterSermonHint(
    chapters: ChapterLite[],
    transcript: Segment[]
): { start: number; end: number; confidence: number } | null {
    const sermonStartPool = chapters
        .filter((c) => Number.isFinite(c.start) && Number.isFinite(c.end) && c.end > c.start)
        .filter((c) => {
            const t = cleanText(c.type ?? '').toLowerCase();
            const title = cleanText(c.title ?? '').toLowerCase();
            return (
                t === 'sermon' ||
                title.includes('sermon introduction') ||
                title.includes('sermon transition') ||
                title.includes('sermón')
            );
        });
    const sermonEndPool = chapters
        .filter((c) => Number.isFinite(c.start) && Number.isFinite(c.end) && c.end > c.start)
        .filter((c) => {
            const t = cleanText(c.type ?? '').toLowerCase();
            const title = cleanText(c.title ?? '').toLowerCase();
            return t === 'sermon' || t === 'post_sermon_response' || title.includes('sermón');
        });
    if (sermonStartPool.length === 0 || sermonEndPool.length === 0) return null;
    const earliestStartChapter = sermonStartPool.reduce((best, cur) => (Number(cur.start) < Number(best.start) ? cur : best));
    const rawStart = Number(earliestStartChapter.start);
    const refinedStart = refineChapterStartFromTranscript(earliestStartChapter, transcript);
    const start = Math.max(rawStart, refinedStart);
    const end = Math.max(...sermonEndPool.map((c) => Number(c.end)));
    const confItems = [...sermonStartPool, ...sermonEndPool];
    let conf = confItems.reduce((a, c) => a + Number(c.confidence ?? 0.75), 0) / confItems.length;
    if (start - rawStart > 45) conf = Math.min(1, conf + 0.03);
    return { start, end, confidence: clamp(conf, 0, 1) };
}

function chapterExcerptNearEnd(chapters: ChapterLite[], end: number, preSec = 900, postSec = 900): string {
    return chapters
        .filter((c) => Number.isFinite(c.start) && Number.isFinite(c.end) && c.end > c.start)
        .filter((c) => c.end >= end - preSec && c.start <= end + postSec)
        .map((c, i) =>
            `${i + 1}. ${Number(c.start).toFixed(2)}-${Number(c.end).toFixed(2)} | ${cleanText(c.type ?? '')} | ${cleanText(
                c.title ?? ''
            )} | conf=${Number(c.confidence ?? 0).toFixed(2)}`
        )
        .join('\n');
}

function transcriptExcerptNearEnd(
    transcript: Segment[],
    end: number,
    preSec = 360,
    postSec = 300,
    maxLines = 140
): string {
    const start = Math.max(0, end - preSec);
    const finish = end + postSec;
    return transcript
        .filter((s) => s.end >= start && s.start <= finish)
        .slice(-maxLines)
        .map((s) => `[${s.start.toFixed(2)}-${s.end.toFixed(2)}] ${cleanText(s.text).slice(0, 180)}`)
        .join('\n');
}

async function llmRefineClosingEnd(
    openai: OpenAI,
    model: string,
    transcript: Segment[],
    chapters: ChapterLite[],
    currentStart: number,
    currentEnd: number,
    duration: number
): Promise<ClosingArbiterDecision | null> {
    const enabled = String(process.env.BOUNDARY_ADJ_ENABLE_CLOSING_ARBITER ?? 'true').toLowerCase() === 'true';
    if (!enabled) return null;

    const preSec = Number(process.env.BOUNDARY_ADJ_CLOSING_PRE_SEC ?? 360);
    const postSec = Number(process.env.BOUNDARY_ADJ_CLOSING_POST_SEC ?? 300);
    const maxShiftBack = Number(process.env.BOUNDARY_ADJ_CLOSING_MAX_BACKWARD_SEC ?? 240);
    const maxShiftForward = Number(process.env.BOUNDARY_ADJ_CLOSING_MAX_FORWARD_SEC ?? 360);

    const excerpt = transcriptExcerptNearEnd(transcript, currentEnd, preSec, postSec);
    if (!excerpt) return null;
    const chapterExcerpt = chapterExcerptNearEnd(chapters, currentEnd, 1200, 1200);
    const minEnd = Math.max(currentStart + 60, currentEnd - maxShiftBack);
    const maxEnd = Math.min(duration, currentEnd + maxShiftForward);

    const prompt = [
        'You are deciding the FINAL sermon closing boundary.',
        'Task: detect the service-state transition from sermon-response flow to non-sermon flow.',
        'Do NOT rely on chapter names only. Use semantic cues in text (altar call, invitation to sing, prayer close, benediction, social chatter, announcements/logistics).',
        'Prefer including: altar/decision call, closing prayer, benediction, response song if still tied to sermon response.',
        'Exclude: post-service social chatter, unrelated announcements, host logistics not tied to sermon response.',
        `Current bounds: start=${currentStart.toFixed(2)} end=${currentEnd.toFixed(2)}`,
        `Allowed output end range: ${minEnd.toFixed(2)}-${maxEnd.toFixed(2)}`,
        '',
        'Return JSON only:',
        '{"end_sec":number,"include_until_reason":"altar_call|closing_prayer|benediction|response_song|handoff|social_transition|other","confidence":number,"reason":"short"}',
        '',
        'Nearby chapters:',
        chapterExcerpt || '[none]',
        '',
        'Transcript around current end:',
        excerpt
    ].join('\n');

    const response = await openai.chat.completions.create({
        model,
        messages: [{ role: 'user', content: prompt }],
        response_format: { type: 'json_object' }
    });
    const content = response.choices[0]?.message?.content ?? '{}';
    const parsed = safeParse<any>(content, null);
    if (!parsed || !Number.isFinite(Number(parsed.end_sec))) return null;
    const endSec = clamp(Number(parsed.end_sec), minEnd, maxEnd);
    return {
        end_sec: endSec,
        include_until_reason: cleanText(String(parsed.include_until_reason ?? '')),
        confidence: clamp(Number(parsed.confidence ?? 0.5), 0, 1),
        reason: cleanText(String(parsed.reason ?? ''))
    };
}

function isAltarLikeChapter(ch: ChapterLite): boolean {
    const t = cleanText(ch.type ?? '').toLowerCase();
    const title = cleanText(ch.title ?? '').toLowerCase();
    const patterns = [
        'altar',
        'llamado',
        'invitación',
        'invitacion',
        'response',
        'respuesta',
        'appeal',
        'decision'
    ];
    return (
        t === 'altar_call' ||
        t === 'post_sermon_response' ||
        patterns.some((k) => title.includes(k))
    );
}

function isPrayerLikeChapter(ch: ChapterLite): boolean {
    const t = cleanText(ch.type ?? '').toLowerCase();
    const title = cleanText(ch.title ?? '').toLowerCase();
    return (
        t === 'prayer' ||
        t === 'closing' ||
        t === 'benediction' ||
        title.includes('prayer') ||
        title.includes('oración') ||
        title.includes('oracion') ||
        title.includes('benediction') ||
        title.includes('bendición') ||
        title.includes('bendicion')
    );
}

function isResponseMusicLikeChapter(ch: ChapterLite): boolean {
    const t = cleanText(ch.type ?? '').toLowerCase();
    const title = cleanText(ch.title ?? '').toLowerCase();
    return (
        t === 'congregational_music' ||
        t === 'special_music' ||
        (t === 'offering' && (title.includes('altar') || title.includes('response') || title.includes('llamado'))) ||
        title.includes('song') ||
        title.includes('canto') ||
        title.includes('alabanza') ||
        title.includes('worship')
    );
}

function extendEndForClosings(
    end: number,
    chapters: ChapterLite[],
    transcript: Segment[],
    opts: {
        includePrayer: boolean;
        includeBenediction: boolean;
        includeAltarCall: boolean;
        includeResponseMusic: boolean;
        maxExtendSec: number;
        chainMaxGapSec: number;
        chainMaxLookaheadSec: number;
    }
): number {
    const eligible = chapters
        .filter((c) => Number.isFinite(c.start) && Number.isFinite(c.end) && c.end > c.start)
        .filter((c) => c.start >= end - 1 && c.start <= end + opts.chainMaxLookaheadSec)
        .sort((a, b) => Number(a.start) - Number(b.start))
        .filter((c) => {
            if (opts.includeAltarCall && isAltarLikeChapter(c)) return true;
            if ((opts.includePrayer || opts.includeBenediction) && isPrayerLikeChapter(c)) return true;
            if (opts.includeResponseMusic && isResponseMusicLikeChapter(c)) return true;
            return false;
        })
        .filter((c) => c.start <= end + opts.maxExtendSec);

    if (!eligible.length) return end;

    let targetEnd = end;
    let cursor = end;
    for (const c of eligible) {
        const gap = Number(c.start) - cursor;
        if (gap > opts.chainMaxGapSec) break;
        targetEnd = Math.max(targetEnd, Number(c.end));
        cursor = Number(c.end);
    }

    if (targetEnd <= end) return end;

    // Snap to nearest transcript segment end to stay aligned with words.
    let best = targetEnd;
    let bestD = Math.abs(best - targetEnd);
    for (const s of transcript) {
        const d = Math.abs(s.end - targetEnd);
        if (d < bestD) {
            best = s.end;
            bestD = d;
        }
    }
    return best;
}

export async function adjudicateBoundariesWithSingleLlm(
    transcript: Segment[],
    local: Boundaries,
    options: { workDir: string }
): Promise<AdjudicationResult | null> {
    const openai = getOpenAIClient();
    if (!openai) return null;
    const workDir = options.workDir;
    const model = process.env.BOUNDARY_LLM_MODEL || process.env.ANALYZE_OPENAI_MODEL || 'gpt-4o-mini';
    const maxStartDrift = Number(process.env.BOUNDARY_ADJ_MAX_START_DRIFT_SEC ?? 600);
    const maxEndDrift = Number(process.env.BOUNDARY_ADJ_MAX_END_DRIFT_SEC ?? 240);
    const maxDrift = Number(process.env.BOUNDARY_ADJ_MAX_DRIFT_SEC ?? Math.max(240, maxStartDrift + maxEndDrift));
    const includePrayer = String(process.env.BOUNDARY_POLICY_INCLUDE_PRAYER ?? 'true').toLowerCase() === 'true';
    const includeBenediction = String(process.env.BOUNDARY_POLICY_INCLUDE_BENEDICTION ?? 'true').toLowerCase() === 'true';
    const includeAltarCall = String(process.env.BOUNDARY_POLICY_INCLUDE_ALTAR_CALL ?? 'true').toLowerCase() === 'true';
    const includeResponseMusic = String(process.env.BOUNDARY_POLICY_INCLUDE_RESPONSE_MUSIC ?? 'true').toLowerCase() === 'true';
    const closingMaxExtendSec = Number(process.env.BOUNDARY_POLICY_CLOSING_MAX_EXTEND_SEC ?? 180);
    const chainMaxGapSec = Number(process.env.BOUNDARY_POLICY_POST_CHAIN_MAX_GAP_SEC ?? 45);
    const chainMaxLookaheadSec = Number(process.env.BOUNDARY_POLICY_POST_CHAIN_MAX_LOOKAHEAD_SEC ?? 900);

    const targetedPath = path.join(workDir, 'sermon.boundaries.targeted-diarization.json');
    const analysisPath = path.join(workDir, 'analysis.doc.json');
    const targeted = fs.existsSync(targetedPath) ? safeParse<any>(fs.readFileSync(targetedPath, 'utf8'), {}) : {};
    const analysis = fs.existsSync(analysisPath) ? safeParse<any>(fs.readFileSync(analysisPath, 'utf8'), {}) : {};

    const scoredStart = Array.isArray(targeted?.candidate_scoring?.start) ? targeted.candidate_scoring.start.slice(0, 4) : [];
    const scoredEnd = Array.isArray(targeted?.candidate_scoring?.end) ? targeted.candidate_scoring.end.slice(0, 4) : [];
    const chapters = (Array.isArray(analysis?.chapters) ? analysis.chapters : []) as ChapterLite[];
    const paragraphs = Array.isArray(analysis?.paragraphs) ? analysis.paragraphs : [];
    const cues = analysis?.cues ?? null;
    const duration = transcript.length > 0 ? transcript[transcript.length - 1].end : local.end;
    const chapterHint = deriveChapterSermonHint(chapters, transcript);
    const chapterWeight = clamp(Number(process.env.BOUNDARY_ADJ_CHAPTER_WEIGHT ?? 0.72), 0, 1);
    const chapterStartWeight = clamp(
        Number(process.env.BOUNDARY_ADJ_CHAPTER_START_WEIGHT ?? Math.max(chapterWeight, 0.95)),
        0,
        1
    );
    const chapterEndWeight = clamp(
        Number(process.env.BOUNDARY_ADJ_CHAPTER_END_WEIGHT ?? Math.max(0.60, Math.min(chapterWeight, 0.78))),
        0,
        1
    );

    const chapterSummary = chapters
        .map((c: any, i: number) => `${i + 1}. ${c.title} | ${c.type} | ${Number(c.start).toFixed(2)}-${Number(c.end).toFixed(2)} | conf=${Number(c.confidence ?? 0).toFixed(2)}`)
        .join('\n');

    const paragraphExcerpt = excerptParagraphsNearBounds(paragraphs, local);
    const prompt = [
        'You are the FINAL boundary adjudicator for sermon clipping.',
        'Decide final sermon bounds using ALL provided evidence.',
        'Return JSON only:',
        '{"start":number,"end":number,"confidence":number,"decision":"keep_local|adjust","reason":"short"}',
        '',
        'Hard policy:',
        '- Keep prayer closing lines (e.g., "Amén, señor, amén.") inside sermon if they are part of closing prayer.',
        '- Include post-sermon altar/decision call when it is contiguous and thematically tied to the sermon.',
        '- Include response song(s) when they are part of that call/closing flow.',
        '- Exclude post-service social/farewell chatter.',
        '- Prefer chapter-labeled sermon spans over local scoring when they are coherent, especially for START.',
        '- Prefer local evidence only when chapter evidence is weak or contradictory.',
        `- Start drift soft limit: ${maxStartDrift}s, end drift soft limit: ${maxEndDrift}s, total drift cap: ${maxDrift}s.`,
        '',
        `Local bounds: start=${local.start.toFixed(2)}, end=${local.end.toFixed(2)}`,
        chapterHint
            ? `Chapter sermon hint: start=${chapterHint.start.toFixed(2)}, end=${chapterHint.end.toFixed(2)}, confidence=${chapterHint.confidence.toFixed(2)}`
            : 'Chapter sermon hint: none',
        `Chapter weight hint: start=${chapterStartWeight.toFixed(2)}, end=${chapterEndWeight.toFixed(2)}`,
        `Duration: ${duration.toFixed(2)}`,
        `Audio cues summary: ${JSON.stringify(cues)}`,
        '',
        'Candidate scoring (start):',
        JSON.stringify(scoredStart, null, 2),
        'Candidate scoring (end):',
        JSON.stringify(scoredEnd, null, 2),
        '',
        'Chapters:',
        chapterSummary || '[none]',
        '',
        'Paragraph excerpts:',
        paragraphExcerpt || '[none]'
    ].join('\n');

    const response = await openai.chat.completions.create({
        model,
        messages: [{ role: 'user', content: prompt }],
        response_format: { type: 'json_object' }
    });

    const content = response.choices[0]?.message?.content ?? '{}';
    const parsed = safeParse<any>(content, {});
    const rawStart = Number(parsed.start);
    const rawEnd = Number(parsed.end);
    const confidence = clamp(Number(parsed.confidence ?? 0.5), 0, 1);
    const decision = parsed.decision === 'adjust' ? 'adjust' : 'keep_local';
    const reason = cleanText(parsed.reason ?? '');
    if (!Number.isFinite(rawStart) || !Number.isFinite(rawEnd) || rawEnd <= rawStart) {
        return {
            start: local.start,
            end: local.end,
            confidence: 0,
            decision: 'keep_local',
            reason: 'invalid_llm_response',
            rejected: 'invalid_bounds'
        };
    }

    let snappedStart = snapToSegmentStart(transcript, clamp(rawStart, 0, duration));
    let snappedEnd = snapToSegmentEnd(transcript, clamp(rawEnd, snappedStart + 30, duration));

    const chapterHardStartConf = Number(process.env.BOUNDARY_ADJ_CHAPTER_HARD_START_CONFIDENCE ?? 0.72);
    const chapterHardEndConf = Number(process.env.BOUNDARY_ADJ_CHAPTER_HARD_END_CONFIDENCE ?? 0.8);
    const chapterHardMode = String(process.env.BOUNDARY_ADJ_CHAPTER_HARD_MODE ?? 'true').toLowerCase() === 'true';
    if (chapterHint) {
        const confidenceFactor = chapterHint.confidence >= 0.65 ? 1 : 0.6;
        const effectiveStartWeight = chapterStartWeight * confidenceFactor;
        const effectiveEndWeight = chapterEndWeight * confidenceFactor;
        const blendedStart = effectiveStartWeight * chapterHint.start + (1 - effectiveStartWeight) * snappedStart;
        const blendedEnd = effectiveEndWeight * chapterHint.end + (1 - effectiveEndWeight) * snappedEnd;
        const chapterStartSnapped = snapToSegmentStart(transcript, clamp(chapterHint.start, 0, duration));
        const chapterEndSnapped = snapToSegmentEnd(transcript, clamp(chapterHint.end, chapterStartSnapped + 30, duration));

        // When chapter confidence is strong, treat chapter boundaries as authoritative.
        if (chapterHardMode && chapterHint.confidence >= chapterHardStartConf) {
            snappedStart = chapterStartSnapped;
        } else {
            snappedStart = snapToSegmentStart(transcript, clamp(blendedStart, 0, duration));
        }

        if (chapterHardMode && chapterHint.confidence >= chapterHardEndConf) {
            snappedEnd = Math.max(snappedStart + 30, chapterEndSnapped);
        } else {
            snappedEnd = snapToSegmentEnd(transcript, clamp(blendedEnd, snappedStart + 30, duration));
        }
    }

    // Optionally extend end to include altar call / prayer / benediction if near.
    const extendedEnd = extendEndForClosings(snappedEnd, chapters, transcript, {
        includePrayer,
        includeBenediction,
        includeAltarCall,
        includeResponseMusic,
        maxExtendSec: closingMaxExtendSec,
        chainMaxGapSec,
        chainMaxLookaheadSec
    });

    snappedEnd = Math.max(snappedEnd, extendedEnd);

    const closingDecision = await llmRefineClosingEnd(
        openai,
        model,
        transcript,
        chapters,
        snappedStart,
        snappedEnd,
        duration
    );
    if (closingDecision && Number.isFinite(closingDecision.end_sec)) {
        const snapped = snapToSegmentEnd(transcript, clamp(closingDecision.end_sec, snappedStart + 30, duration));
        snappedEnd = Math.max(snappedStart + 30, snapped);
    }

    const startDrift = Math.abs(snappedStart - local.start);
    const endDrift = Math.abs(snappedEnd - local.end);
    const drift = startDrift + endDrift;
    const chapterStrongThreshold = Number(process.env.BOUNDARY_ADJ_CHAPTER_STRONG_CONFIDENCE ?? 0.78);
    const chapterStrong = Boolean(chapterHint && chapterHint.confidence >= chapterStrongThreshold);
    const allowedStartDrift = chapterStrong
        ? Number(process.env.BOUNDARY_ADJ_MAX_START_DRIFT_SEC_CHAPTER_STRONG ?? Math.max(maxStartDrift, 1800))
        : maxStartDrift;
    const allowedEndDrift = chapterStrong
        ? Number(process.env.BOUNDARY_ADJ_MAX_END_DRIFT_SEC_CHAPTER_STRONG ?? Math.max(maxEndDrift, 900))
        : maxEndDrift;
    const allowedTotalDrift = chapterStrong
        ? Number(process.env.BOUNDARY_ADJ_MAX_DRIFT_SEC_CHAPTER_STRONG ?? Math.max(maxDrift, allowedStartDrift + allowedEndDrift))
        : maxDrift;

    if (startDrift > allowedStartDrift || endDrift > allowedEndDrift || drift > allowedTotalDrift) {
        return {
            start: local.start,
            end: local.end,
            confidence,
            decision: 'keep_local',
            reason: reason || 'drift_exceeds_limit',
            rejected: `drift_exceeds_limit:start=${startDrift.toFixed(2)}/${allowedStartDrift},end=${endDrift.toFixed(
                2
            )}/${allowedEndDrift},total=${drift.toFixed(2)}/${allowedTotalDrift}${chapterStrong ? ',chapter_strong=true' : ''}`
        };
    }

    const finalDecision: 'keep_local' | 'adjust' =
        Math.abs(snappedStart - local.start) > 0.01 || Math.abs(snappedEnd - local.end) > 0.01 ? 'adjust' : 'keep_local';
    const closingReasonPart =
        closingDecision && cleanText(closingDecision.reason ?? '')
            ? `closing_arbiter(${cleanText(closingDecision.include_until_reason ?? 'n/a')}): ${cleanText(closingDecision.reason ?? '')}`
            : '';
    const baseReason = reason || 'adjudicated';
    const finalReason = closingReasonPart ? `${baseReason}; ${closingReasonPart}` : baseReason;

    return {
        start: snappedStart,
        end: snappedEnd,
        confidence,
        decision: finalDecision,
        reason: finalReason
    };
}
