import fs from 'fs';
import path from 'path';
import { OpenAI } from 'openai';
import type { SermonBoundaries } from './boundaries';
import { emitLlmCueEvents } from './llm-cue-telemetry';

interface Segment {
    start: number;
    end: number;
    text: string;
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

type ChapterType =
    | 'pre_service'
    | 'welcome'
    | 'announcement'
    | 'call_to_worship'
    | 'opening_prayer'
    | 'congregational_music'
    | 'scripture_reading'
    | 'child_presentation'
    | 'children_segment'
    | 'testimony'
    | 'offering'
    | 'prayer'
    | 'sermon'
    | 'ordinance_or_rite'
    | 'post_sermon_response'
    | 'closing'
    | 'post_service'
    | 'other';

interface Chapter {
    start: number;
    end: number;
    title: string;
    type: ChapterType;
    confidence: number;
    reason?: string;
}

const ALLOWED_CHAPTER_TYPES: ChapterType[] = [
    'pre_service',
    'welcome',
    'announcement',
    'call_to_worship',
    'opening_prayer',
    'congregational_music',
    'scripture_reading',
    'child_presentation',
    'children_segment',
    'testimony',
    'offering',
    'prayer',
    'sermon',
    'ordinance_or_rite',
    'post_sermon_response',
    'closing',
    'post_service',
    'other'
];

interface ChapterSignal {
    speech_ratio: number;
    music_ratio: number;
    noenergy_ratio: number;
    has_music: boolean;
    has_long_pause: boolean;
}

interface ParagraphEntry {
    chapter_index: number;
    chapter_title: string;
    start: number;
    end: number;
    text: string;
}

interface AnalysisDoc {
    version: string;
    generated_at: string;
    boundaries: SermonBoundaries;
    chapters: Chapter[];
    chapter_signals: ChapterSignal[];
    cues: {
        music_sections: number;
        noenergy_sections: number;
        long_pause_sections: number;
    };
    ocr: {
        source: string | null;
        segments: number;
        scene_cuts: number;
        speaker_name_cues: number;
        bible_verse_cues: number;
        sermon_title_cues: number;
        song_lyric_cues: number;
    };
    ocr_segments?: OcrEvent[];
    paragraphs: ParagraphEntry[];
    inputs: {
        transcript_segments: number;
        audio_events_source: string | null;
        ocr_events_source: string | null;
    };
}

let warnedLegacyAnalysisOcrDisabled = false;

function cleanText(text: string): string {
    return String(text ?? '')
        .replace(/\s+/g, ' ')
        .trim();
}

function isSabbathGreetingCue(text: string): boolean {
    const t = cleanText(text);
    if (!t) return false;
    return /(buenos d[ií]as[, ]+feliz s[áa]bado|feliz s[áa]bado(?:\s+(?:nuevamente|familia|iglesia|hermanos?|a todos))?)/i.test(
        t
    );
}

function inferTypeFromTextHeuristic(text: string, fallback: ChapterType): ChapterType {
    const t = cleanText(text).toLowerCase();
    if (!t) return fallback;
    if (/(abran? (sus )?biblias|mensaje de hoy|tema de hoy|la biblia|vers[íi]culo|cap[íi]tulo|hoy quiero|quiero compartir|quiero preguntar)/i.test(t)) {
        return 'sermon';
    }
    if (/(oremos|oraci[oó]n|inclinemos|inclinar.*rostro|arrodill|te lo pedimos|en el nombre de (tu hijo )?cristo jes[uú]s|\bam[eé]n\b)/i.test(t)) {
        return 'prayer';
    }
    if (/(vamos a cantar|cantemos|equipo de alabanza|alabemos|worship|himno|coro|reproclamamos|en lo alto est[aá]s)/i.test(t)) {
        return 'congregational_music';
    }
    if (/(diezmo[s]?|ofrenda[s]?|mayordom[ií]a|regresar lo que te pertenece)/i.test(t)) {
        return 'offering';
    }
    if (/(anuncios?|announcement|actividades|recuerden|les invitamos|esta tarde|despu[eé]s del culto|hoy tenemos|les queremos agradecer)/i.test(t)) {
        return 'announcement';
    }
    if (isSabbathGreetingCue(t)) {
        return 'welcome';
    }
    return fallback;
}

function chapterTitleForType(type: ChapterType): string {
    switch (type) {
        case 'sermon':
            return 'Sermón principal';
        case 'prayer':
            return 'Oración congregacional';
        case 'congregational_music':
            return 'Alabanza congregacional';
        case 'offering':
            return 'Ofrendas y diezmos';
        case 'announcement':
            return 'Anuncios';
        case 'welcome':
            return 'Bienvenida';
        default:
            return 'Transición';
    }
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

async function ensureOcrEvents(
    workDir: string,
    durationSec: number,
    videoPath?: string
): Promise<OcrEventPassResult | null> {
    void workDir;
    void durationSec;
    void videoPath;
    const legacyRequested = String(process.env.ANALYSIS_ENABLE_OCR_SIGNALS ?? 'false').toLowerCase() === 'true';
    if (legacyRequested && !warnedLegacyAnalysisOcrDisabled) {
        warnedLegacyAnalysisOcrDisabled = true;
        console.warn(
            '[analysis-doc] Legacy ocr.events* analysis pass is deprecated and disabled. Use slide.events.json cues instead.'
        );
    }
    return null;
}

function getOpenAIClient(): OpenAI | null {
    const key = process.env.OPENAI_API_KEY;
    if (!key) return null;
    return new OpenAI({ apiKey: key });
}

function formatSec(sec: number): string {
    const s = Math.max(0, Math.floor(sec));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const ss = s % 60;
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
}

function trimForPrompt(text: string, maxLen = 160): string {
    const t = cleanText(text);
    if (t.length <= maxLen) return t;
    return `${t.slice(0, maxLen - 1)}…`;
}

function isSpeechLikeLabel(label: string): boolean {
    const l = label.toLowerCase();
    return l === 'speech' || l === 'male' || l === 'female';
}

function getWindowDurations(events: AudioEvent[], start: number, end: number): Map<string, number> {
    const totals = new Map<string, number>();
    for (const e of events) {
        const ol = Math.max(0, Math.min(end, e.end) - Math.max(start, e.start));
        if (ol <= 0) continue;
        totals.set(e.label, (totals.get(e.label) ?? 0) + ol);
    }
    return totals;
}

function buildBins(
    transcript: Segment[],
    audioEvents: AudioEvent[] | null,
    durationSec: number,
    binSec: number
) {
    const bins: Array<{ idx: number; start: number; end: number; text: string; sp: number; mu: number; sil: number }> = [];
    let idx = 0;
    for (let t = 0; t < durationSec; t += binSec) {
        const start = t;
        const end = Math.min(durationSec, t + binSec);
        const segText = transcript
            .filter((s) => s.end >= start && s.start <= end)
            .map((s) => s.text)
            .join(' ');
        const text = trimForPrompt(segText, 180);
        let sp = 0;
        let mu = 0;
        let sil = 0;
        if (audioEvents && audioEvents.length > 0) {
            const d = getWindowDurations(audioEvents, start, end);
            const total = Math.max(0.001, end - start);
            sp =
                [...d.entries()]
                    .filter(([k]) => isSpeechLikeLabel(k))
                    .reduce((acc, [, v]) => acc + v, 0) / total;
            mu = (d.get('music') ?? 0) / total;
            sil = (d.get('noenergy') ?? 0) / total;
        }
        bins.push({
            idx,
            start,
            end,
            text,
            sp: Number(sp.toFixed(3)),
            mu: Number(mu.toFixed(3)),
            sil: Number(sil.toFixed(3))
        });
        idx += 1;
    }
    return bins;
}

function validateChapter(raw: any, durationSec: number): Chapter | null {
    const start = Number(raw?.start);
    const end = Number(raw?.end);
    const title = cleanText(raw?.title ?? '');
    const typeRaw = cleanText(raw?.type ?? 'other').toLowerCase();
    const confidenceRaw = Number(raw?.confidence);
    const legacyTypeMap: Record<string, ChapterType> = {
        song: 'congregational_music',
        prayer: 'prayer',
        announcements: 'announcement',
        transition: 'other',
        child_segment: 'child_presentation',
        child_presentation: 'child_presentation',
        children_segment: 'child_presentation'
    };
    const normalizedTypeRaw = legacyTypeMap[typeRaw] ?? typeRaw;
    const type: ChapterType = ALLOWED_CHAPTER_TYPES.includes(normalizedTypeRaw as ChapterType)
        ? (normalizedTypeRaw as ChapterType)
        : 'other';
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return null;
    if (!title) return null;
    return {
        start: clamp(start, 0, durationSec),
        end: clamp(end, 0, durationSec),
        title,
        type,
        confidence: Number.isFinite(confidenceRaw) ? clamp(confidenceRaw, 0, 1) : 0.6,
        reason: cleanText(raw?.reason ?? '')
    };
}

async function inferChaptersWithLlm(
    transcript: Segment[],
    boundaries: SermonBoundaries,
    audioEvents: AudioEvent[] | null,
    ocrEvents: OcrEventPassResult | null,
    durationSec: number,
    workDir?: string
): Promise<Chapter[] | null> {
    const openai = getOpenAIClient();
    if (!openai) return null;

    const cachePath = workDir ? path.join(workDir, 'analysis.chapters.llm.json') : '';
    if (cachePath && fs.existsSync(cachePath)) {
        const cached = safeJsonParse<{ chapters?: Chapter[] } | null>(fs.readFileSync(cachePath, 'utf8'), null);
        if (cached && Array.isArray(cached.chapters) && cached.chapters.length > 0) {
            const normalizedCached = cached.chapters
                .map((c) => validateChapter(c, durationSec))
                .filter((c): c is Chapter => Boolean(c))
                .sort((a, b) => a.start - b.start);
            if (normalizedCached.length > 0) return normalizedCached;
        }
    }

    const model = process.env.BOUNDARY_LLM_MODEL || process.env.ANALYZE_OPENAI_MODEL || 'gpt-4o-mini';
    const binSec = Number(process.env.ANALYSIS_CHAPTER_BIN_SEC ?? 45);
    const bins = buildBins(transcript, audioEvents, durationSec, Math.max(20, binSec));
    const binLines = bins
        .map((b) => `B${b.idx}|${formatSec(b.start)}-${formatSec(b.end)}|sp=${b.sp}|mu=${b.mu}|sil=${b.sil}|${b.text || '[empty]'}`)
        .join('\n');
    const ocrLines = buildOcrPromptLines(ocrEvents).join('\n');

    const prompt = [
        'Create chapter timeline for a church livestream transcript.',
        'Return JSON only with schema:',
        `{"chapters":[{"start":number,"end":number,"title":string,"type":"${ALLOWED_CHAPTER_TYPES.join('|')}","confidence":number,"reason":"short"}]}`,
        'Rules:',
        '- Keep chapter count between 4 and 14.',
        '- Use clear transitions: music, prayer, sermon start, sermon end, offering, announcements, child presentation.',
        '- In Hispanic SDA services, "feliz sábado" greetings are strong section-boundary cues (even if speaker changes).',
        '- OCR cues from lower-thirds/slides/lyrics are high-value contextual signals: speaker names, sermon title, Bible verses, worship lyrics.',
        '- Do not start a sermon chapter inside prayer or worship lyrics.',
        '- If prayer/worship happens before preaching, start sermon at the first sustained preaching statement.',
        '- The sermon chapter should align with these local bounds unless strong evidence says otherwise.',
        `local_sermon_start=${boundaries.start.toFixed(3)} local_sermon_end=${boundaries.end.toFixed(3)}`,
        `duration_sec=${durationSec.toFixed(3)}`,
        'Bins:',
        binLines,
        '',
        'OCR:',
        ocrLines || '[none]'
    ].join('\n');

    const response = await openai.chat.completions.create({
        model,
        messages: [{ role: 'user', content: prompt }],
        response_format: { type: 'json_object' }
    });

    const content = response.choices[0]?.message?.content ?? '{}';
    const parsed = safeJsonParse<{ chapters?: any[] } | null>(content, null);
    if (!parsed || !Array.isArray(parsed.chapters)) return null;

    const chapters = parsed.chapters
        .map((c) => validateChapter(c, durationSec))
        .filter((c): c is Chapter => Boolean(c))
        .sort((a, b) => a.start - b.start);

    if (chapters.length === 0) return null;
    await emitLlmCueEvents(
        chapters.map((c) => ({
            source_pass: 'analysis_chapters_stage1',
            model,
            section_type: c.type,
            cue_kind: 'chapter_label',
            cue_text: c.title,
            cue_time_sec: c.start,
            confidence: c.confidence,
            metadata: { end_sec: c.end, reason: c.reason ?? null }
        })),
        workDir
    );
    if (cachePath) fs.writeFileSync(cachePath, JSON.stringify({ chapters }, null, 2));
    return chapters;
}

async function renameLowConfidenceChapters(
    transcript: Segment[],
    chapters: Chapter[],
    durationSec: number,
    workDir?: string
): Promise<Chapter[]> {
    const openai = getOpenAIClient();
    if (!openai) return chapters;
    const threshold = Number(process.env.ANALYSIS_CHAPTER_LOW_CONF_THRESHOLD ?? 0.78);
    const model = process.env.BOUNDARY_LLM_MODEL || process.env.ANALYZE_OPENAI_MODEL || 'gpt-4o-mini';
    const out = [...chapters];

    const heuristicTitle = (idx: number): string => {
        const cur = out[idx];
        const prev = idx > 0 ? out[idx - 1] : null;
        const next = idx + 1 < out.length ? out[idx + 1] : null;
        if (cur.type === 'congregational_music' && (prev?.type === 'sermon' || next?.type === 'sermon')) {
            return 'Continuación del sermón';
        }
        if (cur.type === 'other' && prev?.type === 'sermon') return 'Transición posterior al sermón';
        if (cur.type === 'prayer') return 'Oración congregacional';
        return 'Sección por revisar';
    };

    for (let i = 0; i < out.length; i++) {
        const ch = out[i];
        if (ch.confidence >= threshold) continue;
        const winStart = clamp(ch.start - 90, 0, durationSec);
        const winEnd = clamp(ch.end + 90, 0, durationSec);
        const lines = transcript
            .filter((s) => s.end >= winStart && s.start <= winEnd)
            .slice(0, 180)
            .map((s, idx) => `L${idx}|${formatSec(s.start)}-${formatSec(s.end)}|${trimForPrompt(s.text, 150)}`)
            .join('\n');
        const prompt = [
            'Name this church service section with a concise human-readable title (2-8 words).',
            'Return JSON only: {"title":string}',
            `current_type=${ch.type}`,
            `current_title=${ch.title}`,
            `section_start=${ch.start.toFixed(3)} section_end=${ch.end.toFixed(3)}`,
            'Timeline:',
            lines || '[empty]'
        ].join('\n');
        try {
            const response = await openai.chat.completions.create({
                model,
                messages: [{ role: 'user', content: prompt }],
                response_format: { type: 'json_object' }
            });
            const content = response.choices[0]?.message?.content ?? '{}';
            const parsed = safeJsonParse<{ title?: string } | null>(content, null);
            const title = cleanText(parsed?.title ?? '');
            if (title && title.toLowerCase() !== ch.title.toLowerCase()) {
                out[i] = { ...ch, title };
                await emitLlmCueEvents(
                    [
                        {
                            source_pass: 'analysis_chapter_rename',
                            model,
                            section_type: ch.type,
                            cue_kind: 'low_conf_rename',
                            cue_text: title,
                            cue_time_sec: ch.start,
                            confidence: ch.confidence,
                            metadata: { old_title: ch.title, reason: ch.reason ?? null }
                        }
                    ],
                    workDir
                );
            } else {
                out[i] = { ...ch, title: heuristicTitle(i) };
            }
        } catch {
            out[i] = { ...ch, title: heuristicTitle(i) };
        }
    }
    return out;
}

function fallbackChapters(boundaries: SermonBoundaries, durationSec: number): Chapter[] {
    const introEnd = Math.max(0, Math.min(boundaries.start, boundaries.start - 1));
    const outroStart = Math.min(durationSec, boundaries.end);
    const chapters: Chapter[] = [];
    if (introEnd > 0) {
        chapters.push({
            start: 0,
            end: introEnd,
            title: 'Introducción y programa previo',
            type: 'welcome',
            confidence: 0.6,
            reason: 'fallback'
        });
    }
    chapters.push({
        start: boundaries.start,
        end: boundaries.end,
        title: 'Sermón principal',
        type: 'sermon',
        confidence: 0.9,
        reason: 'local-boundaries'
    });
    if (outroStart < durationSec) {
        chapters.push({
            start: outroStart,
            end: durationSec,
            title: 'Cierre y post-sermón',
            type: 'post_service',
            confidence: 0.6,
            reason: 'fallback'
        });
    }
    return chapters;
}

function chapterForTime(chapters: Chapter[], t: number): number {
    for (let i = 0; i < chapters.length; i++) {
        const c = chapters[i];
        const isLast = i === chapters.length - 1;
        if (t >= c.start && (isLast ? t <= c.end : t < c.end)) return i;
    }
    let nearest = 0;
    let best = Number.POSITIVE_INFINITY;
    for (let i = 0; i < chapters.length; i++) {
        const d = Math.min(Math.abs(t - chapters[i].start), Math.abs(t - chapters[i].end));
        if (d < best) {
            best = d;
            nearest = i;
        }
    }
    return nearest;
}

function buildParagraphs(transcript: Segment[], chapters: Chapter[]): ParagraphEntry[] {
    const paragraphs: ParagraphEntry[] = [];
    let current: Segment[] = [];

    const flush = () => {
        if (current.length === 0) return;
        const start = current[0].start;
        const end = current[current.length - 1].end;
        // Use paragraph start as anchoring time so boundary-edge prayer closings
        // like "Amén" stay attached to the section they close.
        const idx = chapterForTime(chapters, start);
        paragraphs.push({
            chapter_index: idx,
            chapter_title: chapters[idx]?.title ?? 'Sin capítulo',
            start,
            end,
            text: current.map((s) => cleanText(s.text)).join(' ')
        });
        current = [];
    };

    for (const s of transcript) {
        const text = cleanText(s.text);
        if (!text) continue;
        const prev = current.length ? current[current.length - 1] : null;
        const gap = prev ? s.start - prev.end : 0;
        const breakByPunctuation = prev ? /[.!?]["')\]]?$/.test(prev.text) : true;
        if (current.length > 0 && (gap > 1.4 || breakByPunctuation)) flush();
        current.push({ ...s, text });
    }
    flush();
    return paragraphs;
}

function buildCueSummary(events: AudioEvent[] | null) {
    if (!events || events.length === 0) {
        return {
            music_sections: 0,
            noenergy_sections: 0,
            long_pause_sections: 0
        };
    }
    const music = events.filter((e) => e.label.toLowerCase() === 'music').length;
    const noenergy = events.filter((e) => e.label.toLowerCase() === 'noenergy').length;
    const longPause = events.filter((e) => e.label.toLowerCase() === 'noenergy' && e.end - e.start >= 4).length;
    return {
        music_sections: music,
        noenergy_sections: noenergy,
        long_pause_sections: longPause
    };
}

function buildOcrSummary(ocr: OcrEventPassResult | null) {
    const segments = ocr?.segments ?? [];
    const count = (type: OcrEvent['type']) => segments.filter((s) => s.type === type).length;
    return {
        source: ocr?.source ?? null,
        segments: segments.length,
        scene_cuts: Array.isArray(ocr?.scene_cuts_sec) ? ocr!.scene_cuts_sec.length : 0,
        speaker_name_cues: count('speaker_name'),
        bible_verse_cues: count('bible_verse'),
        sermon_title_cues: count('sermon_title'),
        song_lyric_cues: count('song_lyric')
    };
}

function buildOcrPromptLines(ocr: OcrEventPassResult | null): string[] {
    if (!ocr || !Array.isArray(ocr.segments) || ocr.segments.length === 0) return [];
    const importantTypes = new Set<OcrEvent['type']>(['speaker_name', 'bible_verse', 'sermon_title', 'song_lyric']);
    const important = ocr.segments
        .filter((s) => importantTypes.has(s.type))
        .sort((a, b) => a.start - b.start)
        .slice(0, 80)
        .map(
            (s, i) =>
                `O${i}|${formatSec(s.start)}-${formatSec(s.end)}|${s.type}|${s.region}|cut=${s.near_scene_cut ? 1 : 0}|${trimForPrompt(s.text, 120)}`
        );
    const sceneCuts = (ocr.scene_cuts_sec ?? [])
        .slice(0, 120)
        .map((t) => formatSec(t))
        .join(', ');
    const header = [
        `OCR source=${ocr.source} segments=${ocr.segments.length} sample_sec=${Number(ocr.sample_sec ?? 0).toFixed(2)}`,
        sceneCuts ? `OCR scene_cuts=${sceneCuts}` : 'OCR scene_cuts=[none]'
    ];
    return header.concat(important.length > 0 ? important : ['OCR cues=[none-important]']);
}

function computeChapterSignal(chapter: Chapter, events: AudioEvent[] | null): ChapterSignal {
    if (!events || events.length === 0) {
        return {
            speech_ratio: 0,
            music_ratio: 0,
            noenergy_ratio: 0,
            has_music: false,
            has_long_pause: false
        };
    }
    const d = getWindowDurations(events, chapter.start, chapter.end);
    const total = Math.max(0.001, chapter.end - chapter.start);
    const speech =
        [...d.entries()]
            .filter(([k]) => isSpeechLikeLabel(k))
            .reduce((acc, [, v]) => acc + v, 0) / total;
    const music = (d.get('music') ?? 0) / total;
    const noenergy = (d.get('noenergy') ?? 0) / total;
    const hasMusic = music >= 0.2;
    const hasLongPause = noenergy >= 0.08;
    return {
        speech_ratio: Number(speech.toFixed(3)),
        music_ratio: Number(music.toFixed(3)),
        noenergy_ratio: Number(noenergy.toFixed(3)),
        has_music: hasMusic,
        has_long_pause: hasLongPause
    };
}

function calibrateChaptersBySignals(chapters: Chapter[], signals: ChapterSignal[]): Chapter[] {
    return chapters.map((ch, i) => {
        const s = signals[i];
        if (!s) return ch;
        let confidence = ch.confidence;
        let reason = ch.reason ?? '';

        // Strong contradiction: chapter marked as music but no music detected.
        if (ch.type === 'congregational_music' && s.music_ratio < 0.12 && s.speech_ratio > 0.7) {
            confidence = Math.min(confidence, 0.45);
            reason = `${reason ? `${reason};` : ''}signal_mismatch_music`;
        }
        // Sermon with mostly music is suspicious.
        if (ch.type === 'sermon' && s.music_ratio > 0.35) {
            confidence = Math.min(confidence, 0.55);
            reason = `${reason ? `${reason};` : ''}signal_mismatch_sermon`;
        }
        // Opening/closing style buckets with mostly silence are weak.
        if ((ch.type === 'welcome' || ch.type === 'closing') && s.noenergy_ratio > 0.65) {
            confidence = Math.min(confidence, 0.5);
            reason = `${reason ? `${reason};` : ''}signal_mismatch_silence`;
        }

        return {
            ...ch,
            confidence: Number(Math.max(0, Math.min(1, confidence)).toFixed(2)),
            reason
        };
    });
}

function relabelChaptersBySignals(
    transcript: Segment[],
    chapters: Chapter[],
    signals: ChapterSignal[]
): Chapter[] {
    const hasSongCue = (text: string) =>
        /(vamos a cantar|cantemos|equipo de alabanza|alabemos|adoraci[oó]n|himno|coro)/i.test(text);
    const prayerCueCount = (text: string) => {
        const cues = [
            /oremos/gi,
            /oraci[oó]n/gi,
            /inclinen.*rostro/gi,
            /padre nuestro/gi,
            /en el nombre de jes[uú]s/gi,
            /te lo pedimos en cristo jes[uú]s/gi,
            /\bam[eé]n\b/gi
        ];
        return cues.reduce((acc, rx) => acc + ((text.match(rx) || []).length > 0 ? 1 : 0), 0);
    };
    const hasPreachingCue = (text: string) =>
        /(la biblia|cap[íi]tulo|vers[íi]culo|hoy quiero decirte|historia|serm[oó]n|predicar|mensaje de hoy)/i.test(text);
    const hasOfferingCue = (text: string) =>
        /(diezmo[s]?|ofrenda[s]?|regresar lo que te pertenece|mayordom[ií]a)/i.test(text);

    const chapterText = (start: number, end: number) =>
        transcript
            .filter((s) => s.end >= start && s.start <= end)
            .map((s) => s.text)
            .join(' ');

    const out = [...chapters];
    for (let i = 0; i < out.length; i++) {
        const ch = out[i];
        const s = signals[i];
        if (!s) continue;
        const prev = i > 0 ? out[i - 1] : null;
        const next = i + 1 < out.length ? out[i + 1] : null;
        const txt = chapterText(ch.start, ch.end);

        if (ch.type === 'congregational_music' && s.music_ratio < 0.12 && s.speech_ratio > 0.7 && !hasSongCue(txt)) {
            const newType: ChapterType = prev?.type === 'sermon' || next?.type === 'sermon' ? 'sermon' : 'other';
            out[i] = {
                ...ch,
                type: newType,
                reason: `${ch.reason ? `${ch.reason};` : ''}relabel_from_music_to_${newType}`
            };
            continue;
        }

        const prayerScore = prayerCueCount(txt);
        const shortSection = (ch.end - ch.start) <= 360;
        const prayerEligibleType =
            ch.type === 'other' || ch.type === 'welcome' || ch.type === 'closing' || ch.type === 'post_service';
        if (
            prayerEligibleType &&
            shortSection &&
            prayerScore >= 2 &&
            s.speech_ratio > 0.75 &&
            !hasPreachingCue(txt) &&
            s.music_ratio < 0.25
        ) {
            out[i] = {
                ...ch,
                type: 'prayer',
                reason: `${ch.reason ? `${ch.reason};` : ''}relabel_to_prayer`
            };
        }

        if (hasOfferingCue(txt) && s.music_ratio < 0.3 && s.speech_ratio > 0.5 && !hasPreachingCue(txt)) {
            out[i] = {
                ...ch,
                type: 'offering',
                reason: `${ch.reason ? `${ch.reason};` : ''}relabel_to_offering`
            };
        }
    }
    return out;
}

function splitMixedAnnouncementSermonIntro(chapters: Chapter[], transcript: Segment[]): Chapter[] {
    const result: Chapter[] = [];

    const sermonIntroCue = (text: string) =>
        isSabbathGreetingCue(text) ||
        /(est[aá]s listo para el mensaje|hoy que vamos a estudiar la palabra|quiero comenzar dici[eé]ndote|abran? sus biblias|mensaje de hoy|tema de hoy)/i.test(
            text
        );
    const announcementCue = (text: string) =>
        /(anuncios?|announcement|actividades|recuerden|les invitamos|esta tarde|despu[eé]s del culto|hoy tenemos|compartir con ustedes|les queremos agradecer)/i.test(
            text
        );

    for (const ch of chapters) {
        const inRange = transcript
            .filter((s) => s.end >= ch.start && s.start <= ch.end)
            .sort((a, b) => a.start - b.start);
        const splitSeg = inRange.find((s) => sermonIntroCue(cleanText(s.text)));
        const splitIdx = splitSeg ? inRange.findIndex((s) => s.start === splitSeg.start && s.end === splitSeg.end) : -1;
        const hasAnnouncementBefore =
            splitIdx > 0 && inRange.slice(0, splitIdx).some((s) => announcementCue(cleanText(s.text)));

        const mixedTitle =
            /announcements?\s*&\s*sermon\s*(introduction|transition)/i.test(ch.title) ||
            /(anuncios?|announcement).*(serm[oó]n|sermon)|(serm[oó]n|sermon).*(anuncios?|announcement)/i.test(ch.title);
        const shouldSplit =
            Boolean(splitSeg) &&
            (mixedTitle || (ch.type === 'announcement' && hasAnnouncementBefore));

        if (!splitSeg) {
            result.push({
                ...ch
            });
            continue;
        }

        if (!shouldSplit) {
            result.push({
                ...ch
            });
            continue;
        }

        const splitAt = clamp(splitSeg.start, ch.start, ch.end);
        const canSplit = splitAt - ch.start >= 10 && ch.end - splitAt >= 10;
        if (!canSplit) {
            result.push({
                ...ch,
                type: 'sermon',
                title: 'Sermon Introduction',
                reason: `${ch.reason ? `${ch.reason};` : ''}retag_mixed_to_sermon_intro`
            });
            continue;
        }

        result.push({
            start: ch.start,
            end: splitAt,
            title: 'Announcements',
            type: 'announcement',
            confidence: Number(Math.max(0.5, ch.confidence - 0.05).toFixed(2)),
            reason: `${ch.reason ? `${ch.reason};` : ''}split_mixed_announcements_sermon_intro`
        });
        result.push({
            start: splitAt,
            end: ch.end,
            title: 'Sermon Introduction',
            type: 'sermon',
            confidence: ch.confidence,
            reason: `${ch.reason ? `${ch.reason};` : ''}split_mixed_announcements_sermon_intro`
        });
    }

    return result.sort((a, b) => a.start - b.start);
}

function enforceSermonIntroType(chapters: Chapter[]): Chapter[] {
    return chapters.map((ch) => {
        if (/(sermon introduction|sermon transition)/i.test(ch.title) && ch.type !== 'sermon') {
            return {
                ...ch,
                type: 'sermon',
                reason: `${ch.reason ? `${ch.reason};` : ''}force_sermon_intro_type`
            };
        }
        return ch;
    });
}

function splitPreSermonIntro(chapters: Chapter[], transcript: Segment[], boundaries: SermonBoundaries): Chapter[] {
    const sermonIntroCue = (text: string) =>
        isSabbathGreetingCue(text) ||
        /(est[aá]s listo para el mensaje|hoy que vamos a estudiar la palabra|quiero comenzar dici[eé]ndote)/i.test(
            text
        );

    const out: Chapter[] = [];
    const targetEnd = boundaries.start;
    const epsilon = 2.0;

    for (const ch of chapters) {
        const isPreSermonBoundaryChapter =
            ch.end >= targetEnd - epsilon && ch.end <= targetEnd + epsilon && ch.start < targetEnd;
        if (!isPreSermonBoundaryChapter || ch.type === 'sermon' || /sermon introduction/i.test(ch.title)) {
            out.push(ch);
            continue;
        }

        const inRange = transcript
            .filter((s) => s.end >= ch.start && s.start <= ch.end)
            .sort((a, b) => a.start - b.start);
        const splitSeg = inRange.find((s) => sermonIntroCue(cleanText(s.text)));
        if (!splitSeg) {
            out.push(ch);
            continue;
        }

        const splitAt = clamp(splitSeg.start, ch.start, ch.end);
        const canSplit = splitAt - ch.start >= 20 && ch.end - splitAt >= 20;
        if (!canSplit) {
            out.push(ch);
            continue;
        }

        out.push({
            ...ch,
            end: splitAt,
            reason: `${ch.reason ? `${ch.reason};` : ''}split_pre_sermon_intro`
        });
        out.push({
            start: splitAt,
            end: ch.end,
            title: 'Sermon Introduction',
            type: 'sermon',
            confidence: Number(Math.max(0.75, ch.confidence).toFixed(2)),
            reason: `${ch.reason ? `${ch.reason};` : ''}split_pre_sermon_intro`
        });
    }

    return out.sort((a, b) => a.start - b.start);
}

function splitBySabbathGreetingCue(chapters: Chapter[], transcript: Segment[]): Chapter[] {
    const minLeadSec = Number(process.env.ANALYSIS_SABBATH_CUE_MIN_LEAD_SEC ?? 20);
    const minTailSec = Number(process.env.ANALYSIS_SABBATH_CUE_MIN_TAIL_SEC ?? 40);
    const contextSec = Number(process.env.ANALYSIS_SABBATH_CUE_CONTEXT_SEC ?? 90);
    const out: Chapter[] = [];

    for (const ch of chapters) {
        const duration = ch.end - ch.start;
        if (duration < minLeadSec + minTailSec + 20) {
            out.push(ch);
            continue;
        }

        const inRange = transcript
            .filter((s) => s.end >= ch.start && s.start <= ch.end)
            .sort((a, b) => a.start - b.start);
        if (inRange.length < 2) {
            out.push(ch);
            continue;
        }

        const splitSeg = inRange.find(
            (s) =>
                isSabbathGreetingCue(cleanText(s.text)) &&
                s.start >= ch.start + minLeadSec &&
                s.end <= ch.end - minTailSec
        );
        if (!splitSeg) {
            out.push(ch);
            continue;
        }

        const splitAt = clamp(splitSeg.start, ch.start, ch.end);
        if (splitAt - ch.start < minLeadSec || ch.end - splitAt < minTailSec) {
            out.push(ch);
            continue;
        }

        const beforeText = inRange
            .filter((s) => s.end >= splitAt - contextSec && s.start < splitAt)
            .map((s) => s.text)
            .join(' ');
        const afterText = inRange
            .filter((s) => s.start <= splitAt + contextSec && s.end > splitAt)
            .map((s) => s.text)
            .join(' ');

        const beforeType = inferTypeFromTextHeuristic(beforeText, ch.type);
        const afterType = inferTypeFromTextHeuristic(afterText, ch.type);
        const transitionLikely =
            beforeType !== afterType ||
            afterType === 'sermon' ||
            beforeType === 'prayer' ||
            beforeType === 'congregational_music' ||
            beforeType === 'announcement' ||
            beforeType === 'offering';

        if (!transitionLikely) {
            out.push(ch);
            continue;
        }

        const firstType = beforeType === afterType ? ch.type : beforeType;
        const secondType = afterType;
        const firstTitle = firstType === ch.type ? ch.title : chapterTitleForType(firstType);
        const secondTitle =
            secondType === ch.type
                ? isSabbathGreetingCue(cleanText(splitSeg.text))
                    ? `${ch.title} (reinicio)`
                    : ch.title
                : chapterTitleForType(secondType);

        out.push({
            ...ch,
            end: splitAt,
            type: firstType,
            title: firstTitle,
            reason: `${ch.reason ? `${ch.reason};` : ''}split_sabbath_greeting_cue`
        });
        out.push({
            start: splitAt,
            end: ch.end,
            type: secondType,
            title: secondTitle,
            confidence: Number(Math.max(0.72, ch.confidence).toFixed(2)),
            reason: `${ch.reason ? `${ch.reason};` : ''}split_sabbath_greeting_cue`
        });
    }

    return out.sort((a, b) => a.start - b.start);
}

function splitLeadingLiturgicalPreludeFromSermon(chapters: Chapter[], transcript: Segment[]): Chapter[] {
    const sermonIntroCue = (text: string) =>
        isSabbathGreetingCue(text) ||
        /(abran? (sus )?biblias|mensaje de hoy|tema de hoy|hoy quiero|quiero compartir|la biblia|vers[íi]culo|cap[íi]tulo)/i.test(
            text
        );
    const prayerCue = (text: string) =>
        /(oremos|oraci[oó]n|inclinemos|inclinar.*rostro|arrodill|en el nombre de (tu hijo )?cristo jes[uú]s|te lo pedimos|am[eé]n)/i.test(
            text
        );
    const worshipCue = (text: string) =>
        /(vamos a cantar|cantemos|equipo de alabanza|alabemos|worship|himno|coro|reproclamamos|en lo alto est[aá]s)/i.test(
            text
        );

    const out: Chapter[] = [];
    for (const ch of chapters) {
        if (ch.type !== 'sermon') {
            out.push(ch);
            continue;
        }

        const inRange = transcript
            .filter((s) => s.end >= ch.start && s.start <= ch.end)
            .sort((a, b) => a.start - b.start);
        if (inRange.length < 6 || ch.end - ch.start < 120) {
            out.push(ch);
            continue;
        }

        const firstSermonSeg = inRange.find((s) => sermonIntroCue(cleanText(s.text)));
        if (!firstSermonSeg) {
            out.push(ch);
            continue;
        }

        // Important Hispanic SDA cue: if speaker says "feliz sábado" again after already speaking,
        // treat it as a likely section reset (often worship/prayer -> sermon transition).
        const greetingMinGapSec = Number(process.env.ANALYSIS_REPEAT_SABBATH_CUE_MIN_GAP_SEC ?? 90);
        const greetingSegments = inRange.filter((s) => isSabbathGreetingCue(cleanText(s.text)));
        let splitCandidate = firstSermonSeg;
        if (greetingSegments.length >= 2) {
            const firstGreeting = greetingSegments[0];
            const repeatedGreeting = greetingSegments.find(
                (s) => s.start > firstGreeting.start + greetingMinGapSec
            );
            if (repeatedGreeting) {
                const between = inRange.filter((s) => s.start > firstGreeting.end && s.end < repeatedGreeting.start);
                const betweenPrayer = between.filter((s) => prayerCue(cleanText(s.text))).length;
                const betweenWorship = between.filter((s) => worshipCue(cleanText(s.text))).length;
                if (betweenPrayer + betweenWorship >= 1) {
                    splitCandidate = repeatedGreeting;
                }
            }
        }

        const splitAt = clamp(splitCandidate.start, ch.start, ch.end);
        if (splitAt - ch.start < 25 || ch.end - splitAt < 90) {
            out.push(ch);
            continue;
        }

        const preSegments = inRange.filter((s) => s.start < splitAt);
        const prayerCount = preSegments.filter((s) => prayerCue(cleanText(s.text))).length;
        const worshipCount = preSegments.filter((s) => worshipCue(cleanText(s.text))).length;
        const liturgicalRatio = preSegments.length > 0 ? (prayerCount + worshipCount) / preSegments.length : 0;

        if (preSegments.length < 2 || (prayerCount + worshipCount) < 1 || liturgicalRatio < 0.25) {
            out.push(ch);
            continue;
        }

        const preType: ChapterType = worshipCount > prayerCount ? 'congregational_music' : 'prayer';
        const preTitle =
            preType === 'congregational_music'
                ? 'Alabanza congregacional'
                : worshipCount > 0
                  ? 'Oración y transición de adoración'
                  : 'Oración congregacional';

        out.push({
            start: ch.start,
            end: splitAt,
            title: preTitle,
            type: preType,
            confidence: Number(Math.max(0.72, ch.confidence - 0.08).toFixed(2)),
            reason: `${ch.reason ? `${ch.reason};` : ''}split_leading_liturgical_prelude`
        });
        out.push({
            start: splitAt,
            end: ch.end,
            title: /sermon/i.test(ch.title) ? ch.title : 'Sermón principal',
            type: 'sermon',
            confidence: ch.confidence,
            reason: `${ch.reason ? `${ch.reason};` : ''}split_leading_liturgical_prelude`
        });
    }

    return out.sort((a, b) => a.start - b.start);
}

function toPolishedMarkdown(doc: AnalysisDoc): string {
    const lines: string[] = [];
    lines.push('# Transcript Polished Review');
    lines.push('');
    lines.push(`Generated: ${doc.generated_at}`);
    lines.push(`Sermon bounds: ${formatSec(doc.boundaries.start)} - ${formatSec(doc.boundaries.end)}`);
    lines.push('');
    for (let i = 0; i < doc.chapters.length; i++) {
        const c = doc.chapters[i];
        lines.push(`## ${i + 1}. ${c.title}`);
        lines.push(`- Type: ${c.type}`);
        lines.push(`- Range: ${formatSec(c.start)} - ${formatSec(c.end)}`);
        lines.push(`- Confidence: ${c.confidence.toFixed(2)}`);
        lines.push('');
        const paras = doc.paragraphs.filter((p) => p.chapter_index === i);
        for (const p of paras) {
            lines.push(`[${formatSec(p.start)} - ${formatSec(p.end)}] ${p.text}`);
            lines.push('');
        }
    }
    return lines.join('\n');
}

function toPolishedMultimodalMarkdown(doc: AnalysisDoc): string {
    const lines: string[] = [];
    lines.push('# Transcript Polished Review (Multimodal)');
    lines.push('');
    lines.push(`Generated: ${doc.generated_at}`);
    lines.push(`Sermon bounds: ${formatSec(doc.boundaries.start)} - ${formatSec(doc.boundaries.end)}`);
    lines.push('');
    lines.push(`Audio cues: music_sections=${doc.cues.music_sections}, noenergy_sections=${doc.cues.noenergy_sections}, long_pause_sections=${doc.cues.long_pause_sections}`);
    lines.push(
        `OCR cues: source=${doc.ocr.source ?? 'none'}, segments=${doc.ocr.segments}, scene_cuts=${doc.ocr.scene_cuts}, speaker_names=${doc.ocr.speaker_name_cues}, bible_verses=${doc.ocr.bible_verse_cues}, sermon_titles=${doc.ocr.sermon_title_cues}, song_lyrics=${doc.ocr.song_lyric_cues}`
    );
    lines.push('');

    for (let i = 0; i < doc.chapters.length; i++) {
        const c = doc.chapters[i];
        const s = doc.chapter_signals[i];
        lines.push(`## ${i + 1}. ${c.title}`);
        lines.push(`- Type: ${c.type}`);
        lines.push(`- Range: ${formatSec(c.start)} - ${formatSec(c.end)}`);
        lines.push(`- Confidence: ${c.confidence.toFixed(2)}`);
        lines.push(`- LLM reason: ${c.reason || 'n/a'}`);
        lines.push(
            `- Signals: speech=${s?.speech_ratio ?? 0}, music=${s?.music_ratio ?? 0}, noenergy=${s?.noenergy_ratio ?? 0}, has_music=${s?.has_music ?? false}, has_long_pause=${s?.has_long_pause ?? false}`
        );
        const ocrCues = (doc.ocr_segments ?? [])
            .filter((o) => o.end >= c.start && o.start <= c.end)
            .filter((o) => o.type !== 'on_screen_text')
            .slice(0, 4);
        if (ocrCues.length > 0) {
            lines.push(
                `- OCR in range: ${ocrCues
                    .map((o) => `${o.type}@${formatSec(o.start)}="${trimForPrompt(o.text, 60)}"`)
                    .join(' | ')}`
            );
        }
        lines.push('');

        const paras = doc.paragraphs.filter((p) => p.chapter_index === i);
        for (const p of paras) {
            lines.push(`[${formatSec(p.start)} - ${formatSec(p.end)}] ${p.text}`);
            lines.push('');
        }
    }
    return lines.join('\n');
}

function extractTextCues(transcript: Segment[]): Array<{ kind: string; section: string; text: string; t: number }> {
    const patterns: Array<{ kind: string; section: string; rx: RegExp }> = [
        { kind: 'sabbath_greeting_boundary', section: 'other', rx: /(buenos d[ií]as[, ]+feliz s[áa]bado|feliz s[áa]bado(?:\s+nuevamente|\s+familia)?)/i },
        { kind: 'invite_pray', section: 'prayer', rx: /(inclinar.*rostro|pong[aá]monos de rodillas|oremos)/i },
        { kind: 'prayer_close', section: 'prayer', rx: /(te lo pedimos en cristo jes[uú]s|en el nombre de jes[uú]s|am[eé]n\\.?$)/i },
        { kind: 'invite_sing', section: 'congregational_music', rx: /(vamos a cantar|cantemos|equipo de alabanza|alabemos)/i },
        { kind: 'offering_invite', section: 'offering', rx: /(diezmo[s]?|ofrenda[s]?|regresar lo que te pertenece|mayordom[ií]a)/i },
        { kind: 'call_to_worship', section: 'call_to_worship', rx: /(llamado a la adoraci[oó]n|pong[aá]monos de pie para adorar)/i },
        { kind: 'child_presentation', section: 'child_presentation', rx: /(niñ[oa]s?|historia infantil|momento de los niñ[oa]s|pasen los niñ[oa]s|presentaci[oó]n infantil)/i }
    ];
    const out: Array<{ kind: string; section: string; text: string; t: number }> = [];
    for (const s of transcript) {
        const text = cleanText(s.text);
        if (!text) continue;
        if (/^am[eé]n\.?$/i.test(text)) continue; // ignore solo amen
        for (const p of patterns) {
            if (p.rx.test(text)) {
                out.push({ kind: p.kind, section: p.section, text, t: s.start });
            }
        }
    }
    return out;
}

export async function generateAnalysisArtifacts(
    transcriptInput: Segment[],
    boundaries: SermonBoundaries,
    options: { workDir?: string; videoPath?: string }
): Promise<void> {
    const workDir = options.workDir;
    if (!workDir) return;
    fs.mkdirSync(workDir, { recursive: true });

    const transcript = transcriptInput
        .filter((s) => Number.isFinite(s.start) && Number.isFinite(s.end))
        .map((s) => ({ start: Number(s.start), end: Number(s.end), text: cleanText(s.text) }))
        .sort((a, b) => a.start - b.start);
    if (transcript.length === 0) return;

    const durationSec = transcript[transcript.length - 1].end;
    const audioEventsPath = path.join(workDir, 'audio.events.json');
    const audioEventsObj = fs.existsSync(audioEventsPath)
        ? safeJsonParse<AudioEventPassResult | null>(fs.readFileSync(audioEventsPath, 'utf8'), null)
        : null;
    const audioEvents = audioEventsObj?.segments ?? null;
    const ocrEventsObj = await ensureOcrEvents(workDir, durationSec, options.videoPath);

    const llmEnabled = String(process.env.ANALYSIS_ENABLE_LLM_CHAPTERS ?? 'true').toLowerCase() === 'true';
    const llmChapters = llmEnabled
        ? await inferChaptersWithLlm(transcript, boundaries, audioEvents, ocrEventsObj, durationSec, workDir)
        : null;
    const baseChapters = llmChapters ?? fallbackChapters(boundaries, durationSec);
    const baseSignals = baseChapters.map((c) => computeChapterSignal(c, audioEvents));
    const calibratedChapters = calibrateChaptersBySignals(baseChapters, baseSignals);
    const relabeledChapters = relabelChaptersBySignals(transcript, calibratedChapters, baseSignals);
    const renamedChapters = await renameLowConfidenceChapters(transcript, relabeledChapters, durationSec, workDir);
    const splitMixed = splitMixedAnnouncementSermonIntro(renamedChapters, transcript);
    const splitIntro = splitPreSermonIntro(splitMixed, transcript, boundaries);
    const splitGreetingCue = splitBySabbathGreetingCue(splitIntro, transcript);
    const splitLiturgicalPrelude = splitLeadingLiturgicalPreludeFromSermon(splitGreetingCue, transcript);
    const chapters = enforceSermonIntroType(splitLiturgicalPrelude);
    const chapterSignals = chapters.map((c) => computeChapterSignal(c, audioEvents));
    const paragraphs = buildParagraphs(transcript, chapters);

    const doc: AnalysisDoc = {
        version: '1.0.0',
        generated_at: new Date().toISOString(),
        boundaries,
        chapters,
        chapter_signals: chapterSignals,
        cues: buildCueSummary(audioEvents),
        ocr: buildOcrSummary(ocrEventsObj),
        ocr_segments: (ocrEventsObj?.segments ?? []).slice(0, 300),
        paragraphs,
        inputs: {
            transcript_segments: transcript.length,
            audio_events_source: audioEventsObj?.source ?? null,
            ocr_events_source: ocrEventsObj?.source ?? null
        }
    };

    const polishedJsonPath = path.join(workDir, 'transcript.polished.json');
    const polishedMdPath = path.join(workDir, 'transcript.polished.md');
    const polishedMultimodalMdPath = path.join(workDir, 'transcript.polished.multimodal.md');
    const analysisDocPath = path.join(workDir, 'analysis.doc.json');

    fs.writeFileSync(analysisDocPath, JSON.stringify(doc, null, 2));
    fs.writeFileSync(polishedJsonPath, JSON.stringify({ chapters: doc.chapters, paragraphs: doc.paragraphs }, null, 2));
    fs.writeFileSync(polishedMdPath, toPolishedMarkdown(doc));
    fs.writeFileSync(polishedMultimodalMdPath, toPolishedMultimodalMarkdown(doc));
    const cueHits = extractTextCues(transcript);
    await emitLlmCueEvents(
        cueHits.map((c) => ({
            source_pass: 'analysis_text_cue_extractor',
            section_type: c.section,
            cue_kind: c.kind,
            cue_text: c.text,
            cue_time_sec: c.t,
            confidence: 0.8,
            metadata: { origin: 'regex' }
        })),
        workDir
    );
    if (ocrEventsObj && Array.isArray(ocrEventsObj.segments) && ocrEventsObj.segments.length > 0) {
        await emitLlmCueEvents(
            ocrEventsObj.segments
                .filter((s) => s.type !== 'on_screen_text')
                .slice(0, 500)
                .map((s) => ({
                    source_pass: 'analysis_ocr_cues',
                    section_type:
                        s.type === 'song_lyric'
                            ? 'congregational_music'
                            : s.type === 'bible_verse'
                              ? 'scripture_reading'
                              : s.type === 'sermon_title'
                                ? 'sermon'
                                : 'other',
                    cue_kind: `ocr_${s.type}`,
                    cue_text: s.text,
                    cue_time_sec: s.start,
                    confidence: Number.isFinite(s.confidence) ? s.confidence : 0.7,
                    metadata: {
                        end_sec: s.end,
                        region: s.region,
                        near_scene_cut: Boolean(s.near_scene_cut)
                    }
                })),
            workDir
        );
    }
    console.log(`Analysis artifact written: ${analysisDocPath}`);
    console.log(`Polished transcript written: ${polishedJsonPath}`);
    console.log(`Polished markdown written: ${polishedMdPath}`);
    console.log(`Polished multimodal markdown written: ${polishedMultimodalMdPath}`);
}
