import { findSermonBoundaries, SermonBoundaries } from './boundaries';
import { findHighlights, HighlightClip } from './highlights';
import { generateAnalysisArtifacts } from './analysis-doc';
import { adjudicateBoundariesWithSingleLlm } from './boundary-adjudicator';
import fs from 'fs';
import path from 'path';

export { type SermonBoundaries, type HighlightClip as ClipCandidate };

interface AnalyzeOptions {
    workDir?: string;
    audioPath?: string;
    videoPath?: string;
}

export async function analyze(
    transcript: any[],
    options: AnalyzeOptions = {}
): Promise<{ boundaries: SermonBoundaries; clips: HighlightClip[] }> {
    console.log('Analyzing transcript...');

    const boundaries = await findSermonBoundaries(transcript, {
        workDir: options.workDir,
        audioPath: options.audioPath,
        videoPath: options.videoPath
    });

    let finalBoundaries = boundaries;
    try {
        await generateAnalysisArtifacts(transcript as { start: number; end: number; text: string }[], finalBoundaries, {
            workDir: options.workDir,
            videoPath: options.videoPath
        });
    } catch (error) {
        console.warn('Analysis artifact generation failed:', error);
    }

    const singleAdjEnabled = String(process.env.BOUNDARY_ENABLE_SINGLE_LLM_ADJUDICATOR ?? 'true').toLowerCase() === 'true';
    if (singleAdjEnabled && options.workDir) {
        try {
            const adjudicated = await adjudicateBoundariesWithSingleLlm(
                transcript as { start: number; end: number; text: string }[],
                finalBoundaries,
                { workDir: options.workDir }
            );
            if (adjudicated && (adjudicated.start !== finalBoundaries.start || adjudicated.end !== finalBoundaries.end)) {
                const padded = applyTranscriptOnlyPadding(
                    adjudicated.start,
                    adjudicated.end,
                    transcript as { start: number; end: number; text: string }[]
                );
                finalBoundaries = { start: padded.clip_start_sec, end: padded.clip_end_sec };
            }

            const targetedPath = path.join(options.workDir, 'sermon.boundaries.targeted-diarization.json');
            if (fs.existsSync(targetedPath)) {
                const raw = JSON.parse(fs.readFileSync(targetedPath, 'utf8')) as any;
                raw.llm_single_adjudicator = adjudicated;
                if (adjudicated) {
                    const padded = applyTranscriptOnlyPadding(
                        adjudicated.start,
                        adjudicated.end,
                        transcript as { start: number; end: number; text: string }[]
                    );
                    raw.final_clip_bounds = {
                        ...(raw.final_clip_bounds ?? {}),
                        clip_start_sec: padded.clip_start_sec,
                        clip_end_sec: padded.clip_end_sec,
                        applied_pre_pad_sec: padded.applied_pre_pad_sec,
                        applied_post_pad_sec: padded.applied_post_pad_sec,
                        padding_source: 'analyze.llm_single_adjudicator'
                    };
                    raw.llm_single_adjudicator_padded = padded;
                }
                fs.writeFileSync(targetedPath, JSON.stringify(raw, null, 2));
            }

            await generateAnalysisArtifacts(transcript as { start: number; end: number; text: string }[], finalBoundaries, {
                workDir: options.workDir,
                videoPath: options.videoPath
            });
        } catch (error) {
            console.warn('Single LLM adjudicator failed; keeping local boundaries:', error);
        }
    }

    // Generate highlights only after final boundaries are settled.
    const sermonTranscript = (transcript as { start: number; end: number; text: string }[])
        .filter((s) => Number.isFinite(s.start) && Number.isFinite(s.end))
        .filter((s) => s.start >= finalBoundaries.start && s.end <= finalBoundaries.end);

    const polishedContext = loadPolishedContext(options.workDir, finalBoundaries);
    const rawClips = await findHighlights(sermonTranscript, 'openai', {
        sermonStart: finalBoundaries.start,
        sermonEnd: finalBoundaries.end,
        paragraphs: polishedContext?.paragraphs ?? null,
        chapters: polishedContext?.chapters ?? null
    });
    const clips = normalizeHighlights(rawClips, sermonTranscript);

    return {
        boundaries: finalBoundaries,
        clips
    };
}

function normalizeHighlights(input: unknown, transcript: { start: number; end: number; text: string }[]): HighlightClip[] {
    if (!Array.isArray(input)) return [];

    const maxEnd = transcript.length > 0 ? transcript[transcript.length - 1].end : 0;
    const results: HighlightClip[] = [];

    for (const item of input) {
        if (!item || typeof item !== 'object') continue;
        const obj = item as Record<string, unknown>;

        const startRaw = typeof obj.start === 'number' ? obj.start : Number(obj.start);
        const endRaw = typeof obj.end === 'number' ? obj.end : Number(obj.end);
        if (!Number.isFinite(startRaw) || !Number.isFinite(endRaw)) continue;

        const start = Math.max(0, startRaw);
        const end = Math.max(start + 1, Math.min(endRaw, maxEnd || endRaw));
        if (end <= start) continue;

        const title = typeof obj.title === 'string' && obj.title.trim() ? obj.title.trim() : 'Sermon Highlight';
        const excerpt = typeof obj.excerpt === 'string' ? obj.excerpt.trim() : '';
        const hook = typeof obj.hook === 'string' ? obj.hook.trim() : '';
        const confRaw = typeof obj.confidence === 'number' ? obj.confidence : Number(obj.confidence);
        const confidence = Number.isFinite(confRaw) ? Math.max(0, Math.min(1, confRaw)) : 0.5;
        const scoreRaw = typeof obj.score === 'number' ? obj.score : Number(obj.score);
        const score = Number.isFinite(scoreRaw) ? Math.max(0, Math.min(1, scoreRaw)) : confidence;
        const scoreBreakdownRaw = obj.score_breakdown;
        const score_breakdown =
            scoreBreakdownRaw && typeof scoreBreakdownRaw === 'object'
                ? (scoreBreakdownRaw as { model_confidence: number; duration_preference: number; ending_completeness: number })
                : undefined;

        results.push({ start, end, title, excerpt, hook, confidence, score, score_breakdown } as HighlightClip);
    }

    return results;
}

function isNonSpeech(text: string): boolean {
    const t = String(text ?? '').trim().toLowerCase();
    if (!t) return true;
    return /^(music|música|musica|piano|instrumental|silence|silencio|applause|aplausos|amen\.?)$/i.test(t);
}

function applyTranscriptOnlyPadding(speakerStart: number, speakerEnd: number, segments: { start: number; end: number; text: string }[]) {
    const normalized = segments
        .filter((s) => Number.isFinite(s.start) && Number.isFinite(s.end))
        .map((s) => ({ start: Number(s.start), end: Number(s.end), text: String(s.text ?? '') }))
        .sort((a, b) => a.start - b.start);
    const speechSegments = normalized.filter((s) => !isNonSpeech(s.text));
    const prevSpeech = [...speechSegments].reverse().find((s) => s.end <= speakerStart);
    const nextSpeech = speechSegments.find((s) => s.start >= speakerEnd);
    const prevBoundary = prevSpeech?.end ?? 0;
    const nextBoundary = nextSpeech?.start ?? Number.POSITIVE_INFINITY;
    const prePad = Math.min(10, Math.max(0, speakerStart - prevBoundary));
    const postPad = Number.isFinite(nextBoundary) ? Math.min(10, Math.max(0, nextBoundary - speakerEnd)) : 10;
    const clipStart = Math.max(0, speakerStart - prePad);
    const clipEnd = Number.isFinite(nextBoundary) ? Math.min(nextBoundary, speakerEnd + postPad) : speakerEnd + postPad;
    return {
        clip_start_sec: clipStart,
        clip_end_sec: clipEnd,
        applied_pre_pad_sec: prePad,
        applied_post_pad_sec: postPad
    };
}

function loadPolishedContext(
    workDir: string | undefined,
    bounds: SermonBoundaries
): { paragraphs: Array<{ chapter_index?: number; chapter_title?: string; start: number; end: number; text: string }>; chapters: Array<{ start: number; end: number; title: string; type?: string; confidence?: number }> } | null {
    if (!workDir) return null;
    const polishedPath = path.join(workDir, 'transcript.polished.json');
    if (!fs.existsSync(polishedPath)) return null;
    try {
        const raw = JSON.parse(fs.readFileSync(polishedPath, 'utf8')) as any;
        const paragraphs = Array.isArray(raw?.paragraphs)
            ? raw.paragraphs
                .filter((p: any) => Number.isFinite(Number(p?.start)) && Number.isFinite(Number(p?.end)))
                .map((p: any) => ({
                    chapter_index: Number.isFinite(Number(p?.chapter_index)) ? Number(p.chapter_index) : undefined,
                    chapter_title: typeof p?.chapter_title === 'string' ? p.chapter_title : '',
                    start: Number(p.start),
                    end: Number(p.end),
                    text: String(p?.text ?? '')
                }))
                .filter((p: any) => p.end >= bounds.start && p.start <= bounds.end)
            : [];
        const chapters = Array.isArray(raw?.chapters)
            ? raw.chapters
                .filter((c: any) => Number.isFinite(Number(c?.start)) && Number.isFinite(Number(c?.end)))
                .map((c: any) => ({
                    start: Number(c.start),
                    end: Number(c.end),
                    title: String(c?.title ?? ''),
                    type: typeof c?.type === 'string' ? c.type : undefined,
                    confidence: Number.isFinite(Number(c?.confidence)) ? Number(c.confidence) : undefined
                }))
                .filter((c: any) => c.end >= bounds.start && c.start <= bounds.end)
            : [];
        return { paragraphs, chapters };
    } catch {
        return null;
    }
}
