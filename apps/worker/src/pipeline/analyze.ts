import { findSermonBoundaries, SermonBoundaries } from './boundaries';
import { findHighlights, HighlightClip } from './highlights';

export { type SermonBoundaries, type HighlightClip as ClipCandidate };

export async function analyze(transcript: any[]): Promise<{ boundaries: SermonBoundaries; clips: HighlightClip[] }> {
    console.log('Analyzing transcript...');

    const boundaries = findSermonBoundaries(transcript);

    // Only search for highlights within the sermon boundaries
    const sermonTranscript = transcript.filter(s => s.start >= boundaries.start && s.end <= boundaries.end);

    const rawClips = await findHighlights(sermonTranscript, 'openai');
    const clips = normalizeHighlights(rawClips, sermonTranscript);

    return {
        boundaries,
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

        results.push({ start, end, title, excerpt, hook, confidence });
    }

    return results;
}
