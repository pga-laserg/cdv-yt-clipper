import { findSermonBoundaries, SermonBoundaries } from './boundaries';
import { findHighlights, HighlightClip } from './highlights';

export { type SermonBoundaries, type HighlightClip as ClipCandidate };

export async function analyze(transcript: any[]): Promise<{ boundaries: SermonBoundaries; clips: HighlightClip[] }> {
    console.log('Analyzing transcript...');

    const boundaries = findSermonBoundaries(transcript);

    // Only search for highlights within the sermon boundaries
    const sermonTranscript = transcript.filter(s => s.start >= boundaries.start && s.end <= boundaries.end);

    const clips = await findHighlights(sermonTranscript, 'openai');

    return {
        boundaries,
        clips
    };
}
