export interface SermonBoundaries {
    start: number;
    end: number;
}

export function findSermonBoundaries(transcript: { start: number; end: number; text: string }[]): SermonBoundaries {
    console.log('Finding sermon boundaries...');

    // Heuristic-based approach
    // Look for keywords at the start and end
    const startKeywords = ['predicación', 'mensaje', 'abramos la palabra', 'lectura de hoy', 'bíblica', 'sermón'];
    const endKeywords = ['oremos', 'vamos a orar', 'despedida', 'anuncios', 'bendición', 'amén'];

    let start = 0;
    let end = transcript.length > 0 ? transcript[transcript.length - 1].end : 0;

    // Find start
    for (const segment of transcript) {
        const text = segment.text.toLowerCase();
        if (startKeywords.some(keyword => text.includes(keyword))) {
            start = segment.start;
            break;
        }
    }

    // Find end (searching backwards)
    for (let i = transcript.length - 1; i >= 0; i--) {
        const segment = transcript[i];
        const text = segment.text.toLowerCase();
        if (endKeywords.some(keyword => text.includes(keyword))) {
            end = segment.end;
            // We don't break immediately because we want the *first* occurrence of an end keyword 
            // after the sermon (searching from end to start, so the last occurrence in the video)
            break;
        }
    }

    // Sanity check
    if (start >= end) {
        start = 0;
        end = transcript.length > 0 ? transcript[transcript.length - 1].end : 0;
    }

    return { start, end };
}
