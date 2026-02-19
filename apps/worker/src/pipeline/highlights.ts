import { OpenAI } from 'openai';
import dotenv from 'dotenv';
import path from 'path';

dotenv.config({ path: path.resolve(__dirname, '../../../.env') });

// Initialize OpenAI only if the API key is present
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
}

export async function findHighlights(
    transcript: { start: number; end: number; text: string }[],
    strategy: 'openai' | 'deepseek' | 'cohere' = 'openai'
): Promise<HighlightClip[]> {
    console.log(`Finding highlights using strategy: ${strategy}`);

    if (strategy === 'openai') {
        return findHighlightsOpenAI(transcript);
    }

    // Fallback logic if API fails or is missing
    const clips: HighlightClip[] = [];

    // Heuristic: Look for markers of meaningful segments
    // (e.g. segments that have high density of religious keywords)
    const religiousKeywords = ['dios', 'jesús', 'cristo', 'señor', 'fe', 'amor', 'biblia', 'palabra', 'amén', 'victoria'];

    for (let i = 0; i < transcript.length - 15; i += 50) { // Sample every 50 segments
        const chunk = transcript.slice(i, i + 15);
        if (chunk.length < 5) continue;

        const text = chunk.map(c => c.text.toLowerCase()).join(' ');
        const score = religiousKeywords.filter(k => text.includes(k)).length;

        if (score >= 2) {
            clips.push({
                start: chunk[0].start,
                end: chunk[chunk.length - 1].end,
                title: `Reflection: ${chunk[0].text.substring(0, 30)}...`,
                excerpt: chunk.map(c => c.text).join(' ').substring(0, 150) + '...',
                hook: "Listen to this powerful reflection.",
                confidence: 0.5 + (score * 0.1)
            });
        }

        if (clips.length >= 6) break;
    }

    return clips;
}

async function findHighlightsOpenAI(transcript: { start: number; end: number; text: string }[]): Promise<HighlightClip[]> {
    // We send a summarized version or chunks of the transcript to ChatGPT
    // to identify the best moments.

    const transcriptText = transcript.map(s => `[${s.start}-${s.end}] ${s.text}`).join('\n');

    const prompt = `
    Analyze the following transcript of a church sermon. 
    Select 6 highlight moments that are:
    - Meaningful out of context
    - Short enough for Reels/Shorts (<60 seconds)
    - Not cut mid-thought
    - Spread across the sermon
    
    Transcript:
    ${transcriptText.substring(0, 10000)} ... (truncated if long)
    
    Return a JSON array of objects with the following structure:
    [
      {
        "start": number,
        "end": number,
        "title": "Short catchy title",
        "excerpt": "A key quote from the selection",
        "hook": "A social media hook/caption",
        "confidence": number (0.0 to 1.0)
      }
    ]
  `;

    const openai = getOpenAIClient();
    if (!openai) {
        console.warn('OpenAI API key missing, skipping AI highlights.');
        return [];
    }

    try {
        const response = await openai.chat.completions.create({
            model: 'gpt-3.5-turbo', // or gpt-4
            messages: [{ role: 'user', content: prompt }],
            response_format: { type: 'json_object' },
        });

        const content = response.choices[0].message.content;
        if (!content) return [];

        const parsed = JSON.parse(content);
        return parsed.highlights || parsed; // Handle different JSON structures
    } catch (error) {
        console.error('OpenAI Error:', error);
        return [];
    }
}
