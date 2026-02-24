import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import dotenv from 'dotenv';
import { OpenAI } from 'openai';

dotenv.config({ path: path.resolve(__dirname, '../../../.env') });
dotenv.config({ path: path.resolve(__dirname, '../../web/.env.local') });

interface DiarizedTurn {
    start: number;
    end: number;
    speaker: string;
    text: string;
    chunk_index: number;
}

function runProcess(cmd: string, args: string[]): Promise<{ stdout: string; stderr: string }> {
    return new Promise((resolve, reject) => {
        const proc = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'] });
        let stdout = '';
        let stderr = '';
        proc.stdout.on('data', (d) => {
            stdout += d.toString();
        });
        proc.stderr.on('data', (d) => {
            stderr += d.toString();
        });
        proc.on('error', reject);
        proc.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`${cmd} exited with code ${code}. stderr: ${stderr.slice(-1200)}`));
                return;
            }
            resolve({ stdout, stderr });
        });
    });
}

async function ensureChunkedAudio(inputVideo: string, chunksDir: string, segmentSec: number): Promise<string[]> {
    fs.mkdirSync(chunksDir, { recursive: true });
    const existing = fs
        .readdirSync(chunksDir)
        .filter((f) => f.endsWith('.m4a'))
        .sort()
        .map((f) => path.join(chunksDir, f));
    if (existing.length > 0) return existing;

    const outPattern = path.join(chunksDir, 'chunk_%03d.m4a');
    await runProcess('ffmpeg', [
        '-y',
        '-i',
        inputVideo,
        '-vn',
        '-ac',
        '1',
        '-ar',
        '16000',
        '-c:a',
        'aac',
        '-b:a',
        '64k',
        '-f',
        'segment',
        '-segment_time',
        String(segmentSec),
        '-reset_timestamps',
        '1',
        outPattern
    ]);

    return fs
        .readdirSync(chunksDir)
        .filter((f) => f.endsWith('.m4a'))
        .sort()
        .map((f) => path.join(chunksDir, f));
}

async function getDurationSec(file: string): Promise<number> {
    const { stdout } = await runProcess('ffprobe', [
        '-v',
        'error',
        '-show_entries',
        'format=duration',
        '-of',
        'default=noprint_wrappers=1:nokey=1',
        file
    ]);
    const n = Number(stdout.trim());
    return Number.isFinite(n) ? n : 0;
}

function normalizeSpeaker(v: unknown): string {
    if (typeof v === 'string' && v.trim()) return v.trim();
    if (typeof v === 'number' && Number.isFinite(v)) return `SPEAKER_${String(v)}`;
    return 'UNKNOWN';
}

function toNumber(v: unknown): number | null {
    const n = typeof v === 'number' ? v : Number(v);
    return Number.isFinite(n) ? n : null;
}

function extractTurnsFromAny(root: unknown): Array<{ start: number; end: number; speaker: string; text: string }> {
    const turns: Array<{ start: number; end: number; speaker: string; text: string }> = [];
    const visit = (node: unknown) => {
        if (!node || typeof node !== 'object') return;
        if (Array.isArray(node)) {
            for (const item of node) visit(item);
            return;
        }
        const obj = node as Record<string, unknown>;
        const start = toNumber(obj.start ?? obj.start_sec ?? obj.from ?? obj.begin);
        const end = toNumber(obj.end ?? obj.end_sec ?? obj.to ?? obj.finish);
        const text = typeof obj.text === 'string' ? obj.text.trim() : '';
        const speaker = normalizeSpeaker(obj.speaker ?? obj.speaker_id ?? obj.speaker_label ?? obj.spk);
        if (start != null && end != null && end > start && text) {
            turns.push({ start, end, speaker, text });
        }
        for (const v of Object.values(obj)) visit(v);
    };
    visit(root);
    return turns;
}

function mergeAdjacent(turns: DiarizedTurn[], maxGap = 0.5): DiarizedTurn[] {
    if (turns.length === 0) return [];
    const sorted = [...turns].sort((a, b) => a.start - b.start);
    const out: DiarizedTurn[] = [sorted[0]];
    for (let i = 1; i < sorted.length; i++) {
        const prev = out[out.length - 1];
        const cur = sorted[i];
        if (cur.speaker === prev.speaker && cur.start - prev.end <= maxGap) {
            prev.end = Math.max(prev.end, cur.end);
            prev.text = `${prev.text} ${cur.text}`.trim();
            continue;
        }
        out.push(cur);
    }
    return out;
}

async function transcribeChunksDiarized(
    client: OpenAI,
    chunks: string[],
    language = 'es'
): Promise<DiarizedTurn[]> {
    const turns: DiarizedTurn[] = [];
    let offset = 0;
    const reqTimeoutMs = Number(process.env.CLOUD_DIAR_REQUEST_TIMEOUT_MS ?? 600000);

    for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        const duration = await getDurationSec(chunk);
        console.log(`Cloud diarize chunk ${i + 1}/${chunks.length}: ${path.basename(chunk)} (${duration.toFixed(1)}s)`);

        const resp = await client.audio.transcriptions.create({
            model: 'gpt-4o-transcribe-diarize',
            file: fs.createReadStream(chunk),
            response_format: 'diarized_json',
            language,
            chunking_strategy: 'auto' as any
        } as any, {
            timeout: reqTimeoutMs
        } as any);

        const parsed = resp as unknown;
        const extracted = extractTurnsFromAny(parsed);
        if (extracted.length === 0) {
            console.warn(`No diarized turns parsed for chunk ${i + 1}.`);
        }

        for (const t of extracted) {
            turns.push({
                start: t.start + offset,
                end: t.end + offset,
                speaker: t.speaker,
                text: t.text,
                chunk_index: i
            });
        }
        offset += duration;
    }

    return mergeAdjacent(turns);
}

async function inferSermonBoundsWithLLM(client: OpenAI, turns: DiarizedTurn[]) {
    const timeline = turns
        .slice(0, 2000)
        .map((t) => `[${t.start.toFixed(2)}-${t.end.toFixed(2)}] ${t.speaker}: ${t.text.slice(0, 220)}`)
        .join('\n');

    const prompt = `
You are analyzing a diarized church service transcript.
Find the MAIN SERMON section boundaries and the MAIN SERMON SPEAKER label.

Return strict JSON:
{
  "sermon_speaker": "speaker label exactly as appears in timeline",
  "speaker_start_sec": number,
  "speaker_end_sec": number,
  "confidence": number,
  "notes": "short"
}

Rules:
- Exclude greetings, music, children story, offering, announcements.
- Exclude speakers before and after sermon.
- Use the same speaker label from timeline for sermon_speaker.

Timeline:
${timeline}
`;

    const response = await client.chat.completions.create({
        model: process.env.ANALYZE_OPENAI_MODEL || 'gpt-4o-mini',
        messages: [{ role: 'user', content: prompt }],
        response_format: { type: 'json_object' }
    });
    const content = response.choices[0]?.message?.content;
    if (!content) throw new Error('Empty sermon boundary response from LLM.');
    return JSON.parse(content) as {
        sermon_speaker: string;
        speaker_start_sec: number;
        speaker_end_sec: number;
        confidence: number;
        notes?: string;
    };
}

function nearestSpeakerBounds(turns: DiarizedTurn[], speaker: string, aroundStart: number, aroundEnd: number) {
    const same = turns.filter((t) => t.speaker === speaker).sort((a, b) => a.start - b.start);
    if (same.length === 0) return null;

    let start = same[0].start;
    let end = same[same.length - 1].end;

    const aroundStartTurn =
        same.find((t) => t.start <= aroundStart && t.end >= aroundStart) ??
        same.reduce((best, t) => (Math.abs(t.start - aroundStart) < Math.abs(best.start - aroundStart) ? t : best), same[0]);
    start = aroundStartTurn.start;

    const aroundEndTurn =
        same.find((t) => t.start <= aroundEnd && t.end >= aroundEnd) ??
        same.reduce((best, t) => (Math.abs(t.end - aroundEnd) < Math.abs(best.end - aroundEnd) ? t : best), same[0]);
    end = aroundEndTurn.end;

    // Add small non-speaking pads (up to 10s) while avoiding speaker overlap.
    const prevOther = [...turns]
        .filter((t) => t.speaker !== speaker && t.end <= start)
        .sort((a, b) => b.end - a.end)[0];
    const nextOther = turns
        .filter((t) => t.speaker !== speaker && t.start >= end)
        .sort((a, b) => a.start - b.start)[0];

    const clipStart = Math.max(0, start - Math.min(10, Math.max(0, start - (prevOther?.end ?? 0))));
    const clipEnd = nextOther ? Math.min(nextOther.start, end + Math.min(10, Math.max(0, nextOther.start - end))) : end + 10;

    return {
        refined_speaker_start_sec: start,
        refined_speaker_end_sec: end,
        clip_start_sec: clipStart,
        clip_end_sec: clipEnd,
        prev_other_speaker: prevOther ?? null,
        next_other_speaker: nextOther ?? null
    };
}

async function main() {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) throw new Error('OPENAI_API_KEY is required.');
    const client = new OpenAI({ apiKey });

    const workDirArg = process.argv[2] || 'apps/test_data/e2e_live_20260219_185221';
    const workDir = path.resolve(process.cwd(), workDirArg);
    const inputVideo = process.argv[3] ? path.resolve(process.cwd(), process.argv[3]) : path.join(workDir, 'source.mp4');
    const segmentSec = Number(process.env.CLOUD_DIAR_SEGMENT_SEC ?? 480);
    const language = process.env.CLOUD_DIAR_LANGUAGE || 'es';
    const maxChunks = Number(process.env.CLOUD_DIAR_MAX_CHUNKS ?? 0);

    if (!fs.existsSync(inputVideo)) throw new Error(`Video file not found: ${inputVideo}`);
    fs.mkdirSync(workDir, { recursive: true });
    const chunksDir = path.join(workDir, `cloud_chunks_${segmentSec}s`);
    let chunks = await ensureChunkedAudio(inputVideo, chunksDir, segmentSec);
    if (chunks.length === 0) throw new Error('No chunks produced.');
    if (maxChunks > 0) chunks = chunks.slice(0, maxChunks);

    const turns = await transcribeChunksDiarized(client, chunks, language);
    const diarizedPath = path.join(workDir, 'transcript.diarized.cloud.json');
    fs.writeFileSync(diarizedPath, JSON.stringify(turns, null, 2));
    console.log(`Wrote ${turns.length} diarized turns -> ${diarizedPath}`);

    const inferred = await inferSermonBoundsWithLLM(client, turns);
    const refined = nearestSpeakerBounds(turns, inferred.sermon_speaker, inferred.speaker_start_sec, inferred.speaker_end_sec);
    if (!refined) throw new Error(`Could not locate inferred speaker "${inferred.sermon_speaker}" in diarized turns.`);

    const out = {
        model: 'gpt-4o-transcribe-diarize',
        chunking: {
            segment_sec: segmentSec,
            chunk_count: chunks.length,
            language
        },
        inferred,
        refined
    };
    const outPath = path.join(workDir, 'sermon.boundaries.cloud-diarize.json');
    fs.writeFileSync(outPath, JSON.stringify(out, null, 2));
    console.log(`Wrote cloud sermon boundaries -> ${outPath}`);
}

main().catch((err) => {
    console.error('test-cloud-sermon-diarization failed:', err);
    process.exit(1);
});
