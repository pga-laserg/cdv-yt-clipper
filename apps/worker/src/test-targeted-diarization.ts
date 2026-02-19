import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import dotenv from 'dotenv';

dotenv.config({ path: path.resolve(__dirname, '../../../.env') });
dotenv.config({ path: path.resolve(__dirname, '../../web/.env.local') });

interface Segment {
    start: number;
    end: number;
    text?: string;
}

interface SpeakerTurn {
    start: number;
    end: number;
    speaker: string;
}

interface GuessBounds {
    start: number;
    end: number;
}

function parseSrtTimestamp(ts: string): number {
    const m = ts.trim().match(/^(\d{2}):(\d{2}):(\d{2}),(\d{3})$/);
    if (!m) return 0;
    const h = Number(m[1]);
    const min = Number(m[2]);
    const s = Number(m[3]);
    const ms = Number(m[4]);
    return h * 3600 + min * 60 + s + ms / 1000;
}

function parseSrtToSegments(srtText: string): Segment[] {
    const blocks = srtText.trim().split(/\r?\n\r?\n+/);
    const segments: Segment[] = [];
    for (const block of blocks) {
        const lines = block.split(/\r?\n/).map((l) => l.trim());
        if (lines.length < 3) continue;
        const time = lines[1];
        const m = time.match(
            /^(\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2},\d{3})$/
        );
        if (!m) continue;
        const text = lines.slice(2).join(' ').trim();
        segments.push({
            start: parseSrtTimestamp(m[1]),
            end: parseSrtTimestamp(m[2]),
            text
        });
    }
    return segments;
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
            const text = d.toString();
            stderr += text;
            if (text.trim()) console.error(text.trimEnd());
        });
        proc.on('error', (err) => reject(err));
        proc.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`${cmd} exited with code ${code}. stderr tail: ${stderr.slice(-1200)}`));
                return;
            }
            resolve({ stdout, stderr });
        });
    });
}

function overlap(aStart: number, aEnd: number, bStart: number, bEnd: number): number {
    return Math.max(0, Math.min(aEnd, bEnd) - Math.max(aStart, bStart));
}

function parseGuess(boundaryJsonPath: string): GuessBounds {
    const raw = JSON.parse(fs.readFileSync(boundaryJsonPath, 'utf8')) as any;
    if (raw?.raw?.speaker_start_sec != null && raw?.raw?.speaker_end_sec != null) {
        return { start: Number(raw.raw.speaker_start_sec), end: Number(raw.raw.speaker_end_sec) };
    }
    if (raw?.speaker_start_sec != null && raw?.speaker_end_sec != null) {
        return { start: Number(raw.speaker_start_sec), end: Number(raw.speaker_end_sec) };
    }
    if (raw?.start != null && raw?.end != null) {
        return { start: Number(raw.start), end: Number(raw.end) };
    }
    throw new Error(`Unsupported boundary JSON format: ${boundaryJsonPath}`);
}

function normalizeSegments(input: any[]): Segment[] {
    return input
        .filter((s) => Number.isFinite(s?.start) && Number.isFinite(s?.end))
        .map((s) => ({ start: Number(s.start), end: Number(s.end), text: typeof s.text === 'string' ? s.text : '' }))
        .sort((a, b) => a.start - b.start);
}

async function extractChunk(audioPath: string, outPath: string, startSec: number, endSec: number): Promise<void> {
    const duration = Math.max(0.01, endSec - startSec);
    await runProcess('ffmpeg', [
        '-y',
        '-i',
        audioPath,
        '-ss',
        String(startSec),
        '-t',
        String(duration),
        '-acodec',
        'pcm_s16le',
        outPath
    ]);
}

async function runDiarize(chunkPath: string): Promise<SpeakerTurn[]> {
    const pythonScript = path.resolve(__dirname, 'pipeline/python/diarize.py');
    const venv311 = path.resolve(__dirname, '../venv311/bin/python3');
    const workerVenv = path.resolve(__dirname, '../venv/bin/python3');
    const pythonBin = process.env.DIARIZATION_PYTHON_BIN
        ? path.resolve(process.env.DIARIZATION_PYTHON_BIN)
        : (fs.existsSync(venv311) ? venv311 : workerVenv);

    const token = process.env.PYANNOTE_ACCESS_TOKEN || process.env.HUGGINGFACE_TOKEN || process.env.HF_TOKEN;
    const args = [pythonScript, chunkPath];
    if (token) args.push('--token', token);

    const { stdout } = await runProcess(pythonBin, args);
    const parsed = JSON.parse(stdout) as SpeakerTurn[];
    if (!Array.isArray(parsed)) throw new Error('Invalid diarization JSON output (expected array).');
    return parsed;
}

function absoluteTurns(turns: SpeakerTurn[], offsetSec: number): SpeakerTurn[] {
    return turns.map((t) => ({ ...t, start: t.start + offsetSec, end: t.end + offsetSec }));
}

function pickAnchorSpeaker(turns: SpeakerTurn[], anchorStart: number, anchorEnd: number): string | null {
    const scores = new Map<string, number>();
    for (const t of turns) {
        const ol = overlap(t.start, t.end, anchorStart, anchorEnd);
        if (ol <= 0) continue;
        scores.set(t.speaker, (scores.get(t.speaker) ?? 0) + ol);
    }
    if (scores.size === 0) return null;
    return [...scores.entries()].sort((a, b) => b[1] - a[1])[0][0];
}

function refineStart(
    turns: SpeakerTurn[],
    speaker: string,
    guessStart: number,
    gapJoinSec = 1.5
): number {
    const same = turns
        .filter((t) => t.speaker === speaker && t.end >= guessStart - 20 && t.start <= guessStart + 20)
        .sort((a, b) => a.start - b.start);
    if (same.length === 0) return guessStart;

    let idx = same.findIndex((t) => t.start <= guessStart && t.end >= guessStart);
    if (idx < 0) idx = same.findIndex((t) => t.start >= guessStart);
    if (idx < 0) idx = same.length - 1;

    let start = same[idx].start;
    for (let i = idx - 1; i >= 0; i--) {
        const gap = same[i + 1].start - same[i].end;
        if (gap > gapJoinSec) break;
        start = same[i].start;
    }
    return start;
}

function refineEnd(
    turns: SpeakerTurn[],
    speaker: string,
    guessEnd: number,
    gapJoinSec = 1.5
): number {
    const same = turns
        .filter((t) => t.speaker === speaker && t.end >= guessEnd - 20 && t.start <= guessEnd + 40)
        .sort((a, b) => a.start - b.start);
    if (same.length === 0) return guessEnd;

    let idx = same.findIndex((t) => t.start <= guessEnd && t.end >= guessEnd);
    if (idx < 0) idx = [...same].reverse().findIndex((t) => t.end <= guessEnd) ;
    if (idx < 0) idx = 0;
    else if (same.findIndex((t) => t.start <= guessEnd && t.end >= guessEnd) < 0) idx = same.length - 1 - idx;

    let end = same[idx].end;
    for (let i = idx + 1; i < same.length; i++) {
        const gap = same[i].start - same[i - 1].end;
        if (gap > gapJoinSec) break;
        end = same[i].end;
    }
    return end;
}

function applyPadding(speakerStart: number, speakerEnd: number, segments: Segment[]) {
    const prevSpeech = [...segments].reverse().find((s) => s.end <= speakerStart);
    const nextSpeech = segments.find((s) => s.start >= speakerEnd);

    const gapBefore = Math.max(0, speakerStart - (prevSpeech?.end ?? 0));
    const prePad = Math.min(10, gapBefore);
    const clipStart = Math.max(0, speakerStart - prePad);

    const gapAfter = nextSpeech ? Math.max(0, nextSpeech.start - speakerEnd) : 10;
    const postPad = Math.min(10, gapAfter);
    const clipEnd = nextSpeech ? Math.min(nextSpeech.start, speakerEnd + postPad) : speakerEnd + postPad;

    return {
        clip_start_sec: clipStart,
        clip_end_sec: clipEnd,
        applied_pre_pad_sec: prePad,
        applied_post_pad_sec: postPad,
        next_speech_start_sec: nextSpeech?.start ?? null
    };
}

async function main() {
    const workDirArg = process.argv[2] || 'apps/test_data/ingest_test';
    const workDir = path.resolve(process.cwd(), workDirArg);
    const audioPath = process.argv[3]
        ? path.resolve(process.cwd(), process.argv[3])
        : path.join(workDir, 'audio.wav');
    const transcriptPath = process.argv[4]
        ? path.resolve(process.cwd(), process.argv[4])
        : path.join(workDir, 'transcript.corrected.json');
    const boundaryPath = process.argv[5]
        ? path.resolve(process.cwd(), process.argv[5])
        : path.join(workDir, 'sermon.boundaries.openai.json');

    if (!fs.existsSync(audioPath)) throw new Error(`Audio not found: ${audioPath}`);
    if (!fs.existsSync(transcriptPath)) throw new Error(`Transcript not found: ${transcriptPath}`);
    if (!fs.existsSync(boundaryPath)) throw new Error(`Boundary guess not found: ${boundaryPath}`);

    const transcriptExt = path.extname(transcriptPath).toLowerCase();
    const transcriptRaw =
        transcriptExt === '.srt'
            ? parseSrtToSegments(fs.readFileSync(transcriptPath, 'utf8'))
            : JSON.parse(fs.readFileSync(transcriptPath, 'utf8'));
    const segments = normalizeSegments(transcriptRaw);
    const guess = parseGuess(boundaryPath);

    const audioEnd = segments.length > 0 ? segments[segments.length - 1].end : guess.end + 1;
    const preScanSec = Number(process.env.TARGET_DIAR_PRE_SCAN_SEC ?? 8 * 60);
    const postScanSec = Number(process.env.TARGET_DIAR_POST_SCAN_SEC ?? 8 * 60);
    const anchorSec = Number(process.env.TARGET_DIAR_ANCHOR_SEC ?? 80);

    const preWindowStart = Math.max(0, guess.start - preScanSec);
    const preWindowEnd = Math.min(audioEnd, guess.start + anchorSec);
    const postWindowStart = Math.max(0, guess.end - anchorSec);
    const postWindowEnd = Math.min(audioEnd, guess.end + postScanSec);

    const tmpDir = path.join(workDir, 'processed');
    fs.mkdirSync(tmpDir, { recursive: true });
    const preChunk = path.join(tmpDir, 'diar.pre.wav');
    const postChunk = path.join(tmpDir, 'diar.post.wav');

    console.log('Extracting pre window chunk...');
    await extractChunk(audioPath, preChunk, preWindowStart, preWindowEnd);
    console.log('Running diarization on pre window...');
    const preTurnsAbs = absoluteTurns(await runDiarize(preChunk), preWindowStart);

    console.log('Extracting post window chunk...');
    await extractChunk(audioPath, postChunk, postWindowStart, postWindowEnd);
    console.log('Running diarization on post window...');
    const postTurnsAbs = absoluteTurns(await runDiarize(postChunk), postWindowStart);

    const preAnchorStart = Math.max(preWindowStart, guess.start + 10);
    const preAnchorEnd = Math.min(preWindowEnd, guess.start + 70);
    const postAnchorStart = Math.max(postWindowStart, guess.end - 70);
    const postAnchorEnd = Math.min(postWindowEnd, guess.end - 10);

    const preSpeaker = pickAnchorSpeaker(preTurnsAbs, preAnchorStart, preAnchorEnd);
    const postSpeaker = pickAnchorSpeaker(postTurnsAbs, postAnchorStart, postAnchorEnd) ?? preSpeaker;
    if (!preSpeaker || !postSpeaker) throw new Error('Failed to resolve anchor speaker from targeted windows.');

    const refinedStart = refineStart(preTurnsAbs, preSpeaker, guess.start);
    const refinedEnd = refineEnd(postTurnsAbs, postSpeaker, guess.end);
    const padded = applyPadding(refinedStart, refinedEnd, segments);

    const output = {
        guess: {
            start_sec: guess.start,
            end_sec: guess.end
        },
        targeted_windows: {
            pre: { start_sec: preWindowStart, end_sec: preWindowEnd, anchor_speaker: preSpeaker, turns: preTurnsAbs },
            post: { start_sec: postWindowStart, end_sec: postWindowEnd, anchor_speaker: postSpeaker, turns: postTurnsAbs }
        },
        refined_speaker_bounds: {
            start_sec: refinedStart,
            end_sec: refinedEnd
        },
        final_clip_bounds: padded
    };

    const outPath = path.join(workDir, 'sermon.boundaries.targeted-diarization.json');
    fs.writeFileSync(outPath, JSON.stringify(output, null, 2));
    console.log(`Wrote targeted diarization boundaries: ${outPath}`);
}

main().catch((err) => {
    console.error('test-targeted-diarization failed:', err);
    process.exit(1);
});
