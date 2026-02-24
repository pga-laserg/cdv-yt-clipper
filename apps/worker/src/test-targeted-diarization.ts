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
interface SpeakerTurnReview extends SpeakerTurn {
    transcript_text: string;
}

interface GuessBounds {
    start: number;
    end: number;
    startCandidates: number[];
    endCandidates: number[];
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

function isNonSpeech(text?: string): boolean {
    const t = (text ?? '').toLowerCase().trim();
    if (!t) return true;
    if (t === 'music' || t === 'música' || t === 'piano' || t === 'silence' || t === 'silencio') return true;
    if (/^\[.*\]$/.test(t) || /^\(.*\)$/.test(t)) return true;
    if (/^(music|música|piano|instrumental|aplausos|applause|silence|silencio)\b/.test(t)) return true;
    return false;
}

function hasOtherSpeakerAround(
    turns: SpeakerTurn[],
    speaker: string,
    start: number,
    end: number,
    minOverlapSec = 0.35
): boolean {
    for (const t of turns) {
        if (t.speaker === speaker) continue;
        if (overlap(t.start, t.end, start, end) >= minOverlapSec) return true;
    }
    return false;
}

function parseGuess(boundaryJsonPath: string): GuessBounds {
    const raw = JSON.parse(fs.readFileSync(boundaryJsonPath, 'utf8')) as any;
    const normalizeCandidates = (items: any[] | undefined, fallback: number): number[] => {
        const values = Array.isArray(items)
            ? items.map((x) => Number(x?.sec ?? x)).filter((v) => Number.isFinite(v))
            : [];
        values.unshift(fallback);
        const out: number[] = [];
        const seen = new Set<number>();
        for (const v of values) {
            const key = Math.round(v * 100) / 100;
            if (seen.has(key)) continue;
            seen.add(key);
            out.push(v);
        }
        return out.slice(0, 3);
    };

    if (raw?.raw?.speaker_start_sec != null && raw?.raw?.speaker_end_sec != null) {
        const start = Number(raw.raw.speaker_start_sec);
        const end = Number(raw.raw.speaker_end_sec);
        return {
            start,
            end,
            startCandidates: normalizeCandidates(raw.raw.start_candidates, start),
            endCandidates: normalizeCandidates(raw.raw.end_candidates, end)
        };
    }
    if (raw?.speaker_start_sec != null && raw?.speaker_end_sec != null) {
        const start = Number(raw.speaker_start_sec);
        const end = Number(raw.speaker_end_sec);
        return {
            start,
            end,
            startCandidates: normalizeCandidates(raw.start_candidates, start),
            endCandidates: normalizeCandidates(raw.end_candidates, end)
        };
    }
    if (raw?.start != null && raw?.end != null) {
        const start = Number(raw.start);
        const end = Number(raw.end);
        const startCandidates = Array.isArray(raw.start_candidates)
            ? raw.start_candidates.map((v: any) => Number(v)).filter((v: number) => Number.isFinite(v))
            : [start];
        const endCandidates = Array.isArray(raw.end_candidates)
            ? raw.end_candidates.map((v: any) => Number(v)).filter((v: number) => Number.isFinite(v))
            : [end];
        return {
            start,
            end,
            startCandidates: startCandidates.length > 0 ? startCandidates : [start],
            endCandidates: endCandidates.length > 0 ? endCandidates : [end]
        };
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
    const pythonScriptCandidates = [
        path.resolve(__dirname, 'pipeline/python/diarize.py'),
        path.resolve(__dirname, '../src/pipeline/python/diarize.py')
    ];
    const pythonScript = pythonScriptCandidates.find((p) => fs.existsSync(p)) ?? pythonScriptCandidates[0];
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

function absoluteTurns(turns: SpeakerTurn[], offsetSec: number, speakerPrefix?: string): SpeakerTurn[] {
    return turns.map((t) => ({
        ...t,
        speaker: speakerPrefix ? `${speakerPrefix}:${t.speaker}` : t.speaker,
        start: t.start + offsetSec,
        end: t.end + offsetSec
    }));
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
    gapJoinSec = 1.5,
    lookbackSec = 120
): number {
    const same = turns
        .filter((t) => t.speaker === speaker && t.end >= guessStart - lookbackSec && t.start <= guessStart + 25)
        .sort((a, b) => a.start - b.start);
    if (same.length === 0) return guessStart;

    let idx = same.findIndex((t) => t.start <= guessStart && t.end >= guessStart);
    if (idx < 0) idx = same.findIndex((t) => t.start >= guessStart);
    if (idx < 0) idx = same.length - 1;

    let start = same[idx].start;
    for (let i = idx - 1; i >= 0; i--) {
        const gap = same[i + 1].start - same[i].end;
        if (gap > gapJoinSec) break;
        const bridgeStart = Math.min(same[i].end, same[i + 1].start) - 0.2;
        const bridgeEnd = Math.max(same[i].end, same[i + 1].start) + 0.2;
        if (hasOtherSpeakerAround(turns, speaker, bridgeStart, bridgeEnd)) break;
        start = same[i].start;
    }
    return start;
}

function refineEnd(
    turns: SpeakerTurn[],
    speaker: string,
    guessEnd: number,
    gapJoinSec = 1.5,
    lookaheadSec = 120
): number {
    const same = turns
        .filter((t) => t.speaker === speaker && t.end >= guessEnd - 40 && t.start <= guessEnd + lookaheadSec)
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
        const bridgeStart = Math.min(same[i - 1].end, same[i].start) - 0.2;
        const bridgeEnd = Math.max(same[i - 1].end, same[i].start) + 0.2;
        if (hasOtherSpeakerAround(turns, speaker, bridgeStart, bridgeEnd)) break;
        end = same[i].end;
    }
    return end;
}

function applyPadding(
    speakerStart: number,
    speakerEnd: number,
    segments: Segment[],
    turns: SpeakerTurn[],
    startSpeaker: string,
    endSpeaker: string
) {
    const speechSegments = segments.filter((s) => !isNonSpeech(s.text));
    const prevSpeech = [...speechSegments].reverse().find((s) => s.end <= speakerStart);
    const nextSpeech = speechSegments.find((s) => s.start >= speakerEnd);

    const prevOtherTurn = [...turns]
        .filter((t) => t.speaker !== startSpeaker && t.end <= speakerStart)
        .sort((a, b) => b.end - a.end)[0];
    const nextOtherTurn = turns
        .filter((t) => t.speaker !== endSpeaker && t.start >= speakerEnd)
        .sort((a, b) => a.start - b.start)[0];

    const prevBoundary = Math.max(prevSpeech?.end ?? 0, prevOtherTurn?.end ?? 0);
    const nextBoundary = Math.min(
        nextSpeech?.start ?? Number.POSITIVE_INFINITY,
        nextOtherTurn?.start ?? Number.POSITIVE_INFINITY
    );

    const gapBefore = Math.max(0, speakerStart - prevBoundary);
    const prePad = Math.min(10, gapBefore);
    const clipStart = Math.max(0, speakerStart - prePad);

    const gapAfter = Number.isFinite(nextBoundary) ? Math.max(0, nextBoundary - speakerEnd) : 10;
    const postPad = Math.min(10, gapAfter);
    const clipEnd = Number.isFinite(nextBoundary)
        ? Math.min(nextBoundary, speakerEnd + postPad)
        : speakerEnd + postPad;

    return {
        clip_start_sec: clipStart,
        clip_end_sec: clipEnd,
        applied_pre_pad_sec: prePad,
        applied_post_pad_sec: postPad,
        next_speech_start_sec: Number.isFinite(nextBoundary) ? nextBoundary : null,
        prev_other_speaker_end_sec: prevOtherTurn?.end ?? null,
        next_other_speaker_start_sec: nextOtherTurn?.start ?? null
    };
}

function transcriptTextForTurn(turn: SpeakerTurn, segments: Segment[]): string {
    const pieces: string[] = [];
    for (const s of segments) {
        const ol = overlap(turn.start, turn.end, s.start, s.end);
        if (ol <= 0.2) continue;
        const t = (s.text ?? '').trim();
        if (!t) continue;
        pieces.push(t);
    }
    return pieces.join(' ').replace(/\s+/g, ' ').trim();
}

function enrichTurnsWithTranscript(turns: SpeakerTurn[], segments: Segment[]): SpeakerTurnReview[] {
    return turns.map((t) => ({
        ...t,
        transcript_text: transcriptTextForTurn(t, segments)
    }));
}

function previousTurnBefore(turns: SpeakerTurn[], t: number): SpeakerTurn | null {
    for (let i = turns.length - 1; i >= 0; i--) {
        if (turns[i].end <= t + 0.001) return turns[i];
    }
    return null;
}

function nextTurnAfter(turns: SpeakerTurn[], t: number): SpeakerTurn | null {
    for (let i = 0; i < turns.length; i++) {
        if (turns[i].start >= t - 0.001) return turns[i];
    }
    return null;
}

function findLatestOtherToTargetStartTransition(
    turns: SpeakerTurn[],
    targetSpeaker: string,
    upperBound: number
): number | null {
    let candidate: number | null = null;
    for (let i = 1; i < turns.length; i++) {
        const prev = turns[i - 1];
        const cur = turns[i];
        if (cur.start > upperBound + 15) break;
        if (prev.speaker !== targetSpeaker && cur.speaker === targetSpeaker) {
            candidate = cur.start;
        }
    }
    return candidate;
}

function findEarliestTargetToOtherEndTransition(
    turns: SpeakerTurn[],
    targetSpeaker: string,
    lowerBound: number
): number | null {
    for (let i = 1; i < turns.length; i++) {
        const prev = turns[i - 1];
        const cur = turns[i];
        if (prev.end < lowerBound - 15) continue;
        if (prev.speaker === targetSpeaker && cur.speaker !== targetSpeaker) {
            return prev.end;
        }
    }
    return null;
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
    const preScanSec = Number(process.env.TARGET_DIAR_PRE_SCAN_SEC ?? 120);
    const postScanSec = Number(process.env.TARGET_DIAR_POST_SCAN_SEC ?? 120);
    const anchorSec = Number(process.env.TARGET_DIAR_ANCHOR_SEC ?? 30);
    const maxConfirmGapSec = Number(process.env.TARGET_DIAR_MAX_CONFIRM_GAP_SEC ?? 30);

    const startCandidates = guess.startCandidates.length > 0 ? guess.startCandidates : [guess.start];
    const endCandidates = guess.endCandidates.length > 0 ? guess.endCandidates : [guess.end];
    const earliestStartSeed = Math.min(...startCandidates);
    const latestStartSeed = Math.max(...startCandidates);
    const earliestEndSeed = Math.min(...endCandidates);
    const latestEndSeed = Math.max(...endCandidates);

    const preWindowStart = Math.max(0, earliestStartSeed - preScanSec);
    const preWindowEnd = Math.min(audioEnd, latestStartSeed + anchorSec);
    const postWindowStart = Math.max(0, earliestEndSeed - anchorSec);
    const postWindowEnd = Math.min(audioEnd, latestEndSeed + postScanSec);

    const tmpDir = path.join(workDir, 'processed');
    fs.mkdirSync(tmpDir, { recursive: true });
    const preChunk = path.join(tmpDir, 'diar.pre.wav');
    const postChunk = path.join(tmpDir, 'diar.post.wav');

    console.log('Extracting pre window chunk...');
    await extractChunk(audioPath, preChunk, preWindowStart, preWindowEnd);
    console.log('Running diarization on pre window...');
    const preTurnsAbs = absoluteTurns(await runDiarize(preChunk), preWindowStart, 'pre');

    console.log('Extracting post window chunk...');
    await extractChunk(audioPath, postChunk, postWindowStart, postWindowEnd);
    console.log('Running diarization on post window...');
    const postTurnsAbs = absoluteTurns(await runDiarize(postChunk), postWindowStart, 'post');

    const preAnchorStart = Math.max(preWindowStart, guess.start + 10);
    const preAnchorEnd = Math.min(preWindowEnd, guess.start + 70);
    const postAnchorStart = Math.max(postWindowStart, guess.end - 70);
    const postAnchorEnd = Math.min(postWindowEnd, guess.end - 10);

    const preSpeaker = pickAnchorSpeaker(preTurnsAbs, preAnchorStart, preAnchorEnd);
    const postSpeaker = pickAnchorSpeaker(postTurnsAbs, postAnchorStart, postAnchorEnd) ?? preSpeaker;
    if (!preSpeaker || !postSpeaker) throw new Error('Failed to resolve anchor speaker from targeted windows.');

    const startSpeaker = preSpeaker;
    const endSpeaker = postSpeaker;
    const allTurns = [...preTurnsAbs, ...postTurnsAbs].sort((a, b) => a.start - b.start);
    let refinedStart = refineStart(preTurnsAbs, startSpeaker, guess.start);
    let refinedEnd = refineEnd(postTurnsAbs, endSpeaker, guess.end);
    let startChosenBy = 'primary';
    let endChosenBy = 'primary';

    for (const c of startCandidates) {
        const candidateStart = refineStart(preTurnsAbs, startSpeaker, c);
        const prev = previousTurnBefore(preTurnsAbs, candidateStart);
        if (prev && prev.speaker !== startSpeaker && candidateStart - prev.end <= maxConfirmGapSec) {
            refinedStart = candidateStart;
            startChosenBy = `candidate:${c}`;
            break;
        }
    }

    if (startChosenBy === 'primary') {
        const fallbackStart = findLatestOtherToTargetStartTransition(preTurnsAbs, startSpeaker, refinedStart);
        if (fallbackStart != null) {
            refinedStart = refineStart(preTurnsAbs, startSpeaker, fallbackStart);
            startChosenBy = 'sequential-fallback';
        }
    }

    for (const c of endCandidates) {
        const candidateEnd = refineEnd(postTurnsAbs, endSpeaker, c);
        const next = nextTurnAfter(postTurnsAbs, candidateEnd);
        if (next && next.speaker !== endSpeaker && next.start - candidateEnd <= maxConfirmGapSec) {
            refinedEnd = candidateEnd;
            endChosenBy = `candidate:${c}`;
            break;
        }
    }

    if (endChosenBy === 'primary') {
        const fallbackEnd = findEarliestTargetToOtherEndTransition(postTurnsAbs, endSpeaker, refinedEnd);
        if (fallbackEnd != null) {
            refinedEnd = refineEnd(postTurnsAbs, endSpeaker, fallbackEnd);
            endChosenBy = 'sequential-fallback';
        }
    }

    const padded = applyPadding(refinedStart, refinedEnd, segments, allTurns, startSpeaker, endSpeaker);
    const prevAtStart = previousTurnBefore(preTurnsAbs, refinedStart);
    const nextAtEnd = nextTurnAfter(postTurnsAbs, refinedEnd);

    const output = {
        guess: {
            start_sec: guess.start,
            end_sec: guess.end,
            start_candidates_sec: startCandidates,
            end_candidates_sec: endCandidates
        },
        selected_target_speaker: {
            start: startSpeaker,
            end: endSpeaker
        },
        boundary_confirmation: {
            start_confirmed_change_of_speaker: Boolean(prevAtStart && prevAtStart.speaker !== startSpeaker),
            end_confirmed_change_of_speaker: Boolean(nextAtEnd && nextAtEnd.speaker !== endSpeaker),
            start_chosen_by: startChosenBy,
            end_chosen_by: endChosenBy,
            start_previous_turn: prevAtStart,
            end_next_turn: nextAtEnd
        },
        targeted_windows: {
            pre: {
                start_sec: preWindowStart,
                end_sec: preWindowEnd,
                anchor_speaker: preSpeaker,
                turns: enrichTurnsWithTranscript(preTurnsAbs, segments)
            },
            post: {
                start_sec: postWindowStart,
                end_sec: postWindowEnd,
                anchor_speaker: postSpeaker,
                turns: enrichTurnsWithTranscript(postTurnsAbs, segments)
            }
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
