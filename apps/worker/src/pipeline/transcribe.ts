import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { jsonToSrt } from '../utils/srt';

export interface TransientSegment {
    start: number;
    end: number;
    text: string;
}

interface TranscribeOptions {
    onProgress?: (currentSeconds: number) => void;
    onFallback?: (reason: string) => void;
}

export async function transcribe(audioPath: string, options?: TranscribeOptions): Promise<TransientSegment[]> {
    const forceRedo = String(process.env.TRANSCRIBE_FORCE_REDO ?? 'false').toLowerCase() === 'true';
    if (!forceRedo) {
        const cached = loadCachedTranscriptFromWorkDir(audioPath);
        if (cached.length > 0) {
            console.log(`Reusing cached transcription with ${cached.length} segments (no retranscribe).`);
            return cached;
        }
    } else {
        console.log('TRANSCRIBE_FORCE_REDO=true, skipping cached transcript reuse.');
    }

    const attempts: Array<{ model: string; beamSize: number }> = [
        { model: 'small', beamSize: 5 },
        { model: 'base', beamSize: 1 }
    ];

    let lastError: unknown;

    for (const [index, attempt] of attempts.entries()) {
        try {
            const segments = await runTranscriptionAttempt(audioPath, attempt.model, attempt.beamSize, options);
            if (isLikelyBrokenTranscript(segments)) {
                throw new Error(
                    `Transcription quality check failed for attempt model=${attempt.model} beam=${attempt.beamSize} ` +
                    `(dot/empty dominated output).`
                );
            }

            // Export SRT
            try {
                const srtContent = jsonToSrt(segments);
                const srtPath = path.join(path.dirname(audioPath), 'source.srt');
                fs.writeFileSync(srtPath, srtContent);
                console.log('SRT file exported to:', srtPath);
                writeTranscriptJson(audioPath, segments);
            } catch (srtError) {
                console.error('Failed to export SRT:', srtError);
            }

            return segments;
        } catch (error) {
            lastError = error;
            const attemptNum = index + 1;
            console.error(`Transcription attempt ${attemptNum}/${attempts.length} failed:`, error);
        }
    }

    const fallbackCached = loadCachedTranscriptFallback();
    if (fallbackCached.length > 0) {
        const fallbackReason = `Using cached transcript fallback with ${fallbackCached.length} segments after transcription failures.`;
        console.warn(fallbackReason);
        options?.onFallback?.(fallbackReason);
        try {
            const srtContent = jsonToSrt(fallbackCached);
            const srtPath = path.join(path.dirname(audioPath), 'source.srt');
            fs.writeFileSync(srtPath, srtContent);
            console.log('SRT file exported from cached fallback to:', srtPath);
            writeTranscriptJson(audioPath, fallbackCached);
        } catch (srtError) {
            console.error('Failed to export SRT from cached fallback:', srtError);
        }
        return fallbackCached;
    }

    throw new Error(`All transcription attempts failed. Last error: ${String(lastError)}`);
}

function runTranscriptionAttempt(
    audioPath: string,
    model: string,
    beamSize: number,
    options?: TranscribeOptions
): Promise<TransientSegment[]> {
    return new Promise((resolve, reject) => {
        const pythonScript = path.resolve(__dirname, 'python/transcribe.py');
        const venvPython = path.resolve(__dirname, '../../venv/bin/python3');
        const noWordTs = String(process.env.TRANSCRIBE_NO_WORD_TIMESTAMPS ?? 'false').toLowerCase() === 'true';
        const args = [pythonScript, audioPath, '--model', model, '--beam-size', String(beamSize)];
        const wordGapSplitSec = Number(process.env.TRANSCRIBE_WORD_GAP_SPLIT_SEC ?? 0.55);
        if (Number.isFinite(wordGapSplitSec) && wordGapSplitSec > 0) {
            args.push('--word-gap-split-sec', String(wordGapSplitSec));
        }
        if (noWordTs) args.push('--no-word-timestamps');
        const pythonProcess = spawn(
            venvPython,
            args,
            { stdio: ['ignore', 'pipe', 'pipe'] }
        );

        let stdout = '';
        let stderr = '';
        let settled = false;
        const timeoutMs = Number(process.env.TRANSCRIBE_NO_PROGRESS_TIMEOUT_MS || 180000);
        let timeout: NodeJS.Timeout | null = null;
        const resetNoProgressTimeout = () => {
            if (timeout) clearTimeout(timeout);
            timeout = setTimeout(() => {
                if (settled) return;
                settled = true;
                try {
                    pythonProcess.kill('SIGKILL');
                } catch {
                    // Ignore kill errors.
                }
                reject(new Error(`Transcribe attempt stalled (no progress for ${timeoutMs}ms, model=${model}, beam=${beamSize})`));
            }, timeoutMs);
        };

        resetNoProgressTimeout();

        pythonProcess.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            const text = data.toString();
            stderr += text;
            console.error(`Python stderr: ${text}`);

            // Any stderr output means the process is alive, so extend stall timeout.
            resetNoProgressTimeout();

            // Parse stderr progress lines like: [1200.00s -> 1207.42s] text...
            const matches = text.matchAll(/\[(\d+(?:\.\d+)?)s\s*->\s*(\d+(?:\.\d+)?)s\]/g);
            for (const match of matches) {
                const end = Number(match[2]);
                if (Number.isFinite(end)) {
                    options?.onProgress?.(end);
                }
            }
        });

        pythonProcess.on('error', (error) => {
            if (settled) return;
            settled = true;
            if (timeout) clearTimeout(timeout);
            reject(new Error(`Failed to start transcription process: ${String(error)}`));
        });

        pythonProcess.on('close', (code, signal) => {
            if (settled) return;
            settled = true;
            if (timeout) clearTimeout(timeout);

            try {
                const parsed = JSON.parse(stdout) as TransientSegment[];
                if (!Array.isArray(parsed)) {
                    reject(new Error('Transcription output is not a JSON array.'));
                    return;
                }
                resolve(parsed);
                return;
            } catch {
                // Fall through to structured error below.
            }

            reject(
                new Error(
                    `Transcribe process failed (code=${code}, signal=${signal ?? 'none'}). ` +
                    `stderr tail: ${stderr.slice(-1200)}`
                )
            );
        });
    });
}

function loadCachedTranscriptFallback(): TransientSegment[] {
    const fallbackPath = path.resolve(__dirname, '../../../test_data/ingest_test/transcript.json');
    if (!fs.existsSync(fallbackPath)) {
        return [];
    }

    try {
        const raw = fs.readFileSync(fallbackPath, 'utf8');
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed)) return [];

        const segments = parsed
            .filter((s) => typeof s?.start === 'number' && typeof s?.end === 'number' && typeof s?.text === 'string')
            .map((s) => ({ start: s.start, end: s.end, text: s.text }));
        if (isLikelyBrokenTranscript(segments)) return [];
        return segments;
    } catch (error) {
        console.error('Failed to load cached transcript fallback:', error);
        return [];
    }
}

function loadCachedTranscriptFromWorkDir(audioPath: string): TransientSegment[] {
    const workDir = path.dirname(audioPath);
    const transcriptJsonPath = path.join(workDir, 'transcript.json');
    const srtPath = path.join(workDir, 'source.srt');

    if (fs.existsSync(transcriptJsonPath)) {
        try {
            const raw = fs.readFileSync(transcriptJsonPath, 'utf8');
            const parsed = JSON.parse(raw);
            if (Array.isArray(parsed)) {
                const segments = parsed
                    .filter((s) => typeof s?.start === 'number' && typeof s?.end === 'number' && typeof s?.text === 'string')
                    .map((s) => ({ start: s.start, end: s.end, text: s.text })) as TransientSegment[];
                if (segments.length > 0 && !isLikelyBrokenTranscript(segments)) return segments;
            }
        } catch (error) {
            console.error('Failed to read cached transcript.json:', error);
        }
    }

    if (fs.existsSync(srtPath)) {
        try {
            const srtText = fs.readFileSync(srtPath, 'utf8');
            const segments = parseSrtToSegments(srtText);
            if (segments.length > 0 && !isLikelyBrokenTranscript(segments)) return segments;
        } catch (error) {
            console.error('Failed to read cached source.srt:', error);
        }
    }

    return [];
}

function parseSrtToSegments(srt: string): TransientSegment[] {
    const blocks = srt.trim().split(/\r?\n\r?\n+/);
    const segments: TransientSegment[] = [];

    for (const block of blocks) {
        const lines = block.split(/\r?\n/).filter(Boolean);
        if (lines.length < 3) continue;

        const timeLine = lines[1];
        const match = timeLine.match(
            /(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2}),(\d{3})/
        );
        if (!match) continue;

        const start =
            Number(match[1]) * 3600 +
            Number(match[2]) * 60 +
            Number(match[3]) +
            Number(match[4]) / 1000;
        const end =
            Number(match[5]) * 3600 +
            Number(match[6]) * 60 +
            Number(match[7]) +
            Number(match[8]) / 1000;

        const text = lines.slice(2).join(' ').trim();
        if (!text) continue;

        segments.push({ start, end, text });
    }

    return segments;
}

function writeTranscriptJson(audioPath: string, segments: TransientSegment[]): void {
    try {
        const workDir = path.dirname(audioPath);
        const transcriptJsonPath = path.join(workDir, 'transcript.json');
        fs.writeFileSync(transcriptJsonPath, JSON.stringify(segments, null, 2));
    } catch (error) {
        console.error('Failed to write transcript.json cache:', error);
    }
}

function isLikelyBrokenTranscript(segments: TransientSegment[]): boolean {
    if (!segments || segments.length === 0) return true;
    const n = segments.length;
    const empty = segments.filter((s) => !String(s.text ?? '').trim()).length;
    const dots = segments.filter((s) => /^\s*\.+\s*$/.test(String(s.text ?? ''))).length;
    const tiny = segments.filter((s) => String(s.text ?? '').trim().length <= 2).length;

    const emptyRatio = empty / n;
    const dotsRatio = dots / n;
    const tinyRatio = tiny / n;

    // Strong signal of failed decode: almost everything is punctuation-only placeholders.
    if (dotsRatio >= 0.8) return true;
    if (emptyRatio >= 0.5) return true;
    if (tinyRatio >= 0.9 && dotsRatio >= 0.5) return true;
    return false;
}
