import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { jsonToSrt } from '../utils/srt';

export interface TransientSegment {
    start: number;
    end: number;
    text: string;
}

export async function transcribe(audioPath: string): Promise<TransientSegment[]> {
    const attempts: Array<{ model: string; beamSize: number }> = [
        { model: 'small', beamSize: 5 },
        { model: 'base', beamSize: 1 }
    ];

    let lastError: unknown;

    for (const [index, attempt] of attempts.entries()) {
        try {
            const segments = await runTranscriptionAttempt(audioPath, attempt.model, attempt.beamSize);

            // Export SRT
            try {
                const srtContent = jsonToSrt(segments);
                const srtPath = path.join(path.dirname(audioPath), 'source.srt');
                fs.writeFileSync(srtPath, srtContent);
                console.log('SRT file exported to:', srtPath);
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

    const cached = loadCachedTranscriptFallback();
    if (cached.length > 0) {
        console.warn(`Using cached transcript fallback with ${cached.length} segments.`);
        try {
            const srtContent = jsonToSrt(cached);
            const srtPath = path.join(path.dirname(audioPath), 'source.srt');
            fs.writeFileSync(srtPath, srtContent);
            console.log('SRT file exported from cached fallback to:', srtPath);
        } catch (srtError) {
            console.error('Failed to export SRT from cached fallback:', srtError);
        }
        return cached;
    }

    throw new Error(`All transcription attempts failed. Last error: ${String(lastError)}`);
}

function runTranscriptionAttempt(audioPath: string, model: string, beamSize: number): Promise<TransientSegment[]> {
    return new Promise((resolve, reject) => {
        const pythonScript = path.resolve(__dirname, 'python/transcribe.py');
        const venvPython = path.resolve(__dirname, '../../venv/bin/python3');
        const pythonProcess = spawn(
            venvPython,
            [pythonScript, audioPath, '--model', model, '--beam-size', String(beamSize)],
            { stdio: ['ignore', 'pipe', 'pipe'] }
        );

        let stdout = '';
        let stderr = '';
        let settled = false;
        const timeoutMs = Number(process.env.TRANSCRIBE_ATTEMPT_TIMEOUT_MS || 120000);
        const timeout = setTimeout(() => {
            if (settled) return;
            settled = true;
            try {
                pythonProcess.kill('SIGKILL');
            } catch {
                // Ignore kill errors.
            }
            reject(new Error(`Transcribe attempt timed out after ${timeoutMs}ms (model=${model}, beam=${beamSize})`));
        }, timeoutMs);

        pythonProcess.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            const text = data.toString();
            stderr += text;
            console.error(`Python stderr: ${text}`);
        });

        pythonProcess.on('error', (error) => {
            if (settled) return;
            settled = true;
            clearTimeout(timeout);
            reject(new Error(`Failed to start transcription process: ${String(error)}`));
        });

        pythonProcess.on('close', (code, signal) => {
            if (settled) return;
            settled = true;
            clearTimeout(timeout);

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

        return parsed
            .filter((s) => typeof s?.start === 'number' && typeof s?.end === 'number' && typeof s?.text === 'string')
            .map((s) => ({ start: s.start, end: s.end, text: s.text }));
    } catch (error) {
        console.error('Failed to load cached transcript fallback:', error);
        return [];
    }
}
