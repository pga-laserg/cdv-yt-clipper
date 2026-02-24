import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import dotenv from 'dotenv';

dotenv.config({ path: path.resolve(__dirname, '../../../.env') });
dotenv.config({ path: path.resolve(__dirname, '../../web/.env.local') });

function runProcess(cmd: string, args: string[]): Promise<{ stdout: string; stderr: string }> {
    return new Promise((resolve, reject) => {
        const proc = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'] });
        let stdout = '';
        let stderr = '';
        proc.stdout.on('data', (d) => {
            const text = d.toString();
            stdout += text;
            if (text.trim()) console.log(text.trimEnd());
        });
        proc.stderr.on('data', (d) => {
            const text = d.toString();
            stderr += text;
            if (text.trim()) console.error(text.trimEnd());
        });
        proc.on('error', reject);
        proc.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`${cmd} exited with code ${code}. stderr tail: ${stderr.slice(-1200)}`));
                return;
            }
            resolve({ stdout, stderr });
        });
    });
}

async function ensureAudioWav(videoPath: string, audioPath: string) {
    if (fs.existsSync(audioPath)) return;
    fs.mkdirSync(path.dirname(audioPath), { recursive: true });
    await runProcess('ffmpeg', [
        '-y',
        '-i',
        videoPath,
        '-vn',
        '-ac',
        '1',
        '-ar',
        '16000',
        '-acodec',
        'pcm_s16le',
        audioPath
    ]);
}

function resolvePythonBin(): string {
    if (process.env.FACE_PYTHON_BIN) return path.resolve(process.env.FACE_PYTHON_BIN);
    const venv311 = path.resolve(__dirname, '../venv311/bin/python3');
    if (fs.existsSync(venv311)) return venv311;
    const workerVenv = path.resolve(__dirname, '../venv/bin/python3');
    if (fs.existsSync(workerVenv)) return workerVenv;
    return 'python3';
}

function resolveExistingPath(candidates: string[]): string {
    for (const p of candidates) {
        if (fs.existsSync(p)) return p;
    }
    return candidates[0];
}

async function runWhisperTranscription(audioPath: string, workDir: string): Promise<string> {
    const pythonBin =
        process.env.WHISPER_PYTHON_BIN
            ? path.resolve(process.env.WHISPER_PYTHON_BIN)
            : resolveExistingPath([
                  path.resolve(__dirname, '../venv/bin/python3'),
                  path.resolve(__dirname, '../venv311/bin/python3'),
                  'python3'
              ]);
    const script = resolveExistingPath([
        path.resolve(__dirname, 'pipeline/python/transcribe.py'),
        path.resolve(__dirname, '../src/pipeline/python/transcribe.py')
    ]);
    const model = process.env.WHISPER_MODEL || 'small';
    const beamSize = Number(process.env.WHISPER_BEAM_SIZE ?? 5);
    const { stdout } = await runProcess(pythonBin, [
        script,
        audioPath,
        '--model',
        model,
        '--beam-size',
        String(beamSize)
    ]);
    const parsed = JSON.parse(stdout);
    if (!Array.isArray(parsed)) throw new Error('Whisper transcription output is not an array.');
    const outPath = path.join(workDir, 'transcript.whisper.json');
    fs.writeFileSync(outPath, JSON.stringify(parsed, null, 2));
    return outPath;
}

async function maybeClipVideo(workDir: string, sourceVideo: string, boundaryJsonPath: string) {
    if (String(process.env.FACE_TARGETED_SKIP_CLIP ?? 'false') === 'true') return;
    const raw = JSON.parse(fs.readFileSync(boundaryJsonPath, 'utf8')) as any;
    const clipStart = Number(raw?.final_clip_bounds?.clip_start_sec);
    const clipEnd = Number(raw?.final_clip_bounds?.clip_end_sec);
    if (!Number.isFinite(clipStart) || !Number.isFinite(clipEnd) || clipEnd <= clipStart) return;

    const processedDir = path.join(workDir, 'processed');
    fs.mkdirSync(processedDir, { recursive: true });
    const webmOut = path.join(processedDir, 'sermon_horizontal.face-targeted.webm');
    await runProcess('ffmpeg', [
        '-y',
        '-ss',
        String(clipStart),
        '-to',
        String(clipEnd),
        '-i',
        sourceVideo,
        '-c',
        'copy',
        webmOut
    ]);
}

function hasPyannoteToken(): boolean {
    const token =
        process.env.PYANNOTE_ACCESS_TOKEN || process.env.HUGGINGFACE_TOKEN || process.env.HF_TOKEN || '';
    return token.trim().length > 0;
}

async function main() {
    const workDirArg = process.argv[2] || 'apps/test_data/e2e_live_youtube_redownload';
    const workDir = path.resolve(process.cwd(), workDirArg);
    const sourceVideo = process.argv[3]
        ? path.resolve(process.cwd(), process.argv[3])
        : path.join(workDir, 'source.mp4');
    const transcriptArgPath = process.argv[4]
        ? path.resolve(process.cwd(), process.argv[4])
        : path.join(workDir, 'transcript.whisper.json');
    const audioPath = process.argv[5]
        ? path.resolve(process.cwd(), process.argv[5])
        : path.join(workDir, 'audio.wav');

    if (!fs.existsSync(sourceVideo)) throw new Error(`Video not found: ${sourceVideo}`);
    if (transcriptArgPath && process.argv[4] && !fs.existsSync(transcriptArgPath)) {
        throw new Error(`Transcript not found: ${transcriptArgPath}`);
    }

    await ensureAudioWav(sourceVideo, audioPath);
    const transcriptPath =
        process.argv[4] && fs.existsSync(transcriptArgPath)
            ? transcriptArgPath
            : await runWhisperTranscription(audioPath, workDir);

    const faceOut = path.join(workDir, 'sermon.boundaries.face-pass.json');
    const faceScript = resolveExistingPath([
        path.resolve(__dirname, 'pipeline/python/face_sermon_bounds.py'),
        path.resolve(__dirname, '../src/pipeline/python/face_sermon_bounds.py')
    ]);
    const pythonBin = resolvePythonBin();
    await runProcess(pythonBin, [faceScript, sourceVideo, '--out', faceOut]);

    if (!hasPyannoteToken()) {
        console.warn(
            'Skipping targeted diarization: missing PYANNOTE_ACCESS_TOKEN/HUGGINGFACE_TOKEN/HF_TOKEN. Face-pass output is ready.'
        );
        console.log(`Face-only output: ${faceOut}`);
        return;
    }

    await runProcess('npm', ['run', 'test:targeted-diarization', '--', workDir, audioPath, transcriptPath, faceOut]);

    const targetedOut = path.join(workDir, 'sermon.boundaries.targeted-diarization.json');
    await maybeClipVideo(workDir, sourceVideo, targetedOut);
    console.log(`Face+targeted run complete. Output: ${targetedOut}`);
}

main().catch((err) => {
    console.error('test-face-targeted-sermon failed:', err);
    process.exit(1);
});
