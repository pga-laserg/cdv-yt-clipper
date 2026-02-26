import path from 'path';
import fs from 'fs';
import { spawn } from 'child_process';
import { extractOcrProgressEvents, summarizeOcrProgress } from './utils/ocr-progress';

function runProcess(cmd: string, args: string[]): Promise<{ stdout: string; stderr: string }> {
    return new Promise((resolve, reject) => {
        const proc = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'] });
        let stdout = '';
        let stderr = '';
        proc.stdout.on('data', (d) => {
            const s = d.toString();
            stdout += s;
            process.stdout.write(s);
        });
        proc.stderr.on('data', (d) => {
            const s = d.toString();
            stderr += s;
            process.stderr.write(s);
        });
        proc.on('error', reject);
        proc.on('close', (code) => {
            if (code !== 0) return reject(new Error(`${cmd} exited ${code}. stderr tail: ${stderr.slice(-1200)}`));
            resolve({ stdout, stderr });
        });
    });
}

async function main() {
    const workDirArg = process.argv[2];
    if (!workDirArg) {
        throw new Error('Usage: npm run test:ocr-events -- <workDir> [videoPath]');
    }
    const workDir = path.resolve(workDirArg);
    const videoPath = process.argv[3]
        ? path.resolve(process.argv[3])
        : path.join(workDir, 'source.mp4');
    const outPath = path.join(workDir, 'ocr.events.json');
    const scriptPath = path.resolve(__dirname, 'pipeline/python/ocr_events.py');
    const explicitPython = process.env.OCR_PYTHON_BIN || process.env.DIARIZATION_PYTHON_BIN;
    const pythonCandidates = [
        explicitPython ? path.resolve(explicitPython) : '',
        path.resolve(__dirname, '../venv311/bin/python3'),
        path.resolve(__dirname, '../venv/bin/python3'),
        path.resolve(process.cwd(), 'apps/worker/venv311/bin/python3'),
        path.resolve(process.cwd(), 'apps/worker/venv/bin/python3'),
        'python3',
    ].filter(Boolean);
    const pythonBin = pythonCandidates.find((candidate) => candidate === 'python3' || fs.existsSync(candidate)) || 'python3';

    if (!fs.existsSync(videoPath)) throw new Error(`video not found: ${videoPath}`);
    if (!fs.existsSync(scriptPath)) throw new Error(`ocr script not found: ${scriptPath}`);
    fs.mkdirSync(workDir, { recursive: true });

    const { stderr } = await runProcess(pythonBin, [scriptPath, videoPath, '--out', outPath]);
    const parsed = JSON.parse(fs.readFileSync(outPath, 'utf8'));
    const segs = Array.isArray(parsed?.segments) ? parsed.segments.length : 0;
    const progressEvents = extractOcrProgressEvents(stderr);
    const progressSummary = summarizeOcrProgress(progressEvents);
    if (progressSummary.events > 0) {
        console.log(
            `OCR progress events=${progressSummary.events} first=${progressSummary.firstPercent?.toFixed(1)}% last=${progressSummary.lastPercent?.toFixed(1)}% eta_last=${progressSummary.lastEtaSec?.toFixed(2)}s speed_last=${progressSummary.lastVideoSpeedX?.toFixed(2)}x`
        );
    }
    console.log(`Wrote OCR events -> ${outPath} segments=${segs}`);
}

main().catch((err) => {
    console.error('test-ocr-events failed:', err);
    process.exit(1);
});
