import path from 'path';
import fs from 'fs';
import { spawn } from 'child_process';

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
            if (code !== 0) return reject(new Error(`${cmd} exited ${code}. stderr tail: ${stderr.slice(-2000)}`));
            resolve({ stdout, stderr });
        });
    });
}

function resolvePythonBin(): string {
    const explicit = process.env.OCR_PYTHON_BIN || process.env.DIARIZATION_PYTHON_BIN || process.env.FACE_PYTHON_BIN;
    const candidates = [
        explicit ? path.resolve(explicit) : '',
        path.resolve(__dirname, '../venv311/bin/python3'),
        path.resolve(__dirname, '../venv/bin/python3'),
        path.resolve(process.cwd(), 'apps/worker/venv311/bin/python3'),
        path.resolve(process.cwd(), 'apps/worker/venv/bin/python3'),
        'python3',
    ].filter(Boolean);
    return candidates.find((candidate) => candidate === 'python3' || fs.existsSync(candidate)) || 'python3';
}

function resolveInputPath(raw: string): string {
    const direct = path.resolve(raw);
    if (fs.existsSync(direct)) return direct;
    const oneUp = path.resolve(process.cwd(), '..', raw);
    if (fs.existsSync(oneUp)) return oneUp;
    const twoUp = path.resolve(process.cwd(), '..', '..', raw);
    if (fs.existsSync(twoUp)) return twoUp;
    return direct;
}

async function main() {
    const workDirArg = process.argv[2];
    if (!workDirArg) {
        throw new Error('Usage: npm run test:slide-ocr-v2 -- <workDir> [lowVideoPath] [hqVideoPath]');
    }
    const workDir = resolveInputPath(workDirArg);
    const lowVideoPath = process.argv[3]
        ? resolveInputPath(process.argv[3])
        : (fs.existsSync(path.join(workDir, 'source.light.mp4')) ? path.join(workDir, 'source.light.mp4') : path.join(workDir, 'source.mp4'));
    const hqVideoPath = process.argv[4]
        ? resolveInputPath(process.argv[4])
        : (fs.existsSync(path.join(workDir, 'source.mp4')) ? path.join(workDir, 'source.mp4') : lowVideoPath);
    const outPath = path.join(workDir, 'slide.events.json');
    const summaryMdPath = path.join(workDir, 'slide.ocr.positive-findings.md');
    const scriptPath = path.resolve(__dirname, 'pipeline/python/slide_ocr_v2.py');
    const pythonBin = resolvePythonBin();

    if (!fs.existsSync(lowVideoPath)) throw new Error(`low video not found: ${lowVideoPath}`);
    if (!fs.existsSync(hqVideoPath)) throw new Error(`hq video not found: ${hqVideoPath}`);
    if (!fs.existsSync(scriptPath)) throw new Error(`slide_ocr_v2 script not found: ${scriptPath}`);
    fs.mkdirSync(workDir, { recursive: true });

    const args = [scriptPath, lowVideoPath, '--hq-video', hqVideoPath, '--out', outPath, '--summary-md', summaryMdPath];
    await runProcess(pythonBin, args);
    const parsed = JSON.parse(fs.readFileSync(outPath, 'utf8'));
    const events = Array.isArray(parsed?.events) ? parsed.events.length : 0;
    const candidates = Array.isArray(parsed?.candidates) ? parsed.candidates.filter((c: any) => Boolean(c?.selected)).length : 0;
    console.log(`Wrote slide OCR events -> ${outPath} selected_candidates=${candidates} events=${events}`);
}

main().catch((err) => {
    console.error('test-slide-ocr-v2 failed:', err);
    process.exit(1);
});
