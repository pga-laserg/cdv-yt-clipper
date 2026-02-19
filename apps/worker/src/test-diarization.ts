import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import dotenv from 'dotenv';

dotenv.config({ path: path.resolve(__dirname, '../../../.env') });
dotenv.config({ path: path.resolve(__dirname, '../../web/.env.local') });

interface SpeakerTurn {
    start: number;
    end: number;
    speaker: string;
}

function runDiarization(audioPath: string): Promise<SpeakerTurn[]> {
    return new Promise((resolve, reject) => {
        const pythonScript = path.resolve(__dirname, 'pipeline/python/diarize.py');
        const configuredPython = process.env.DIARIZATION_PYTHON_BIN;
        const venv311Python = path.resolve(__dirname, '../venv311/bin/python3');
        const workerVenvPython = path.resolve(__dirname, '../venv/bin/python3');
        const venvPython = configuredPython
            ? path.resolve(configuredPython)
            : (fs.existsSync(venv311Python) ? venv311Python : workerVenvPython);
        const token = process.env.PYANNOTE_ACCESS_TOKEN || process.env.HUGGINGFACE_TOKEN || process.env.HF_TOKEN;

        const args = [pythonScript, audioPath];
        if (token) args.push('--token', token);

        const proc = spawn(venvPython, args, { stdio: ['ignore', 'pipe', 'pipe'] });

        let stdout = '';
        let stderr = '';

        proc.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        proc.stderr.on('data', (data) => {
            const text = data.toString();
            stderr += text;
            console.error(`[diarize.py] ${text.trimEnd()}`);
        });

        proc.on('error', (err) => reject(new Error(`Failed to start diarization process: ${String(err)}`)));

        proc.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`Diarization failed with code=${code}. stderr tail: ${stderr.slice(-1500)}`));
                return;
            }
            try {
                const parsed = JSON.parse(stdout) as SpeakerTurn[];
                if (!Array.isArray(parsed)) {
                    reject(new Error('Invalid diarization output format (expected JSON array).'));
                    return;
                }
                resolve(parsed);
            } catch (err) {
                reject(new Error(`Failed to parse diarization JSON output: ${String(err)}`));
            }
        });
    });
}

async function main() {
    const inputArg = process.argv[2] || 'apps/worker/work_dir/e9f7b0ae-242a-4392-aa93-f46d52a298b2/audio.wav';
    const audioPath = path.resolve(process.cwd(), inputArg);

    if (!fs.existsSync(audioPath)) {
        throw new Error(`Audio file not found: ${audioPath}`);
    }

    console.log(`Running diarization on ${audioPath}`);
    const turns = await runDiarization(audioPath);

    const outputPath = path.join(path.dirname(audioPath), 'diarization.pyannote3.1.json');
    fs.writeFileSync(outputPath, JSON.stringify(turns, null, 2));
    console.log(`Wrote ${turns.length} speaker turns -> ${outputPath}`);
}

main().catch((error) => {
    console.error('test-diarization failed:', error);
    process.exit(1);
});
