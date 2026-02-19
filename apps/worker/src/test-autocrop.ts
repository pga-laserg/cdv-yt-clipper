import { spawn } from 'child_process';
import path from 'path';

async function testAutocrop() {
    const videoPath = path.resolve(__dirname, '../../test_data/ingest_test/source.mp4');
    const pythonScript = path.resolve(__dirname, 'pipeline/python/autocrop.py');
    const venvPython = path.resolve(__dirname, '../venv/bin/python3');

    console.log(`Testing autocrop on: ${videoPath}`);

    return new Promise<void>((resolve, reject) => {
        const process = spawn(venvPython, [pythonScript, videoPath]);
        let dataString = '';

        process.stdout.on('data', (data) => {
            dataString += data.toString();
            console.log(`stdout: ${data}`);
        });

        process.stderr.on('data', (data) => {
            console.error(`stderr: ${data}`);
        });

        process.on('close', (code) => {
            console.log(`Process exited with code ${code}`);
            if (code === 0) {
                try {
                    const result = JSON.parse(dataString);
                    console.log('Result:', result);
                } catch (e) {
                    console.error('Failed to parse result:', e);
                }
            }
            resolve();
        });
    });
}

testAutocrop();
