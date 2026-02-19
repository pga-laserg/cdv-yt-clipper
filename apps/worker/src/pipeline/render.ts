import ffmpeg from 'fluent-ffmpeg';
import path from 'path';
import fs from 'fs';
import { spawn } from 'child_process';

export async function render(
    videoPath: string,
    boundaries: { start: number; end: number },
    clips: { start: number; end: number; id: string }[]
): Promise<string[]> {
    console.log('Rendering clips...');

    const outputDir = path.join(path.dirname(videoPath), 'processed');
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    const results: string[] = [];

    // 1. Render Sermon Horizontal
    const sermonPath = path.join(outputDir, 'sermon_horizontal.mp4');
    await cutVideo(videoPath, boundaries.start, boundaries.end, sermonPath);
    results.push(sermonPath);

    // 2. Render Vertical Shorts
    // First, analyze video to find face center to crop around
    let centerX = 0.5; // Default center
    try {
        centerX = await detectSpeakerCenter(videoPath);
        console.log(`Detected speaker center at relative X: ${centerX}`);
    } catch (e) {
        console.error('Failed to detect speaker, defaulting to center.', e);
    }

    for (const clip of clips) {
        const clipPath = path.join(outputDir, `${clip.id}.mp4`);
        await cutVideoVertical(videoPath, clip.start, clip.end, clipPath, centerX);
        results.push(clipPath);
    }

    return results;
}

async function detectSpeakerCenter(videoPath: string): Promise<number> {
    return new Promise((resolve, reject) => {
        const pythonScript = path.resolve(__dirname, 'python/autocrop.py');
        const venvPython = path.resolve(__dirname, '../../venv/bin/python3');

        const process = spawn(venvPython, [pythonScript, videoPath, '--limit_seconds', '300']);
        let dataString = '';

        process.stdout.on('data', (data) => {
            dataString += data.toString();
        });

        process.stderr.on('data', (data) => {
            console.error(`Autocrop stderr: ${data}`);
        });

        process.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`Autocrop process exited with code ${code}`));
                return;
            }
            try {
                const result = JSON.parse(dataString);
                resolve(result.center_x);
            } catch (err) {
                reject(err);
            }
        });
    });
}

function cutVideo(input: string, start: number, end: number, output: string): Promise<void> {
    return new Promise((resolve, reject) => {
        console.log(`Cutting sermon: ${start}s to ${end}s`);
        ffmpeg(input)
            .setStartTime(start)
            .setDuration(end - start)
            .videoCodec('copy')
            .audioCodec('copy')
            .output(output)
            .on('end', () => resolve())
            .on('error', (err) => reject(err))
            .run();
    });
}

function cutVideoVertical(input: string, start: number, end: number, output: string, centerX: number): Promise<void> {
    return new Promise((resolve, reject) => {
        console.log(`Cutting vertical clip: ${start}s to ${end}s with center ${centerX}`);
        // Use ffmpeg expression-based crop to avoid runtime ffprobe dependency.
        // Crop width = ih*9/16, then clamp X around detected centerX.
        const xExpr = `max(min(${centerX}*iw-(ih*9/16)/2\\,iw-(ih*9/16))\\,0)`;

        ffmpeg(input)
            .setStartTime(start)
            .setDuration(end - start)
            .videoFilters([
                `crop=ih*9/16:ih:${xExpr}:0`,
                `scale=1080:1920`
            ])
            .output(output)
            .on('stderr', (line) => console.log(`ffmpeg: ${line}`))
            .on('end', () => resolve())
            .on('error', (err) => reject(err))
            .run();
    });
}
