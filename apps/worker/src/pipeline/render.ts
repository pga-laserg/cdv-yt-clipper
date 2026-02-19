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

        ffmpeg.ffprobe(input, (err, metadata) => {
            if (err) {
                reject(err);
                return;
            }

            const stream = metadata.streams.find(s => s.codec_type === 'video');
            if (!stream || !stream.width || !stream.height) {
                reject(new Error('No video stream found or resolution missing'));
                return;
            }

            const inputWidth = stream.width;
            const inputHeight = stream.height;

            // Calculate vertical crop width based on aspect ratio
            const targetHeight = inputHeight;
            let targetWidth = Math.floor(inputHeight * (9 / 16));
            if (targetWidth % 2 !== 0) targetWidth -= 1;

            let x = Math.floor((centerX * inputWidth) - (targetWidth / 2));
            if (x % 2 !== 0) x -= 1;
            if (x < 0) x = 0;
            if (x + targetWidth > inputWidth) x = inputWidth - targetWidth;

            ffmpeg(input)
                .setStartTime(start)
                .setDuration(end - start)
                .videoFilters([
                    `crop=${targetWidth}:${targetHeight}:${x}:0`,
                    `scale=1080:1920`
                ])
                .output(output)
                .on('stderr', (line) => console.log(`ffmpeg: ${line}`))
                .on('end', () => resolve())
                .on('error', (err) => reject(err))
                .run();
        });
    });
}
