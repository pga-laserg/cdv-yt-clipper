import path from 'path';
import fs from 'fs';
import { spawn } from 'child_process';
import ffmpeg from 'fluent-ffmpeg';

export async function ingest(source: string, outputDir: string): Promise<{ videoPath: string; audioPath: string }> {
    console.log(`Ingesting source: ${source}`);

    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    const videoPath = path.join(outputDir, 'source.mp4');
    const audioPath = path.join(outputDir, 'audio.wav');
    const cacheDir = path.resolve(__dirname, '../../work_tmp');
    const cachedVideoPath = path.join(cacheDir, 'source.mp4');
    const cachedAudioPath = path.join(cacheDir, 'audio.wav');

    // Reuse files if they already exist in the job output.
    if (!fs.existsSync(videoPath)) {
        // Check if it's a URL or local file
        if (source.startsWith('http')) {
            if (fs.existsSync(cachedVideoPath)) {
                console.log(`Reusing cached video from ${cachedVideoPath}`);
                fs.copyFileSync(cachedVideoPath, videoPath);
            } else {
                await downloadYouTube(source, videoPath);
            }
        } else {
            // Local file copy
            if (fs.existsSync(source)) {
                fs.copyFileSync(source, videoPath);
            } else {
                throw new Error(`Local file not found: ${source}`);
            }
        }
    } else {
        console.log(`Reusing existing video at ${videoPath}`);
    }

    if (!fs.existsSync(audioPath)) {
        if (source.startsWith('http') && fs.existsSync(cachedAudioPath)) {
            console.log(`Reusing cached audio from ${cachedAudioPath}`);
            fs.copyFileSync(cachedAudioPath, audioPath);
        } else {
            // Extract Audio
            await extractAudio(videoPath, audioPath);
        }
    } else {
        console.log(`Reusing existing audio at ${audioPath}`);
    }

    return {
        videoPath,
        audioPath
    };
}

function downloadYouTube(url: string, outputPath: string): Promise<void> {
    return new Promise((resolve, reject) => {
        console.log(`Downloading YouTube video to ${outputPath}...`);

        // Using yt-dlp directly with android client to avoid 403s
        const ytDlp = spawn('yt-dlp', [
            '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            '--extractor-args', 'youtube:player_client=android',
            '-o', outputPath,
            url
        ]);

        ytDlp.stdout.on('data', (data) => console.log(`yt-dlp: ${data}`));
        ytDlp.stderr.on('data', (data) => console.error(`yt-dlp stderr: ${data}`));

        ytDlp.on('close', (code) => {
            if (code === 0) {
                // Verify file exists
                if (fs.existsSync(outputPath)) {
                    console.log('Download complete.');
                    resolve();
                } else {
                    // Sometimes yt-dlp appends .mp4 or .mkv if not forced. 
                    // The -o option should hopefully force it, but let's check.
                    reject(new Error('Output file not found after download.'));
                }
            } else {
                reject(new Error(`yt-dlp exited with code ${code}`));
            }
        });
    });
}

function extractAudio(videoPath: string, audioPath: string): Promise<void> {
    return new Promise((resolve, reject) => {
        console.log(`Extracting audio to ${audioPath}...`);
        ffmpeg(videoPath)
            .toFormat('wav')
            .audioFrequency(16000) // Whisper prefers 16kHz
            .audioChannels(1)      // Mono
            .on('end', () => {
                console.log('Audio extraction complete.');
                resolve();
            })
            .on('error', (err) => {
                console.error('ffmpeg error:', err);
                reject(err);
            })
            .save(audioPath);
    });
}
