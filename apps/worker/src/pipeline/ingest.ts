import path from 'path';
import fs from 'fs';
import { spawn, spawnSync } from 'child_process';
import ffmpeg from 'fluent-ffmpeg';

export interface IngestResult {
    videoPath: string;
    audioPath: string;
    videoPathOriginal: string;
    videoPathHQ: string;
    videoPathLight: string;
}

interface DownloadAttempt {
    name: string;
    args: string[];
}

interface VideoResolution {
    width: number;
    height: number;
}

export async function ingest(source: string, outputDir: string): Promise<IngestResult> {
    console.log(`Ingesting source: ${source}`);

    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    const originalVideoTemplate = path.join(outputDir, 'source.original.%(ext)s');
    const videoPath = path.join(outputDir, 'source.mp4'); // HQ normalized mp4
    const videoPathLight = path.join(outputDir, 'source.light.mp4');
    const audioPath = path.join(outputDir, 'audio.wav');
    let videoPathOriginal = path.join(outputDir, 'source.original.mp4');

    if (source.startsWith('http')) {
        removeExistingOriginalVideos(outputDir);
        videoPathOriginal = await downloadYouTubeHighest(source, originalVideoTemplate, outputDir);
    } else {
        if (!fs.existsSync(source)) {
            throw new Error(`Local file not found: ${source}`);
        }
        const ext = path.extname(source) || '.mp4';
        const localOriginal = path.join(outputDir, `source.original${ext}`);
        if (!fs.existsSync(localOriginal)) {
            fs.copyFileSync(source, localOriginal);
        }
        videoPathOriginal = localOriginal;
    }

    if (!fs.existsSync(videoPath)) {
        await normalizeHighQualityMp4(videoPathOriginal, videoPath);
    } else {
        console.log(`Reusing existing HQ video at ${videoPath}`);
    }

    if (!fs.existsSync(videoPathLight)) {
        await createLightweightVideo(videoPath, videoPathLight);
    } else {
        console.log(`Reusing existing lightweight video at ${videoPathLight}`);
    }

    if (!fs.existsSync(audioPath)) {
        // Extract from HQ normalized source.
        await extractAudio(videoPath, audioPath);
    } else {
        console.log(`Reusing existing audio at ${audioPath}`);
    }

    return {
        videoPath,
        audioPath,
        videoPathOriginal,
        videoPathHQ: videoPath,
        videoPathLight
    };
}

function findExistingOriginalVideo(outputDir: string): string | null {
    const entries = fs.readdirSync(outputDir).filter((f) => f.startsWith('source.original.'));
    if (entries.length === 0) return null;
    // Prefer mp4 if available, then the first deterministic entry.
    const sorted = entries.sort((a, b) => {
        const amp4 = a.endsWith('.mp4') ? 0 : 1;
        const bmp4 = b.endsWith('.mp4') ? 0 : 1;
        if (amp4 !== bmp4) return amp4 - bmp4;
        return a.localeCompare(b);
    });
    return path.join(outputDir, sorted[0]);
}

function removeExistingOriginalVideos(outputDir: string): void {
    const entries = fs.readdirSync(outputDir).filter((f) => f.startsWith('source.original.'));
    for (const file of entries) {
        try {
            fs.unlinkSync(path.join(outputDir, file));
        } catch (error) {
            console.warn(`Failed to remove old original video ${file}:`, error);
        }
    }
}

function downloadYouTubeHighest(url: string, outputTemplate: string, outputDir: string): Promise<string> {
    const ytDlpBin = resolveBinary(['/opt/homebrew/bin/yt-dlp', '/usr/local/bin/yt-dlp', 'yt-dlp']);
    if (!ytDlpBin) {
        throw new Error('yt-dlp binary not found. Install yt-dlp and ensure it is on PATH.');
    }

    const ffmpegBin = resolveBinary(['/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg', 'ffmpeg']);
    const ffprobeBin = resolveBinary(['/opt/homebrew/bin/ffprobe', '/usr/local/bin/ffprobe', 'ffprobe']);
    const nodeBin = resolveBinary(['/opt/homebrew/bin/node', '/usr/local/bin/node', 'node']);

    const minPreferredHeight = readIntEnv('YTDLP_MIN_PREFERRED_HEIGHT', 720);
    const allowLowQualityFallback = readBoolEnv('YTDLP_ALLOW_LOW_QUALITY_FALLBACK', true);
    const cookiesFromBrowser = (process.env.YTDLP_COOKIES_FROM_BROWSER ?? '').trim();
    const enableAndroidFallback = readBoolEnv('YTDLP_ENABLE_ANDROID_FALLBACK', true);

    const attempts: DownloadAttempt[] = [
        {
            name: 'highest-mp4-m4a-web',
            args: [
                '-f',
                'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                '--merge-output-format',
                'mp4'
            ]
        },
        {
            name: 'highest-any-container',
            args: ['-f', 'bv*+ba/b', '--merge-output-format', 'mkv']
        },
        {
            name: 'best-progressive-mp4',
            args: ['-f', 'best[ext=mp4]/best']
        }
    ];
    if (enableAndroidFallback) {
        attempts.push({
            name: 'highest-mp4-m4a-android-fallback',
            args: [
                '-f',
                'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                '--extractor-args',
                'youtube:player_client=android',
                '--merge-output-format',
                'mp4'
            ]
        });
    }

    const commonArgs: string[] = ['--newline', '--no-playlist'];
    if (ffmpegBin) {
        commonArgs.push('--ffmpeg-location', ffmpegBin);
    }
    if (nodeBin) {
        commonArgs.push('--js-runtimes', `node:${nodeBin}`);
    }
    if (cookiesFromBrowser) {
        commonArgs.push('--cookies-from-browser', cookiesFromBrowser);
    }

    const runAttempt = (attempt: DownloadAttempt): Promise<string | null> =>
        new Promise((resolve) => {
            console.log(`yt-dlp attempt: ${attempt.name}`);
            removeExistingOriginalVideos(outputDir);
            const ytDlp = spawn(ytDlpBin, [...commonArgs, ...attempt.args, '-o', outputTemplate, url], {
                env: buildSpawnEnv()
            });
            ytDlp.stdout.on('data', (data) => console.log(`yt-dlp: ${data}`));
            ytDlp.stderr.on('data', (data) => console.error(`yt-dlp stderr: ${data}`));
            ytDlp.on('close', (code) => {
                if (code !== 0) return resolve(null);
                const output = findExistingOriginalVideo(outputDir);
                resolve(output ?? null);
            });
        });

    return (async () => {
        console.log('Downloading YouTube video at highest available quality...');
        console.log(`yt-dlp output template: ${outputTemplate}`);
        console.log(`yt-dlp binary: ${ytDlpBin}`);
        console.log(`yt-dlp min preferred height: ${minPreferredHeight}`);
        console.log(`yt-dlp low-quality fallback enabled: ${allowLowQualityFallback}`);
        if (cookiesFromBrowser) {
            console.log(`yt-dlp cookies-from-browser enabled (${cookiesFromBrowser})`);
        }
        if (!ffprobeBin) {
            console.warn('ffprobe not found; quality gate disabled (accepting first successful download).');
        }

        let bestLowQualityFallback: { path: string; height: number; attempt: string } | null = null;
        for (const attempt of attempts) {
            const out = await runAttempt(attempt);
            if (!out) continue;

            const resolution = probeVideoResolution(out, ffprobeBin);
            const width = resolution?.width ?? 0;
            const height = resolution?.height ?? 0;
            if (resolution) {
                console.log(`Downloaded resolution via ${attempt.name}: ${width}x${height}`);
            }

            const qualitySatisfied = !ffprobeBin || minPreferredHeight <= 0 || height >= minPreferredHeight;
            if (qualitySatisfied) {
                console.log(`Download complete via ${attempt.name}: ${out}`);
                return out;
            }

            const preserved = preserveLowQualityCandidate(out, outputDir);
            if (!bestLowQualityFallback || height > bestLowQualityFallback.height) {
                bestLowQualityFallback = { path: preserved, height, attempt: attempt.name };
            }
            console.warn(
                `Rejecting low-quality download (${width}x${height}) from ${attempt.name}; trying next strategy...`
            );
        }

        if (bestLowQualityFallback && allowLowQualityFallback) {
            console.warn(
                `No preferred-quality download succeeded. Falling back to ${bestLowQualityFallback.height}p from ${bestLowQualityFallback.attempt}: ${bestLowQualityFallback.path}`
            );
            return bestLowQualityFallback.path;
        }

        throw new Error('yt-dlp failed for all download strategies');
    })();
}

function normalizeHighQualityMp4(inputPath: string, outputPath: string): Promise<void> {
    return new Promise((resolve, reject) => {
        console.log(`Normalizing HQ MP4 from ${inputPath} -> ${outputPath}`);
        ffmpeg(inputPath)
            .videoCodec('libx264')
            .audioCodec('aac')
            .outputOptions([
                '-preset', 'medium',
                '-crf', '20',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-b:a', '192k'
            ])
            .on('end', () => {
                console.log('HQ normalization complete.');
                resolve();
            })
            .on('error', (err) => {
                console.error('HQ normalization error:', err);
                reject(err);
            })
            .save(outputPath);
    });
}

function createLightweightVideo(inputPath: string, outputPath: string): Promise<void> {
    return new Promise((resolve, reject) => {
        console.log(`Creating lightweight MP4 from ${inputPath} -> ${outputPath}`);
        ffmpeg(inputPath)
            .videoCodec('libx264')
            .audioCodec('aac')
            .size('640x?')
            .outputOptions([
                '-preset', 'veryfast',
                '-crf', '31',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-b:a', '96k'
            ])
            .on('end', () => {
                console.log('Lightweight video complete.');
                resolve();
            })
            .on('error', (err) => {
                console.error('Lightweight video error:', err);
                reject(err);
            })
            .save(outputPath);
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

function resolveBinary(candidates: string[]): string | null {
    for (const candidate of candidates) {
        if (candidate.includes('/')) {
            if (fs.existsSync(candidate)) return candidate;
            continue;
        }
        const result = spawnSync('which', [candidate], { encoding: 'utf8' });
        if (result.status === 0) {
            const resolved = (result.stdout ?? '').trim().split('\n')[0];
            if (resolved) return resolved;
        }
    }
    return null;
}

function readBoolEnv(name: string, defaultValue: boolean): boolean {
    const value = (process.env[name] ?? '').trim().toLowerCase();
    if (!value) return defaultValue;
    return value === '1' || value === 'true' || value === 'yes' || value === 'on';
}

function readIntEnv(name: string, defaultValue: number): number {
    const raw = (process.env[name] ?? '').trim();
    if (!raw) return defaultValue;
    const parsed = Number.parseInt(raw, 10);
    if (Number.isNaN(parsed)) return defaultValue;
    return parsed;
}

function buildSpawnEnv(): NodeJS.ProcessEnv {
    const existingPath = process.env.PATH ?? '';
    const preferred = ['/opt/homebrew/bin', '/usr/local/bin'];
    const mergedPath = [...preferred, existingPath].filter(Boolean).join(':');
    return { ...process.env, PATH: mergedPath };
}

function probeVideoResolution(videoPath: string, ffprobeBin: string | null): VideoResolution | null {
    if (!ffprobeBin) return null;
    try {
        const result = spawnSync(
            ffprobeBin,
            [
                '-v',
                'error',
                '-select_streams',
                'v:0',
                '-show_entries',
                'stream=width,height',
                '-of',
                'csv=p=0:s=x',
                videoPath
            ],
            { encoding: 'utf8' }
        );
        if (result.status !== 0) return null;
        const raw = (result.stdout ?? '').trim();
        if (!raw.includes('x')) return null;
        const [w, h] = raw.split('x');
        const width = Number.parseInt(w, 10);
        const height = Number.parseInt(h, 10);
        if (Number.isNaN(width) || Number.isNaN(height)) return null;
        return { width, height };
    } catch {
        return null;
    }
}

function preserveLowQualityCandidate(sourcePath: string, outputDir: string): string {
    const ext = path.extname(sourcePath) || '.mp4';
    const preserved = path.join(outputDir, `source.lowq${ext}`);
    try {
        if (fs.existsSync(preserved)) fs.unlinkSync(preserved);
        fs.copyFileSync(sourcePath, preserved);
        return preserved;
    } catch (error) {
        console.warn(`Failed to preserve low-quality fallback at ${preserved}:`, error);
        return sourcePath;
    }
}
