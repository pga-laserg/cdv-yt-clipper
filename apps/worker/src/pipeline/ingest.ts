import path from 'path';
import fs from 'fs';
import { spawn, spawnSync } from 'child_process';
import ffmpeg from 'fluent-ffmpeg';

export interface IngestResult {
    videoPath: string;
    audioPath: string;
    videoPathOriginal: string;
    videoPathHQ?: string;
    videoPathLight: string;
    videoPathPreferredRender: string;
    ingestProfile: IngestProfile;
}

interface DownloadAttempt {
    name: string;
    args: string[];
}

interface VideoResolution {
    width: number;
    height: number;
}

interface DurationIntegrityStatus {
    ffprobeAvailable: boolean;
    originalDurationSec: number | null;
    hqDurationSec: number | null;
    lightDurationSec: number | null;
    hqValid: boolean;
    lightValid: boolean;
    issues: string[];
}

export type IngestMode = 'audio_first' | 'eager' | 'hybrid';

export interface IngestProfile {
    mode: IngestMode;
    source_type: 'remote' | 'local';
    plan: {
        requires_hq: boolean;
        light_from_hq: boolean;
        audio_from_hq: boolean;
    };
    download_duration_ms: number;
    hq_transcode_duration_ms: number;
    light_transcode_duration_ms: number;
    audio_extract_duration_ms: number;
    hq_transcode_performed: boolean;
}

interface IngestPlan {
    mode: IngestMode;
    requiresHQ: boolean;
    lightFromHQ: boolean;
    audioFromHQ: boolean;
}

export async function ingest(source: string, outputDir: string): Promise<IngestResult> {
    console.log(`Ingesting source: ${source}`);

    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    const ingestStartedAt = Date.now();
    const isHttp = source.startsWith('http://') || source.startsWith('https://');
    const sourceType: 'remote' | 'local' = isHttp ? 'remote' : 'local';
    const ingestMode = parseIngestMode(process.env.INGEST_MODE, process.env.PIPELINE_MODE);
    const plan = planIngestArtifacts(ingestMode, sourceType, {
        analyzeRequiresLightVideo: true
    });

    const originalVideoTemplate = path.join(outputDir, 'source.original.%(ext)s');
    const videoPath = path.join(outputDir, 'source.mp4'); // eager-mode HQ normalized mp4
    const videoPathLight = path.join(outputDir, 'source.light.mp4');
    const audioPath = path.join(outputDir, 'audio.wav');
    let videoPathOriginal = path.join(outputDir, 'source.original.mp4');
    let videoPathHQ: string | undefined;

    const downloadStartedAt = Date.now();

    if (source.startsWith('http')) {
        removeExistingOriginalVideos(outputDir);
        if (source.includes('youtube.com') || source.includes('youtu.be')) {
            videoPathOriginal = await downloadYouTubeHighest(source, originalVideoTemplate, outputDir);
        } else if (isGoogleDriveUrl(source)) {
            console.log(`Google Drive source detected: ${source}`);
            videoPathOriginal = await downloadGoogleDriveVideo(source, path.join(outputDir, 'source.original.mp4'));
        } else {
            console.log(`Direct download for generic source: ${source}`);
            videoPathOriginal = await downloadGenericVideo(source, path.join(outputDir, 'source.original.mp4'));
        }
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
    const downloadDurationMs = Date.now() - downloadStartedAt;

    const hqStartedAt = Date.now();
    let hqTranscodePerformed = false;
    if (plan.requiresHQ) {
        videoPathHQ = videoPath;
        if (!fs.existsSync(videoPathHQ)) {
            await normalizeHighQualityMp4(videoPathOriginal, videoPathHQ);
            hqTranscodePerformed = true;
        } else {
            console.log(`Reusing existing HQ video at ${videoPathHQ}`);
        }
    }
    const hqTranscodeDurationMs = Date.now() - hqStartedAt;

    const lightInputPath = plan.lightFromHQ && videoPathHQ ? videoPathHQ : videoPathOriginal;
    const lightStartedAt = Date.now();
    if (!fs.existsSync(videoPathLight)) {
        await createLightweightVideo(lightInputPath, videoPathLight);
    } else {
        console.log(`Reusing existing lightweight video at ${videoPathLight}`);
    }
    const lightTranscodeDurationMs = Date.now() - lightStartedAt;

    await ensureDerivedVideoIntegrity(videoPathOriginal, videoPathHQ ?? null, videoPathLight);

    const audioInputPath = plan.audioFromHQ && videoPathHQ ? videoPathHQ : videoPathOriginal;
    const audioStartedAt = Date.now();
    if (!fs.existsSync(audioPath)) {
        await extractAudio(audioInputPath, audioPath);
    } else {
        console.log(`Reusing existing audio at ${audioPath}`);
    }
    const audioExtractDurationMs = Date.now() - audioStartedAt;

    const videoPathPreferredRender = videoPathOriginal;
    const ingestProfile: IngestProfile = {
        mode: plan.mode,
        source_type: sourceType,
        plan: {
            requires_hq: plan.requiresHQ,
            light_from_hq: plan.lightFromHQ,
            audio_from_hq: plan.audioFromHQ
        },
        download_duration_ms: downloadDurationMs,
        hq_transcode_duration_ms: hqTranscodeDurationMs,
        light_transcode_duration_ms: lightTranscodeDurationMs,
        audio_extract_duration_ms: audioExtractDurationMs,
        hq_transcode_performed: hqTranscodePerformed
    };
    console.log(
        `[ingest-profile] mode=${ingestProfile.mode} source=${ingestProfile.source_type} ` +
            `download=${ingestProfile.download_duration_ms}ms hq=${ingestProfile.hq_transcode_duration_ms}ms ` +
            `light=${ingestProfile.light_transcode_duration_ms}ms audio=${ingestProfile.audio_extract_duration_ms}ms ` +
            `hq_performed=${ingestProfile.hq_transcode_performed} total=${Date.now() - ingestStartedAt}ms`
    );

    return {
        videoPath: videoPathPreferredRender,
        audioPath,
        videoPathOriginal,
        videoPathHQ,
        videoPathLight,
        videoPathPreferredRender,
        ingestProfile
    };
}

export async function ensureHighQualityRenderSource(videoPathOriginal: string, outputDir: string): Promise<string> {
    const fallbackPath = path.join(outputDir, 'source.hq.fallback.mp4');
    if (fs.existsSync(fallbackPath)) {
        console.log(`[render-fallback] Reusing existing HQ fallback at ${fallbackPath}`);
        return fallbackPath;
    }
    await normalizeHighQualityMp4(videoPathOriginal, fallbackPath);
    return fallbackPath;
}

function planIngestArtifacts(
    mode: IngestMode,
    _sourceType: 'remote' | 'local',
    features: { analyzeRequiresLightVideo: boolean }
): IngestPlan {
    const needsLight = features.analyzeRequiresLightVideo;
    if (mode === 'eager') {
        return {
            mode,
            requiresHQ: true,
            lightFromHQ: needsLight,
            audioFromHQ: true
        };
    }

    if (mode === 'hybrid') {
        return {
            mode,
            requiresHQ: false,
            lightFromHQ: false,
            audioFromHQ: false
        };
    }

    return {
        mode: 'audio_first',
        requiresHQ: false,
        lightFromHQ: false,
        audioFromHQ: false
    };
}

function parseIngestMode(rawMode: string | undefined, pipelineModeRaw: string | undefined): IngestMode {
    const mode = String(rawMode ?? 'audio_first').trim().toLowerCase();
    if (mode === 'audio_first' || mode === 'eager' || mode === 'hybrid') {
        return mode;
    }
    if (mode && mode !== 'audio_first') {
        console.warn(`Invalid INGEST_MODE="${rawMode}". Falling back to default.`);
    }

    const pipelineMode = String(pipelineModeRaw ?? 'prod').trim().toLowerCase();
    return pipelineMode === 'local' ? 'hybrid' : 'audio_first';
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
    const jsRuntimeArg = resolveYtDlpJsRuntimeArg();

    const minPreferredHeight = readIntEnv('YTDLP_MIN_PREFERRED_HEIGHT', 720);
    const allowLowQualityFallback = readBoolEnv('YTDLP_ALLOW_LOW_QUALITY_FALLBACK', true);
    const requireFfprobeForQuality = readBoolEnv('YTDLP_REQUIRE_FFPROBE_FOR_QUALITY', true);
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
    if (jsRuntimeArg) {
        commonArgs.push('--js-runtimes', jsRuntimeArg);
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
        console.log(`yt-dlp js runtime: ${jsRuntimeArg ?? 'none'}`);
        if (cookiesFromBrowser) {
            console.log(`yt-dlp cookies-from-browser enabled (${cookiesFromBrowser})`);
        }
        if (!ffprobeBin && minPreferredHeight > 0 && requireFfprobeForQuality) {
            throw new Error(
                'ffprobe not found; cannot enforce YTDLP_MIN_PREFERRED_HEIGHT. Install ffprobe or set YTDLP_REQUIRE_FFPROBE_FOR_QUALITY=false.'
            );
        }
        if (!ffprobeBin && minPreferredHeight > 0) {
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

async function ensureDerivedVideoIntegrity(
    videoPathOriginal: string,
    videoPathHQ: string | null,
    videoPathLight: string
): Promise<void> {
    const requireDurationCheck = readBoolEnv('INGEST_REQUIRE_DURATION_CHECK', true);
    const initial = inspectDerivedVideoIntegrity(videoPathOriginal, videoPathHQ, videoPathLight);
    if (!initial.ffprobeAvailable) {
        const message = '[ingest] ffprobe unavailable; cannot run derived-video duration integrity checks.';
        if (requireDurationCheck) throw new Error(message);
        console.warn(`${message} Continuing because INGEST_REQUIRE_DURATION_CHECK=false.`);
        return;
    }
    if (initial.issues.length === 0) {
        console.log(
            `[ingest] duration integrity ok original=${formatDuration(initial.originalDurationSec)} ` +
                `hq=${formatDuration(initial.hqDurationSec)} light=${formatDuration(initial.lightDurationSec)}`
        );
        return;
    }

    console.warn('[ingest] detected derived-video duration integrity issues:');
    for (const issue of initial.issues) console.warn(`  - ${issue}`);

    if (!initial.hqValid && videoPathHQ) {
        console.warn('[ingest] regenerating HQ + light videos from original source...');
        safeUnlink(videoPathHQ);
        await normalizeHighQualityMp4(videoPathOriginal, videoPathHQ);
        safeUnlink(videoPathLight);
        await createLightweightVideo(videoPathHQ, videoPathLight);
    } else if (!initial.lightValid) {
        const regenSource = videoPathHQ && fs.existsSync(videoPathHQ) ? videoPathHQ : videoPathOriginal;
        console.warn(`[ingest] regenerating lightweight video from ${path.basename(regenSource)}...`);
        safeUnlink(videoPathLight);
        await createLightweightVideo(regenSource, videoPathLight);
    }

    const repaired = inspectDerivedVideoIntegrity(videoPathOriginal, videoPathHQ, videoPathLight);
    if (!repaired.ffprobeAvailable) {
        console.warn('[ingest] ffprobe unavailable after regeneration; cannot verify duration integrity.');
        return;
    }
    if (repaired.issues.length > 0) {
        throw new Error(
            '[ingest] derived-video duration integrity failed after regeneration:\n' +
                repaired.issues.map((issue) => `- ${issue}`).join('\n')
        );
    }
    console.log(
        `[ingest] duration integrity repaired original=${formatDuration(repaired.originalDurationSec)} ` +
            `hq=${formatDuration(repaired.hqDurationSec)} light=${formatDuration(repaired.lightDurationSec)}`
    );
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

function resolveYtDlpJsRuntimeArg(): string | null {
    const raw = (process.env.YTDLP_JS_RUNTIME ?? 'auto').trim();
    const selected = raw.toLowerCase();

    const denoBin = () => resolveBinary(['/opt/homebrew/bin/deno', '/usr/local/bin/deno', 'deno']);
    const nodeBin = () => resolveBinary(['/opt/homebrew/bin/node', '/usr/local/bin/node', 'node']);

    const auto = () => {
        const deno = denoBin();
        if (deno) return `deno:${deno}`;
        const node = nodeBin();
        if (node) return `node:${node}`;
        return null;
    };

    if (!selected || selected === 'auto') return auto();
    if (selected === 'none' || selected === 'off' || selected === 'false') return null;
    if (selected.includes(':')) return raw;

    if (selected === 'deno') {
        const deno = denoBin();
        return deno ? `deno:${deno}` : null;
    }
    if (selected === 'node') {
        const node = nodeBin();
        return node ? `node:${node}` : null;
    }

    const fallback = auto();
    console.warn(`Unknown YTDLP_JS_RUNTIME="${raw}". Falling back to auto runtime resolution.`);
    return fallback;
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

function readFloatEnv(name: string, defaultValue: number): number {
    const raw = (process.env[name] ?? '').trim();
    if (!raw) return defaultValue;
    const parsed = Number.parseFloat(raw);
    if (!Number.isFinite(parsed)) return defaultValue;
    return parsed;
}

function buildSpawnEnv(): NodeJS.ProcessEnv {
    const existingPath = process.env.PATH ?? '';
    const preferred = ['/opt/homebrew/bin', '/usr/local/bin'];
    const mergedPath = [...preferred, existingPath].filter(Boolean).join(':');
    return { ...process.env, PATH: mergedPath };
}

function formatDuration(value: number | null): string {
    if (!Number.isFinite(value as number)) return 'n/a';
    return `${(value as number).toFixed(2)}s`;
}

function safeUnlink(filePath: string): void {
    if (!fs.existsSync(filePath)) return;
    try {
        fs.unlinkSync(filePath);
    } catch (error) {
        throw new Error(`Unable to remove file ${filePath}: ${String(error)}`);
    }
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

function probeVideoDuration(videoPath: string, ffprobeBin: string | null): number | null {
    if (!ffprobeBin) return null;
    try {
        const result = spawnSync(
            ffprobeBin,
            [
                '-v',
                'error',
                '-show_entries',
                'format=duration',
                '-of',
                'default=noprint_wrappers=1:nokey=1',
                videoPath
            ],
            { encoding: 'utf8' }
        );
        if (result.status !== 0) return null;
        const raw = (result.stdout ?? '').trim();
        const duration = Number.parseFloat(raw);
        if (!Number.isFinite(duration) || duration <= 0) return null;
        return duration;
    } catch {
        return null;
    }
}

function inspectDerivedVideoIntegrity(
    videoPathOriginal: string,
    videoPathHQ: string | null,
    videoPathLight: string
): DurationIntegrityStatus {
    const ffprobeBin = resolveBinary(['/opt/homebrew/bin/ffprobe', '/usr/local/bin/ffprobe', 'ffprobe']);
    if (!ffprobeBin) {
        return {
            ffprobeAvailable: false,
            originalDurationSec: null,
            hqDurationSec: null,
            lightDurationSec: null,
            hqValid: true,
            lightValid: true,
            issues: ['ffprobe binary not found']
        };
    }

    const minHqRatio = Math.min(1.0, Math.max(0.5, readFloatEnv('INGEST_HQ_DURATION_MIN_RATIO', 0.98)));
    const minLightRatio = Math.min(1.0, Math.max(0.5, readFloatEnv('INGEST_LIGHT_DURATION_MIN_RATIO', 0.98)));

    const originalDurationSec = probeVideoDuration(videoPathOriginal, ffprobeBin);
    const hqDurationSec = videoPathHQ ? probeVideoDuration(videoPathHQ, ffprobeBin) : null;
    const lightDurationSec = probeVideoDuration(videoPathLight, ffprobeBin);

    const issues: string[] = [];
    let hqValid = true;
    let lightValid = true;

    if (originalDurationSec == null) {
        issues.push(`cannot probe original duration: ${videoPathOriginal}`);
    }
    if (videoPathHQ && hqDurationSec == null) {
        issues.push(`cannot probe HQ duration: ${videoPathHQ}`);
        hqValid = false;
    }
    if (lightDurationSec == null) {
        issues.push(`cannot probe light duration: ${videoPathLight}`);
        lightValid = false;
    }

    if (videoPathHQ && originalDurationSec != null && hqDurationSec != null) {
        const hqRatio = hqDurationSec / originalDurationSec;
        if (hqRatio < minHqRatio) {
            issues.push(
                `hq duration too short (${hqDurationSec.toFixed(2)}s < ${(originalDurationSec * minHqRatio).toFixed(2)}s ` +
                    `ratio=${hqRatio.toFixed(4)} threshold=${minHqRatio.toFixed(4)})`
            );
            hqValid = false;
        }
    }

    if (lightDurationSec != null && (hqDurationSec != null || originalDurationSec != null)) {
        const reference = hqDurationSec ?? originalDurationSec!;
        const lightRatio = lightDurationSec / reference;
        if (lightRatio < minLightRatio) {
            issues.push(
                `light duration too short (${lightDurationSec.toFixed(2)}s < ${(reference * minLightRatio).toFixed(2)}s ` +
                    `ratio=${lightRatio.toFixed(4)} threshold=${minLightRatio.toFixed(4)})`
            );
            lightValid = false;
        }
    }

    return {
        ffprobeAvailable: true,
        originalDurationSec,
        hqDurationSec,
        lightDurationSec,
        hqValid,
        lightValid,
        issues
    };
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

/**
 * Returns true if the URL looks like a Google Drive share/view/uc link.
 * Handles:
 *   https://drive.google.com/file/d/{id}/view?usp=sharing
 *   https://drive.google.com/open?id={id}
 *   https://drive.google.com/uc?id={id}&export=download
 */
function isGoogleDriveUrl(url: string): boolean {
    return url.includes('drive.google.com');
}

/**
 * Extracts the Google Drive file ID from common share URL formats.
 */
function extractGoogleDriveFileId(url: string): string | null {
    // Format: /file/d/{id}/...
    const fileDMatch = url.match(/\/file\/d\/([a-zA-Z0-9_-]+)/);
    if (fileDMatch) return fileDMatch[1];

    // Format: ?id={id} or &id={id}
    const idParamMatch = url.match(/[?&]id=([a-zA-Z0-9_-]+)/);
    if (idParamMatch) return idParamMatch[1];

    return null;
}

/**
 * Downloads a Google Drive file to the output path.
 * Uses gdown (Python) for large-file-safe downloads with virus-scan bypass.
 * Falls back to curl with the direct download URL if gdown is not found.
 */
function downloadGoogleDriveVideo(url: string, outputPath: string): Promise<string> {
    return new Promise((resolve, reject) => {
        const fileId = extractGoogleDriveFileId(url);
        if (!fileId) {
            return reject(new Error(`Could not extract Google Drive file ID from: ${url}`));
        }

        console.log(`Google Drive file ID: ${fileId}`);

        // Try gdown first — handles large file quota bypass automatically
        const gdownBin = resolveBinary(['gdown', '/opt/homebrew/bin/gdown', '/usr/local/bin/gdown']);
        if (gdownBin) {
            console.log(`Using gdown (${gdownBin}) to download Drive file...`);
            const proc = spawn(gdownBin, [
                `https://drive.google.com/uc?id=${fileId}`,
                '-O', outputPath,
                '--fuzzy'
            ]);
            proc.stdout?.on('data', (d) => console.log(`gdown: ${d}`));
            proc.stderr?.on('data', (d) => console.error(`gdown: ${d}`));
            proc.on('close', (code) => {
                if (code === 0 && fs.existsSync(outputPath)) {
                    console.log(`Google Drive download complete via gdown: ${outputPath}`);
                    return resolve(outputPath);
                }
                // gdown failed — try python -m gdown as fallback
                console.warn(`gdown exited with code ${code}. Falling back to python -m gdown...`);
                const python = spawn('python3', ['-m', 'gdown',
                    `https://drive.google.com/uc?id=${fileId}`,
                    '-O', outputPath, '--fuzzy']);
                python.on('close', (pyCode) => {
                    if (pyCode === 0 && fs.existsSync(outputPath)) {
                        return resolve(outputPath);
                    }
                    // Final fallback: curl direct download URL
                    downloadWithCurlDrive(fileId, outputPath, resolve, reject);
                });
            });
        } else {
            // No gdown available — use curl with the direct download URL
            downloadWithCurlDrive(fileId, outputPath, resolve, reject);
        }
    });
}

function downloadWithCurlDrive(
    fileId: string,
    outputPath: string,
    resolve: (v: string) => void,
    reject: (e: Error) => void
): void {
    const directUrl = `https://drive.google.com/uc?export=download&id=${fileId}&confirm=t`;
    console.log(`Falling back to curl for Drive download (may fail for large files): ${directUrl}`);
    const curl = spawn('curl', ['-L', '-o', outputPath, directUrl]);
    curl.on('close', (code) => {
        if (code === 0 && fs.existsSync(outputPath)) {
            resolve(outputPath);
        } else {
            reject(new Error(
                `All Google Drive download methods failed for file ID: ${fileId}. ` +
                `Install gdown for large file support: pip install gdown`
            ));
        }
    });
}

function downloadGenericVideo(url: string, outputPath: string): Promise<string> {
    return new Promise((resolve, reject) => {
        console.log(`Downloading generic video from ${url} to ${outputPath}...`);
        const curl = spawn('curl', ['-L', '-o', outputPath, url]);
        curl.on('close', (code) => {
            if (code === 0 && fs.existsSync(outputPath)) {
                resolve(outputPath);
            } else {
                reject(new Error(`Failed to download generic video from ${url} (code ${code})`));
            }
        });
    });
}
