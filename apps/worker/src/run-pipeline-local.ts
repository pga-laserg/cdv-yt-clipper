#!/usr/bin/env tsx
/**
 * Standalone script to run the entire pipeline except for ingestion.
 * Uses an existing video file from local storage.
 * 
 * Usage:
 *   tsx apps/worker/src/run-pipeline-local.ts <input-video-path> [output-dir]
 * 
 * Example:
 *   tsx apps/worker/src/run-pipeline-local.ts /Users/pablogallardo/Downloads/Reyes\ del\ dia\ Lunes.mp4
 */

import path from 'path';
import fs from 'fs';
import dotenv from 'dotenv';
import ffmpeg from 'fluent-ffmpeg';
import { transcribe } from './pipeline/transcribe';
import { analyze } from './pipeline/analyze';
import { render } from './pipeline/render';
import { runBlogArtifactPostProcess } from './pipeline/blog-artifact';

dotenv.config({ path: path.resolve(__dirname, '../../../.env') });
dotenv.config({ path: path.resolve(__dirname, '../../web/.env.local') });

interface IngestResult {
    videoPathOriginal: string;
    videoPathLight: string;
    audioPath: string;
    videoPathPreferredRender: string;
}

async function createLightweightVideo(inputPath: string, outputPath: string): Promise<void> {
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

async function extractAudio(videoPath: string, audioPath: string): Promise<void> {
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

async function prepareIngestArtifacts(sourceVideoPath: string, outputDir: string): Promise<IngestResult> {
    console.log(`Preparing ingest artifacts from local file: ${sourceVideoPath}`);

    if (!fs.existsSync(sourceVideoPath)) {
        throw new Error(`Source video file not found: ${sourceVideoPath}`);
    }

    // Create output directory if it doesn't exist
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    // Copy source video to work directory
    const ext = path.extname(sourceVideoPath) || '.mp4';
    const videoPathOriginal = path.join(outputDir, `source.original${ext}`);

    if (!fs.existsSync(videoPathOriginal)) {
        console.log(`Copying source video to ${videoPathOriginal}`);
        fs.copyFileSync(sourceVideoPath, videoPathOriginal);
    } else {
        console.log(`Using existing video at ${videoPathOriginal}`);
    }

    // Create lightweight video for analysis
    const videoPathLight = path.join(outputDir, 'source.light.mp4');
    if (!fs.existsSync(videoPathLight)) {
        await createLightweightVideo(videoPathOriginal, videoPathLight);
    } else {
        console.log(`Using existing lightweight video at ${videoPathLight}`);
    }

    // Extract audio for transcription
    const audioPath = path.join(outputDir, 'audio.wav');
    if (!fs.existsSync(audioPath)) {
        await extractAudio(videoPathOriginal, audioPath);
    } else {
        console.log(`Using existing audio at ${audioPath}`);
    }

    return {
        videoPathOriginal,
        videoPathLight,
        audioPath,
        videoPathPreferredRender: videoPathOriginal
    };
}

async function runPipeline() {
    const args = process.argv.slice(2);

    if (args.length < 1) {
        console.error('Usage: tsx apps/worker/src/run-pipeline-local.ts <input-video-path> [output-dir]');
        console.error('Example: tsx apps/worker/src/run-pipeline-local.ts /Users/pablogallardo/Downloads/Reyes\\ del\\ dia\\ Lunes.mp4');
        process.exit(1);
    }

    const inputVideoPath = args[0];
    const outputDir = args[1]
        ? path.resolve(process.cwd(), args[1])
        : path.resolve(__dirname, '../../work_dir/local_pipeline_' + Date.now());

    console.log('='.repeat(80));
    console.log('Starting local pipeline execution');
    console.log('='.repeat(80));
    console.log(`Input video: ${inputVideoPath}`);
    console.log(`Output directory: ${outputDir}`);
    console.log('='.repeat(80));

    const startedAt = Date.now();

    try {
        // Prepare ingest artifacts (skip actual download/ingest)
        console.log('\n--- Stage 1: Prepare Ingest Artifacts ---');
        const { videoPathOriginal, videoPathLight, audioPath, videoPathPreferredRender } =
            await prepareIngestArtifacts(inputVideoPath, outputDir);
        console.log(`Stage 1 completed in ${Math.round((Date.now() - startedAt) / 1000)}s`);

        // Transcribe
        console.log('\n--- Stage 2: Transcribe ---');
        const totalDurationSeconds = await getMediaDurationSeconds(audioPath);
        let lastProgressSecond = -1;
        let lastProgressAt = 0;
        const segments = await transcribe(audioPath, {
            onProgress: (currentSeconds) => {
                if (!totalDurationSeconds || !Number.isFinite(totalDurationSeconds)) return;
                const current = Math.min(Math.max(0, Math.floor(currentSeconds)), Math.floor(totalDurationSeconds));
                const now = Date.now();

                // Throttle console output
                if (current <= lastProgressSecond || now - lastProgressAt < 5000) return;
                lastProgressSecond = current;
                lastProgressAt = now;
                console.log(`Transcribing: ${current}/${Math.floor(totalDurationSeconds)}s`);
            },
            onFallback: (reason) => {
                console.warn(reason);
            }
        });
        console.log(`Stage 2 completed in ${Math.round((Date.now() - startedAt) / 1000)}s`);
        console.log(`Generated ${segments.length} transcript segments`);

        // Analyze
        console.log('\n--- Stage 3: Analyze ---');
        const { boundaries, clips } = await analyze(segments, { workDir: outputDir, audioPath, videoPath: videoPathLight });
        console.log(`Stage 3 completed in ${Math.round((Date.now() - startedAt) / 1000)}s`);
        console.log(`Found sermon boundaries: ${boundaries.start.toFixed(2)}s - ${boundaries.end.toFixed(2)}s`);
        console.log(`Identified ${clips.length} clips`);

        // Render
        console.log('\n--- Stage 4: Render ---');
        const clipData = clips.map((c, i) => ({ ...c, id: `local_clip_${i + 1}` }));
        const renderedFiles = await render(videoPathPreferredRender, boundaries, clipData, {
            trackingVideoPath: videoPathLight,
            horizontalVideoPath: videoPathOriginal
        });
        console.log(`Stage 4 completed in ${Math.round((Date.now() - startedAt) / 1000)}s`);
        console.log(`Rendered files:`, renderedFiles);

        // Blog Artifact
        console.log('\n--- Stage 5: Blog Artifact ---');
        try {
            await runBlogArtifactPostProcess({
                jobId: 'local_job',
                organizationId: 'local',
                workDir: outputDir,
                youtubeUrl: inputVideoPath,
                boundaries,
                transcriptSegments: segments,
                clips: clips.map((clip) => ({
                    start: clip.start,
                    end: clip.end,
                    title: clip.title,
                    excerpt: clip.excerpt,
                    confidence: clip.confidence
                }))
            });
            console.log(`Stage 5 completed in ${Math.round((Date.now() - startedAt) / 1000)}s`);
        } catch (blogError) {
            console.error('Blog artifact stage failed:', blogError);
            console.log('Continuing despite blog artifact failure...');
        }

        console.log('\n' + '='.repeat(80));
        console.log(`Pipeline finished successfully in ${Math.round((Date.now() - startedAt) / 1000)}s`);
        console.log('='.repeat(80));
        console.log(`\nOutput directory: ${outputDir}`);
        console.log(`\nGenerated files:`);
        console.log(`  - Source video: ${videoPathOriginal}`);
        console.log(`  - Light video: ${videoPathLight}`);
        console.log(`  - Audio: ${audioPath}`);
        console.log(`  - Rendered clips: ${renderedFiles.length} files`);

    } catch (error) {
        console.error('\n' + '='.repeat(80));
        console.error(`Pipeline failed after ${Math.round((Date.now() - startedAt) / 1000)}s`);
        console.error('='.repeat(80));
        console.error(error);
        process.exit(1);
    }
}

function getMediaDurationSeconds(mediaPath: string): Promise<number> {
    return new Promise((resolve, reject) => {
        ffmpeg.ffprobe(mediaPath, (err, metadata) => {
            if (err) {
                reject(err);
                return;
            }
            const duration = metadata.format.duration || 0;
            resolve(duration);
        });
    });
}

runPipeline().catch((error) => {
    console.error('Unhandled error:', error);
    process.exit(1);
});
