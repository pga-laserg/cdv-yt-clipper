import { ingest } from './pipeline/ingest';
import { transcribe } from './pipeline/transcribe';
import { analyze } from './pipeline/analyze';
import { render } from './pipeline/render';
import { uploadFile, UploadFailedError } from './pipeline/store';
import { supabase } from './lib/supabase';
import path from 'path';
import fs from 'fs';
import ffmpeg from 'fluent-ffmpeg';

async function runPipeline(jobId: string, source: string) {
    const workDir = path.resolve(__dirname, `../work_dir/${jobId}`);
    if (!fs.existsSync(workDir)) fs.mkdirSync(workDir, { recursive: true });
    const startedAt = Date.now();
    let currentStage = 'initializing';

    try {
        // 1. Ingest
        console.log('--- Stage 1: Ingest ---');
        currentStage = 'ingest';
        await updateJobStatus(jobId, 'processing:ingest');
        const { videoPath, audioPath } = await ingest(source, workDir);
        console.log(`Stage ingest completed in ${Math.round((Date.now() - startedAt) / 1000)}s`);

        // 2. Transcribe
        console.log('--- Stage 2: Transcribe ---');
        currentStage = 'transcribe';
        await updateJobStatus(jobId, 'processing:transcribe');
        const totalDurationSeconds = await getMediaDurationSeconds(audioPath);
        let lastProgressSecond = -1;
        let lastProgressAt = 0;
        const segments = await transcribe(audioPath, {
            onProgress: (currentSeconds) => {
                if (!totalDurationSeconds || !Number.isFinite(totalDurationSeconds)) return;
                const current = Math.min(Math.max(0, Math.floor(currentSeconds)), Math.floor(totalDurationSeconds));
                const now = Date.now();

                // Throttle DB writes while still giving useful movement in UI.
                if (current <= lastProgressSecond || now - lastProgressAt < 5000) return;
                lastProgressSecond = current;
                lastProgressAt = now;
                void updateJobStatus(jobId, `processing:transcribe:${current}/${Math.floor(totalDurationSeconds)}`);
            },
            onFallback: (reason) => {
                console.warn(reason);
                void updateJobStatus(jobId, 'processing:transcribe:fallback');
            }
        });
        const srtPath = path.join(path.dirname(audioPath), 'source.srt');
        console.log(`Stage transcribe completed in ${Math.round((Date.now() - startedAt) / 1000)}s`);

        // 3. Analyze
        console.log('--- Stage 3: Analyze ---');
        currentStage = 'analyze';
        await updateJobStatus(jobId, 'processing:analyze');
        const { boundaries, clips } = await analyze(segments);
        console.log(`Stage analyze completed in ${Math.round((Date.now() - startedAt) / 1000)}s`);

        // 4. Render
        console.log('--- Stage 4: Render ---');
        currentStage = 'render';
        await updateJobStatus(jobId, 'processing:render');
        const clipData = clips.map((c, i) => ({ ...c, id: `clip_${Date.now()}_${i + 1}` }));
        const renderedFiles = await render(videoPath, boundaries, clipData);
        console.log(`Stage render completed in ${Math.round((Date.now() - startedAt) / 1000)}s`);

        // 5. Store & Metadata
        console.log('--- Stage 5: Store ---');
        currentStage = 'store';
        await updateJobStatus(jobId, 'processing:store');

        // Upload SRT
        let srtUrl = '';
        if (fs.existsSync(srtPath)) {
            srtUrl = await uploadAssetWithFallback(srtPath, 'assets', `jobs/${jobId}/transcript.srt`);
        }

        // Upload Horizontal Sermon
        let horizontalUrl = '';
        const horizontalLocal = renderedFiles.find(f => f.includes('sermon_horizontal.mp4'));
        if (horizontalLocal && fs.existsSync(horizontalLocal)) {
            horizontalUrl = await uploadAssetWithFallback(horizontalLocal, 'assets', `jobs/${jobId}/sermon_horizontal.mp4`);
        }

        // Upload thumbnail frame at 80% of the horizontal sermon duration.
        if (horizontalLocal && fs.existsSync(horizontalLocal)) {
            try {
                const thumbnailLocal = path.join(path.dirname(horizontalLocal), 'thumbnail.jpg');
                const sermonDuration = Math.max(1, boundaries.end - boundaries.start);
                const thumbnailAt = sermonDuration * 0.8;
                await createVideoThumbnail(horizontalLocal, thumbnailLocal, thumbnailAt);
                await uploadAssetWithFallback(thumbnailLocal, 'assets', `jobs/${jobId}/thumbnail.jpg`);
            } catch (thumbnailError) {
                console.error('Thumbnail generation/upload failed:', thumbnailError);
            }
        }

        // Upload Clips and save to DB
        await savePipelineResults(jobId, boundaries, clipData, renderedFiles, srtUrl, horizontalUrl, horizontalLocal ?? '');

        await updateJobStatus(jobId, 'completed');
        console.log(`Pipeline finished successfully in ${Math.round((Date.now() - startedAt) / 1000)}s`);

    } catch (error) {
        console.error(`Pipeline failed at stage "${currentStage}" after ${Math.round((Date.now() - startedAt) / 1000)}s:`, error);
        await updateJobStatus(jobId, 'failed');
    }
}

async function updateJobStatus(id: string, status: string) {
    await supabase.from('jobs').update({ status }).eq('id', id);
}

function getMediaDurationSeconds(filePath: string): Promise<number | null> {
    return Promise.resolve(getWavDurationSeconds(filePath));
}

function getWavDurationSeconds(filePath: string): number | null {
    try {
        const stats = fs.statSync(filePath);
        if (!stats.isFile() || stats.size < 44) return null;

        const fd = fs.openSync(filePath, 'r');
        const header = Buffer.alloc(44);
        fs.readSync(fd, header, 0, 44, 0);
        fs.closeSync(fd);

        const isRiff = header.toString('ascii', 0, 4) === 'RIFF';
        const isWave = header.toString('ascii', 8, 12) === 'WAVE';
        if (!isRiff || !isWave) return null;

        const byteRate = header.readUInt32LE(28);
        if (!byteRate) return null;

        const duration = Math.max(0, (stats.size - 44) / byteRate);
        return Number.isFinite(duration) ? duration : null;
    } catch (error) {
        console.error(`Failed to parse WAV duration for ${filePath}:`, error);
        return null;
    }
}

function createVideoThumbnail(videoPath: string, outputPath: string, second: number): Promise<void> {
    return new Promise((resolve, reject) => {
        const at = Math.max(0, Math.floor(second));
        ffmpeg(videoPath)
            .seekInput(at)
            .frames(1)
            .outputOptions(['-q:v 2'])
            .output(outputPath)
            .on('end', () => resolve())
            .on('error', (err) => reject(err))
            .run();
    });
}

async function savePipelineResults(
    jobId: string,
    boundaries: any,
    clips: any[],
    renderedFiles: string[],
    srtUrl: string,
    horizontalUrl: string,
    horizontalLocalPath: string
) {
    const { data: existingJob } = await supabase
        .from('jobs')
        .select('metadata')
        .eq('id', jobId)
        .single();
    const metadata = {
        ...(existingJob?.metadata || {}),
        full_res_video_path: horizontalLocalPath || null
    };

    // Update Job
    await supabase.from('jobs').update({
        sermon_start_seconds: boundaries.start,
        sermon_end_seconds: boundaries.end,
        video_url: horizontalUrl,
        srt_url: srtUrl,
        metadata
    }).eq('id', jobId);

    // Upload each clip and insert into clips table
    for (let i = 0; i < clips.length; i++) {
        await updateJobStatus(jobId, `processing:store:${i + 1}/${clips.length}`);
        const clip = clips[i];
        const localPath = renderedFiles.find(f => f.includes(clip.id));

        let videoUrl = '';
        if (localPath && fs.existsSync(localPath)) {
            videoUrl = await uploadAssetWithFallback(localPath, 'assets', `jobs/${jobId}/clips/${clip.id}.mp4`);
        }

        await supabase.from('clips').insert({
            job_id: jobId,
            start_seconds: clip.start,
            end_seconds: clip.end,
            title: clip.title,
            transcript_excerpt: clip.excerpt,
            confidence_score: clip.confidence,
            video_url: videoUrl,
            status: 'draft'
        });
    }
}

async function uploadAssetWithFallback(filePath: string, bucket: string, destination: string): Promise<string> {
    try {
        return await uploadFile(filePath, bucket, destination);
    } catch (error) {
        if (!(error instanceof UploadFailedError)) throw error;
        if (error.code !== 'OBJECT_TOO_LARGE' || path.extname(filePath).toLowerCase() !== '.mp4') {
            console.warn(`Skipping upload for ${filePath}: ${error.message}`);
            return '';
        }

        console.warn(`File too large, retrying compressed upload for ${filePath}`);
        const compressedPath = filePath.replace(/\.mp4$/i, '.upload-compressed.mp4');
        await transcodeForUpload(filePath, compressedPath);

        try {
            return await uploadFile(compressedPath, bucket, destination);
        } catch (retryError) {
            const message = retryError instanceof Error ? retryError.message : String(retryError);
            console.warn(`Compressed upload failed for ${filePath}: ${message}`);
            return '';
        }
    }
}

function transcodeForUpload(inputPath: string, outputPath: string): Promise<void> {
    return new Promise((resolve, reject) => {
        ffmpeg(inputPath)
            .videoCodec('libx264')
            .audioCodec('aac')
            .outputOptions([
                '-preset veryfast',
                '-crf 33',
                '-movflags +faststart',
                '-b:a 96k'
            ])
            .output(outputPath)
            .on('end', () => resolve())
            .on('error', (err) => reject(err))
            .run();
    });
}

async function startWorker() {
    console.log('Worker started. Polling for jobs...');

    while (true) {
        try {
            const { data: jobs, error } = await supabase
                .from('jobs')
                .select('*')
                .eq('status', 'pending')
                .limit(1);

            if (error) {
                console.error('Error fetching jobs:', error);
            } else if (jobs && jobs.length > 0) {
                const job = jobs[0];
                console.log(`Found job: ${job.id}. Starting pipeline...`);
                await runPipeline(job.id, job.youtube_url);
            }
        } catch (e) {
            console.error('Worker loop error:', e);
        }

        // Wait 10 seconds before polling again
        await new Promise(resolve => setTimeout(resolve, 10000));
    }
}

// Simple test trigger if run directly
if (require.main === module) {
    startWorker();
}

export { runPipeline };
