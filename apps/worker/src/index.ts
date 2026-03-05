import { ingest } from './pipeline/ingest';
import { transcribe } from './pipeline/transcribe';
import { analyze } from './pipeline/analyze';
import { render } from './pipeline/render';
import { uploadFile, UploadFailedError } from './pipeline/store';
import { runBlogArtifactPostProcess } from './pipeline/blog-artifact';
import { supabase } from './lib/supabase';
import path from 'path';
import fs from 'fs';
import ffmpeg from 'fluent-ffmpeg';
import { execFile } from 'child_process';
import os from 'os';
import { promisify } from 'util';

interface PipelineJob {
    id: string;
    youtube_url: string;
    organization_id: string;
    claim_token?: string | null;
}

type PipelineMode = 'local' | 'cloud_limited' | 'prod';
type FullResStorageProvider = 'none' | 'proxmox' | 'supabase' | 's3';

const WORKER_ID = (process.env.WORKER_ID || `worker-${process.pid}`).trim();
const RPC_CLAIM_ENABLED = String(process.env.WORKER_RPC_CLAIM_ENABLED ?? 'true').toLowerCase() === 'true';
const JOB_LEASE_SECONDS = parsePositiveInt(process.env.WORKER_JOB_LEASE_SECONDS, 120, 30);
const POLL_INTERVAL_MS = parsePositiveInt(process.env.WORKER_POLL_INTERVAL_MS, 10_000, 500);
const PIPELINE_MODE = parsePipelineMode(process.env.PIPELINE_MODE);
const FULLRES_STORAGE_ENABLED = String(process.env.FULLRES_STORAGE_ENABLED ?? 'false').toLowerCase() === 'true';
const FULLRES_STORAGE_PROVIDER = parseFullResStorageProvider(
    process.env.FULLRES_STORAGE_PROVIDER,
    FULLRES_STORAGE_ENABLED ? 'proxmox' : 'none'
);
const FULLRES_STORAGE_SSH_HOST = (process.env.FULLRES_STORAGE_SSH_HOST || '').trim();
const FULLRES_STORAGE_SSH_USER = (process.env.FULLRES_STORAGE_SSH_USER || 'uploader').trim();
const FULLRES_STORAGE_SSH_PORT = parsePositiveInt(process.env.FULLRES_STORAGE_SSH_PORT, 22, 1);
const FULLRES_STORAGE_SSH_PATH = (process.env.FULLRES_STORAGE_SSH_PATH || '/home/uploader/uploads').trim();
const FULLRES_STORAGE_PUBLIC_BASE_URL = (process.env.FULLRES_STORAGE_PUBLIC_BASE_URL || '').trim();
const FULLRES_STORAGE_SSH_IDENTITY_FILE = (process.env.FULLRES_STORAGE_SSH_IDENTITY_FILE || '').trim();
const ENABLE_RAILWAY_VERTICAL = readBoolEnv('ENABLE_RAILWAY_VERTICAL', PIPELINE_MODE !== 'local');
const ENABLE_RAILWAY_HORIZONTAL_LOWRES = readBoolEnv('ENABLE_RAILWAY_HORIZONTAL_LOWRES', PIPELINE_MODE !== 'local');
const ENABLE_PROXMOX_FULLRES = readBoolEnv(
    'ENABLE_PROXMOX_FULLRES',
    PIPELINE_MODE === 'cloud_limited'
        ? FULLRES_STORAGE_ENABLED || Boolean(FULLRES_STORAGE_SSH_HOST)
        : FULLRES_STORAGE_ENABLED
);
const ENABLE_BLOG_ARTIFACT_STAGE = readBoolEnv('ENABLE_BLOG_ARTIFACT_STAGE', true);
const execFileAsync = promisify(execFile);

async function runPipeline(job: PipelineJob) {
    const jobId = job.id;
    const source = job.youtube_url;
    const organizationId = job.organization_id;
    const workDir = path.resolve(__dirname, `../work_dir/${jobId}`);
    if (!fs.existsSync(workDir)) fs.mkdirSync(workDir, { recursive: true });
    const startedAt = Date.now();
    let currentStage = 'initializing';

    try {
        // 1. Ingest
        console.log('--- Stage 1: Ingest ---');
        currentStage = 'ingest';
        await updateJobStatus(jobId, organizationId, 'processing:ingest');
        const { videoPathHQ, videoPathLight, audioPath } = await ingest(source, workDir);
        console.log(`Stage ingest completed in ${Math.round((Date.now() - startedAt) / 1000)}s`);

        // 2. Transcribe
        console.log('--- Stage 2: Transcribe ---');
        currentStage = 'transcribe';
        await updateJobStatus(jobId, organizationId, 'processing:transcribe');
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
                void updateJobStatus(jobId, organizationId, `processing:transcribe:${current}/${Math.floor(totalDurationSeconds)}`);
            },
            onFallback: (reason) => {
                console.warn(reason);
                void updateJobStatus(jobId, organizationId, 'processing:transcribe:fallback');
            }
        });
        const srtPath = path.join(path.dirname(audioPath), 'source.srt');
        console.log(`Stage transcribe completed in ${Math.round((Date.now() - startedAt) / 1000)}s`);

        // 3. Analyze
        console.log('--- Stage 3: Analyze ---');
        currentStage = 'analyze';
        await updateJobStatus(jobId, organizationId, 'processing:analyze');
        const { boundaries, clips } = await analyze(segments, { workDir, audioPath, videoPath: videoPathLight });
        console.log(`Stage analyze completed in ${Math.round((Date.now() - startedAt) / 1000)}s`);

        // 4. Render
        console.log('--- Stage 4: Render ---');
        currentStage = 'render';
        await updateJobStatus(jobId, organizationId, 'processing:render');
        const clipData = clips.map((c, i) => ({ ...c, id: `clip_${Date.now()}_${i + 1}` }));
        const renderedFiles = await render(videoPathHQ, boundaries, clipData, { trackingVideoPath: videoPathLight });
        console.log(`Stage render completed in ${Math.round((Date.now() - startedAt) / 1000)}s`);

        // 5. Store & Metadata
        console.log('--- Stage 5: Store ---');
        currentStage = 'store';
        await updateJobStatus(jobId, organizationId, 'processing:store');

        const shouldUploadSharedAssets = ENABLE_RAILWAY_VERTICAL || ENABLE_RAILWAY_HORIZONTAL_LOWRES;

        // Upload SRT
        let srtUrl = '';
        if (shouldUploadSharedAssets && fs.existsSync(srtPath)) {
            srtUrl = await uploadAssetWithFallback(srtPath, 'assets', `jobs/${jobId}/transcript.srt`);
        }

        // Upload Horizontal Sermon
        let horizontalUrl = '';
        const horizontalLocal = renderedFiles.find(f => f.includes('sermon_horizontal.mp4'));
        if (ENABLE_RAILWAY_HORIZONTAL_LOWRES && horizontalLocal && fs.existsSync(horizontalLocal)) {
            horizontalUrl = await uploadAssetWithFallback(horizontalLocal, 'assets', `jobs/${jobId}/sermon_horizontal.mp4`);
        } else if (horizontalLocal && fs.existsSync(horizontalLocal)) {
            console.log(`Skipping horizontal low-res upload for job ${jobId} (ENABLE_RAILWAY_HORIZONTAL_LOWRES=false).`);
        }

        // Upload thumbnail frame at 80% of the horizontal sermon duration.
        if (shouldUploadSharedAssets && horizontalLocal && fs.existsSync(horizontalLocal)) {
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
        await savePipelineResults(jobId, organizationId, boundaries, clipData, renderedFiles, srtUrl, horizontalUrl, horizontalLocal ?? '');

        // 6. Blog Artifact (non-blocking for final success state)
        if (ENABLE_BLOG_ARTIFACT_STAGE) {
            console.log('--- Stage 6: Blog Artifact ---');
            currentStage = 'blog';
            try {
                await runBlogArtifactPostProcess({
                    jobId,
                    organizationId,
                    workDir,
                    youtubeUrl: source,
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
            } catch (blogError) {
                console.error('Blog artifact stage failed (pipeline will still complete):', blogError);
            }
        } else {
            console.log('--- Stage 6: Blog Artifact (disabled) ---');
        }

        console.log(`Pipeline finished successfully in ${Math.round((Date.now() - startedAt) / 1000)}s`);

    } catch (error) {
        console.error(`Pipeline failed at stage "${currentStage}" after ${Math.round((Date.now() - startedAt) / 1000)}s:`, error);
        throw error;
    }
}

async function updateJobStatus(id: string, organizationId: string, status: string) {
    await supabase
        .from('jobs')
        .update({ status })
        .eq('id', id)
        .eq('organization_id', organizationId);
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
    organizationId: string,
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
        .eq('organization_id', organizationId)
        .single();
    let fullResVideoUrl = '';
    if (horizontalLocalPath && fs.existsSync(horizontalLocalPath)) {
        fullResVideoUrl = await uploadFullResToConfiguredStorage(horizontalLocalPath, jobId);
    }

    const metadata = {
        ...(existingJob?.metadata || {}),
        full_res_video_path: horizontalLocalPath || null,
        full_res_video_url: fullResVideoUrl || null,
        pipeline_mode: PIPELINE_MODE,
        artifact_routing: {
            railway_vertical: ENABLE_RAILWAY_VERTICAL,
            railway_horizontal_lowres: ENABLE_RAILWAY_HORIZONTAL_LOWRES,
            proxmox_fullres: ENABLE_PROXMOX_FULLRES,
            fullres_storage_provider: FULLRES_STORAGE_PROVIDER
        }
    };

    // Update Job
    await supabase.from('jobs').update({
        sermon_start_seconds: boundaries.start,
        sermon_end_seconds: boundaries.end,
        video_url: horizontalUrl,
        srt_url: srtUrl,
        metadata
    }).eq('id', jobId).eq('organization_id', organizationId);

    // Upload each clip and insert into clips table
    if (!ENABLE_RAILWAY_VERTICAL) {
        console.log(`Skipping vertical clip uploads for job ${jobId} (ENABLE_RAILWAY_VERTICAL=false).`);
    }
    for (let i = 0; i < clips.length; i++) {
        await updateJobStatus(jobId, organizationId, `processing:store:${i + 1}/${clips.length}`);
        const clip = clips[i];
        const localPath = renderedFiles.find(f => f.includes(clip.id));

        let videoUrl = '';
        if (ENABLE_RAILWAY_VERTICAL && localPath && fs.existsSync(localPath)) {
            videoUrl = await uploadAssetWithFallback(localPath, 'assets', `jobs/${jobId}/clips/${clip.id}.mp4`);
        }

        await supabase.from('clips').insert({
            organization_id: organizationId,
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

async function uploadFullResToConfiguredStorage(filePath: string, jobId: string): Promise<string> {
    if (!ENABLE_PROXMOX_FULLRES) return '';
    if (FULLRES_STORAGE_PROVIDER !== 'proxmox') {
        console.warn(`Full-res provider "${FULLRES_STORAGE_PROVIDER}" is not implemented yet. Expected "proxmox".`);
        return '';
    }

    if (!FULLRES_STORAGE_SSH_HOST || !FULLRES_STORAGE_PUBLIC_BASE_URL) {
        console.warn('Full-res proxmox upload enabled but FULLRES_STORAGE_SSH_HOST or FULLRES_STORAGE_PUBLIC_BASE_URL is missing.');
        return '';
    }

    const remoteRelativePath = `jobs/${jobId}/sermon_fullres.mp4`;
    const remoteAbsolutePath = `${FULLRES_STORAGE_SSH_PATH}/${remoteRelativePath}`;
    const sshTarget = `${FULLRES_STORAGE_SSH_USER}@${FULLRES_STORAGE_SSH_HOST}`;
    const sshArgs = ['-p', String(FULLRES_STORAGE_SSH_PORT)];
    const scpArgs = ['-P', String(FULLRES_STORAGE_SSH_PORT)];

    if (FULLRES_STORAGE_SSH_IDENTITY_FILE) {
        sshArgs.push('-i', FULLRES_STORAGE_SSH_IDENTITY_FILE);
        scpArgs.push('-i', FULLRES_STORAGE_SSH_IDENTITY_FILE);
    }

    sshArgs.push(`${sshTarget}`, `mkdir -p ${shellEscape(path.posix.dirname(remoteAbsolutePath))}`);

    try {
        await execFileAsync('ssh', sshArgs, { timeout: 45_000 });
        await execFileAsync('scp', [
            ...scpArgs,
            filePath,
            `${sshTarget}:${remoteAbsolutePath}`
        ], { timeout: 10 * 60_000, maxBuffer: 1024 * 1024 });

        const base = FULLRES_STORAGE_PUBLIC_BASE_URL.replace(/\/$/, '');
        return `${base}/${remoteRelativePath}`;
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.warn(`Failed to upload full-res video to remote storage: ${message}`);
        return '';
    }
}

function shellEscape(value: string): string {
    return `'${value.replace(/'/g, `'\"'\"'`)}'`;
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

function parsePositiveInt(value: string | undefined, fallback: number, min: number): number {
    const parsed = Number(value ?? fallback);
    if (!Number.isFinite(parsed)) return fallback;
    return Math.max(min, Math.floor(parsed));
}

function parsePipelineMode(value: string | undefined): PipelineMode {
    const normalized = String(value ?? 'prod').trim().toLowerCase();
    if (normalized === 'local' || normalized === 'cloud_limited' || normalized === 'prod') {
        return normalized;
    }
    console.warn(`Invalid PIPELINE_MODE="${value}". Falling back to "prod".`);
    return 'prod';
}

function parseFullResStorageProvider(value: string | undefined, fallback: FullResStorageProvider): FullResStorageProvider {
    const normalized = String(value ?? fallback).trim().toLowerCase();
    if (normalized === 'none' || normalized === 'proxmox' || normalized === 'supabase' || normalized === 's3') {
        return normalized;
    }
    console.warn(`Invalid FULLRES_STORAGE_PROVIDER="${value}". Falling back to "${fallback}".`);
    return fallback;
}

function readBoolEnv(name: string, fallback: boolean): boolean {
    const raw = process.env[name];
    if (raw == null || raw.trim() === '') return fallback;
    return raw.trim().toLowerCase() === 'true';
}

function normalizeClaimRow(row: any): PipelineJob | null {
    if (!row || typeof row !== 'object') return null;
    const id = typeof row.id === 'string' ? row.id.trim() : '';
    const youtubeUrl = typeof row.youtube_url === 'string' ? row.youtube_url.trim() : '';
    const organizationId = typeof row.organization_id === 'string' ? row.organization_id.trim() : '';
    if (!id || !youtubeUrl || !organizationId) return null;
    return {
        id,
        youtube_url: youtubeUrl,
        organization_id: organizationId,
        claim_token: typeof row.claim_token === 'string' ? row.claim_token : null
    };
}

async function claimNextJobRpc(workerId: string): Promise<PipelineJob | null> {
    const { data, error } = await supabase.rpc('claim_next_job', {
        p_worker_id: workerId,
        p_lease_seconds: JOB_LEASE_SECONDS
    });

    if (error) {
        console.error('Error calling claim_next_job RPC:', error);
        return null;
    }

    const row = Array.isArray(data) ? data[0] : data;
    return normalizeClaimRow(row);
}

async function claimNextJobLegacy(): Promise<PipelineJob | null> {
    const { data: jobs, error } = await supabase
        .from('jobs')
        .select('id,youtube_url,organization_id')
        .eq('status', 'pending')
        .order('created_at', { ascending: true })
        .limit(1);

    if (error) {
        console.error('Error fetching jobs:', error);
        return null;
    }

    if (!jobs || jobs.length === 0) return null;
    return normalizeClaimRow(jobs[0]);
}

function startClaimHeartbeat(job: PipelineJob): NodeJS.Timeout | null {
    if (!RPC_CLAIM_ENABLED || !job.claim_token) return null;

    const heartbeatEveryMs = Math.max(15_000, Math.floor((JOB_LEASE_SECONDS * 1000) / 2));
    const timer = setInterval(async () => {
        try {
            const { data, error } = await supabase.rpc('heartbeat_job_claim', {
                p_job_id: job.id,
                p_claim_token: job.claim_token,
                p_extend_seconds: JOB_LEASE_SECONDS
            });

            if (error) {
                console.error(`heartbeat_job_claim failed for job ${job.id}:`, error);
                return;
            }

            if (data !== true) {
                console.warn(`heartbeat_job_claim returned false for job ${job.id}; lease may be lost.`);
            }
        } catch (error) {
            console.error(`heartbeat_job_claim crashed for job ${job.id}:`, error);
        }
    }, heartbeatEveryMs);

    timer.unref?.();
    return timer;
}

async function completeJob(job: PipelineJob, finalStatus: 'completed' | 'failed', errorMessage: string | null): Promise<void> {
    if (RPC_CLAIM_ENABLED && job.claim_token) {
        const { data, error } = await supabase.rpc('complete_job_claim', {
            p_job_id: job.id,
            p_claim_token: job.claim_token,
            p_final_status: finalStatus,
            p_error: errorMessage
        });

        if (error) {
            console.error(`complete_job_claim failed for job ${job.id}:`, error);
        } else if (data !== true) {
            console.warn(`complete_job_claim returned false for job ${job.id}; claim token mismatch or stale lease.`);
            return;
        } else {
            return;
        }
    }

    const payload: Record<string, unknown> = {
        status: finalStatus,
        last_error: errorMessage
    };

    if (RPC_CLAIM_ENABLED) {
        payload.claim_token = null;
        payload.claimed_by = null;
        payload.claimed_at = null;
        payload.lease_expires_at = null;
    }

    await supabase
        .from('jobs')
        .update(payload)
        .eq('id', job.id)
        .eq('organization_id', job.organization_id);
}

async function processClaimedJob(job: PipelineJob): Promise<void> {
    const heartbeat = startClaimHeartbeat(job);
    let failure: string | null = null;

    try {
        console.log(`Found job: ${job.id}. Starting pipeline...`);
        await runPipeline(job);
        await completeJob(job, 'completed', null);
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        failure = message;
        console.error(`Pipeline failed for job ${job.id}:`, error);
        await completeJob(job, 'failed', message);
    } finally {
        if (heartbeat) clearInterval(heartbeat);
    }

    if (failure) {
        console.warn(`Job ${job.id} finished as failed: ${failure}`);
    }
}

async function startWorker() {
    console.log(
        `Worker started. claim_mode=${RPC_CLAIM_ENABLED ? 'rpc' : 'legacy'} ` +
        `worker_id=${WORKER_ID} pipeline_mode=${PIPELINE_MODE} ` +
        `vertical_upload=${ENABLE_RAILWAY_VERTICAL} horizontal_upload=${ENABLE_RAILWAY_HORIZONTAL_LOWRES} ` +
        `fullres_upload=${ENABLE_PROXMOX_FULLRES} fullres_provider=${FULLRES_STORAGE_PROVIDER} ` +
        `blog_stage=${ENABLE_BLOG_ARTIFACT_STAGE}`
    );

    while (true) {
        try {
            const job = RPC_CLAIM_ENABLED
                ? await claimNextJobRpc(WORKER_ID)
                : await claimNextJobLegacy();

            if (job) {
                await processClaimedJob(job);
            }
        } catch (e) {
            console.error('Worker loop error:', e);
        }

        await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL_MS));
    }
}

// Simple test trigger if run directly
if (require.main === module) {
    startWorker();
}

export { runPipeline };
