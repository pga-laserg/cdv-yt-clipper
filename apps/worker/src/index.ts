import { ingest } from './pipeline/ingest';
import { transcribe } from './pipeline/transcribe';
import { analyze } from './pipeline/analyze';
import { render } from './pipeline/render';
import { uploadFile } from './pipeline/store';
import { supabase } from './lib/supabase';
import path from 'path';
import fs from 'fs';

async function runPipeline(jobId: string, source: string) {
    const workDir = path.resolve(__dirname, `../work_dir/${jobId}`);
    if (!fs.existsSync(workDir)) fs.mkdirSync(workDir, { recursive: true });

    try {
        // 1. Ingest
        console.log('--- Stage 1: Ingest ---');
        await updateJobStatus(jobId, 'processing:ingest');
        const { videoPath, audioPath } = await ingest(source, workDir);

        // 2. Transcribe
        console.log('--- Stage 2: Transcribe ---');
        await updateJobStatus(jobId, 'processing:transcribe');
        const segments = await transcribe(audioPath);
        const srtPath = path.join(path.dirname(audioPath), 'source.srt');

        // 3. Analyze
        console.log('--- Stage 3: Analyze ---');
        await updateJobStatus(jobId, 'processing:analyze');
        const { boundaries, clips } = await analyze(segments);

        // 4. Render
        console.log('--- Stage 4: Render ---');
        await updateJobStatus(jobId, 'processing:render');
        const clipData = clips.map((c, i) => ({ ...c, id: `clip_${Date.now()}_${i + 1}` }));
        const renderedFiles = await render(videoPath, boundaries, clipData);

        // 5. Store & Metadata
        console.log('--- Stage 5: Store ---');
        await updateJobStatus(jobId, 'processing:store');

        // Upload SRT
        let srtUrl = '';
        if (fs.existsSync(srtPath)) {
            srtUrl = await uploadFile(srtPath, 'assets', `jobs/${jobId}/transcript.srt`);
        }

        // Upload Horizontal Sermon
        let horizontalUrl = '';
        const horizontalLocal = renderedFiles.find(f => f.includes('sermon_horizontal.mp4'));
        if (horizontalLocal && fs.existsSync(horizontalLocal)) {
            horizontalUrl = await uploadFile(horizontalLocal, 'assets', `jobs/${jobId}/sermon_horizontal.mp4`);
        }

        // Upload Clips and save to DB
        await savePipelineResults(jobId, boundaries, clipData, renderedFiles, srtUrl, horizontalUrl);

        await updateJobStatus(jobId, 'completed');
        console.log('Pipeline finished successfully!');

    } catch (error) {
        console.error('Pipeline failed:', error);
        await updateJobStatus(jobId, 'failed');
    }
}

async function updateJobStatus(id: string, status: string) {
    await supabase.from('jobs').update({ status }).eq('id', id);
}

async function savePipelineResults(jobId: string, boundaries: any, clips: any[], renderedFiles: string[], srtUrl: string, horizontalUrl: string) {
    // Update Job
    await supabase.from('jobs').update({
        sermon_start_seconds: boundaries.start,
        sermon_end_seconds: boundaries.end,
        video_url: horizontalUrl,
        srt_url: srtUrl
    }).eq('id', jobId);

    // Upload each clip and insert into clips table
    for (let i = 0; i < clips.length; i++) {
        await updateJobStatus(jobId, `processing:store:${i + 1}/${clips.length}`);
        const clip = clips[i];
        const localPath = renderedFiles.find(f => f.includes(clip.id));

        let videoUrl = '';
        if (localPath && fs.existsSync(localPath)) {
            videoUrl = await uploadFile(localPath, 'assets', `jobs/${jobId}/clips/${clip.id}.mp4`);
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
