import path from 'path';
import fs from 'fs';
import dotenv from 'dotenv';
import { ingest } from './pipeline/ingest';
import { transcribe } from './pipeline/transcribe';
import { analyze } from './pipeline/analyze';
import { render } from './pipeline/render';

dotenv.config({ path: path.resolve(__dirname, '../../../.env') });
dotenv.config({ path: path.resolve(__dirname, '../../web/.env.local') });

async function run() {
    const source = process.argv[2] || 'https://www.youtube.com/live/-COEISq_Y3w?si=9uMphvrKsBriAKFA';
    const outDir = process.argv[3]
        ? path.resolve(process.cwd(), process.argv[3])
        : path.resolve(__dirname, '../../test_data/e2e_live_test');

    fs.mkdirSync(outDir, { recursive: true });
    console.log(`E2E source: ${source}`);
    console.log(`E2E outDir: ${outDir}`);

    const { videoPathOriginal, videoPathHQ, videoPathLight, videoPathPreferredRender, audioPath } = await ingest(source, outDir);
    console.log('Ingest complete:', { videoPathOriginal, videoPathHQ, videoPathLight, videoPathPreferredRender, audioPath });

    const transcript = await transcribe(audioPath);
    console.log(`Transcribe complete. Segments: ${transcript.length}`);

    const { boundaries, clips } = await analyze(transcript, { workDir: outDir, audioPath, videoPath: videoPathLight });
    console.log('Analyze complete:', { boundaries, clips: clips.length });

    const clipData = clips.map((c, i) => ({ ...c, id: `e2e_clip_${i + 1}` }));
    const rendered = await render(videoPathPreferredRender, boundaries, clipData, {
        trackingVideoPath: videoPathLight,
        horizontalVideoPath: videoPathOriginal
    });
    console.log('Render complete:', rendered);
}

run().catch((err) => {
    console.error('test-e2e failed:', err);
    process.exit(1);
});
