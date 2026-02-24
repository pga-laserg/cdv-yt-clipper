import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import { generateAnalysisArtifacts } from './pipeline/analysis-doc';

dotenv.config({ path: path.resolve(__dirname, '../../../.env') });
dotenv.config({ path: path.resolve(__dirname, '../../web/.env.local') });

interface Segment {
    start: number;
    end: number;
    text: string;
}

function readJson<T>(filePath: string): T {
    return JSON.parse(fs.readFileSync(filePath, 'utf8')) as T;
}

async function run() {
    const workDir = process.argv[2]
        ? path.resolve(process.cwd(), process.argv[2])
        : path.resolve(__dirname, '../../test_data/e2e_live_4fHXiEHXT3I_rerun2');
    const transcriptPath = path.join(workDir, 'transcript.json');
    const targetedPath = path.join(workDir, 'sermon.boundaries.targeted-diarization.json');

    if (!fs.existsSync(transcriptPath)) {
        throw new Error(`Missing transcript: ${transcriptPath}`);
    }
    if (!fs.existsSync(targetedPath)) {
        throw new Error(`Missing boundaries artifact: ${targetedPath}`);
    }

    const transcript = readJson<Segment[]>(transcriptPath);
    const targeted = readJson<any>(targetedPath);
    const start = Number(targeted?.final_clip_bounds?.clip_start_sec);
    const end = Number(targeted?.final_clip_bounds?.clip_end_sec);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
        throw new Error('Invalid final_clip_bounds in sermon.boundaries.targeted-diarization.json');
    }

    await generateAnalysisArtifacts(transcript, { start, end }, { workDir });
    console.log(
        JSON.stringify(
            {
                workDir,
                boundaries: { start, end },
                outputs: [
                    path.join(workDir, 'analysis.doc.json'),
                    path.join(workDir, 'transcript.polished.json'),
                    path.join(workDir, 'transcript.polished.md'),
                    path.join(workDir, 'transcript.polished.multimodal.md')
                ]
            },
            null,
            2
        )
    );
}

run().catch((err) => {
    console.error('test-analysis-doc failed:', err);
    process.exit(1);
});

