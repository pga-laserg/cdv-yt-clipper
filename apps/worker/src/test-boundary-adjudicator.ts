import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import { adjudicateBoundariesWithSingleLlm } from './pipeline/boundary-adjudicator';

dotenv.config({ path: path.resolve(__dirname, '../../../.env') });
dotenv.config({ path: path.resolve(__dirname, '../../web/.env.local') });

interface Segment {
    start: number;
    end: number;
    text: string;
}

async function run() {
    const workDir = process.argv[2]
        ? path.resolve(process.cwd(), process.argv[2])
        : path.resolve(__dirname, '../../test_data/e2e_live_4fHXiEHXT3I_rerun2');
    const transcriptPath = path.join(workDir, 'transcript.json');
    const targetedPath = path.join(workDir, 'sermon.boundaries.targeted-diarization.json');
    if (!fs.existsSync(transcriptPath) || !fs.existsSync(targetedPath)) {
        throw new Error(`Missing cached artifacts in ${workDir}`);
    }

    const transcript = JSON.parse(fs.readFileSync(transcriptPath, 'utf8')) as Segment[];
    const targeted = JSON.parse(fs.readFileSync(targetedPath, 'utf8')) as any;
    const local = {
        start: Number(targeted?.final_clip_bounds?.clip_start_sec),
        end: Number(targeted?.final_clip_bounds?.clip_end_sec)
    };

    if (!Number.isFinite(local.start) || !Number.isFinite(local.end) || local.end <= local.start) {
        throw new Error('Invalid local bounds in sermon.boundaries.targeted-diarization.json');
    }

    const result = await adjudicateBoundariesWithSingleLlm(transcript, local, { workDir });
    console.log(
        JSON.stringify(
            {
                workDir,
                local,
                adjudicated: result
            },
            null,
            2
        )
    );
}

run().catch((err) => {
    console.error('test-boundary-adjudicator failed:', err);
    process.exit(1);
});

