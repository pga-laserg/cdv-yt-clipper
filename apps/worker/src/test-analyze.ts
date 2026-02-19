import { analyze } from './pipeline/analyze';
import path from 'path';
import fs from 'fs';

async function testAnalyze() {
    const transcriptPath = path.resolve(__dirname, '../../test_data/ingest_test/transcript.json');

    if (!fs.existsSync(transcriptPath)) {
        console.error('Transcript file not found. Run transcript test first.');
        process.exit(1);
    }

    try {
        const transcript = JSON.parse(fs.readFileSync(transcriptPath, 'utf8'));
        console.log(`Loaded ${transcript.length} segments.`);

        const result = await analyze(transcript);

        console.log('Analysis Result:');
        console.log('Sermon Boundaries:', result.boundaries);
        console.log('Clips Found:', result.clips.length);

        if (result.clips.length > 0) {
            console.log('First Clip:', result.clips[0]);
        }

    } catch (error) {
        console.error('Analysis Failed:', error);
    }
}

testAnalyze();
