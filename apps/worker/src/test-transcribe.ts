import { transcribe } from './pipeline/transcribe';
import path from 'path';
import fs from 'fs';

async function testTranscribe() {
    // Expecting the audio file from the ingestion test
    const audioPath = path.resolve(__dirname, '../../test_data/ingest_test/audio.wav');

    if (!fs.existsSync(audioPath)) {
        console.error('Audio file not found. Run ingest test first.');
        process.exit(1);
    }

    try {
        console.log('Starting transcription...');
        const start = Date.now();
        const segments = await transcribe(audioPath);
        const duration = (Date.now() - start) / 1000;

        console.log(`Transcription complete in ${duration}s`);
        console.log(`Generated ${segments.length} segments.`);

        fs.writeFileSync(path.join(path.dirname(audioPath), 'transcript.json'), JSON.stringify(segments, null, 2));
        console.log('Transcript saved to transcript.json');

        if (segments.length > 0) {
            console.log('First segment:', segments[0]);
            console.log('Last segment:', segments[segments.length - 1]);
        }
    } catch (error) {
        console.error('Transcription Failed:', error);
    }
}

testTranscribe();
