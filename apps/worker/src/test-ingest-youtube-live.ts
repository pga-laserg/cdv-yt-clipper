import { ingest } from './pipeline/ingest';
import path from 'path';

async function testIngest() {
    const url = 'https://www.youtube.com/live/orTFeWZ1j4E';
    // Use a temp directory for testing
    const outputDir = path.resolve(__dirname, '../../test_data/ingest_test_youtube_live');

    try {
        console.log('Starting ingestion test for:', url);
        console.log('Output directory:', outputDir);
        const result = await ingest(url, outputDir);
        console.log('Ingest Result:', result);
    } catch (error) {
        console.error('Ingest Failed:', error);
    }
}

testIngest();
