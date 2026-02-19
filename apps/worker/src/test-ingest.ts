import { ingest } from './pipeline/ingest';
import path from 'path';

async function testIngest() {
    const url = 'https://www.youtube.com/watch?v=Jg51MCpDf0w';
    // Use a temp directory for testing
    const outputDir = path.resolve(__dirname, '../../test_data/ingest_test');

    try {
        const result = await ingest(url, outputDir);
        console.log('Ingest Result:', result);
    } catch (error) {
        console.error('Ingest Failed:', error);
    }
}

testIngest();
