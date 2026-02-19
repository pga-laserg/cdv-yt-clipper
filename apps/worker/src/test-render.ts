import { render } from './pipeline/render';
import path from 'path';

async function testRender() {
    const videoPath = path.resolve(__dirname, '../../test_data/ingest_test/source.mp4');
    const boundaries = { start: 600, end: 1200 }; // 10 min to 20 min
    const clips = [
        { id: 'test_clip_1', start: 660, end: 680, title: 'Test Clip' }
    ];

    console.log('Starting test render...');
    try {
        const results = await render(videoPath, boundaries, clips);
        console.log('Render complete:', results);
    } catch (e) {
        console.error('Render failed:', e);
    }
}

testRender();
