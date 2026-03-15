import { transcribe } from './pipeline/transcribe';
import fs from 'fs';
import path from 'path';

async function main() {
    const testDir = path.resolve(__dirname, '../tmp/test_defuddle');
    if (!fs.existsSync(testDir)) fs.mkdirSync(testDir, { recursive: true });
    
    // Create dummy audio file
    const audioPath = path.join(testDir, 'audio.wav');
    fs.writeFileSync(audioPath, 'dummy audio content for test');
    
    // Create metadata.json with YouTube URL from the request
    const source = 'https://www.youtube.com/watch?v=orTFeWZ1j4E';
    fs.writeFileSync(path.join(testDir, 'metadata.json'), JSON.stringify({ source }, null, 2));
    
    console.log(`Testing Defuddle integration for source: ${source}`);
    console.log(`Working in: ${testDir}`);

    // Set env to ensure we don't accidentally call ElevenLabs or other expensive things if fast-path fails
    // But we want to test the fast-path specifically.
    process.env.TRANSCRIBE_PROVIDER = 'auto';
    process.env.TRANSCRIBE_FORCE_REDO = 'true';

    try {
        const segments = await transcribe(audioPath);
        console.log(`\n--- TEST RESULTS ---`);
        console.log(`Success! Extracted ${segments.length} segments.`);
        
        // Check for artifacts
        const transcriptPath = path.join(testDir, 'transcript.json');
        const srtPath = path.join(testDir, 'source.srt');
        const eventsPath = path.join(testDir, 'audio.events.defuddle.json');
        
        if (fs.existsSync(transcriptPath)) {
            console.log('✅ transcript.json created.');
        } else {
            console.error('❌ transcript.json NOT created.');
        }

        if (fs.existsSync(srtPath)) {
            console.log('✅ source.srt created.');
        } else {
            console.error('❌ source.srt NOT created.');
        }
        
        if (fs.existsSync(eventsPath)) {
            const events = JSON.parse(fs.readFileSync(eventsPath, 'utf8'));
            console.log(`✅ audio.events.defuddle.json created with ${events.segments.length} music events.`);
            if (events.segments.length > 0) {
                console.log(`First music event: ${events.segments[0].start}s - ${events.segments[0].end}s`);
            }
        } else {
            console.error('❌ audio.events.defuddle.json NOT created.');
        }
        
    } catch (error) {
        console.error('❌ Integration test failed:', error);
    }
}

main();
