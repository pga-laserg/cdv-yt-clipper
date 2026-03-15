import { getProcessingCache, upsertProcessingCache, generateTranscriptHash } from './lib/cache';
import { loadWorkerEnv } from './lib/load-env';

async function testCache() {
    loadWorkerEnv();
    
    const sourceUrl = `https://www.youtube.com/watch?v=test_${Date.now()}`;
    const segments = [
        { start: 0, end: 10, text: "Hello world" },
        { start: 10, end: 20, text: "This is a test" }
    ];
    const analysis = {
        boundaries: { start: 0, end: 20 },
        clips: [
            { id: "clip_1", start: 0, end: 10, title: "Test Clip" }
        ]
    };

    console.log(`--- Testing Cache for ${sourceUrl} ---`);

    // 1. Initial lookup (should be empty)
    const initial = await getProcessingCache(sourceUrl);
    console.log('Initial lookup:', initial ? 'FOUND' : 'NOT FOUND');

    // 2. Upsert segments only
    console.log('Upserting segments...');
    await upsertProcessingCache(sourceUrl, { segments });

    // 3. Verify segments cached
    const afterSegments = await getProcessingCache(sourceUrl);
    console.log('After segments lookup:', afterSegments ? `FOUND ${afterSegments.segments.length} segments` : 'NOT FOUND');
    console.log('Transcript hash:', afterSegments?.transcript_hash);
    console.log('Analysis cached:', Boolean(afterSegments?.analysis_json));

    // 4. Upsert full analysis
    console.log('Upserting full analysis...');
    await upsertProcessingCache(sourceUrl, { segments, analysis });

    // 5. Verify full hit
    const final = await getProcessingCache(sourceUrl);
    console.log('Final lookup:', final ? `FOUND ${final.segments.length} segments and analysis` : 'NOT FOUND');
    console.log('Analysis boundaries:', final?.analysis_json?.boundaries);
    console.log('Analysis clips:', final?.analysis_json?.clips.length);

    // 6. Test hash validation
    const hashMatches = final?.transcript_hash === generateTranscriptHash(segments);
    console.log('Hash validation:', hashMatches ? 'PASS' : 'FAIL');

    if (final && hashMatches) {
        console.log('\n✅ Cache logic verified successfully!');
    } else {
        console.error('\n❌ Cache logic verification failed.');
        process.exit(1);
    }
}

testCache().catch(err => {
    console.error(err);
    process.exit(1);
});
