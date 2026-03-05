import path from 'path';
import dotenv from 'dotenv';
import { normalizeBlogDraft, parseProviderJson, selectProfileCandidate, slugify } from './pipeline/blog-artifact';

dotenv.config({ path: path.resolve(__dirname, '../../../.env') });

function assert(condition: unknown, message: string): void {
    if (!condition) throw new Error(message);
}

function run() {
    const selected = selectProfileCandidate('client-a', [
        { client_id: 'default', llm_provider: 'openai', llm_model: 'gpt-5-mini' },
        { client_id: 'client-a', llm_provider: 'openai', llm_model: 'gpt-5-mini', default_author_name: 'Client Author' }
    ]);
    assert(selected?.client_id === 'client-a', 'should pick explicit client profile');

    const fallback = selectProfileCandidate('unknown', [
        { client_id: 'default', llm_provider: 'openai', llm_model: 'gpt-5-mini' }
    ]);
    assert(fallback?.client_id === 'default', 'should fallback to default profile');

    const slug = slugify('Fe, Coraje y Proceso!');
    assert(slug === 'fe-coraje-y-proceso', `unexpected slug ${slug}`);

    const artifact = normalizeBlogDraft({
        raw: {
            title: 'Fe, coraje y proceso',
            tags: ['fe', 'esperanza'],
            scripture_refs: 'Santiago 1:2-4, Romanos 5:3-5',
            confidence: 0.9
        },
        context: {
            jobId: 'job1234567890',
            workDir: '/tmp',
            youtubeUrl: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
            boundaries: { start: 0, end: 60 },
            transcriptSegments: [{ start: 0, end: 5, text: 'Contenido de prueba' }],
            clips: []
        },
        profile: {
            client_id: 'default',
            enabled: true,
            llm_provider: 'openai',
            llm_model: 'gpt-5-mini',
            prompt_version: 'blog-v1',
            system_prompt: null,
            user_prompt_template: null,
            default_author_name: 'Daniel Orellana',
            default_status: 'draft',
            sync_enabled: true,
            sync_endpoint: null,
            sync_token: null,
            preserve_published_fields: true,
            field_rules: {}
        },
        generatedAt: '2026-03-04T00:00:00.000Z'
    });

    assert(artifact.post.id === 'sermon-job1234567890', 'deterministic post id is required');
    assert(artifact.post.status === 'draft', 'status should be draft');
    assert(Array.isArray(artifact.post.tags) && artifact.post.tags.length === 2, 'tags should normalize to string[]');
    assert(Array.isArray(artifact.post.scripture_refs) && artifact.post.scripture_refs.length === 2, 'scripture refs should normalize to string[]');
    assert(artifact.validation.valid, 'normalized artifact should be valid');

    const parsed = parseProviderJson('{"title":"ok"}');
    assert(parsed.title === 'ok', 'provider json parse should work');

    let parseError = false;
    try {
        parseProviderJson('not-json');
    } catch {
        parseError = true;
    }
    assert(parseError, 'invalid provider response must throw');

    console.log('blog artifact unit checks passed');
}

run();
