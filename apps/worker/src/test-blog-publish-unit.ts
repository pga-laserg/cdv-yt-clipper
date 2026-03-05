import { parseProviderJson, selectProfileCandidate, normalizeBlogDraft } from './pipeline/blog-artifact';

function assert(condition: unknown, message: string): void {
    if (!condition) throw new Error(message);
}

function run() {
    const profile = selectProfileCandidate('client-z', [
        { client_id: 'default', default_author_name: 'Fallback Author', llm_provider: 'openai', llm_model: 'gpt-5-mini' }
    ]);

    assert(profile?.client_id === 'default', 'should fallback to default profile');

    const draft = normalizeBlogDraft({
        raw: {
            title: 'Esperanza en tiempos difíciles',
            tags: ['esperanza'],
            scripture_refs: ['Romanos 5:3-5']
        },
        context: {
            jobId: 'job-publish-1234',
            workDir: '/tmp',
            youtubeUrl: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
            boundaries: { start: 0, end: 60 },
            transcriptSegments: [{ start: 0, end: 5, text: 'Mensaje principal del sermón.' }],
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
            default_author_name: 'Fallback Author',
            default_status: 'draft',
            sync_enabled: true,
            sync_endpoint: null,
            sync_token: null,
            preserve_published_fields: true,
            field_rules: {}
        },
        generatedAt: '2026-03-04T00:00:00.000Z'
    });

    assert(draft.post.author_name === 'Fallback Author', 'author should resolve from client profile default');
    assert(draft.post.id === 'sermon-job-publish-1234', 'deterministic post id should remain stable');

    const parsed = parseProviderJson('{"ok":true}');
    assert(parsed.ok === true, 'provider JSON should parse');

    console.log('blog publish unit checks passed');
}

run();
