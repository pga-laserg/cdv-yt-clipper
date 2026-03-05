import path from 'path';
import dotenv from 'dotenv';
import { supabase } from './lib/supabase';
import { BlogPostDraftRow, upsertBlogPostDraft } from './pipeline/blog-artifact';

dotenv.config({ path: path.resolve(__dirname, '../../../.env') });

function assert(condition: unknown, message: string): void {
    if (!condition) throw new Error(message);
}

async function run() {
    const { data: orgRow, error: orgError } = await supabase
        .from('organizations')
        .select('id')
        .eq('slug', 'default-org')
        .maybeSingle();
    if (orgError || !orgRow?.id) {
        throw new Error(`failed to resolve default organization: ${orgError?.message || 'missing default-org row'}`);
    }
    const organizationId = orgRow.id as string;

    const id = `sermon-integration-${Date.now()}`;
    const now = new Date().toISOString();

    const draft1: BlogPostDraftRow = {
        id,
        organization_id: organizationId,
        slug: `integration-${Date.now()}`,
        status: 'draft',
        title: 'Integration Draft 1',
        excerpt: 'Initial draft excerpt',
        content_markdown: '## Draft 1\n\nBody',
        seo_title: 'Integration Draft 1',
        seo_description: 'Integration description 1',
        focus_keyword: 'integration',
        tags: ['integration'],
        scripture_refs: ['Santiago 1:2-4'],
        youtube_url: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
        youtube_video_id: 'dQw4w9WgXcQ',
        youtube_thumbnail: 'https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg',
        sermon_date: null,
        published_at: null,
        author_name: 'Integration Bot',
        canonical_url: '',
        hero_image_url: '',
        created_at: now,
        updated_at: now
    };

    const first = await upsertBlogPostDraft(draft1, {
        organizationId,
        preservePublishedFields: true
    });
    assert(first.id === id, 'first upsert should return same id');

    const draft2 = {
        ...draft1,
        title: 'Integration Draft 2',
        excerpt: 'Updated draft excerpt',
        content_markdown: '## Draft 2\n\nUpdated body'
    };

    const second = await upsertBlogPostDraft(draft2, {
        organizationId,
        preservePublishedFields: true
    });
    assert(second.id === id, 'second upsert should return same id');

    const { data: row, error } = await supabase
        .from('posts')
        .select('id,title,excerpt,content_markdown,status')
        .eq('id', id)
        .eq('organization_id', organizationId)
        .single();

    if (error) throw new Error(`failed to fetch upserted row: ${error.message}`);

    assert(row.id === id, 'row id mismatch');
    assert(row.title === 'Integration Draft 2', 'row title should reflect second upsert');
    assert(row.status === 'draft', 'row status should stay draft');

    const cleanup = await supabase
        .from('posts')
        .delete()
        .eq('id', id)
        .eq('organization_id', organizationId);
    if (cleanup.error) {
        console.warn(`cleanup failed for ${id}: ${cleanup.error.message}`);
    }

    console.log(`blog artifact integration checks passed for ${id}`);
}

run().catch((error) => {
    console.error('test-blog-artifact-integration failed:', error instanceof Error ? error.message : String(error));
    process.exit(1);
});
