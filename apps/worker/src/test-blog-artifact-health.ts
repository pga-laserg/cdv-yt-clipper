import path from 'path';
import dotenv from 'dotenv';
import { supabase } from './lib/supabase';

dotenv.config({ path: path.resolve(__dirname, '../../../.env') });

async function assertTableReadable(table: string, column: string) {
    const res = await supabase.from(table).select(column, { head: true }).limit(1);
    if (res.error) {
        throw new Error(`${table} probe failed: ${res.error.message}`);
    }
}

async function run() {
    const { data: orgRow, error: orgError } = await supabase
        .from('organizations')
        .select('id')
        .eq('slug', 'default-org')
        .maybeSingle();
    if (orgError || !orgRow?.id) {
        throw new Error(`default-org lookup failed: ${orgError?.message || 'missing row'}`);
    }
    const organizationId = orgRow.id as string;

    await assertTableReadable('posts', 'id');
    await assertTableReadable('client_blog_profiles', 'client_id');
    await assertTableReadable('blog_generation_events', 'id');
    await assertTableReadable('client_publish_destinations', 'id');
    await assertTableReadable('blog_publish_events', 'id');

    const event = await supabase
        .from('blog_generation_events')
        .insert({
            organization_id: organizationId,
            job_id: `health-${Date.now()}`,
            client_id: 'default',
            post_id: null,
            status: 'health_check',
            stage: 'health',
            provider: 'openai',
            model: process.env.BLOG_OPENAI_MODEL || process.env.ANALYZE_OPENAI_MODEL || 'gpt-5-mini',
            prompt_version: process.env.BLOG_PROMPT_VERSION || 'blog-v1',
            attempt: 1,
            duration_ms: 1,
            metadata: { health_check: true }
        })
        .select('id')
        .single();

    if (event.error) {
        throw new Error(`blog_generation_events insert failed: ${event.error.message}`);
    }

    const id = event.data?.id;
    if (id) {
        const cleanup = await supabase
            .from('blog_generation_events')
            .delete()
            .eq('id', id)
            .eq('organization_id', organizationId);
        if (cleanup.error) {
            console.warn(`health check cleanup failed: ${cleanup.error.message}`);
        }
    }

    const publishEvent = await supabase
        .from('blog_publish_events')
        .insert({
            organization_id: organizationId,
            destination_id: null,
            destination_type: 'webhook',
            job_id: `health-${Date.now()}`,
            client_id: 'default',
            post_id: 'health-post',
            status: 'health_check',
            attempt: 1,
            duration_ms: 1,
            metadata: { health_check: true }
        })
        .select('id')
        .single();

    if (publishEvent.error) {
        throw new Error(`blog_publish_events insert failed: ${publishEvent.error.message}`);
    }

    const publishId = publishEvent.data?.id;
    if (publishId) {
        const publishCleanup = await supabase
            .from('blog_publish_events')
            .delete()
            .eq('id', publishId)
            .eq('organization_id', organizationId);
        if (publishCleanup.error) {
            console.warn(`publish health cleanup failed: ${publishCleanup.error.message}`);
        }
    }

    console.log(
        JSON.stringify(
            {
                ok: true,
                tables: ['posts', 'client_blog_profiles', 'blog_generation_events', 'client_publish_destinations', 'blog_publish_events'],
                insert_probe: 'ok',
                cleanup: id ? 'attempted' : 'skipped'
            },
            null,
            2
        )
    );
}

run().catch((error) => {
    console.error('test-blog-artifact-health failed:', error instanceof Error ? error.message : String(error));
    process.exit(1);
});
