import path from 'path';
import dotenv from 'dotenv';
import { supabase } from './lib/supabase';

dotenv.config({ path: path.resolve(__dirname, '../../../.env') });
dotenv.config({ path: path.resolve(__dirname, '../../web/.env.local') });

async function main() {
    if (!process.env.SUPABASE_URL || !process.env.SUPABASE_SERVICE_ROLE_KEY) {
        throw new Error('Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in env.');
    }

    const probe = await supabase.from('llm_cue_events').select('id', { head: true }).limit(1);
    if (probe.error) {
        if ((probe.error as any).code === 'PGRST205') {
            throw new Error(
                "Table public.llm_cue_events does not exist. Apply migration supabase/migrations/20260224_001_create_llm_cue_events.sql."
            );
        }
        throw new Error(`Select probe failed: ${probe.error.message}`);
    }

    const runId = `health-${Date.now()}`;
    const inserted = await supabase
        .from('llm_cue_events')
        .insert({
            run_id: runId,
            source_pass: 'health_check',
            model: process.env.BOUNDARY_LLM_MODEL || process.env.ANALYZE_OPENAI_MODEL || null,
            language: process.env.SERVICE_LANGUAGE || process.env.BOUNDARY_LANGUAGE || 'spanish',
            denomination: process.env.CHURCH_DENOMINATION || 'seventh_day_adventist',
            cue_kind: 'health_ping',
            cue_text: 'llm cue telemetry insert health check',
            confidence: 1,
            metadata: { health_check: true }
        })
        .select('id')
        .single();

    if (inserted.error) {
        throw new Error(`Insert probe failed: ${inserted.error.message}`);
    }

    const rowId = inserted.data?.id;
    if (rowId) {
        const del = await supabase.from('llm_cue_events').delete().eq('id', rowId);
        if (del.error) {
            console.warn(`Health row inserted but cleanup delete failed: ${del.error.message}`);
        }
    }

    console.log(
        JSON.stringify(
            {
                ok: true,
                table: 'public.llm_cue_events',
                select_probe: 'ok',
                insert_probe: 'ok',
                cleanup: rowId ? 'attempted' : 'skipped'
            },
            null,
            2
        )
    );
}

main().catch((err) => {
    console.error('llm cue telemetry health check failed:', err instanceof Error ? err.message : String(err));
    process.exit(1);
});

