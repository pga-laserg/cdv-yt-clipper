import fs from 'fs';
import path from 'path';
import { supabase } from '../lib/supabase';

export interface LlmCueEvent {
    job_id?: string | null;
    run_id?: string | null;
    source_pass: string;
    model?: string | null;
    language?: string | null;
    denomination?: string | null;
    service_date?: string | null;
    section_type?: string | null;
    cue_kind: string;
    cue_text: string;
    cue_time_sec?: number | null;
    confidence?: number | null;
    metadata?: Record<string, unknown> | null;
}

let cueTableAvailability: 'unknown' | 'available' | 'missing' = 'unknown';
let missingTableWarningLogged = false;

function resolveJobId(workDir?: string): string | null {
    if (!workDir) return null;
    const base = path.basename(workDir);
    return base || null;
}

async function isCueTableAvailable(): Promise<boolean> {
    if (cueTableAvailability === 'available') return true;
    if (cueTableAvailability === 'missing') return false;

    try {
        const { error } = await supabase.from('llm_cue_events').select('id', { head: true }).limit(1);
        if (error) {
            if ((error as any).code === 'PGRST205') {
                cueTableAvailability = 'missing';
                if (!missingTableWarningLogged) {
                    missingTableWarningLogged = true;
                    console.warn(
                        "Supabase table 'public.llm_cue_events' is missing. Apply supabase/schema.sql (or migration) to enable cloud cue telemetry."
                    );
                }
                return false;
            }
            // Unknown API error: do not lock state, allow future retries.
            return true;
        }
        cueTableAvailability = 'available';
        return true;
    } catch {
        // Network/runtime errors should not permanently disable attempts.
        return true;
    }
}

export async function emitLlmCueEvents(events: LlmCueEvent[], workDir?: string): Promise<void> {
    if (!events.length) return;
    const language = process.env.SERVICE_LANGUAGE || process.env.BOUNDARY_LANGUAGE || 'spanish';
    const denomination = process.env.CHURCH_DENOMINATION || 'seventh_day_adventist';
    const serviceDate = process.env.SERVICE_DATE_HINT || null;
    const runId = `${Date.now()}`;
    const jobId = resolveJobId(workDir);

    const rows = events.map((e) => ({
        job_id: e.job_id ?? jobId,
        run_id: e.run_id ?? runId,
        source_pass: e.source_pass,
        model: e.model ?? null,
        language: e.language ?? language,
        denomination: e.denomination ?? denomination,
        service_date: e.service_date ?? serviceDate,
        section_type: e.section_type ?? null,
        cue_kind: e.cue_kind,
        cue_text: e.cue_text,
        cue_time_sec: Number.isFinite(Number(e.cue_time_sec)) ? Number(e.cue_time_sec) : null,
        confidence: Number.isFinite(Number(e.confidence)) ? Number(e.confidence) : null,
        metadata: e.metadata ?? {}
    }));

    if (workDir) {
        const out = path.join(workDir, 'llm.cues.jsonl');
        const lines = rows.map((r) => JSON.stringify({ created_at: new Date().toISOString(), ...r })).join('\n') + '\n';
        fs.appendFileSync(out, lines);
    }

    try {
        const available = await isCueTableAvailable();
        if (!available) return;
        const { error } = await supabase.from('llm_cue_events').insert(rows);
        if (error) {
            if ((error as any).code === 'PGRST205') {
                cueTableAvailability = 'missing';
                if (!missingTableWarningLogged) {
                    missingTableWarningLogged = true;
                    console.warn(
                        "Supabase table 'public.llm_cue_events' is missing. Apply supabase/schema.sql (or migration) to enable cloud cue telemetry."
                    );
                }
                return;
            }
            console.warn('Failed to insert llm cue telemetry:', error.message);
        }
    } catch (error) {
        console.warn('Failed to persist llm cue telemetry:', error);
    }
}
