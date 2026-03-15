import { supabase } from './supabase';
import crypto from 'crypto';

export interface ProcessingCacheEntry {
    source_url: string;
    transcript_hash: string;
    segments: any[];
    analysis_json: {
        boundaries: { start: number; end: number };
        clips: any[];
    } | null;
}

/**
 * Generates a stable hash for transcript segments to detect changes.
 */
export function generateTranscriptHash(segments: any[]): string {
    const seed = JSON.stringify(segments.map(s => ({ start: s.start, end: s.end, text: s.text })));
    return crypto.createHash('md5').update(seed).digest('hex');
}

/**
 * Retrieves a cached result for a source URL.
 */
export async function getProcessingCache(sourceUrl: string): Promise<ProcessingCacheEntry | null> {
    const { data, error } = await supabase
        .from('processing_cache')
        .select('*')
        .eq('source_url', sourceUrl)
        .maybeSingle();

    if (error) {
        console.warn(`[cache] Failed to fetch cache for ${sourceUrl}:`, error.message);
        return null;
    }

    return data as ProcessingCacheEntry | null;
}

/**
 * Upserts a processing result into the cache.
 */
export async function upsertProcessingCache(
    sourceUrl: string, 
    data: { 
        segments: any[]; 
        analysis?: { boundaries: { start: number; end: number }; clips: any[] } 
    }
): Promise<void> {
    const transcript_hash = generateTranscriptHash(data.segments);
    
    const { error } = await supabase
        .from('processing_cache')
        .upsert({
            source_url: sourceUrl,
            transcript_hash,
            segments: data.segments,
            analysis_json: data.analysis || null,
            updated_at: new Date().toISOString()
        }, { onConflict: 'source_url' });

    if (error) {
        console.warn(`[cache] Failed to upsert cache for ${sourceUrl}:`, error.message);
    } else {
        console.log(`[cache] Persisted results for ${sourceUrl} (hash=${transcript_hash})`);
    }
}
