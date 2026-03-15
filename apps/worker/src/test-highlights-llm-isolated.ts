import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import { findHighlights, HighlightClip } from './pipeline/highlights';

dotenv.config({ path: path.resolve(__dirname, '../../../.env') });
dotenv.config({ path: path.resolve(__dirname, '../../web/.env.local') });

interface Segment {
    start: number;
    end: number;
    text: string;
}

interface Paragraph {
    chapter_index?: number;
    chapter_title?: string;
    start: number;
    end: number;
    text: string;
}

interface Chapter {
    start: number;
    end: number;
    title: string;
    type?: string;
    confidence?: number;
}

function readJson<T>(filePath: string): T {
    return JSON.parse(fs.readFileSync(filePath, 'utf8')) as T;
}

function cleanText(text: string): string {
    return String(text ?? '').replace(/\s+/g, ' ').trim();
}

function fmt(sec: number): string {
    const s = Math.max(0, Math.floor(sec));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const ss = s % 60;
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
}

function overlap(aStart: number, aEnd: number, bStart: number, bEnd: number): number {
    return Math.max(0, Math.min(aEnd, bEnd) - Math.max(aStart, bStart));
}

function nearestByStart<T extends { start: number }>(items: T[], t: number): T | null {
    let best: T | null = null;
    let bestD = Number.POSITIVE_INFINITY;
    for (const item of items) {
        const d = Math.abs(item.start - t);
        if (d < bestD) {
            bestD = d;
            best = item;
        }
    }
    return best;
}

async function run() {
    const t0 = Date.now();
    const workDir = process.argv[2]
        ? path.resolve(process.cwd(), process.argv[2])
        : path.resolve(__dirname, '../../test_data/e2e_live_4fHXiEHXT3I_light');

    const transcriptPath = path.join(workDir, 'transcript.json');
    const polishedPath = path.join(workDir, 'transcript.polished.json');
    const targetedPath = path.join(workDir, 'sermon.boundaries.targeted-diarization.json');

    if (!fs.existsSync(transcriptPath)) throw new Error(`Missing transcript: ${transcriptPath}`);
    if (!fs.existsSync(polishedPath)) throw new Error(`Missing polished transcript: ${polishedPath}`);
    if (!fs.existsSync(targetedPath)) throw new Error(`Missing bounds: ${targetedPath}`);

    const transcript = readJson<Segment[]>(transcriptPath);
    const polished = readJson<{ paragraphs?: Paragraph[]; chapters?: Chapter[] }>(polishedPath);
    const targeted = readJson<any>(targetedPath);

    const start = Number(targeted?.final_clip_bounds?.clip_start_sec);
    const end = Number(targeted?.final_clip_bounds?.clip_end_sec);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
        throw new Error('Invalid final_clip_bounds in sermon.boundaries.targeted-diarization.json');
    }

    const sermonTranscript = transcript
        .filter((s) => Number.isFinite(s.start) && Number.isFinite(s.end))
        .filter((s) => s.start >= start && s.end <= end)
        .map((s) => ({ start: Number(s.start), end: Number(s.end), text: cleanText(s.text) }));

    const paragraphs = Array.isArray(polished?.paragraphs)
        ? polished.paragraphs
            .filter((p) => Number.isFinite(Number(p.start)) && Number.isFinite(Number(p.end)))
            .map((p) => ({
                chapter_index: Number.isFinite(Number(p.chapter_index)) ? Number(p.chapter_index) : undefined,
                chapter_title: typeof p.chapter_title === 'string' ? p.chapter_title : '',
                start: Number(p.start),
                end: Number(p.end),
                text: cleanText(p.text)
            }))
            .filter((p) => p.end >= start && p.start <= end)
        : [];

    const chapters = Array.isArray(polished?.chapters)
        ? polished.chapters
            .filter((c) => Number.isFinite(Number(c.start)) && Number.isFinite(Number(c.end)))
            .map((c) => ({
                start: Number(c.start),
                end: Number(c.end),
                title: cleanText(c.title),
                type: typeof c.type === 'string' ? c.type : undefined,
                confidence: Number.isFinite(Number(c.confidence)) ? Number(c.confidence) : undefined
            }))
            .filter((c) => c.end >= start && c.start <= end)
        : [];

    const clips = await findHighlights(sermonTranscript, 'openai', {
        sermonStart: start,
        sermonEnd: end,
        paragraphs,
        chapters
    });

    const detailed = clips.map((clip, idx) => {
        const overlappingParagraphs = paragraphs
            .map((p) => ({
                ...p,
                overlap_sec: Number(overlap(clip.start, clip.end, p.start, p.end).toFixed(2))
            }))
            .filter((p) => p.overlap_sec > 0)
            .sort((a, b) => a.start - b.start);

        const transcriptExcerpt = sermonTranscript
            .filter((s) => overlap(clip.start, clip.end, s.start, s.end) > 0)
            .map((s) => `[${s.start.toFixed(2)}-${s.end.toFixed(2)}] ${s.text}`)
            .join(' ');

        const nearStart = nearestByStart(paragraphs, clip.start);
        const nearEnd = nearestByStart(paragraphs, clip.end);

        return {
            index: idx + 1,
            clip,
            duration_sec: Number((clip.end - clip.start).toFixed(2)),
            paragraph_count: overlappingParagraphs.length,
            overlapping_paragraphs: overlappingParagraphs,
            nearest_start_paragraph: nearStart,
            nearest_end_paragraph: nearEnd,
            transcript_excerpt: transcriptExcerpt
        };
    });

    const meta = {
        generated_at: new Date().toISOString(),
        work_dir: workDir,
        model: process.env.HIGHLIGHTS_OPENAI_MODEL || process.env.ANALYZE_OPENAI_MODEL || 'gpt-5-mini',
        bounds: { start, end, duration_sec: Number((end - start).toFixed(2)) },
        input_counts: {
            transcript_segments_in_bounds: sermonTranscript.length,
            polished_paragraphs_in_bounds: paragraphs.length,
            polished_chapters_in_bounds: chapters.length
        },
        clips
    };

    const outJson = path.join(workDir, 'highlights.sections.llm.json');
    const outMetaJson = path.join(workDir, 'highlights.sections.llm.paragraph-metadata.json');
    const outMd = path.join(workDir, 'highlights.sections.llm.dev.md');

    fs.writeFileSync(outJson, JSON.stringify(meta, null, 2));
    fs.writeFileSync(outMetaJson, JSON.stringify({ ...meta, detailed }, null, 2));

    const md = [
        '# Highlights LLM (Isolated)',
        '',
        `- Generated: ${meta.generated_at}`,
        `- Model: ${meta.model}`,
        `- Bounds: ${start.toFixed(2)} - ${end.toFixed(2)} (${fmt(start)} - ${fmt(end)})`,
        `- Inputs: segments=${sermonTranscript.length}, paragraphs=${paragraphs.length}, chapters=${chapters.length}`,
        ''
    ];

    for (const item of detailed) {
        const c = item.clip;
        const sb = c.score_breakdown;
        md.push(`## Clip ${item.index}: ${c.title}`);
        md.push(`- Range: ${c.start.toFixed(2)} - ${c.end.toFixed(2)} (${fmt(c.start)} - ${fmt(c.end)})`);
        md.push(`- Duration: ${item.duration_sec}s`);
        md.push(`- Score: ${c.score}`);
        md.push(`- Hook type: ${c.hook_type ?? '[none]'}`);
        md.push(`- Hook text: ${c.hook || '[empty]'}`);
        if (sb) {
            md.push(`- Score breakdown:`);
            md.push(`  - hook_strength:       ${sb.hook_strength}  (weight 20%)`);
            md.push(`  - spiritual_impact:    ${sb.spiritual_impact}  (weight 30%)`);
            md.push(`  - shareability:        ${sb.shareability}  (weight 15%)`);
            md.push(`  - ending_completeness: ${sb.ending_completeness}  (weight 35%)`);
            md.push(`  - model_confidence:    ${sb.model_confidence}  (metadata only)`);
        }
        if (c.broll_cues && c.broll_cues.length) {
            md.push(`- B-roll cues (${c.broll_cues.length}):`);
            for (const cue of c.broll_cues) {
                md.push(`  - @${cue.offset_sec.toFixed(1)}s: ${cue.description} [${cue.keywords.join(', ')}]`);
            }
        }
        md.push(`- Excerpt: ${c.excerpt || '[empty]'}`);
        md.push(`- Paragraph overlaps: ${item.paragraph_count}`);
        if (item.nearest_start_paragraph) {
            md.push(
                `- Nearest start paragraph: [${item.nearest_start_paragraph.start.toFixed(2)}-${item.nearest_start_paragraph.end.toFixed(
                    2
                )}] ${item.nearest_start_paragraph.chapter_title || '[no chapter]'}`
            );
        }
        if (item.nearest_end_paragraph) {
            md.push(
                `- Nearest end paragraph: [${item.nearest_end_paragraph.start.toFixed(2)}-${item.nearest_end_paragraph.end.toFixed(
                    2
                )}] ${item.nearest_end_paragraph.chapter_title || '[no chapter]'}`
            );
        }
        md.push('- Transcript excerpt:');
        md.push(`  ${item.transcript_excerpt || '[empty]'}`);
        md.push('');
    }

    fs.writeFileSync(outMd, md.join('\n'));

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    console.log(
        JSON.stringify(
            {
                workDir,
                elapsed_sec: Number(elapsed),
                outputs: [outJson, outMetaJson, outMd],
                clip_count: clips.length
            },
            null,
            2
        )
    );
}

run().catch((err) => {
    console.error('test-highlights-llm-isolated failed:', err);
    process.exit(1);
});
