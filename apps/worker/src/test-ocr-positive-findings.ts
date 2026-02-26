import fs from 'fs';
import path from 'path';

interface OcrSegment {
    start: number;
    end: number;
    text: string;
    type?: string;
    lane?: string;
    region?: string;
    confidence?: number;
}

interface OcrEventsFile {
    source?: string;
    duration_sec?: number;
    segments?: OcrSegment[];
}

function formatSec(sec: number): string {
    const s = Math.max(0, Math.floor(sec));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const ss = s % 60;
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
}

function cleanText(text: string): string {
    return String(text ?? '').replace(/\s+/g, ' ').trim();
}

function dedupeByTextAndTime(segments: OcrSegment[], maxGapSec = 15): OcrSegment[] {
    const sorted = [...segments].sort((a, b) => Number(a.start) - Number(b.start));
    const out: OcrSegment[] = [];
    for (const seg of sorted) {
        const text = cleanText(seg.text).toLowerCase();
        const last = out[out.length - 1];
        if (!last) {
            out.push(seg);
            continue;
        }
        const lastText = cleanText(last.text).toLowerCase();
        const gap = Number(seg.start) - Number(last.end);
        if (text && text === lastText && gap <= maxGapSec) {
            last.end = Math.max(Number(last.end), Number(seg.end));
            last.confidence = Math.max(Number(last.confidence ?? 0), Number(seg.confidence ?? 0));
            continue;
        }
        out.push(seg);
    }
    return out;
}

function isSermonTitleCandidate(seg: OcrSegment): boolean {
    const text = cleanText(seg.text);
    if (!text) return false;
    if (String(seg.type ?? '').toLowerCase() === 'sermon_title') return true;
    if (String(seg.lane ?? '').toLowerCase() !== 'slides') return false;
    return /\b(tema|t[íi]tulo|mensaje|serm[óo]n|predicaci[óo]n)\b/i.test(text);
}

function isHighConfidenceSpeaker(seg: OcrSegment, minConfidence: number): boolean {
    return (
        String(seg.type ?? '').toLowerCase() === 'speaker_name' &&
        Number(seg.confidence ?? 0) >= minConfidence &&
        cleanText(seg.text).length >= 3
    );
}

async function main() {
    const workDirArg = process.argv[2];
    if (!workDirArg) {
        throw new Error('Usage: npm run report:ocr-positive -- <workDir> [ocrFileName] [minSpeakerConf]');
    }
    const workDir = path.resolve(workDirArg);
    const ocrFileName = process.argv[3] || 'ocr.events.v3.json';
    const minSpeakerConf = Number(process.argv[4] ?? '0.85');

    const ocrPath = path.join(workDir, ocrFileName);
    if (!fs.existsSync(ocrPath)) throw new Error(`OCR file not found: ${ocrPath}`);

    const parsed = JSON.parse(fs.readFileSync(ocrPath, 'utf8')) as OcrEventsFile;
    const segments = Array.isArray(parsed.segments) ? parsed.segments : [];

    const sermonTitleCandidates = dedupeByTextAndTime(segments.filter((s) => isSermonTitleCandidate(s)));
    const highConfidenceSpeakers = dedupeByTextAndTime(
        segments.filter((s) => isHighConfidenceSpeaker(s, minSpeakerConf))
    );

    const outPath = path.join(workDir, 'ocr.positive-findings.md');
    const lines: string[] = [];
    lines.push('# OCR Positive Findings');
    lines.push('');
    lines.push(`- Source: \`${ocrFileName}\``);
    lines.push(`- OCR source: ${parsed.source ?? 'unknown'}`);
    if (Number.isFinite(Number(parsed.duration_sec))) {
        lines.push(`- Duration: ${formatSec(Number(parsed.duration_sec))}`);
    }
    lines.push(`- Min speaker confidence: ${minSpeakerConf.toFixed(2)}`);
    lines.push('');

    lines.push('## Sermon Title Slide Findings');
    if (sermonTitleCandidates.length === 0) {
        lines.push('- No positive sermon-title slide findings detected.');
    } else {
        for (const seg of sermonTitleCandidates) {
            lines.push(
                `- [${formatSec(Number(seg.start))} - ${formatSec(Number(seg.end))}] c=${Number(seg.confidence ?? 0).toFixed(3)} :: ${cleanText(seg.text)}`
            );
        }
    }
    lines.push('');

    lines.push('## High-Confidence Speaker Name Findings');
    if (highConfidenceSpeakers.length === 0) {
        lines.push('- No high-confidence speaker-name findings detected.');
    } else {
        for (const seg of highConfidenceSpeakers) {
            lines.push(
                `- [${formatSec(Number(seg.start))} - ${formatSec(Number(seg.end))}] c=${Number(seg.confidence ?? 0).toFixed(3)} :: ${cleanText(seg.text)}`
            );
        }
    }
    lines.push('');

    fs.writeFileSync(outPath, `${lines.join('\n')}\n`, 'utf8');
    console.log(`Wrote OCR positive findings report: ${outPath}`);
    console.log(
        JSON.stringify(
            {
                ok: true,
                out: outPath,
                sermon_title_findings: sermonTitleCandidates.length,
                high_conf_speaker_findings: highConfidenceSpeakers.length,
            },
            null,
            2
        )
    );
}

main().catch((err) => {
    console.error('report:ocr-positive failed:', err);
    process.exit(1);
});

