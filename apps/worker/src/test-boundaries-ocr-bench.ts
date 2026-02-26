import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import dotenv from 'dotenv';
import { findSermonBoundaries } from './pipeline/boundaries';
import { generateAnalysisArtifacts } from './pipeline/analysis-doc';

dotenv.config({ path: path.resolve(__dirname, '../../../.env') });
dotenv.config({ path: path.resolve(__dirname, '../../web/.env.local') });

interface Segment {
    start: number;
    end: number;
    text: string;
}

function runProcess(cmd: string, args: string[]): Promise<{ stdout: string; stderr: string }> {
    return new Promise((resolve, reject) => {
        const proc = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'] });
        let stdout = '';
        let stderr = '';
        proc.stdout.on('data', (d) => {
            const s = d.toString();
            stdout += s;
            if (s.trim()) console.log(s.trimEnd());
        });
        proc.stderr.on('data', (d) => {
            const s = d.toString();
            stderr += s;
            if (s.trim()) console.error(s.trimEnd());
        });
        proc.on('error', reject);
        proc.on('close', (code) => {
            if (code !== 0) return reject(new Error(`${cmd} exited ${code}. stderr tail: ${stderr.slice(-1200)}`));
            resolve({ stdout, stderr });
        });
    });
}

function nowStamp(): string {
    const d = new Date();
    const pad = (n: number) => String(n).padStart(2, '0');
    return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

async function ensureAudioWav(videoPath: string, audioPath: string): Promise<void> {
    if (fs.existsSync(audioPath)) return;
    fs.mkdirSync(path.dirname(audioPath), { recursive: true });
    await runProcess('ffmpeg', [
        '-y',
        '-i',
        videoPath,
        '-vn',
        '-ac',
        '1',
        '-ar',
        '16000',
        '-acodec',
        'pcm_s16le',
        audioPath
    ]);
}

function readJson<T>(p: string): T {
    return JSON.parse(fs.readFileSync(p, 'utf8')) as T;
}

function sec(ms: number): number {
    return Number((ms / 1000).toFixed(3));
}

function resolveOcrEventsPath(workDir: string): string {
    const rawPipeline = String(process.env.BOUNDARY_OCR_PIPELINE ?? process.env.OCR_PIPELINE ?? 'v1')
        .trim()
        .toLowerCase();
    const preferV2 = rawPipeline === 'v2' || rawPipeline === '2' || rawPipeline === 'ocr_v2';
    const preferV3 = rawPipeline === 'v3' || rawPipeline === '3' || rawPipeline === 'ocr_v3';
    const v1 = path.join(workDir, 'ocr.events.json');
    const v2 = path.join(workDir, 'ocr.events.v2.json');
    const v3 = path.join(workDir, 'ocr.events.v3.json');
    if (preferV3) {
        if (fs.existsSync(v3)) return v3;
        if (fs.existsSync(v2)) return v2;
        return v1;
    }
    if (preferV2) return fs.existsSync(v2) ? v2 : v1;
    if (fs.existsSync(v1)) return v1;
    if (fs.existsSync(v2)) return v2;
    return v3;
}

async function main() {
    const sourceDirArg = process.argv[2] || 'apps/test_data/e2e_live_wbkSOmlo1fw_light';
    const sourceDir = path.resolve(process.cwd(), sourceDirArg);
    const outDirArg = process.argv[3] || `apps/test_data/e2e_live_wbkSOmlo1fw_light_ocrbench_${nowStamp()}`;
    const outDir = path.resolve(process.cwd(), outDirArg);

    const sourceVideo = path.join(sourceDir, 'source.mp4');
    const sourceTranscript = path.join(sourceDir, 'transcript.json');
    if (!fs.existsSync(sourceVideo)) throw new Error(`Missing source video: ${sourceVideo}`);
    if (!fs.existsSync(sourceTranscript)) throw new Error(`Missing transcript: ${sourceTranscript}`);

    fs.mkdirSync(outDir, { recursive: true });
    const benchVideo = path.join(outDir, 'source.mp4');
    const benchTranscript = path.join(outDir, 'transcript.json');
    const benchAudio = path.join(outDir, 'audio.wav');
    const sourceFacePass = path.join(sourceDir, 'sermon.boundaries.face-pass.json');
    const benchFacePass = path.join(outDir, 'sermon.boundaries.face-pass.json');

    // Keep original folder untouched: symlink video, copy transcript.
    if (!fs.existsSync(benchVideo)) fs.symlinkSync(sourceVideo, benchVideo);
    if (!fs.existsSync(benchTranscript)) fs.copyFileSync(sourceTranscript, benchTranscript);
    if (fs.existsSync(sourceFacePass) && !fs.existsSync(benchFacePass)) {
        fs.copyFileSync(sourceFacePass, benchFacePass);
        console.log(`Reused face-pass seed: ${sourceFacePass}`);
    }

    const transcript = readJson<Segment[]>(benchTranscript);
    const tAudio0 = Date.now();
    await ensureAudioWav(benchVideo, benchAudio);
    const tAudio1 = Date.now();

    const tBound0 = Date.now();
    const boundaries = await findSermonBoundaries(transcript, {
        workDir: outDir,
        audioPath: benchAudio,
        videoPath: benchVideo
    });
    const tBound1 = Date.now();

    const tAnalysis0 = Date.now();
    await generateAnalysisArtifacts(transcript, boundaries, {
        workDir: outDir,
        videoPath: benchVideo
    });
    const tAnalysis1 = Date.now();

    const targetedPath = path.join(outDir, 'sermon.boundaries.targeted-diarization.json');
    const analysisDocPath = path.join(outDir, 'analysis.doc.json');
    const ocrEventsPath = resolveOcrEventsPath(outDir);
    const audioEventsPath = path.join(outDir, 'audio.events.json');

    const targeted = fs.existsSync(targetedPath) ? readJson<any>(targetedPath) : null;
    const analysisDoc = fs.existsSync(analysisDocPath) ? readJson<any>(analysisDocPath) : null;
    const ocrEvents = fs.existsSync(ocrEventsPath) ? readJson<any>(ocrEventsPath) : null;
    const audioEvents = fs.existsSync(audioEventsPath) ? readJson<any>(audioEventsPath) : null;

    const summary = {
        source_dir: sourceDir,
        output_dir: outDir,
        boundaries,
        timings_sec: {
            audio_extract: sec(tAudio1 - tAudio0),
            boundaries: sec(tBound1 - tBound0),
            analysis_doc: sec(tAnalysis1 - tAnalysis0),
            total: sec(tAnalysis1 - tAudio0)
        },
        extracted_signals: {
            ocr: {
                source: ocrEvents?.source ?? null,
                segments: Array.isArray(ocrEvents?.segments) ? ocrEvents.segments.length : 0,
                scene_cuts: Array.isArray(ocrEvents?.scene_cuts_sec) ? ocrEvents.scene_cuts_sec.length : 0,
                by_type: Array.isArray(ocrEvents?.segments)
                    ? ocrEvents.segments.reduce((acc: Record<string, number>, s: any) => {
                          const k = String(s?.type ?? 'unknown');
                          acc[k] = (acc[k] ?? 0) + 1;
                          return acc;
                      }, {})
                    : {}
            },
            audio: {
                source: audioEvents?.source ?? null,
                segments: Array.isArray(audioEvents?.segments) ? audioEvents.segments.length : 0,
                by_label: Array.isArray(audioEvents?.segments)
                    ? audioEvents.segments.reduce((acc: Record<string, number>, s: any) => {
                          const k = String(s?.label ?? 'unknown');
                          acc[k] = (acc[k] ?? 0) + 1;
                          return acc;
                      }, {})
                    : {}
            },
            boundary_debug: {
                start_confirmed: Boolean(targeted?.boundary_confirmation?.start_confirmed_change_of_speaker),
                end_confirmed: Boolean(targeted?.boundary_confirmation?.end_confirmed_change_of_speaker),
                start_chosen_by: targeted?.boundary_confirmation?.start_chosen_by ?? null,
                end_chosen_by: targeted?.boundary_confirmation?.end_chosen_by ?? null,
                start_ocr_signal: targeted?.boundary_confirmation?.start_ocr_signal ?? null,
                end_ocr_signal: targeted?.boundary_confirmation?.end_ocr_signal ?? null
            },
            chaptering: {
                chapters: Array.isArray(analysisDoc?.chapters) ? analysisDoc.chapters.length : 0,
                ocr_summary: analysisDoc?.ocr ?? null
            }
        },
        outputs: {
            targeted_boundaries: targetedPath,
            analysis_doc: analysisDocPath,
            ocr_events: ocrEventsPath,
            audio_events: audioEventsPath,
            polished_multimodal: path.join(outDir, 'transcript.polished.multimodal.md')
        }
    };

    const summaryJsonPath = path.join(outDir, 'benchmark.ocr-boundary.summary.json');
    const summaryMdPath = path.join(outDir, 'benchmark.ocr-boundary.summary.md');
    fs.writeFileSync(summaryJsonPath, JSON.stringify(summary, null, 2));

    const md = [
        '# OCR Boundary Benchmark',
        '',
        `- Source: \`${sourceDir}\``,
        `- Output: \`${outDir}\``,
        '',
        '## Timings (sec)',
        `- Audio extract: ${summary.timings_sec.audio_extract}`,
        `- Boundaries: ${summary.timings_sec.boundaries}`,
        `- Analysis doc: ${summary.timings_sec.analysis_doc}`,
        `- Total: ${summary.timings_sec.total}`,
        '',
        '## OCR Signals',
        `- Source: ${summary.extracted_signals.ocr.source ?? 'n/a'}`,
        `- Segments: ${summary.extracted_signals.ocr.segments}`,
        `- Scene cuts: ${summary.extracted_signals.ocr.scene_cuts}`,
        `- By type: \`${JSON.stringify(summary.extracted_signals.ocr.by_type)}\``,
        '',
        '## Audio Signals',
        `- Source: ${summary.extracted_signals.audio.source ?? 'n/a'}`,
        `- Segments: ${summary.extracted_signals.audio.segments}`,
        `- By label: \`${JSON.stringify(summary.extracted_signals.audio.by_label)}\``,
        '',
        '## Boundary Debug',
        `- Start confirmed: ${summary.extracted_signals.boundary_debug.start_confirmed}`,
        `- End confirmed: ${summary.extracted_signals.boundary_debug.end_confirmed}`,
        `- Start chosen by: ${summary.extracted_signals.boundary_debug.start_chosen_by ?? 'n/a'}`,
        `- End chosen by: ${summary.extracted_signals.boundary_debug.end_chosen_by ?? 'n/a'}`,
        '',
        '## Output Files',
        `- ${summary.outputs.targeted_boundaries}`,
        `- ${summary.outputs.analysis_doc}`,
        `- ${summary.outputs.ocr_events}`,
        `- ${summary.outputs.audio_events}`,
        `- ${summary.outputs.polished_multimodal}`,
        ''
    ].join('\n');
    fs.writeFileSync(summaryMdPath, md);

    console.log(JSON.stringify(summary, null, 2));
    console.log(`Wrote summary: ${summaryJsonPath}`);
    console.log(`Wrote summary: ${summaryMdPath}`);
}

main().catch((err) => {
    console.error('test-boundaries-ocr-bench failed:', err);
    process.exit(1);
});
