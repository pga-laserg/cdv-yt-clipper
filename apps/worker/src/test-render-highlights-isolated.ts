import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import ffmpeg from 'fluent-ffmpeg';
import { spawn } from 'child_process';

dotenv.config({ path: path.resolve(__dirname, '../../../.env') });
dotenv.config({ path: path.resolve(__dirname, '../../web/.env.local') });

interface HighlightClip {
    start: number;
    end: number;
    title?: string;
    excerpt?: string;
    hook?: string;
    confidence?: number;
    score?: number;
}

interface HighlightsDoc {
    clips: HighlightClip[];
}

interface CropTrackPoint {
    t: number;
    center_x: number;
}

function parseLastJson<T>(raw: string): T {
    const lines = raw
        .split(/\r?\n/)
        .map((l) => l.trim())
        .filter(Boolean);
    for (let i = lines.length - 1; i >= 0; i--) {
        try {
            return JSON.parse(lines[i]) as T;
        } catch {
            // continue
        }
    }
    return JSON.parse(raw) as T;
}

function perpendicularDistance(p: CropTrackPoint, a: CropTrackPoint, b: CropTrackPoint): number {
    const x = p.t;
    const y = p.center_x;
    const x1 = a.t;
    const y1 = a.center_x;
    const x2 = b.t;
    const y2 = b.center_x;
    const dx = x2 - x1;
    const dy = y2 - y1;
    if (Math.abs(dx) < 1e-9 && Math.abs(dy) < 1e-9) return Math.hypot(x - x1, y - y1);
    const num = Math.abs(dy * x - dx * y + x2 * y1 - y2 * x1);
    const den = Math.hypot(dx, dy);
    return num / den;
}

function simplifyRdp(points: CropTrackPoint[], epsilon: number): CropTrackPoint[] {
    if (points.length <= 2) return points;
    const first = points[0];
    const last = points[points.length - 1];
    let idx = -1;
    let maxDist = -1;
    for (let i = 1; i < points.length - 1; i++) {
        const d = perpendicularDistance(points[i], first, last);
        if (d > maxDist) {
            maxDist = d;
            idx = i;
        }
    }
    if (idx > 0 && maxDist > epsilon) {
        const left = simplifyRdp(points.slice(0, idx + 1), epsilon);
        const right = simplifyRdp(points.slice(idx), epsilon);
        return left.slice(0, -1).concat(right);
    }
    return [first, last];
}

function readJson<T>(filePath: string): T {
    return JSON.parse(fs.readFileSync(filePath, 'utf8')) as T;
}

async function detectSpeakerCenter(videoPath: string): Promise<number> {
    return new Promise((resolve, reject) => {
        const scriptCandidates = [
            path.resolve(__dirname, 'pipeline/python/autocrop.py'),
            path.resolve(process.cwd(), 'apps/worker/src/pipeline/python/autocrop.py'),
            path.resolve(process.cwd(), 'src/pipeline/python/autocrop.py')
        ];
        const pythonScript = scriptCandidates.find((p) => fs.existsSync(p));
        if (!pythonScript) return reject(new Error(`autocrop.py not found. Tried: ${scriptCandidates.join(', ')}`));

        const candidates = [
            path.resolve(__dirname, '../venv311/bin/python3'),
            path.resolve(__dirname, '../venv/bin/python3'),
            path.resolve(__dirname, '../../venv/bin/python3'),
            path.resolve(process.cwd(), 'apps/worker/venv/bin/python3'),
            path.resolve(process.cwd(), 'apps/worker/venv311/bin/python3'),
            path.resolve(process.cwd(), 'venv311/bin/python3'),
            path.resolve(process.cwd(), 'venv/bin/python3')
        ];
        const venvPython = candidates.find((p) => fs.existsSync(p));
        if (!venvPython) return reject(new Error(`python3 not found in expected venv paths: ${candidates.join(', ')}`));

        const proc = spawn(venvPython, [pythonScript, videoPath, '--limit_seconds', '300']);
        let out = '';

        proc.stdout.on('data', (d) => {
            out += d.toString();
        });
        proc.stderr.on('data', (d) => {
            process.stderr.write(`[autocrop] ${d}`);
        });
        proc.on('close', (code) => {
            if (code !== 0) return reject(new Error(`autocrop exited ${code}`));
            try {
                const parsed = parseLastJson<{ center_x: number }>(out);
                const cx = Number(parsed.center_x);
                if (!Number.isFinite(cx)) return reject(new Error('autocrop center_x missing'));
                resolve(cx);
            } catch (err) {
                reject(err);
            }
        });
    });
}

async function detectSpeakerTrack(videoPath: string, start: number, end: number): Promise<CropTrackPoint[] | null> {
    return new Promise((resolve, reject) => {
        const scriptCandidates = [
            path.resolve(__dirname, 'pipeline/python/autocrop.py'),
            path.resolve(process.cwd(), 'apps/worker/src/pipeline/python/autocrop.py'),
            path.resolve(process.cwd(), 'src/pipeline/python/autocrop.py')
        ];
        const pythonScript = scriptCandidates.find((p) => fs.existsSync(p));
        if (!pythonScript) return reject(new Error(`autocrop.py not found. Tried: ${scriptCandidates.join(', ')}`));

        const candidates = [
            path.resolve(__dirname, '../venv311/bin/python3'),
            path.resolve(__dirname, '../venv/bin/python3'),
            path.resolve(__dirname, '../../venv/bin/python3'),
            path.resolve(process.cwd(), 'apps/worker/venv/bin/python3'),
            path.resolve(process.cwd(), 'apps/worker/venv311/bin/python3'),
            path.resolve(process.cwd(), 'venv311/bin/python3'),
            path.resolve(process.cwd(), 'venv/bin/python3')
        ];
        const venvPython = candidates.find((p) => fs.existsSync(p));
        if (!venvPython) return reject(new Error(`python3 not found in expected venv paths: ${candidates.join(', ')}`));

        const duration = Math.max(0.1, end - start);
        const windowSec = Number(process.env.VERTICAL_DYNAMIC_CROP_WINDOW_SEC ?? 1.0);
        const smoothAlpha = Number(process.env.VERTICAL_DYNAMIC_CROP_SMOOTH_ALPHA ?? 0.35);
        const maxDeltaPerSec = Number(process.env.VERTICAL_DYNAMIC_CROP_MAX_DELTA_PER_SEC ?? 0.12);
        const useFacePassId = String(process.env.VERTICAL_DYNAMIC_CROP_USE_FACEPASS_ID ?? 'true').toLowerCase() === 'true';
        const facePassPath = path.join(path.dirname(videoPath), 'sermon.boundaries.face-pass.json');
        const identityMinSim = Number(process.env.VERTICAL_DYNAMIC_CROP_IDENTITY_MIN_SIM ?? 0.35);
        const identityDetSize = Number(process.env.VERTICAL_DYNAMIC_CROP_IDENTITY_DET_SIZE ?? 256);
        const deadband = Number(process.env.VERTICAL_DYNAMIC_CROP_DEADBAND ?? 0.015);
        const holdWindows = Number(process.env.VERTICAL_DYNAMIC_CROP_HOLD_WINDOWS ?? 2);
        const motionDt = Number(process.env.VERTICAL_DYNAMIC_CROP_MOTION_DT ?? 0.10);
        const maxAccelPerSec2 = Number(process.env.VERTICAL_DYNAMIC_CROP_MAX_ACCEL_PER_SEC2 ?? 0.06);
        const followKp = Number(process.env.VERTICAL_DYNAMIC_CROP_FOLLOW_KP ?? 7.0);
        const followKd = Number(process.env.VERTICAL_DYNAMIC_CROP_FOLLOW_KD ?? 4.0);
        const compositionFaceWeight = Number(process.env.VERTICAL_DYNAMIC_CROP_COMPOSITION_FACE_WEIGHT ?? 0.75);
        const lookaheadSec = Number(process.env.VERTICAL_DYNAMIC_CROP_LOOKAHEAD_SEC ?? 1.2);
        const keyframeSec = Number(process.env.VERTICAL_DYNAMIC_CROP_KEYFRAME_SEC ?? 1.5);
        const keyframeMinMove = Number(process.env.VERTICAL_DYNAMIC_CROP_KEYFRAME_MIN_MOVE ?? 0.012);
        const keyframeMaxHoldSec = Number(process.env.VERTICAL_DYNAMIC_CROP_KEYFRAME_MAX_HOLD_SEC ?? 4.0);
        const proc = spawn(venvPython, [
            pythonScript,
            videoPath,
            '--track',
            '--start_sec',
            String(start),
            '--duration_sec',
            String(duration),
            '--window_sec',
            String(windowSec),
            '--smooth_alpha',
            String(smoothAlpha),
            '--target_center',
            '0.5',
            '--max_delta_per_sec',
            String(maxDeltaPerSec),
            '--deadband',
            String(deadband),
            '--hold_windows',
            String(holdWindows),
            '--motion_dt',
            String(motionDt),
            '--max_accel_per_sec2',
            String(maxAccelPerSec2),
            '--follow_kp',
            String(followKp),
            '--follow_kd',
            String(followKd),
            '--composition_face_weight',
            String(compositionFaceWeight),
            '--lookahead_sec',
            String(lookaheadSec),
            '--keyframe_sec',
            String(keyframeSec),
            '--keyframe_min_move',
            String(keyframeMinMove),
            '--keyframe_max_hold_sec',
            String(keyframeMaxHoldSec),
            ...(useFacePassId && fs.existsSync(facePassPath)
                ? [
                    '--face_pass_json',
                    facePassPath,
                    '--identity_min_sim',
                    String(identityMinSim),
                    '--identity_det_size',
                    String(identityDetSize)
                ]
                : [])
        ]);
        let out = '';

        proc.stdout.on('data', (d) => {
            out += d.toString();
        });
        proc.stderr.on('data', (d) => {
            process.stderr.write(`[autocrop] ${d}`);
        });
        proc.on('close', (code) => {
            if (code !== 0) return reject(new Error(`autocrop track exited ${code}`));
            try {
                const parsed = parseLastJson<{ track?: Array<{ t: number; center_x: number }>; camera_keyframes?: Array<{ t: number; center_x: number }> }>(out);
                const useKeyframes = String(process.env.VERTICAL_DYNAMIC_CROP_USE_KEYFRAMES ?? 'false').toLowerCase() === 'true';
                const raw = useKeyframes && Array.isArray(parsed?.camera_keyframes) && parsed.camera_keyframes.length >= 2
                    ? parsed.camera_keyframes
                    : (Array.isArray(parsed?.track) ? parsed.track : []);
                const track = Array.isArray(raw)
                    ? raw
                        .filter((p) => Number.isFinite(Number(p.t)) && Number.isFinite(Number(p.center_x)))
                        .map((p) => ({ t: Number(p.t), center_x: Math.max(0, Math.min(1, Number(p.center_x))) }))
                        .sort((a, b) => a.t - b.t)
                    : [];
                resolve(track.length > 0 ? track : null);
            } catch (err) {
                reject(err);
            }
        });
    });
}

function buildCenterExpr(track: CropTrackPoint[]): string {
    if (!track.length) return '0.5';
    const sortedRaw = [...track].sort((a, b) => a.t - b.t);
    const maxPoints = Number(process.env.VERTICAL_DYNAMIC_CROP_MAX_EXPR_POINTS ?? 420);
    const eps = Number(process.env.VERTICAL_DYNAMIC_CROP_EXPR_SIMPLIFY_EPS ?? 0.0015);
    let sorted = simplifyRdp(sortedRaw, Math.max(0.0001, eps));
    if (sorted.length > maxPoints) {
        const stride = Math.ceil(sorted.length / maxPoints);
        sorted = sorted.filter((_, i) => i % stride === 0);
        if (sorted[sorted.length - 1].t !== sortedRaw[sortedRaw.length - 1].t) {
            sorted.push(sortedRaw[sortedRaw.length - 1]);
        }
    }
    if (sorted.length === 1) return `${sorted[0].center_x.toFixed(6)}`;

    // C1-continuous spline interpolation (Catmull-Rom/Hermite form) to avoid step-like velocity changes.
    let expr = `${sorted[sorted.length - 1].center_x.toFixed(6)}`;
    for (let i = sorted.length - 2; i >= 0; i--) {
        const p0 = sorted[Math.max(0, i - 1)];
        const p1 = sorted[i];
        const p2 = sorted[i + 1];
        const p3 = sorted[Math.min(sorted.length - 1, i + 2)];
        const dt = Math.max(0.001, p2.t - p1.t);
        const u = `max(0\\,min(1\\,(t-${p1.t.toFixed(3)})/${dt.toFixed(6)}))`;
        const u2 = `((${u})*(${u}))`;
        const u3 = `((${u2})*(${u}))`;
        const m1 = ((p2.center_x - p0.center_x) * 0.5) * dt;
        const m2 = ((p3.center_x - p1.center_x) * 0.5) * dt;
        const h00 = `(2*${u3}-3*${u2}+1)`;
        const h10 = `(${u3}-2*${u2}+${u})`;
        const h01 = `(-2*${u3}+3*${u2})`;
        const h11 = `(${u3}-${u2})`;
        const seg = `(${h00}*${p1.center_x.toFixed(6)}+${h10}*${m1.toFixed(6)}+${h01}*${p2.center_x.toFixed(6)}+${h11}*${m2.toFixed(6)})`;
        expr = `if(lt(t\\,${p2.t.toFixed(3)})\\,${seg}\\,${expr})`;
    }
    return expr;
}

function computeTrackStats(track: CropTrackPoint[] | null): { points: number; avg_step: number; max_step: number } {
    if (!track || track.length < 2) return { points: track?.length ?? 0, avg_step: 0, max_step: 0 };
    let sum = 0;
    let n = 0;
    let maxStep = 0;
    for (let i = 1; i < track.length; i++) {
        const d = Math.abs(track[i].center_x - track[i - 1].center_x);
        sum += d;
        n += 1;
        if (d > maxStep) maxStep = d;
    }
    return { points: track.length, avg_step: Number((sum / Math.max(1, n)).toFixed(4)), max_step: Number(maxStep.toFixed(4)) };
}

function renderVerticalClip(
    input: string,
    output: string,
    start: number,
    end: number,
    centerX: number,
    centerTrack: CropTrackPoint[] | null = null
): Promise<void> {
    return new Promise((resolve, reject) => {
        const duration = Math.max(0.1, end - start);
        const centerExpr = centerTrack && centerTrack.length > 1 ? buildCenterExpr(centerTrack) : `${centerX}`;
        const xExpr = `max(min((${centerExpr})*iw-(ih*9/16)/2\\,iw-(ih*9/16))\\,0)`;

        ffmpeg(input)
            .setStartTime(start)
            .setDuration(duration)
            .videoCodec('libx264')
            .audioCodec('aac')
            .outputOptions(['-preset veryfast', '-crf 20', '-movflags +faststart', '-b:a 160k'])
            .videoFilters([
                `crop=ih*9/16:ih:${xExpr}:0`,
                'scale=1080:1920'
            ])
            .output(output)
            .on('end', () => resolve())
            .on('error', (err) => reject(err))
            .run();
    });
}

async function run() {
    const t0 = Date.now();
    const workDir = process.argv[2]
        ? path.resolve(process.cwd(), process.argv[2])
        : path.resolve(__dirname, '../../test_data/e2e_live_4fHXiEHXT3I_light');

    const highlightsPath = path.join(workDir, 'highlights.sections.llm.json');
    const videoPath = path.join(workDir, 'source.mp4');
    const outDir = path.join(workDir, 'processed', 'highlights_vertical');

    if (!fs.existsSync(videoPath)) throw new Error(`Missing source video: ${videoPath}`);
    if (!fs.existsSync(highlightsPath)) throw new Error(`Missing highlights file: ${highlightsPath}`);
    fs.mkdirSync(outDir, { recursive: true });

    const doc = readJson<HighlightsDoc>(highlightsPath);
    const clips = Array.isArray(doc?.clips) ? doc.clips : [];
    if (!clips.length) throw new Error('No clips found in highlights.sections.llm.json');
    const maxRender = Number(process.env.HIGHLIGHTS_RENDER_COUNT ?? 5);
    const selected = [...clips]
        .sort((a, b) => {
            const sa = Number.isFinite(Number(a.score)) ? Number(a.score) : Number(a.confidence ?? 0);
            const sb = Number.isFinite(Number(b.score)) ? Number(b.score) : Number(b.confidence ?? 0);
            return sb - sa;
        })
        .slice(0, Math.max(0, maxRender));

    console.log(`[render-highlights] clips=${clips.length}, selected=${selected.length}`);
    console.log('[render-highlights] detecting center...');
    let centerX = 0.5;
    try {
        centerX = await detectSpeakerCenter(videoPath);
    } catch (err) {
        console.warn(`[render-highlights] autocrop failed; using center 0.5 (${String(err)})`);
    }
    console.log(`[render-highlights] centerX=${centerX}`);

    const outputs: string[] = [];
    const trackStats: Array<{ clip_index: number; track: { points: number; avg_step: number; max_step: number } }> = [];
    for (let i = 0; i < selected.length; i++) {
        const c = selected[i];
        const start = Number(c.start);
        const end = Number(c.end);
        if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
            console.warn(`[render-highlights] skip invalid clip ${i + 1}`);
            continue;
        }
        const out = path.join(outDir, `highlight_${String(i + 1).padStart(2, '0')}.vertical.mp4`);
        console.log(`[render-highlights] (${i + 1}/${selected.length}) ${start.toFixed(2)}-${end.toFixed(2)} -> ${path.basename(out)}`);
        let centerTrack: CropTrackPoint[] | null = null;
        const dynamicCropEnabled = String(process.env.VERTICAL_DYNAMIC_CROP_ENABLED ?? 'true').toLowerCase() === 'true';
        if (dynamicCropEnabled) {
            try {
                centerTrack = await detectSpeakerTrack(videoPath, start, end);
                if (centerTrack && centerTrack.length > 0) {
                    console.log(`[render-highlights] track points=${centerTrack.length} for ${path.basename(out)}`);
                }
            } catch (err) {
                console.warn(`[render-highlights] track failed for ${path.basename(out)} (${String(err)})`);
            }
        }
        trackStats.push({ clip_index: i + 1, track: computeTrackStats(centerTrack) });
        await renderVerticalClip(videoPath, out, start, end, centerX, centerTrack);
        outputs.push(out);
    }

    const summary = {
        generated_at: new Date().toISOString(),
        elapsed_sec: Number(((Date.now() - t0) / 1000).toFixed(1)),
        center_x: centerX,
        clips_rendered: outputs.length,
        track_stats: trackStats,
        outputs
    };
    const summaryPath = path.join(workDir, 'processed', 'highlights_vertical.render-summary.json');
    fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));

    console.log(JSON.stringify({ workDir, summaryPath, ...summary }, null, 2));
}

run().catch((err) => {
    console.error('test-render-highlights-isolated failed:', err);
    process.exit(1);
});
