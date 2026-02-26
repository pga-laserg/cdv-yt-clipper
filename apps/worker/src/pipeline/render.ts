import ffmpeg from 'fluent-ffmpeg';
import path from 'path';
import fs from 'fs';
import { spawn } from 'child_process';

interface CropTrackPoint {
    t: number;
    center_x: number;
    center_y?: number;
    face_h?: number;
}

interface CropTrackData {
    track: CropTrackPoint[];
    scene_cuts_sec: number[];
}

interface TrackStats {
    points: number;
    avg_step: number;
    max_step: number;
}

interface PathMetrics {
    velocity_rms_x: number;
    acceleration_rms_x: number;
    jerk_rms_x: number;
    velocity_rms_y: number;
    acceleration_rms_y: number;
    jerk_rms_y: number;
}

interface VerticalClipDebug {
    raw_track_points: number;
    processed_track_points: number;
    scene_cuts_sec: number[];
    applied_scene_resets: number;
    zoom_const: number;
    anchor_y: number;
    expressions: {
        center_x: string;
        x: string;
        y: string;
        crop_w: string;
        crop_h: string;
    };
    filters: string[];
    track_stats: TrackStats;
    path_metrics: PathMetrics;
    raw_track: CropTrackPoint[];
    processed_track: CropTrackPoint[];
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

function getWorkerPythonCandidates(): string[] {
    const override = String(process.env.WORKER_PYTHON_BIN ?? '').trim();
    const candidates = [
        path.resolve(__dirname, '../../venv311/bin/python3'),
        path.resolve(__dirname, '../../venv/bin/python3'),
        path.resolve(process.cwd(), 'apps/worker/venv311/bin/python3'),
        path.resolve(process.cwd(), 'apps/worker/venv/bin/python3'),
        path.resolve(process.cwd(), 'venv311/bin/python3'),
        path.resolve(process.cwd(), 'venv/bin/python3')
    ];
    return override ? [override, ...candidates] : candidates;
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

function resampleTrackUniform(points: CropTrackPoint[], dt: number): CropTrackPoint[] {
    if (!points.length) return [];
    const sorted = [...points].sort((a, b) => a.t - b.t);
    const startT = sorted[0].t;
    const endT = sorted[sorted.length - 1].t;
    const out: CropTrackPoint[] = [];
    let idx = 0;
    for (let t = startT; t <= endT + 1e-6; t += dt) {
        while (idx + 1 < sorted.length && sorted[idx + 1].t < t) idx++;
        const a = sorted[idx];
        const b = sorted[Math.min(sorted.length - 1, idx + 1)];
        const span = Math.max(1e-6, b.t - a.t);
        const u = Math.max(0, Math.min(1, (t - a.t) / span));
        const cx = a.center_x + (b.center_x - a.center_x) * u;
        out.push({ t, center_x: cx });
    }
    return out;
}

function ema(data: number[], alpha: number): number[] {
    const out: number[] = [];
    let prev = data[0];
    for (const v of data) {
        prev = alpha * v + (1 - alpha) * prev;
        out.push(prev);
    }
    return out;
}

function zeroPhaseSmooth(points: CropTrackPoint[], dt = 0.05, alpha = 0.2): CropTrackPoint[] {
    if (points.length < 2) return points;
    const uniform = resampleTrackUniform(points, dt);
    const xs = uniform.map((p) => p.center_x);
    const fwd = ema(xs, alpha);
    const rev = ema([...fwd].reverse(), alpha).reverse();
    const smoothed = rev.map((v, i) => (v + fwd[i]) * 0.5);
    return uniform.map((p, i) => ({ t: p.t, center_x: Math.max(0, Math.min(1, smoothed[i])) }));
}

function interpolateTrackField(
    track: CropTrackPoint[],
    t: number,
    key: 'center_y' | 'face_h',
    fallback: number
): number {
    if (!track.length) return fallback;
    const sorted = [...track].sort((a, b) => a.t - b.t);
    if (t <= sorted[0].t) return Number.isFinite(Number(sorted[0][key])) ? Number(sorted[0][key]) : fallback;
    if (t >= sorted[sorted.length - 1].t) {
        return Number.isFinite(Number(sorted[sorted.length - 1][key])) ? Number(sorted[sorted.length - 1][key]) : fallback;
    }
    let i = 0;
    while (i + 1 < sorted.length && sorted[i + 1].t < t) i++;
    const a = sorted[i];
    const b = sorted[Math.min(sorted.length - 1, i + 1)];
    const av = Number.isFinite(Number(a[key])) ? Number(a[key]) : fallback;
    const bv = Number.isFinite(Number(b[key])) ? Number(b[key]) : fallback;
    const span = Math.max(1e-6, b.t - a.t);
    const u = Math.max(0, Math.min(1, (t - a.t) / span));
    return av + (bv - av) * u;
}

function interpolateScalar(track: CropTrackPoint[], t: number, getValue: (p: CropTrackPoint) => number, fallback: number): number {
    if (!track.length) return fallback;
    const sorted = [...track].sort((a, b) => a.t - b.t);
    if (t <= sorted[0].t) return Number.isFinite(getValue(sorted[0])) ? getValue(sorted[0]) : fallback;
    if (t >= sorted[sorted.length - 1].t) {
        return Number.isFinite(getValue(sorted[sorted.length - 1])) ? getValue(sorted[sorted.length - 1]) : fallback;
    }
    let i = 0;
    while (i + 1 < sorted.length && sorted[i + 1].t < t) i++;
    const a = sorted[i];
    const b = sorted[Math.min(sorted.length - 1, i + 1)];
    const av = Number.isFinite(getValue(a)) ? getValue(a) : fallback;
    const bv = Number.isFinite(getValue(b)) ? getValue(b) : fallback;
    const span = Math.max(1e-6, b.t - a.t);
    const u = Math.max(0, Math.min(1, (t - a.t) / span));
    return av + (bv - av) * u;
}

function interpolatePoint(track: CropTrackPoint[], t: number): CropTrackPoint {
    const x = interpolateScalar(track, t, (p) => p.center_x, 0.5);
    const y = interpolateTrackField(track, t, 'center_y', 0.5);
    const h = interpolateTrackField(track, t, 'face_h', 0.18);
    return {
        t,
        center_x: Math.max(0, Math.min(1, x)),
        center_y: Math.max(0, Math.min(1, y)),
        face_h: Math.max(0.01, Math.min(0.8, h)),
    };
}

function sanitizeSceneCuts(cuts: number[], duration: number, minSegSec: number): number[] {
    const valid = [...cuts]
        .map((v) => Number(v))
        .filter((v) => Number.isFinite(v) && v > 0 && v < duration)
        .sort((a, b) => a - b);
    if (!valid.length) return [];
    const out: number[] = [];
    let prev = 0;
    for (const c of valid) {
        if ((c - prev) < minSegSec) continue;
        out.push(c);
        prev = c;
    }
    if (out.length && (duration - out[out.length - 1]) < minSegSec) {
        out.pop();
    }
    return out;
}

function splitTrackByCuts(track: CropTrackPoint[], cuts: number[], duration: number): Array<{ start: number; end: number; points: CropTrackPoint[] }> {
    if (!track.length) return [];
    const sorted = [...track].sort((a, b) => a.t - b.t);
    const boundaries = [0, ...cuts, duration].filter((v, i, arr) => i === 0 || v > arr[i - 1]);
    const segments: Array<{ start: number; end: number; points: CropTrackPoint[] }> = [];
    for (let i = 0; i < boundaries.length - 1; i++) {
        const t0 = boundaries[i];
        const t1 = boundaries[i + 1];
        if (t1 <= t0) continue;
        const inSeg = sorted.filter((p) => p.t >= t0 && p.t <= t1);
        const p0 = interpolatePoint(sorted, t0);
        const p1 = interpolatePoint(sorted, t1);
        const pts: CropTrackPoint[] = [p0];
        for (const p of inSeg) {
            if (p.t > t0 && p.t < t1) pts.push(p);
        }
        pts.push(p1);
        segments.push({ start: t0, end: t1, points: pts });
    }
    return segments;
}

function smoothTrackWithSceneResets(
    track: CropTrackPoint[],
    sceneCutsSec: number[],
    duration: number,
    dt: number,
    alpha: number
): CropTrackPoint[] {
    if (!track.length) return [];
    const minSegSec = Math.max(0.2, Number(process.env.VERTICAL_SHOT_RESET_MIN_SEG_SEC ?? 0.6));
    const cuts = sanitizeSceneCuts(sceneCutsSec, duration, minSegSec);
    if (!cuts.length) return zeroPhaseSmooth(track, dt, alpha);

    const segments = splitTrackByCuts(track, cuts, duration);
    const out: CropTrackPoint[] = [];
    for (const seg of segments) {
        const sm = zeroPhaseSmooth(seg.points, dt, alpha);
        for (const p of sm) {
            if (!out.length || p.t > out[out.length - 1].t + 1e-6) {
                out.push(p);
            }
        }
    }
    return out.length ? out : zeroPhaseSmooth(track, dt, alpha);
}

function quantile(values: number[], q: number): number {
    if (!values.length) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const idx = Math.max(0, Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1))));
    return sorted[idx];
}

interface ZoomPoint {
    t: number;
    z: number;
}

function buildZoomTrack(points: CropTrackPoint[], speedThreshold = 0.012, maxZoomOut = 1.3): ZoomPoint[] {
    if (points.length < 2) return points.map((p) => ({ t: p.t, z: 1 }));
    const zs: ZoomPoint[] = [];
    for (let i = 1; i < points.length; i++) {
        const prev = points[i - 1];
        const curr = points[i];
        const dt = Math.max(1e-6, curr.t - prev.t);
        const speed = Math.abs(curr.center_x - prev.center_x) / dt;
        const scale = 1 + Math.min(1, speed / speedThreshold) * (maxZoomOut - 1);
        zs.push({ t: curr.t, z: Math.min(maxZoomOut, Math.max(1, scale)) });
    }
    if (zs.length === 0) zs.push({ t: points[0].t, z: 1 });
    return zs;
}

function computeTrackStats(track: CropTrackPoint[] | null): TrackStats {
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
    return {
        points: track.length,
        avg_step: Number((sum / Math.max(1, n)).toFixed(5)),
        max_step: Number(maxStep.toFixed(5)),
    };
}

function rms(values: number[]): number {
    if (!values.length) return 0;
    const sumSq = values.reduce((acc, v) => acc + v * v, 0);
    return Math.sqrt(sumSq / values.length);
}

function derivative(values: number[], times: number[]): number[] {
    if (values.length < 2 || times.length < 2) return [];
    const out: number[] = [];
    for (let i = 1; i < values.length; i++) {
        const dt = Math.max(1e-6, times[i] - times[i - 1]);
        out.push((values[i] - values[i - 1]) / dt);
    }
    return out;
}

function computePathMetrics(track: CropTrackPoint[] | null): PathMetrics {
    if (!track || track.length < 4) {
        return {
            velocity_rms_x: 0,
            acceleration_rms_x: 0,
            jerk_rms_x: 0,
            velocity_rms_y: 0,
            acceleration_rms_y: 0,
            jerk_rms_y: 0,
        };
    }

    const sorted = [...track].sort((a, b) => a.t - b.t);
    const tx = sorted.map((p) => p.t);
    const xs = sorted.map((p) => p.center_x);
    const ys = sorted.map((p) => p.center_y ?? 0.5);

    const vx = derivative(xs, tx);
    const ax = derivative(vx, tx.slice(1));
    const jx = derivative(ax, tx.slice(2));

    const vy = derivative(ys, tx);
    const ay = derivative(vy, tx.slice(1));
    const jy = derivative(ay, tx.slice(2));

    return {
        velocity_rms_x: Number(rms(vx).toFixed(6)),
        acceleration_rms_x: Number(rms(ax).toFixed(6)),
        jerk_rms_x: Number(rms(jx).toFixed(6)),
        velocity_rms_y: Number(rms(vy).toFixed(6)),
        acceleration_rms_y: Number(rms(ay).toFixed(6)),
        jerk_rms_y: Number(rms(jy).toFixed(6)),
    };
}

function buildCenterExpr(track: CropTrackPoint[]): string {
    if (!track.length) return '0.5';
    const sortedRaw = [...track].sort((a, b) => a.t - b.t);
    const maxPoints = Number(process.env.VERTICAL_DYNAMIC_CROP_MAX_EXPR_POINTS ?? 180);
    const eps = Number(process.env.VERTICAL_DYNAMIC_CROP_EXPR_SIMPLIFY_EPS ?? 0.006);
    let sorted = simplifyRdp(sortedRaw, Math.max(0.0001, eps));
    if (sorted.length > maxPoints) {
        const stride = Math.ceil(sorted.length / maxPoints);
        sorted = sorted.filter((_, i) => i % stride === 0);
        if (sorted[sorted.length - 1].t !== sortedRaw[sortedRaw.length - 1].t) {
            sorted.push(sortedRaw[sortedRaw.length - 1]);
        }
    }
    if (sorted.length === 1) return `${sorted[0].center_x.toFixed(6)}`;

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

export async function render(
    videoPath: string,
    boundaries: { start: number; end: number },
    clips: { start: number; end: number; id: string; score?: number; confidence?: number }[],
    options: { trackingVideoPath?: string } = {}
): Promise<string[]> {
    console.log('Rendering clips...');
    const trackingVideoPath = options.trackingVideoPath && fs.existsSync(options.trackingVideoPath)
        ? options.trackingVideoPath
        : videoPath;

    const outputDir = path.join(path.dirname(videoPath), 'processed');
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    const results: string[] = [];
    const summary: {
        version: string;
        generated_at: string;
        video_path: string;
        tracking_video_path: string;
        center_x: number;
        clips_selected: number;
        clips_rendered: number;
        clips: Array<{
            id: string;
            start: number;
            end: number;
            output: string;
            path_debug_json: string;
            track_stats: TrackStats;
            path_metrics: PathMetrics;
        }>;
    } = {
        version: 'vertical-clipper-v3',
        generated_at: new Date().toISOString(),
        video_path: videoPath,
        tracking_video_path: trackingVideoPath,
        center_x: 0.5,
        clips_selected: 0,
        clips_rendered: 0,
        clips: [],
    };

    const horizontalFadeEnabled = String(process.env.HORIZONTAL_SERMON_FADE_ENABLED ?? 'true').toLowerCase() === 'true';
    const horizontalFadeInSecRaw = Number(process.env.HORIZONTAL_SERMON_FADE_IN_SEC ?? 1);
    const horizontalFadeOutSecRaw = Number(process.env.HORIZONTAL_SERMON_FADE_OUT_SEC ?? 1);
    const horizontalFadeInSec = horizontalFadeEnabled && Number.isFinite(horizontalFadeInSecRaw) ? Math.max(0, horizontalFadeInSecRaw) : 0;
    const horizontalFadeOutSec = horizontalFadeEnabled && Number.isFinite(horizontalFadeOutSecRaw) ? Math.max(0, horizontalFadeOutSecRaw) : 0;

    const sermonPath = path.join(outputDir, 'sermon_horizontal.mp4');
    await cutVideo(videoPath, boundaries.start, boundaries.end, sermonPath, {
        fadeInSec: horizontalFadeInSec,
        fadeOutSec: horizontalFadeOutSec,
    });
    results.push(sermonPath);

    let centerX = 0.5;
    try {
        centerX = await detectSpeakerCenter(trackingVideoPath);
        console.log(`Detected speaker center at relative X: ${centerX}`);
    } catch (e) {
        console.error('Failed to detect speaker, defaulting to center.', e);
    }
    summary.center_x = centerX;

    const maxVerticalRenders = Number(process.env.HIGHLIGHTS_RENDER_COUNT ?? 5);
    const selectedClips = [...clips]
        .sort((a, b) => {
            const sa = Number.isFinite(Number(a.score)) ? Number(a.score) : Number(a.confidence ?? 0);
            const sb = Number.isFinite(Number(b.score)) ? Number(b.score) : Number(b.confidence ?? 0);
            return sb - sa;
        })
        .slice(0, Math.max(0, maxVerticalRenders));

    summary.clips_selected = selectedClips.length;
    console.log(`Rendering vertical highlights: selected ${selectedClips.length}/${clips.length}`);

    for (const clip of selectedClips) {
        const clipPath = path.join(outputDir, `${clip.id}.mp4`);
        const dynamicCropEnabled = String(process.env.VERTICAL_DYNAMIC_CROP_ENABLED ?? 'true').toLowerCase() === 'true';
        let trackData: CropTrackData | null = null;
        if (dynamicCropEnabled) {
            try {
                trackData = await detectSpeakerTrack(trackingVideoPath, clip.start, clip.end);
                if (trackData?.track && trackData.track.length > 0) {
                    const cuts = trackData.scene_cuts_sec?.length ?? 0;
                    console.log(`Detected dynamic center track points=${trackData.track.length}, scene_cuts=${cuts} for ${clip.id}`);
                }
            } catch (e) {
                console.error(`Dynamic center track failed for ${clip.id}, using static center.`, e);
            }
        }

        const debug = await cutVideoVertical(
            videoPath,
            clip.start,
            clip.end,
            clipPath,
            centerX,
            trackData?.track ?? null,
            trackData?.scene_cuts_sec ?? []
        );
        results.push(clipPath);

        const pathDebugPath = path.join(outputDir, `${clip.id}.vertical.path.v3.json`);
        fs.writeFileSync(
            pathDebugPath,
            JSON.stringify(
                {
                    version: 'vertical-clipper-v3',
                    clip: {
                        id: clip.id,
                        start: clip.start,
                        end: clip.end,
                        duration_sec: Math.max(0, clip.end - clip.start),
                    },
                    source: {
                        video_path: videoPath,
                        tracking_video_path: trackingVideoPath,
                        center_x: centerX,
                    },
                    ...debug,
                },
                null,
                2
            )
        );

        summary.clips_rendered += 1;
        summary.clips.push({
            id: clip.id,
            start: clip.start,
            end: clip.end,
            output: clipPath,
            path_debug_json: pathDebugPath,
            track_stats: debug.track_stats,
            path_metrics: debug.path_metrics,
        });
    }

    const summaryPath = path.join(outputDir, 'highlights_vertical.v3.render-summary.json');
    fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));
    console.log(`Vertical v3 summary: ${summaryPath}`);

    return results;
}

async function detectSpeakerCenter(videoPath: string): Promise<number> {
    return new Promise((resolve, reject) => {
        const scriptCandidates = [
            path.resolve(__dirname, 'python/autocrop.py'),
            path.resolve(__dirname, '../../src/pipeline/python/autocrop.py'),
            path.resolve(process.cwd(), 'apps/worker/src/pipeline/python/autocrop.py'),
            path.resolve(process.cwd(), 'src/pipeline/python/autocrop.py')
        ];
        const pythonScript = scriptCandidates.find((p) => fs.existsSync(p));
        if (!pythonScript) {
            return reject(new Error(`autocrop.py not found. Tried: ${scriptCandidates.join(', ')}`));
        }

        const pyCandidates = getWorkerPythonCandidates();
        const venvPython = pyCandidates.find((p) => fs.existsSync(p));
        if (!venvPython) {
            return reject(new Error(`python3 not found in expected venv paths: ${pyCandidates.join(', ')}`));
        }

        const child = spawn(venvPython, [pythonScript, videoPath, '--limit_seconds', '300']);
        let dataString = '';

        child.stdout.on('data', (data) => {
            dataString += data.toString();
        });
        child.stderr.on('data', (data) => {
            console.error(`Autocrop stderr: ${data}`);
        });
        child.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`Autocrop process exited with code ${code}`));
                return;
            }
            try {
                const result = parseLastJson<{ center_x: number }>(dataString);
                resolve(result.center_x);
            } catch (err) {
                reject(err);
            }
        });
    });
}

async function detectSpeakerTrack(videoPath: string, start: number, end: number): Promise<CropTrackData | null> {
    return new Promise((resolve, reject) => {
        const scriptCandidates = [
            path.resolve(__dirname, 'python/autocrop.py'),
            path.resolve(__dirname, '../../src/pipeline/python/autocrop.py'),
            path.resolve(process.cwd(), 'apps/worker/src/pipeline/python/autocrop.py'),
            path.resolve(process.cwd(), 'src/pipeline/python/autocrop.py')
        ];
        const pythonScript = scriptCandidates.find((p) => fs.existsSync(p));
        if (!pythonScript) {
            return reject(new Error(`autocrop.py not found. Tried: ${scriptCandidates.join(', ')}`));
        }

        const pyCandidates = getWorkerPythonCandidates();
        const venvPython = pyCandidates.find((p) => fs.existsSync(p));
        if (!venvPython) {
            return reject(new Error(`python3 not found in expected venv paths: ${pyCandidates.join(', ')}`));
        }

        const duration = Math.max(0.1, end - start);
        const windowSec = Number(process.env.VERTICAL_DYNAMIC_CROP_WINDOW_SEC ?? 2.8);
        const smoothAlpha = Number(process.env.VERTICAL_DYNAMIC_CROP_SMOOTH_ALPHA ?? 0.18);
        const maxDeltaPerSec = Number(process.env.VERTICAL_DYNAMIC_CROP_MAX_DELTA_PER_SEC ?? 0.05);
        const useFacePassId = String(process.env.VERTICAL_DYNAMIC_CROP_USE_FACEPASS_ID ?? 'true').toLowerCase() === 'true';
        const facePassPath = path.join(path.dirname(videoPath), 'sermon.boundaries.face-pass.json');
        const identityMinSim = Number(process.env.VERTICAL_DYNAMIC_CROP_IDENTITY_MIN_SIM ?? 0.35);
        const identityDetSize = Number(process.env.VERTICAL_DYNAMIC_CROP_IDENTITY_DET_SIZE ?? 256);
        const deadband = Number(process.env.VERTICAL_DYNAMIC_CROP_DEADBAND ?? 0.03);
        const holdWindows = Number(process.env.VERTICAL_DYNAMIC_CROP_HOLD_WINDOWS ?? 4);
        const motionDt = Number(process.env.VERTICAL_DYNAMIC_CROP_MOTION_DT ?? 0.22);
        const maxAccelPerSec2 = Number(process.env.VERTICAL_DYNAMIC_CROP_MAX_ACCEL_PER_SEC2 ?? 0.025);
        const followKp = Number(process.env.VERTICAL_DYNAMIC_CROP_FOLLOW_KP ?? 2.5);
        const followKd = Number(process.env.VERTICAL_DYNAMIC_CROP_FOLLOW_KD ?? 1.6);
        const activeIdWeight = Number(process.env.VERTICAL_DYNAMIC_CROP_ACTIVE_ID_WEIGHT ?? 0.20);
        const talkWeight = Number(process.env.VERTICAL_DYNAMIC_CROP_TALK_WEIGHT ?? 0.15);
        const compositionFaceWeight = Number(process.env.VERTICAL_DYNAMIC_CROP_COMPOSITION_FACE_WEIGHT ?? 0.95);
        const lookaheadSec = Number(process.env.VERTICAL_DYNAMIC_CROP_LOOKAHEAD_SEC ?? 0.8);
        const keyframeSec = Number(process.env.VERTICAL_DYNAMIC_CROP_KEYFRAME_SEC ?? 3.0);
        const keyframeMinMove = Number(process.env.VERTICAL_DYNAMIC_CROP_KEYFRAME_MIN_MOVE ?? 0.03);
        const keyframeMaxHoldSec = Number(process.env.VERTICAL_DYNAMIC_CROP_KEYFRAME_MAX_HOLD_SEC ?? 8.0);

        const child = spawn(venvPython, [
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
            '--active_id_weight',
            String(activeIdWeight),
            '--talk_weight',
            String(talkWeight),
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

        let dataString = '';
        child.stdout.on('data', (data) => {
            dataString += data.toString();
        });
        child.stderr.on('data', (data) => {
            console.error(`Autocrop stderr: ${data}`);
        });
        child.on('close', (code) => {
            if (code !== 0) return reject(new Error(`Autocrop process exited with code ${code}`));
            try {
                const result = parseLastJson<{
                    track?: Array<{ t: number; center_x: number; center_y?: number; face_h?: number }>;
                    camera_keyframes?: Array<{ t: number; center_x: number }>;
                    scene_cuts_sec?: number[];
                }>(dataString);
                const useKeyframes = String(process.env.VERTICAL_DYNAMIC_CROP_USE_KEYFRAMES ?? 'false').toLowerCase() === 'true';
                const raw = useKeyframes && Array.isArray(result?.camera_keyframes) && result.camera_keyframes.length >= 2
                    ? result.camera_keyframes
                    : (Array.isArray(result?.track) ? result.track : []);
                const track = Array.isArray(raw)
                    ? raw
                        .filter((p) => Number.isFinite(Number(p.t)) && Number.isFinite(Number(p.center_x)))
                        .map((p) => ({
                            t: Number(p.t),
                            center_x: Math.max(0, Math.min(1, Number(p.center_x))),
                            center_y: Number.isFinite(Number((p as any).center_y))
                                ? Math.max(0, Math.min(1, Number((p as any).center_y)))
                                : undefined,
                            face_h: Number.isFinite(Number((p as any).face_h))
                                ? Math.max(0, Math.min(1, Number((p as any).face_h)))
                                : undefined
                        }))
                        .sort((a, b) => a.t - b.t)
                    : [];
                const sceneCuts = Array.isArray(result?.scene_cuts_sec)
                    ? result.scene_cuts_sec
                        .map((v) => Number(v))
                        .filter((v) => Number.isFinite(v) && v > 0)
                    : [];
                resolve(track.length > 0 ? { track, scene_cuts_sec: sceneCuts } : null);
            } catch (err) {
                reject(err);
            }
        });
    });
}

type FadeOpts = { fadeInSec?: number; fadeOutSec?: number };

function cutVideo(input: string, start: number, end: number, output: string, fades: FadeOpts = {}): Promise<void> {
    return new Promise((resolve, reject) => {
        console.log(`Cutting sermon: ${start}s to ${end}s`);
        const duration = end - start;
        const fadeIn = Math.max(0, Math.min(fades.fadeInSec ?? 0, duration / 2));
        const fadeOut = Math.max(0, Math.min(fades.fadeOutSec ?? 0, duration / 2));

        const vf: string[] = [];
        const af: string[] = [];

        if (fadeIn > 0) {
            vf.push(`fade=t=in:st=0:d=${fadeIn}`);
            af.push(`afade=t=in:st=0:d=${fadeIn}`);
        }
        if (fadeOut > 0) {
            vf.push(`fade=t=out:st=${Math.max(0, duration - fadeOut)}:d=${fadeOut}`);
            af.push(`afade=t=out:st=${Math.max(0, duration - fadeOut)}:d=${fadeOut}`);
        }

        const cmd = ffmpeg(input)
            .setStartTime(start)
            .setDuration(duration)
            .videoCodec('libx264')
            .audioCodec('aac')
            .outputOptions([
                '-preset veryfast',
                '-crf 20',
                '-movflags +faststart',
                '-b:a 192k'
            ])
            .output(output)
            .on('end', () => resolve())
            .on('error', (err) => reject(err));

        if (vf.length) cmd.videoFilters(vf.join(','));
        if (af.length) cmd.audioFilters(af.join(','));

        cmd.run();
    });
}

function cutVideoVertical(
    input: string,
    start: number,
    end: number,
    output: string,
    centerX: number,
    centerTrack: CropTrackPoint[] | null = null,
    sceneCutsSec: number[] = []
): Promise<VerticalClipDebug> {
    return new Promise((resolve, reject) => {
        console.log(`Cutting vertical clip v3: ${start}s to ${end}s with center ${centerX}`);
        const duration = Math.max(0.1, end - start);

        let processedTrack: CropTrackPoint[] | null = null;
        if (centerTrack && centerTrack.length > 1) {
            const dt = Math.max(0.02, Number(process.env.VERTICAL_V3_SMOOTH_DT ?? 0.05));
            const alpha = Math.max(0.01, Math.min(0.99, Number(process.env.VERTICAL_V3_SMOOTH_ALPHA ?? 0.18)));
            const shotResetEnabled = String(process.env.VERTICAL_SHOT_RESETS_ENABLED ?? 'true').toLowerCase() === 'true';
            const smoothedX = shotResetEnabled
                ? smoothTrackWithSceneResets(centerTrack, sceneCutsSec, duration, dt, alpha)
                : zeroPhaseSmooth(centerTrack, dt, alpha);
            processedTrack = smoothedX.map((p) => ({
                ...p,
                center_y: interpolateTrackField(centerTrack, p.t, 'center_y', 0.5),
                face_h: interpolateTrackField(centerTrack, p.t, 'face_h', 0.18),
            }));
        }

        const usableTrack = (processedTrack && processedTrack.length > 1)
            ? processedTrack
            : (centerTrack || [{ t: 0, center_x: centerX, center_y: 0.5, face_h: 0.18 }]);

        const centerExprRaw = usableTrack.length > 1
            ? buildCenterExpr(usableTrack)
            : `${centerX.toFixed(6)}`;
        const xs = usableTrack.map((p) => p.center_x);
        const torsoXAnchor = quantile(xs, 0.5);
        const torsoXPull = Math.max(0, Math.min(0.8, Number(process.env.VERTICAL_TORSO_X_CENTER_PULL ?? 0.40)));
        const centerExpr = Number.isFinite(torsoXAnchor) && torsoXPull > 0
            ? `((${(1 - torsoXPull).toFixed(4)})*(${centerExprRaw})+(${torsoXPull.toFixed(4)})*${torsoXAnchor.toFixed(6)})`
            : centerExprRaw;

        const zoomTrack = usableTrack.length > 1 ? buildZoomTrack(usableTrack) : [{ t: 0, z: 1 }];
        const motionSignal = Math.max(0, quantile(zoomTrack.map((p) => p.z), 0.8) - 1);
        const zoomBase = Math.max(1, Number(process.env.VERTICAL_CROP_ZOOM_BASE ?? 1.00));
        const zoomMotionGain = Math.max(0, Number(process.env.VERTICAL_CROP_ZOOM_MOTION_GAIN ?? 0.08));
        const zoomMax = Math.max(1, Number(process.env.VERTICAL_CROP_ZOOM_MAX ?? 1.04));
        const zoomConst = Math.max(1, Math.min(zoomMax, zoomBase + zoomMotionGain * motionSignal));

        const cropWidthExpr = `(ih/${zoomConst.toFixed(4)})*9/16`;
        const cropHeightExpr = `ih/${zoomConst.toFixed(4)}`;
        const xExpr = `max(min((${centerExpr})*iw-(${cropWidthExpr})/2\\,iw-(${cropWidthExpr}))\\,0)`;

        const ys = usableTrack.map((p) => p.center_y ?? 0.5);
        const hs = usableTrack.map((p) => p.face_h ?? 0.18);
        const torsoBias = Number(process.env.VERTICAL_TORSO_BIAS_MULT ?? 1.10);
        const headroomPos = Number(process.env.VERTICAL_TORSO_HEADROOM_POS ?? 0.26);
        const anchorY = Math.max(0, Math.min(1, quantile(ys, 0.5) + torsoBias * quantile(hs, 0.5)));
        const yExpr = `max(min(${anchorY.toFixed(4)}*ih-(${cropHeightExpr})*${headroomPos.toFixed(4)}\\,ih-(${cropHeightExpr}))\\,0)`;

        const filters = [
            'scale=iw*4:ih*4',
            `crop=${cropWidthExpr}:${cropHeightExpr}:${xExpr}:${yExpr}`,
            'scale=1080:1920',
            'setsar=1'
        ];

        ffmpeg(input)
            .setStartTime(start)
            .setDuration(duration)
            .videoCodec('libx264')
            .audioCodec('aac')
            .outputOptions([
                '-preset veryfast',
                '-crf 20',
                '-movflags +faststart',
                '-b:a 160k'
            ])
            .videoFilters(filters)
            .output(output)
            .on('stderr', (line) => console.log(`ffmpeg: ${line}`))
            .on('end', () => {
                const debug: VerticalClipDebug = {
                    raw_track_points: centerTrack?.length ?? 0,
                    processed_track_points: processedTrack?.length ?? 0,
                    scene_cuts_sec: sanitizeSceneCuts(sceneCutsSec, duration, Math.max(0.2, Number(process.env.VERTICAL_SHOT_RESET_MIN_SEG_SEC ?? 0.6))),
                    applied_scene_resets: sanitizeSceneCuts(sceneCutsSec, duration, Math.max(0.2, Number(process.env.VERTICAL_SHOT_RESET_MIN_SEG_SEC ?? 0.6))).length,
                    zoom_const: Number(zoomConst.toFixed(4)),
                    anchor_y: Number(anchorY.toFixed(4)),
                    expressions: {
                        center_x: centerExpr,
                        x: xExpr,
                        y: yExpr,
                        crop_w: cropWidthExpr,
                        crop_h: cropHeightExpr,
                    },
                    filters,
                    track_stats: computeTrackStats(usableTrack),
                    path_metrics: computePathMetrics(usableTrack),
                    raw_track: centerTrack || [],
                    processed_track: processedTrack || usableTrack,
                };
                resolve(debug);
            })
            .on('error', (err) => reject(err))
            .run();
    });
}
