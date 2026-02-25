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
    center_y?: number;
    face_h?: number;
}

function resampleTrackUniform(points: CropTrackPoint[], dt: number): CropTrackPoint[] {
    if (!points.length) return [];
    const sorted = [...points].sort((a, b) => a.t - b.t);
    const endT = sorted[sorted.length - 1].t;
    const out: CropTrackPoint[] = [];
    let idx = 0;
    for (let t = 0; t <= endT + 1e-6; t += dt) {
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

function minJerkSmooth(points: CropTrackPoint[], dt = 0.05): CropTrackPoint[] {
    // Solve argmin ||y - x||^2 + λ||D1 y||^2 + μ||D2 y||^2 + ν||D3 y||^2
    // Simple banded system; tuned for small tracks (<1k points).
    if (points.length < 4) return points;
    let np: any;
    try {
        np = require('numpy');
    } catch {
        return points; // fallback if numpy unavailable
    }
    const uniform = resampleTrackUniform(points, dt);
    const x = np.array(uniform.map((p) => p.center_x), 'float64');
    const n = x.shape[0];
    const lam = Number(process.env.VERTICAL_MINJERK_LAMBDA ?? 20);
    const mu = Number(process.env.VERTICAL_MINJERK_MU ?? 200);
    const nu = Number(process.env.VERTICAL_MINJERK_NU ?? 800);

    // Build T = I + lam*D1^T D1 + mu*D2^T D2 + nu*D3^T D3
    const T = np.zeros([n, n], 'float64');
    for (let i = 0; i < n; i++) T[i][i] = 1.0;

    // D1
    for (let i = 0; i < n - 1; i++) {
        T[i][i] += lam;
        T[i + 1][i + 1] += lam;
        T[i][i + 1] -= lam;
        T[i + 1][i] -= lam;
    }
    // D2
    for (let i = 0; i < n - 2; i++) {
        T[i][i] += mu;
        T[i + 1][i + 1] += 4 * mu;
        T[i + 2][i + 2] += mu;
        T[i][i + 1] -= 2 * mu;
        T[i + 1][i] -= 2 * mu;
        T[i + 1][i + 2] -= 2 * mu;
        T[i + 2][i + 1] -= 2 * mu;
        T[i][i + 2] += mu;
        T[i + 2][i] += mu;
    }
    // D3
    for (let i = 0; i < n - 3; i++) {
        T[i][i] += nu;
        T[i + 1][i + 1] += 9 * nu;
        T[i + 2][i + 2] += 9 * nu;
        T[i + 3][i + 3] += nu;

        T[i][i + 1] -= 3 * nu;
        T[i + 1][i] -= 3 * nu;
        T[i + 2][i + 3] -= 3 * nu;
        T[i + 3][i + 2] -= 3 * nu;

        T[i][i + 2] += 3 * nu;
        T[i + 2][i] += 3 * nu;
        T[i + 1][i + 3] += 3 * nu;
        T[i + 3][i + 1] += 3 * nu;

        T[i][i + 3] -= nu;
        T[i + 3][i] -= nu;
    }

    const y = np.linalg.solve(T, x);
    const out: CropTrackPoint[] = [];
    for (let i = 0; i < n; i++) {
        out.push({ t: uniform[i].t, center_x: Math.max(0, Math.min(1, Number(y[i]))) });
    }
    return out;
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
                const parsed = parseLastJson<{ track?: Array<{ t: number; center_x: number; center_y?: number; face_h?: number }>; camera_keyframes?: Array<{ t: number; center_x: number }> }>(out);
                const useKeyframes = String(process.env.VERTICAL_DYNAMIC_CROP_USE_KEYFRAMES ?? 'false').toLowerCase() === 'true';
                const raw = useKeyframes && Array.isArray(parsed?.camera_keyframes) && parsed.camera_keyframes.length >= 2
                    ? parsed.camera_keyframes
                    : (Array.isArray(parsed?.track) ? parsed.track : []);
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

function buildZoomExpr(track: ZoomPoint[]): string {
    if (!track.length) return '1';
    const sorted = [...track].sort((a, b) => a.t - b.t);
    let expr = `${sorted[sorted.length - 1].z.toFixed(4)}`;
    for (let i = sorted.length - 2; i >= 0; i--) {
        const p1 = sorted[i];
        const p2 = sorted[i + 1];
        const dt = Math.max(0.001, p2.t - p1.t);
        const u = `max(0\\,min(1\\,(t-${p1.t.toFixed(3)})/${dt.toFixed(6)}))`;
        const seg = `${p1.z.toFixed(4)}+(${(p2.z - p1.z).toFixed(4)})*${u}`;
        expr = `if(lt(t\,${p2.t.toFixed(3)})\,${seg}\,${expr})`;
    }
    return expr;
}

function quantile(values: number[], q: number): number {
    if (!values.length) return 1;
    const sorted = [...values].sort((a, b) => a - b);
    const idx = Math.max(0, Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1))));
    return sorted[idx];
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
        let processedTrack: CropTrackPoint[] | null = null;
        if (centerTrack && centerTrack.length > 1) {
            const mj = minJerkSmooth(centerTrack, 0.05);
            processedTrack = mj && mj.length > 1 ? mj : zeroPhaseSmooth(centerTrack, 0.05, 0.18);
            // Keep X smoothing, but restore Y/face-size from the original track so torso framing remains valid.
            processedTrack = processedTrack.map((p) => ({
                ...p,
                center_y: interpolateTrackField(centerTrack, p.t, 'center_y', 0.5),
                face_h: interpolateTrackField(centerTrack, p.t, 'face_h', 0.18),
            }));
        }
        const centerExprRaw = processedTrack && processedTrack.length > 1
            ? buildCenterExpr(processedTrack)
            : buildCenterExpr(centerTrack || [{ t: 0, center_x: centerX }]);
        const xs = (processedTrack || centerTrack || []).map((p) => p.center_x);
        const torsoXAnchor = quantile(xs, 0.5);
        const torsoXPull = Math.max(0, Math.min(0.8, Number(process.env.VERTICAL_TORSO_X_CENTER_PULL ?? 0.28)));
        const centerExpr = Number.isFinite(torsoXAnchor) && torsoXPull > 0
            ? `((${(1 - torsoXPull).toFixed(4)})*(${centerExprRaw})+(${torsoXPull.toFixed(4)})*${torsoXAnchor.toFixed(6)})`
            : centerExprRaw;
        const zoomTrack = processedTrack && processedTrack.length > 1 ? buildZoomTrack(processedTrack) : [{ t: 0, z: 1 }];
        // Looser vertical framing by default, with only mild motion-driven tighten when needed.
        const motionSignal = Math.max(0, quantile(zoomTrack.map((p) => p.z), 0.8) - 1);
        const zoomBase = Math.max(1, Number(process.env.VERTICAL_CROP_ZOOM_BASE ?? 1.0));
        const zoomMotionGain = Math.max(0, Number(process.env.VERTICAL_CROP_ZOOM_MOTION_GAIN ?? 0.15));
        const zoomMax = Math.max(1, Number(process.env.VERTICAL_CROP_ZOOM_MAX ?? 1.10));
        const zoomConst = Math.max(1, Math.min(zoomMax, zoomBase + zoomMotionGain * motionSignal));
        const cropWidthExpr = `(ih/${zoomConst.toFixed(4)})*9/16`;
        const cropHeightExpr = `ih/${zoomConst.toFixed(4)}`;
        const xExpr = `max(min((${centerExpr})*iw-(${cropWidthExpr})/2\\,iw-(${cropWidthExpr}))\\,0)`;
        const ys = (processedTrack || centerTrack || []).map((p) => p.center_y ?? 0.5);
        const hs = (processedTrack || centerTrack || []).map((p) => p.face_h ?? 0.18);
        const torsoBias = Number(process.env.VERTICAL_TORSO_BIAS_MULT ?? 0.95);
        const headroomPos = Number(process.env.VERTICAL_TORSO_HEADROOM_POS ?? 0.30);
        const anchorY = Math.max(0, Math.min(1, quantile(ys, 0.5) + torsoBias * quantile(hs, 0.5)));
        const yExpr = `max(min(${anchorY.toFixed(4)}*ih-(${cropHeightExpr})*${headroomPos.toFixed(4)}\\,ih-(${cropHeightExpr}))\\,0)`;

        ffmpeg(input)
            .setStartTime(start)
            .setDuration(duration)
            .videoCodec('libx264')
            .audioCodec('aac')
            .outputOptions(['-preset veryfast', '-crf 20', '-movflags +faststart', '-b:a 160k'])
            .videoFilters([
                'scale=iw*4:ih*4',
                `crop=${cropWidthExpr}:${cropHeightExpr}:${xExpr}:${yExpr}`,
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
