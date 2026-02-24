import ffmpeg from 'fluent-ffmpeg';
import path from 'path';
import fs from 'fs';
import { spawn } from 'child_process';

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

export async function render(
    videoPath: string,
    boundaries: { start: number; end: number },
    clips: { start: number; end: number; id: string; score?: number; confidence?: number }[]
): Promise<string[]> {
    console.log('Rendering clips...');

    const outputDir = path.join(path.dirname(videoPath), 'processed');
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    const results: string[] = [];

    // 1. Render Sermon Horizontal
    const sermonPath = path.join(outputDir, 'sermon_horizontal.mp4');
    await cutVideo(videoPath, boundaries.start, boundaries.end, sermonPath, {
        fadeInSec: 1,
        fadeOutSec: 1,
    });
    results.push(sermonPath);

    // 2. Render Vertical Shorts
    // First, analyze video to find face center to crop around
    let centerX = 0.5; // Default center
    try {
        centerX = await detectSpeakerCenter(videoPath);
        console.log(`Detected speaker center at relative X: ${centerX}`);
    } catch (e) {
        console.error('Failed to detect speaker, defaulting to center.', e);
    }

    const maxVerticalRenders = Number(process.env.HIGHLIGHTS_RENDER_COUNT ?? 5);
    const selectedClips = [...clips]
        .sort((a, b) => {
            const sa = Number.isFinite(Number(a.score)) ? Number(a.score) : Number(a.confidence ?? 0);
            const sb = Number.isFinite(Number(b.score)) ? Number(b.score) : Number(b.confidence ?? 0);
            return sb - sa;
        })
        .slice(0, Math.max(0, maxVerticalRenders));

    console.log(`Rendering vertical highlights: selected ${selectedClips.length}/${clips.length}`);
    for (const clip of selectedClips) {
        const clipPath = path.join(outputDir, `${clip.id}.mp4`);
        const dynamicCropEnabled = String(process.env.VERTICAL_DYNAMIC_CROP_ENABLED ?? 'true').toLowerCase() === 'true';
        let centerTrack: CropTrackPoint[] | null = null;
        if (dynamicCropEnabled) {
            try {
                centerTrack = await detectSpeakerTrack(videoPath, clip.start, clip.end);
                if (centerTrack && centerTrack.length > 0) {
                    console.log(`Detected dynamic center track points=${centerTrack.length} for ${clip.id}`);
                }
            } catch (e) {
                console.error(`Dynamic center track failed for ${clip.id}, using static center.`, e);
            }
        }
        await cutVideoVertical(videoPath, clip.start, clip.end, clipPath, centerX, centerTrack);
        results.push(clipPath);
    }

    return results;
}

async function detectSpeakerCenter(videoPath: string): Promise<number> {
    return new Promise((resolve, reject) => {
        const scriptCandidates = [
            // ts-node: apps/worker/src/pipeline -> src/pipeline/python/autocrop.py
            path.resolve(__dirname, 'python/autocrop.py'),
            // built JS: apps/worker/dist/pipeline -> src/pipeline/python/autocrop.py
            path.resolve(__dirname, '../../src/pipeline/python/autocrop.py'),
            // repo-root and app-root fallbacks
            path.resolve(process.cwd(), 'apps/worker/src/pipeline/python/autocrop.py'),
            path.resolve(process.cwd(), 'src/pipeline/python/autocrop.py')
        ];
        const pythonScript = scriptCandidates.find((p) => fs.existsSync(p));
        if (!pythonScript) {
            return reject(new Error(`autocrop.py not found. Tried: ${scriptCandidates.join(', ')}`));
        }

        const pyCandidates = [
            // prefer venv311 (insightface/pyannote stack), then fallback
            path.resolve(__dirname, '../../../venv311/bin/python3'),
            path.resolve(__dirname, '../../../venv/bin/python3'),
            path.resolve(process.cwd(), 'apps/worker/venv/bin/python3'),
            path.resolve(process.cwd(), 'apps/worker/venv311/bin/python3'),
            path.resolve(process.cwd(), 'venv311/bin/python3'),
            path.resolve(process.cwd(), 'venv/bin/python3')
        ];
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

async function detectSpeakerTrack(videoPath: string, start: number, end: number): Promise<CropTrackPoint[] | null> {
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

        const pyCandidates = [
            path.resolve(__dirname, '../../../venv311/bin/python3'),
            path.resolve(__dirname, '../../../venv/bin/python3'),
            path.resolve(process.cwd(), 'apps/worker/venv/bin/python3'),
            path.resolve(process.cwd(), 'apps/worker/venv311/bin/python3'),
            path.resolve(process.cwd(), 'venv311/bin/python3'),
            path.resolve(process.cwd(), 'venv/bin/python3')
        ];
        const venvPython = pyCandidates.find((p) => fs.existsSync(p));
        if (!venvPython) {
            return reject(new Error(`python3 not found in expected venv paths: ${pyCandidates.join(', ')}`));
        }

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
                const result = parseLastJson<{ track?: Array<{ t: number; center_x: number }>; camera_keyframes?: Array<{ t: number; center_x: number }> }>(dataString);
                const useKeyframes = String(process.env.VERTICAL_DYNAMIC_CROP_USE_KEYFRAMES ?? 'false').toLowerCase() === 'true';
                const raw = useKeyframes && Array.isArray(result?.camera_keyframes) && result.camera_keyframes.length >= 2
                    ? result.camera_keyframes
                    : (Array.isArray(result?.track) ? result.track : []);
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

function cutVideoVertical(
    input: string,
    start: number,
    end: number,
    output: string,
    centerX: number,
    centerTrack: CropTrackPoint[] | null = null
): Promise<void> {
    return new Promise((resolve, reject) => {
        console.log(`Cutting vertical clip: ${start}s to ${end}s with center ${centerX}`);
        // Use ffmpeg expression-based crop to avoid runtime ffprobe dependency.
        // Crop width = ih*9/16, then clamp X around detected centerX.
        const centerExpr = centerTrack && centerTrack.length > 1 ? buildCenterExpr(centerTrack) : `${centerX}`;
        const xExpr = `max(min((${centerExpr})*iw-(ih*9/16)/2\\,iw-(ih*9/16))\\,0)`;

        const duration = end - start;
        const vf: string[] = [
            `crop=ih*9/16:ih:${xExpr}:0`,
            `scale=1080:1920`
        ];

        const cmd = ffmpeg(input)
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
            .videoFilters(vf)
            .output(output)
            .on('stderr', (line) => console.log(`ffmpeg: ${line}`))
            .on('end', () => resolve())
            .on('error', (err) => reject(err));

        cmd.run();
    });
}
