export const OCR_PROGRESS_PREFIX = '[ocr-events-progress]';

export interface OcrProgressEvent {
    type: 'ocr_progress';
    percent: number;
    sampled: number;
    total_samples: number;
    video_time_sec: number | null;
    video_duration_sec: number | null;
    observations: number;
    scene_cuts: number;
    ocr_calls: number;
    roi_candidates: number;
    elapsed_sec: number | null;
    eta_sec: number | null;
    samples_per_sec: number;
    frames_per_sec: number;
    video_speed_x: number;
    targeted: boolean;
}

export interface OcrProgressSummary {
    events: number;
    firstPercent: number | null;
    lastPercent: number | null;
    lastEtaSec: number | null;
    lastVideoSpeedX: number | null;
}

function isFiniteNumber(value: unknown): value is number {
    return typeof value === 'number' && Number.isFinite(value);
}

function isNumberOrNull(value: unknown): value is number | null {
    return value === null || isFiniteNumber(value);
}

function toEvent(parsed: unknown): OcrProgressEvent | null {
    if (!parsed || typeof parsed !== 'object') return null;
    const obj = parsed as Record<string, unknown>;
    if (obj.type !== 'ocr_progress') return null;
    if (
        !isFiniteNumber(obj.percent) ||
        !isFiniteNumber(obj.sampled) ||
        !isFiniteNumber(obj.total_samples) ||
        !isNumberOrNull(obj.video_time_sec) ||
        !isNumberOrNull(obj.video_duration_sec) ||
        !isFiniteNumber(obj.observations) ||
        !isFiniteNumber(obj.scene_cuts) ||
        !isFiniteNumber(obj.ocr_calls) ||
        !isFiniteNumber(obj.roi_candidates) ||
        !isNumberOrNull(obj.elapsed_sec) ||
        !isNumberOrNull(obj.eta_sec) ||
        !isFiniteNumber(obj.samples_per_sec) ||
        !isFiniteNumber(obj.frames_per_sec) ||
        !isFiniteNumber(obj.video_speed_x) ||
        typeof obj.targeted !== 'boolean'
    ) {
        return null;
    }
    return {
        type: 'ocr_progress',
        percent: obj.percent,
        sampled: obj.sampled,
        total_samples: obj.total_samples,
        video_time_sec: obj.video_time_sec,
        video_duration_sec: obj.video_duration_sec,
        observations: obj.observations,
        scene_cuts: obj.scene_cuts,
        ocr_calls: obj.ocr_calls,
        roi_candidates: obj.roi_candidates,
        elapsed_sec: obj.elapsed_sec,
        eta_sec: obj.eta_sec,
        samples_per_sec: obj.samples_per_sec,
        frames_per_sec: obj.frames_per_sec,
        video_speed_x: obj.video_speed_x,
        targeted: obj.targeted,
    };
}

export function parseOcrProgressLine(line: string): OcrProgressEvent | null {
    const idx = line.indexOf(OCR_PROGRESS_PREFIX);
    if (idx < 0) return null;
    const jsonPart = line.slice(idx + OCR_PROGRESS_PREFIX.length).trim();
    if (!jsonPart.startsWith('{')) return null;
    try {
        return toEvent(JSON.parse(jsonPart));
    } catch {
        return null;
    }
}

export function extractOcrProgressEvents(stderrText: string): OcrProgressEvent[] {
    const out: OcrProgressEvent[] = [];
    for (const line of stderrText.split(/\r?\n/)) {
        const event = parseOcrProgressLine(line);
        if (event) out.push(event);
    }
    return out;
}

export function summarizeOcrProgress(events: OcrProgressEvent[]): OcrProgressSummary {
    const first = events[0] ?? null;
    const last = events.length > 0 ? events[events.length - 1] : null;
    return {
        events: events.length,
        firstPercent: first?.percent ?? null,
        lastPercent: last?.percent ?? null,
        lastEtaSec: last?.eta_sec ?? null,
        lastVideoSpeedX: last?.video_speed_x ?? null,
    };
}

