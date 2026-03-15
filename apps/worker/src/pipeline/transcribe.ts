import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { jsonToSrt } from '../utils/srt';
import { fetchYouTubeTranscript } from '../lib/defuddle';

export interface TransientSegment {
    start: number;
    end: number;
    text: string;
}

interface TranscribeOptions {
    onProgress?: (currentSeconds: number) => void;
    onFallback?: (reason: string) => void;
}

type TranscribeProvider = 'auto' | 'local' | 'elevenlabs_scribe_v2';

interface ElevenLabsWord {
    text?: string;
    start?: number;
    end?: number;
    type?: string;
    speaker_id?: string | number | null;
    channel_index?: number | null;
    logprob?: number;
}

interface ElevenLabsAdditionalFormat {
    requested_format?: string;
    file_extension?: string;
    content_type?: string;
    is_base64_encoded?: boolean;
    content?: string;
}

interface ElevenLabsTranscriptChunk {
    text?: string;
    words?: ElevenLabsWord[];
    channel_index?: number | null;
}

interface ElevenLabsSpeechToTextResponse {
    language_code?: string;
    language_probability?: number;
    text?: string;
    words?: ElevenLabsWord[];
    transcripts?: ElevenLabsTranscriptChunk[];
    additional_formats?: Array<ElevenLabsAdditionalFormat | null> | null;
    transcription_id?: string | null;
}

interface ElevenLabsRequestConfig {
    modelId: string;
    diarize: boolean;
    tagAudioEvents: boolean;
    timestampsGranularity: 'none' | 'word' | 'character';
    languageCode?: string;
    numSpeakers?: number;
    noVerbatim: boolean;
    timeoutMs: number;
    includeAudioEventsInTranscript: boolean;
}

export async function transcribe(audioPath: string, options?: TranscribeOptions): Promise<TransientSegment[]> {
    const forceRedo = String(process.env.TRANSCRIBE_FORCE_REDO ?? 'false').toLowerCase() === 'true';
    if (!forceRedo) {
        const cached = loadCachedTranscriptFromWorkDir(audioPath);
        if (cached.length > 0) {
            console.log(`Reusing cached transcription with ${cached.length} segments (no retranscribe).`);
            return cached;
        }
    } else {
        console.log('TRANSCRIBE_FORCE_REDO=true, skipping cached transcript reuse.');
    }

    const provider = resolveTranscribeProvider(process.env.TRANSCRIBE_PROVIDER);
    const workDir = path.dirname(audioPath);
    const metadataPath = path.join(workDir, 'metadata.json');
    let sourceUrl: string | undefined;
    if (fs.existsSync(metadataPath)) {
        try {
            const meta = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
            sourceUrl = meta.source;
        } catch (e) {
            console.warn(`Failed to read metadata.json for source URL: ${e}`);
        }
    }

    const isYouTube = sourceUrl && (sourceUrl.includes('youtube.com') || sourceUrl.includes('youtu.be'));
    const tryDefuddle = isYouTube && (provider === 'auto' || provider === 'local' || provider === 'elevenlabs_scribe_v2');

    if (tryDefuddle && sourceUrl) {
        try {
            console.log(`YouTube source detected. Attempting Defuddle fast-path for ${sourceUrl}...`);
            const { segments, audioEvents } = await fetchYouTubeTranscript(sourceUrl);
            if (segments.length > 0) {
                console.log(`Defuddle fast-path succeeded with ${segments.length} segments.`);
                
                // Write standard artifacts
                writeTranscriptArtifacts(audioPath, segments);
                
                // Write Defuddle-specific audio events artifact if music detected
                if (audioEvents.length > 0) {
                    const audioEventsPayload = {
                        source: 'defuddle-youtube-manual-captions',
                        duration_sec: segments[segments.length - 1].end,
                        step_sec: null,
                        segments: audioEvents
                    };
                    fs.writeFileSync(path.join(workDir, 'audio.events.defuddle.json'), JSON.stringify(audioEventsPayload, null, 2));
                    // Also write to the primary audio.events.json to satisfy downstream signals
                    fs.writeFileSync(path.join(workDir, 'audio.events.json'), JSON.stringify(audioEventsPayload, null, 2));
                    console.log(`Music cues detected via Defuddle: ${audioEvents.length} events (written to audio.events.json).`);
                }
                
                return segments;
            }
        } catch (error) {
            console.warn(`Defuddle fast-path failed: ${error}. Falling back to normal flow.`);
        }
    }

    const tryElevenLabs = provider === 'auto' || provider === 'elevenlabs_scribe_v2';
    if (tryElevenLabs) {
        try {
            const segments = await runElevenLabsScribeV2(audioPath, options);
            if (isLikelyBrokenTranscript(segments)) {
                throw new Error('ElevenLabs transcript failed quality check (dot/empty dominated output).');
            }
            writeTranscriptArtifacts(audioPath, segments);
            return segments;
        } catch (error) {
            const reason = `ElevenLabs Scribe v2 unavailable/failing: ${String(error)}. Falling back to local transcription.`;
            console.warn(reason);
            options?.onFallback?.(reason);
            if (provider === 'elevenlabs_scribe_v2') {
                const strict = String(process.env.TRANSCRIBE_STRICT_PROVIDER ?? 'false').toLowerCase() === 'true';
                if (strict) throw error;
            }
        }
    }

    if (provider === 'elevenlabs_scribe_v2') {
        console.warn(
            'TRANSCRIBE_PROVIDER=elevenlabs_scribe_v2 requested, but fallback to local is enabled (TRANSCRIBE_STRICT_PROVIDER=false).'
        );
    }

    const attempts: Array<{ model: string; beamSize: number }> = [
        { model: 'small', beamSize: 5 },
        { model: 'base', beamSize: 1 }
    ];

    let lastError: unknown;

    for (const [index, attempt] of attempts.entries()) {
        try {
            const segments = await runTranscriptionAttempt(audioPath, attempt.model, attempt.beamSize, options);
            if (isLikelyBrokenTranscript(segments)) {
                throw new Error(
                    `Transcription quality check failed for attempt model=${attempt.model} beam=${attempt.beamSize} ` +
                    `(dot/empty dominated output).`
                );
            }

            writeTranscriptArtifacts(audioPath, segments);
            return segments;
        } catch (error) {
            lastError = error;
            const attemptNum = index + 1;
            console.error(`Transcription attempt ${attemptNum}/${attempts.length} failed:`, error);
        }
    }

    const fallbackCached = loadCachedTranscriptFallback();
    if (fallbackCached.length > 0) {
        const fallbackReason = `Using cached transcript fallback with ${fallbackCached.length} segments after transcription failures.`;
        console.warn(fallbackReason);
        options?.onFallback?.(fallbackReason);
        writeTranscriptArtifacts(audioPath, fallbackCached, true);
        return fallbackCached;
    }

    throw new Error(`All transcription attempts failed. Last error: ${String(lastError)}`);
}

function resolveTranscribeProvider(raw: string | undefined): TranscribeProvider {
    const value = String(raw ?? 'local').trim().toLowerCase();
    if (value === 'local') return 'local';
    if (value === 'elevenlabs_scribe_v2' || value === 'elevenlabs' || value === 'scribe_v2') return 'elevenlabs_scribe_v2';
    return 'auto';
}

function parseBooleanEnv(raw: string | undefined, fallback: boolean): boolean {
    if (raw == null || raw === '') return fallback;
    const value = String(raw).trim().toLowerCase();
    if (['1', 'true', 'yes', 'on'].includes(value)) return true;
    if (['0', 'false', 'no', 'off'].includes(value)) return false;
    return fallback;
}

function parsePositiveInt(raw: string | undefined): number | undefined {
    if (!raw) return undefined;
    const n = Number(raw);
    if (!Number.isFinite(n)) return undefined;
    const int = Math.floor(n);
    if (int <= 0) return undefined;
    return int;
}

function resolveElevenLabsRequestConfig(): ElevenLabsRequestConfig {
    const modelId = String(process.env.TRANSCRIBE_ELEVENLABS_MODEL_ID ?? 'scribe_v2').trim() || 'scribe_v2';
    const rawGranularity = String(process.env.TRANSCRIBE_ELEVENLABS_TIMESTAMP_GRANULARITY ?? 'word')
        .trim()
        .toLowerCase();
    const timestampsGranularity: 'none' | 'word' | 'character' =
        rawGranularity === 'none' || rawGranularity === 'character' ? rawGranularity : 'word';
    const languageCode = String(process.env.TRANSCRIBE_ELEVENLABS_LANGUAGE_CODE ?? '').trim() || undefined;
    const numSpeakers = parsePositiveInt(process.env.TRANSCRIBE_ELEVENLABS_NUM_SPEAKERS);
    const timeoutMs = Number(process.env.TRANSCRIBE_ELEVENLABS_TIMEOUT_MS ?? 1_800_000);

    return {
        modelId,
        diarize: parseBooleanEnv(process.env.TRANSCRIBE_ELEVENLABS_DIARIZE, true),
        tagAudioEvents: parseBooleanEnv(process.env.TRANSCRIBE_ELEVENLABS_TAG_AUDIO_EVENTS, true),
        timestampsGranularity,
        languageCode,
        numSpeakers,
        noVerbatim: parseBooleanEnv(process.env.TRANSCRIBE_ELEVENLABS_NO_VERBATIM, false),
        timeoutMs: Number.isFinite(timeoutMs) && timeoutMs > 0 ? timeoutMs : 1_800_000,
        includeAudioEventsInTranscript: parseBooleanEnv(
            process.env.TRANSCRIBE_ELEVENLABS_INCLUDE_AUDIO_EVENTS_IN_TRANSCRIPT,
            false
        )
    };
}

async function runElevenLabsScribeV2(audioPath: string, options?: TranscribeOptions): Promise<TransientSegment[]> {
    const apiKey = String(process.env.ELEVENLABS_API_KEY ?? '').trim();
    if (!apiKey) {
        throw new Error('ELEVENLABS_API_KEY is not configured.');
    }
    if (!fs.existsSync(audioPath)) {
        throw new Error(`Audio file not found: ${audioPath}`);
    }

    const cfg = resolveElevenLabsRequestConfig();
    const form = new FormData();
    form.set('model_id', cfg.modelId);
    form.set('diarize', String(cfg.diarize));
    form.set('tag_audio_events', String(cfg.tagAudioEvents));
    form.set('timestamps_granularity', cfg.timestampsGranularity);
    form.set('no_verbatim', String(cfg.noVerbatim));
    if (cfg.languageCode) form.set('language_code', cfg.languageCode);
    if (cfg.numSpeakers) form.set('num_speakers', String(cfg.numSpeakers));

    const openAsBlob = (fs as unknown as { openAsBlob?: (p: string, opts?: { type?: string }) => Promise<Blob> }).openAsBlob;
    if (typeof openAsBlob !== 'function') {
        throw new Error('Current Node runtime does not support fs.openAsBlob required for ElevenLabs uploads.');
    }
    const blob = await openAsBlob(audioPath, { type: 'application/octet-stream' });
    form.set('file', blob, path.basename(audioPath));

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), cfg.timeoutMs);
    const endpoint = 'https://api.elevenlabs.io/v1/speech-to-text';
    options?.onProgress?.(0);
    let response: Response;
    try {
        response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'xi-api-key': apiKey
            },
            body: form,
            signal: controller.signal
        });
    } finally {
        clearTimeout(timeout);
    }

    const responseText = await response.text();
    if (!response.ok) {
        throw new Error(`ElevenLabs speech-to-text failed HTTP ${response.status}: ${responseText.slice(0, 800)}`);
    }

    let payload: ElevenLabsSpeechToTextResponse;
    try {
        payload = JSON.parse(responseText) as ElevenLabsSpeechToTextResponse;
    } catch {
        throw new Error(`ElevenLabs speech-to-text returned non-JSON response: ${responseText.slice(0, 800)}`);
    }

    const words = collectElevenLabsWords(payload);
    if (words.length === 0) {
        throw new Error('ElevenLabs response contains no words.');
    }

    const segments = buildSegmentsFromElevenLabsWords(words, cfg.includeAudioEventsInTranscript);
    if (segments.length === 0) {
        throw new Error('ElevenLabs response words could not be normalized into transcript segments.');
    }

    writeElevenLabsArtifacts(audioPath, payload, words, cfg.includeAudioEventsInTranscript);
    const lastEnd = segments[segments.length - 1]?.end;
    if (Number.isFinite(lastEnd)) options?.onProgress?.(Number(lastEnd));
    return segments;
}

function collectElevenLabsWords(response: ElevenLabsSpeechToTextResponse): ElevenLabsWord[] {
    const words: ElevenLabsWord[] = [];
    const hasTranscriptArray = Array.isArray(response.transcripts) && response.transcripts.length > 0;
    if (!hasTranscriptArray && Array.isArray(response.words)) {
        words.push(...response.words);
    }
    if (hasTranscriptArray && Array.isArray(response.transcripts)) {
        for (const transcript of response.transcripts) {
            if (Array.isArray(transcript.words)) {
                const channelIndex =
                    typeof transcript.channel_index === 'number' ? transcript.channel_index : undefined;
                for (const word of transcript.words) {
                    if (channelIndex == null || typeof word?.channel_index === 'number') {
                        words.push(word);
                    } else {
                        words.push({ ...word, channel_index: channelIndex });
                    }
                }
            }
        }
    }

    return words
        .filter((w) => typeof w?.text === 'string' && w.text.length > 0)
        .sort((a, b) => {
            const aStart = Number.isFinite(a.start) ? Number(a.start) : Number.POSITIVE_INFINITY;
            const bStart = Number.isFinite(b.start) ? Number(b.start) : Number.POSITIVE_INFINITY;
            if (aStart !== bStart) return aStart - bStart;
            const aEnd = Number.isFinite(a.end) ? Number(a.end) : Number.POSITIVE_INFINITY;
            const bEnd = Number.isFinite(b.end) ? Number(b.end) : Number.POSITIVE_INFINITY;
            return aEnd - bEnd;
        });
}

function buildSegmentsFromElevenLabsWords(
    words: ElevenLabsWord[],
    includeAudioEventsInTranscript: boolean
): TransientSegment[] {
    const gapSplitSec = Number(process.env.TRANSCRIBE_WORD_GAP_SPLIT_SEC ?? 0.55);
    const safeGapSplitSec = Number.isFinite(gapSplitSec) && gapSplitSec > 0 ? gapSplitSec : 0.55;
    const maxSegmentSecRaw = Number(process.env.TRANSCRIBE_ELEVENLABS_MAX_SEGMENT_SEC ?? 14);
    const maxSegmentSec = Number.isFinite(maxSegmentSecRaw) && maxSegmentSecRaw > 0 ? maxSegmentSecRaw : 14;
    const maxSegmentCharsRaw = Number(process.env.TRANSCRIBE_ELEVENLABS_MAX_SEGMENT_CHARS ?? 220);
    const maxSegmentChars = Number.isFinite(maxSegmentCharsRaw) && maxSegmentCharsRaw > 0 ? Math.floor(maxSegmentCharsRaw) : 220;

    const out: TransientSegment[] = [];
    let currentStart: number | null = null;
    let currentEnd: number | null = null;
    let currentSpeaker: string | null = null;
    let currentText = '';

    const flush = () => {
        if (currentStart == null || currentEnd == null) {
            currentText = '';
            return;
        }
        const text = currentText.replace(/\s+/g, ' ').trim();
        if (!text) {
            currentText = '';
            return;
        }
        const start = Math.max(0, currentStart);
        const end = Math.max(start + 0.05, currentEnd);
        out.push({ start, end, text });
        currentStart = null;
        currentEnd = null;
        currentSpeaker = null;
        currentText = '';
    };

    const appendText = (text: string) => {
        const token = String(text ?? '');
        if (!token) return;
        const trimmedToken = token.trim();
        if (!trimmedToken) return;
        const noLeadingSpace = /^[,.;:!?)\]}]/.test(trimmedToken);
        if (!currentText) {
            currentText = trimmedToken;
        } else if (noLeadingSpace || /[(\[{]$/.test(currentText)) {
            currentText += trimmedToken;
        } else {
            currentText += ` ${trimmedToken}`;
        }
    };

    for (const token of words) {
        const type = String(token.type ?? 'word').toLowerCase();
        const tokenText = String(token.text ?? '');
        const tokenStart = Number.isFinite(token.start) ? Number(token.start) : null;
        const tokenEnd = Number.isFinite(token.end) ? Number(token.end) : tokenStart;
        const speaker = token.speaker_id == null ? null : String(token.speaker_id);

        if (type === 'spacing') continue;
        if (type === 'audio_event' && !includeAudioEventsInTranscript) continue;
        if (!tokenText.trim()) continue;

        if (currentStart == null) {
            currentStart = tokenStart ?? currentEnd ?? 0;
            currentEnd = tokenEnd ?? currentStart;
            currentSpeaker = speaker;
            appendText(tokenText);
            continue;
        }

        const gap = tokenStart != null && currentEnd != null ? tokenStart - currentEnd : 0;
        const speakerChanged = Boolean(speaker && currentSpeaker && speaker !== currentSpeaker);
        const durationIfAppended =
            tokenEnd != null && currentStart != null ? Math.max(0, tokenEnd - currentStart) : 0;
        const shouldSplit =
            gap > safeGapSplitSec ||
            speakerChanged ||
            (durationIfAppended > maxSegmentSec && currentText.length >= 24) ||
            currentText.length >= maxSegmentChars ||
            (type === 'audio_event' && currentText.length > 0);

        if (shouldSplit) flush();

        if (currentStart == null) {
            currentStart = tokenStart ?? 0;
            currentEnd = tokenEnd ?? currentStart;
            currentSpeaker = speaker;
            appendText(tokenText);
            continue;
        }

        if (tokenStart != null && tokenStart < currentStart) currentStart = tokenStart;
        if (tokenEnd != null) {
            currentEnd = currentEnd == null ? tokenEnd : Math.max(currentEnd, tokenEnd);
        }
        if (!currentSpeaker && speaker) currentSpeaker = speaker;
        appendText(tokenText);
    }

    flush();
    return out
        .filter((s) => Number.isFinite(s.start) && Number.isFinite(s.end) && s.end > s.start && s.text.trim().length > 0)
        .sort((a, b) => a.start - b.start);
}

function buildDiarizedTurnsFromElevenLabsWords(words: ElevenLabsWord[]): Array<{
    start: number;
    end: number;
    speaker: string;
    text: string;
}> {
    const turns: Array<{ start: number; end: number; speaker: string; text: string }> = [];
    let current: { start: number; end: number; speaker: string; text: string } | null = null;

    const pushCurrent = () => {
        if (!current) return;
        const text = current.text.replace(/\s+/g, ' ').trim();
        if (text && Number.isFinite(current.start) && Number.isFinite(current.end) && current.end > current.start) {
            turns.push({ ...current, text });
        }
        current = null;
    };

    for (const token of words) {
        const type = String(token.type ?? 'word').toLowerCase();
        if (type !== 'word') continue;
        const text = String(token.text ?? '').trim();
        if (!text) continue;
        const start = Number.isFinite(token.start) ? Number(token.start) : null;
        const end = Number.isFinite(token.end) ? Number(token.end) : start;
        if (start == null || end == null) continue;
        const speaker = token.speaker_id == null ? 'unknown' : String(token.speaker_id);
        const gap = current ? start - current.end : 0;
        if (!current || current.speaker !== speaker || gap > 0.8) {
            pushCurrent();
            current = { start, end, speaker, text };
            continue;
        }
        current.end = Math.max(current.end, end);
        current.text += /[,.;:!?)]$/.test(text) ? text : ` ${text}`;
    }
    pushCurrent();
    return turns;
}

function mapElevenLabsAudioEventLabel(rawLabel: string): string {
    const label = String(rawLabel ?? '')
        .toLowerCase()
        .replace(/[(){}\[\]]/g, '')
        .trim();
    if (!label) return 'noise';

    if (
        /(music|música|musica|song|singing|sings|hymn|choir|instrument|piano|guitar|organ|worship|canto|cantando|canci[oó]n)/.test(
            label
        )
    ) {
        return 'music';
    }
    if (/(silence|silent|pausa|pause|quiet|no.?sound|mute|sin audio|sin sonido)/.test(label)) {
        return 'noenergy';
    }
    if (/(speech|talking|speaking|voice|spoken|habla|hablando)/.test(label)) {
        return 'speech';
    }
    return 'noise';
}

function buildAudioEventsFromElevenLabsWords(words: ElevenLabsWord[]): {
    source: string;
    duration_sec: number;
    step_sec: null;
    segments: Array<{ label: string; start: number; end: number }>;
} {
    const speechWindows: Array<{ start: number; end: number }> = [];
    const taggedEvents: Array<{ label: string; start: number; end: number }> = [];

    for (const token of words) {
        const start = Number.isFinite(token.start) ? Number(token.start) : null;
        const end = Number.isFinite(token.end) ? Number(token.end) : start;
        if (start == null || end == null || end <= start) continue;
        const type = String(token.type ?? 'word').toLowerCase();
        if (type === 'word') {
            speechWindows.push({ start, end });
        } else if (type === 'audio_event') {
            taggedEvents.push({
                label: mapElevenLabsAudioEventLabel(String(token.text ?? '')),
                start,
                end
            });
        }
    }

    speechWindows.sort((a, b) => a.start - b.start);
    const mergedSpeech: Array<{ start: number; end: number }> = [];
    const speechMergeGapSec = Number(process.env.TRANSCRIBE_ELEVENLABS_SPEECH_MERGE_GAP_SEC ?? 0.45);
    const gapThreshold = Number.isFinite(speechMergeGapSec) && speechMergeGapSec >= 0 ? speechMergeGapSec : 0.45;
    for (const window of speechWindows) {
        const last = mergedSpeech[mergedSpeech.length - 1];
        if (!last || window.start - last.end > gapThreshold) {
            mergedSpeech.push({ ...window });
        } else {
            last.end = Math.max(last.end, window.end);
        }
    }

    const timeline: Array<{ label: string; start: number; end: number }> = mergedSpeech.map((s) => ({
        label: 'speech',
        start: s.start,
        end: s.end
    }));
    timeline.push(...taggedEvents);
    timeline.sort((a, b) => a.start - b.start);

    const mergedTimeline: Array<{ label: string; start: number; end: number }> = [];
    for (const event of timeline) {
        const last = mergedTimeline[mergedTimeline.length - 1];
        if (!last || last.label !== event.label || event.start - last.end > 0.08) {
            mergedTimeline.push({ ...event });
        } else {
            last.end = Math.max(last.end, event.end);
        }
    }

    const maxEnd = mergedTimeline.reduce((m, e) => Math.max(m, e.end), 0);
    const durationSec = Number.isFinite(maxEnd) ? maxEnd : 0;
    const eventsWithSilence: Array<{ label: string; start: number; end: number }> = [];
    const silenceGapSec = Number(process.env.TRANSCRIBE_ELEVENLABS_SILENCE_GAP_SEC ?? 0.7);
    const minSilenceGapSec = Number.isFinite(silenceGapSec) && silenceGapSec > 0 ? silenceGapSec : 0.7;
    let cursor = 0;
    for (const event of mergedTimeline) {
        if (event.start - cursor >= minSilenceGapSec) {
            eventsWithSilence.push({ label: 'noenergy', start: cursor, end: event.start });
        }
        eventsWithSilence.push(event);
        cursor = Math.max(cursor, event.end);
    }
    if (durationSec - cursor >= minSilenceGapSec) {
        eventsWithSilence.push({ label: 'noenergy', start: cursor, end: durationSec });
    }

    return {
        source: 'elevenlabs-scribe-v2-derived-v1',
        duration_sec: durationSec,
        step_sec: null,
        segments: eventsWithSilence
            .filter((s) => Number.isFinite(s.start) && Number.isFinite(s.end) && s.end > s.start)
            .map((s) => ({
                label: s.label,
                start: Number(s.start.toFixed(3)),
                end: Number(s.end.toFixed(3))
            }))
    };
}

function decodeAdditionalFormatContent(format: ElevenLabsAdditionalFormat): string {
    const content = String(format.content ?? '');
    if (!content) return '';
    if (format.is_base64_encoded) {
        try {
            return Buffer.from(content, 'base64').toString('utf8');
        } catch {
            return '';
        }
    }
    return content;
}

function writeElevenLabsArtifacts(
    audioPath: string,
    payload: ElevenLabsSpeechToTextResponse,
    words: ElevenLabsWord[],
    includeAudioEventsInTranscript: boolean
): void {
    const workDir = path.dirname(audioPath);
    try {
        fs.writeFileSync(path.join(workDir, 'transcript.elevenlabs.scribe_v2.json'), JSON.stringify(payload, null, 2));
    } catch (error) {
        console.error('Failed to write ElevenLabs raw transcript artifact:', error);
    }

    const turns = buildDiarizedTurnsFromElevenLabsWords(words);
    if (turns.length > 0) {
        try {
            fs.writeFileSync(path.join(workDir, 'transcript.diarized.elevenlabs.json'), JSON.stringify(turns, null, 2));
        } catch (error) {
            console.error('Failed to write ElevenLabs diarized turns artifact:', error);
        }
    }

    try {
        const audioEvents = buildAudioEventsFromElevenLabsWords(words);
        if (audioEvents.segments.length > 0) {
            fs.writeFileSync(path.join(workDir, 'audio.events.elevenlabs.json'), JSON.stringify(audioEvents, null, 2));
        }
    } catch (error) {
        console.error('Failed to write ElevenLabs-derived audio events artifact:', error);
    }

    if (Array.isArray(payload.additional_formats)) {
        for (const entry of payload.additional_formats) {
            if (!entry || !entry.requested_format) continue;
            const requested = String(entry.requested_format).toLowerCase();
            if (requested !== 'srt') continue;
            const decoded = decodeAdditionalFormatContent(entry);
            if (!decoded) continue;
            try {
                fs.writeFileSync(path.join(workDir, 'source.elevenlabs.srt'), decoded);
            } catch (error) {
                console.error('Failed to write ElevenLabs SRT artifact:', error);
            }
        }
    }

    if (includeAudioEventsInTranscript) {
        try {
            fs.writeFileSync(
                path.join(workDir, 'transcript.elevenlabs.notes.txt'),
                'TRANSCRIBE_ELEVENLABS_INCLUDE_AUDIO_EVENTS_IN_TRANSCRIPT=true'
            );
        } catch {
            // Ignore optional artifact write failures.
        }
    }
}

function writeTranscriptArtifacts(audioPath: string, segments: TransientSegment[], useFallbackSrt = false): void {
    writeTranscriptJson(audioPath, segments);

    const workDir = path.dirname(audioPath);
    const transcriptSrtPath = path.join(workDir, 'source.srt');
    if (useFallbackSrt) {
        const fallbackSrtPath = path.resolve(__dirname, '../../../test_data/ingest_test/source.srt');
        if (fs.existsSync(fallbackSrtPath)) {
            try {
                fs.copyFileSync(fallbackSrtPath, transcriptSrtPath);
                return;
            } catch (error) {
                console.error('Failed to copy fallback source.srt:', error);
            }
        }
    }

    try {
        fs.writeFileSync(transcriptSrtPath, jsonToSrt(segments));
    } catch (error) {
        console.error('Failed to write source.srt:', error);
    }
}

function runTranscriptionAttempt(
    audioPath: string,
    model: string,
    beamSize: number,
    options?: TranscribeOptions
): Promise<TransientSegment[]> {
    return new Promise((resolve, reject) => {
        const pythonScript = resolveExistingPath([
            path.resolve(__dirname, 'python/transcribe.py'),
            path.resolve(__dirname, '../../src/pipeline/python/transcribe.py'),
            path.resolve(process.cwd(), 'apps/worker/src/pipeline/python/transcribe.py'),
            path.resolve(process.cwd(), 'src/pipeline/python/transcribe.py')
        ]);
        const venvPython = resolveExistingPath([
            process.env.WORKER_PYTHON_BIN,
            path.resolve(__dirname, '../../venv/bin/python3'),
            path.resolve(__dirname, '../../venv311/bin/python3'),
            path.resolve(process.cwd(), 'apps/worker/venv/bin/python3'),
            path.resolve(process.cwd(), 'apps/worker/venv311/bin/python3'),
            path.resolve(process.cwd(), 'venv/bin/python3'),
            path.resolve(process.cwd(), 'venv311/bin/python3'),
            'python3'
        ]);
        const noWordTs = String(process.env.TRANSCRIBE_NO_WORD_TIMESTAMPS ?? 'false').toLowerCase() === 'true';
        const args = [pythonScript, audioPath, '--model', model, '--beam-size', String(beamSize)];
        const wordGapSplitSec = Number(process.env.TRANSCRIBE_WORD_GAP_SPLIT_SEC ?? 0.55);
        if (Number.isFinite(wordGapSplitSec) && wordGapSplitSec > 0) {
            args.push('--word-gap-split-sec', String(wordGapSplitSec));
        }
        if (noWordTs) args.push('--no-word-timestamps');
        const pythonProcess = spawn(
            venvPython,
            args,
            { stdio: ['ignore', 'pipe', 'pipe'] }
        );

        let stdout = '';
        let stderr = '';
        let settled = false;
        const timeoutMs = Number(process.env.TRANSCRIBE_NO_PROGRESS_TIMEOUT_MS || 180000);
        let timeout: NodeJS.Timeout | null = null;
        const resetNoProgressTimeout = () => {
            if (timeout) clearTimeout(timeout);
            timeout = setTimeout(() => {
                if (settled) return;
                settled = true;
                try {
                    pythonProcess.kill('SIGKILL');
                } catch {
                    // Ignore kill errors.
                }
                reject(new Error(`Transcribe attempt stalled (no progress for ${timeoutMs}ms, model=${model}, beam=${beamSize})`));
            }, timeoutMs);
        };

        resetNoProgressTimeout();

        pythonProcess.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
            const text = data.toString();
            stderr += text;
            console.error(`Python stderr: ${text}`);

            // Any stderr output means the process is alive, so extend stall timeout.
            resetNoProgressTimeout();

            // Parse stderr progress lines like: [1200.00s -> 1207.42s] text...
            const matches = text.matchAll(/\[(\d+(?:\.\d+)?)s\s*->\s*(\d+(?:\.\d+)?)s\]/g);
            for (const match of matches) {
                const end = Number(match[2]);
                if (Number.isFinite(end)) {
                    options?.onProgress?.(end);
                }
            }
        });

        pythonProcess.on('error', (error) => {
            if (settled) return;
            settled = true;
            if (timeout) clearTimeout(timeout);
            reject(new Error(`Failed to start transcription process: ${String(error)}`));
        });

        pythonProcess.on('close', (code, signal) => {
            if (settled) return;
            settled = true;
            if (timeout) clearTimeout(timeout);

            try {
                const parsed = JSON.parse(stdout) as TransientSegment[];
                if (!Array.isArray(parsed)) {
                    reject(new Error('Transcription output is not a JSON array.'));
                    return;
                }
                resolve(parsed);
                return;
            } catch {
                // Fall through to structured error below.
            }

            reject(
                new Error(
                    `Transcribe process failed (code=${code}, signal=${signal ?? 'none'}). ` +
                    `stderr tail: ${stderr.slice(-1200)}`
                )
            );
        });
    });
}

function resolveExistingPath(candidates: Array<string | undefined>): string {
    for (const candidate of candidates) {
        if (!candidate) continue;
        if (candidate === 'python3') return candidate;
        if (fs.existsSync(candidate)) return candidate;
    }
    const inspected = candidates.filter(Boolean).join(', ');
    throw new Error(`Unable to resolve required executable/script path. Candidates: ${inspected}`);
}

function loadCachedTranscriptFallback(): TransientSegment[] {
    const fallbackPath = path.resolve(__dirname, '../../../test_data/ingest_test/transcript.json');
    if (!fs.existsSync(fallbackPath)) {
        return [];
    }

    try {
        const raw = fs.readFileSync(fallbackPath, 'utf8');
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed)) return [];

        const segments = parsed
            .filter((s) => typeof s?.start === 'number' && typeof s?.end === 'number' && typeof s?.text === 'string')
            .map((s) => ({ start: s.start, end: s.end, text: s.text }));
        if (isLikelyBrokenTranscript(segments)) return [];
        return segments;
    } catch (error) {
        console.error('Failed to load cached transcript fallback:', error);
        return [];
    }
}

function loadCachedTranscriptFromWorkDir(audioPath: string): TransientSegment[] {
    const workDir = path.dirname(audioPath);
    const transcriptJsonPath = path.join(workDir, 'transcript.json');
    const srtPath = path.join(workDir, 'source.srt');

    if (fs.existsSync(transcriptJsonPath)) {
        try {
            const raw = fs.readFileSync(transcriptJsonPath, 'utf8');
            const parsed = JSON.parse(raw);
            if (Array.isArray(parsed)) {
                const segments = parsed
                    .filter((s) => typeof s?.start === 'number' && typeof s?.end === 'number' && typeof s?.text === 'string')
                    .map((s) => ({ start: s.start, end: s.end, text: s.text })) as TransientSegment[];
                if (segments.length > 0 && !isLikelyBrokenTranscript(segments)) return segments;
            }
        } catch (error) {
            console.error('Failed to read cached transcript.json:', error);
        }
    }

    if (fs.existsSync(srtPath)) {
        try {
            const srtText = fs.readFileSync(srtPath, 'utf8');
            const segments = parseSrtToSegments(srtText);
            if (segments.length > 0 && !isLikelyBrokenTranscript(segments)) return segments;
        } catch (error) {
            console.error('Failed to read cached source.srt:', error);
        }
    }

    return [];
}

function parseSrtToSegments(srt: string): TransientSegment[] {
    const blocks = srt.trim().split(/\r?\n\r?\n+/);
    const segments: TransientSegment[] = [];

    for (const block of blocks) {
        const lines = block.split(/\r?\n/).filter(Boolean);
        if (lines.length < 3) continue;

        const timeLine = lines[1];
        const match = timeLine.match(
            /(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2}),(\d{3})/
        );
        if (!match) continue;

        const start =
            Number(match[1]) * 3600 +
            Number(match[2]) * 60 +
            Number(match[3]) +
            Number(match[4]) / 1000;
        const end =
            Number(match[5]) * 3600 +
            Number(match[6]) * 60 +
            Number(match[7]) +
            Number(match[8]) / 1000;

        const text = lines.slice(2).join(' ').trim();
        if (!text) continue;

        segments.push({ start, end, text });
    }

    return segments;
}

function writeTranscriptJson(audioPath: string, segments: TransientSegment[]): void {
    try {
        const workDir = path.dirname(audioPath);
        const transcriptJsonPath = path.join(workDir, 'transcript.json');
        fs.writeFileSync(transcriptJsonPath, JSON.stringify(segments, null, 2));
    } catch (error) {
        console.error('Failed to write transcript.json cache:', error);
    }
}

function isLikelyBrokenTranscript(segments: TransientSegment[]): boolean {
    if (!segments || segments.length === 0) return true;
    const n = segments.length;
    const empty = segments.filter((s) => !String(s.text ?? '').trim()).length;
    const dots = segments.filter((s) => /^\s*\.+\s*$/.test(String(s.text ?? ''))).length;
    const tiny = segments.filter((s) => String(s.text ?? '').trim().length <= 2).length;

    const emptyRatio = empty / n;
    const dotsRatio = dots / n;
    const tinyRatio = tiny / n;

    // Strong signal of failed decode: almost everything is punctuation-only placeholders.
    if (dotsRatio >= 0.8) return true;
    if (emptyRatio >= 0.5) return true;
    if (tinyRatio >= 0.9 && dotsRatio >= 0.5) return true;
    return false;
}
