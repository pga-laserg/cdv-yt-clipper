import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import dotenv from 'dotenv';
import { v2 as speechV2 } from '@google-cloud/speech';
import { Storage } from '@google-cloud/storage';
import { OpenAI } from 'openai';

dotenv.config({ path: path.resolve(__dirname, '../../../.env') });
dotenv.config({ path: path.resolve(__dirname, '../../web/.env.local') });

interface DiarizedTurn {
  start: number;
  end: number;
  speaker: string;
  text: string;
}

interface WordTurn {
  word: string;
  start: number;
  end: number;
  speaker: string;
}

interface AudioChunk {
  filePath: string;
  startOffsetSec: number;
  durationSec: number;
}

function runProcess(cmd: string, args: string[]): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    proc.stdout.on('data', (d) => {
      stdout += d.toString();
    });
    proc.stderr.on('data', (d) => {
      stderr += d.toString();
    });
    proc.on('error', reject);
    proc.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`${cmd} exited with code ${code}. stderr: ${stderr.slice(-1200)}`));
        return;
      }
      resolve({ stdout, stderr });
    });
  });
}

async function getDurationSec(filePath: string): Promise<number> {
  const { stdout } = await runProcess('ffprobe', [
    '-v',
    'error',
    '-show_entries',
    'format=duration',
    '-of',
    'default=noprint_wrappers=1:nokey=1',
    filePath
  ]);
  const n = Number(stdout.trim());
  return Number.isFinite(n) ? n : 0;
}

async function ensureChunkedAudioMp3(inputVideo: string, chunksDir: string, segmentSec: number): Promise<AudioChunk[]> {
  fs.mkdirSync(chunksDir, { recursive: true });
  const existing = fs
    .readdirSync(chunksDir)
    .filter((f) => f.endsWith('.mp3'))
    .sort()
    .map((f) => path.join(chunksDir, f));

  const files = existing.length
    ? existing
    : (() => {
        const outPattern = path.join(chunksDir, 'chunk_%03d.mp3');
        return outPattern;
      })();

  if (!existing.length) {
    await runProcess('ffmpeg', [
      '-y',
      '-i',
      inputVideo,
      '-vn',
      '-ac',
      '1',
      '-ar',
      '16000',
      '-c:a',
      'libmp3lame',
      '-b:a',
      '64k',
      '-f',
      'segment',
      '-segment_time',
      String(segmentSec),
      '-reset_timestamps',
      '1',
      files as string
    ]);
  }

  const finalFiles = fs
    .readdirSync(chunksDir)
    .filter((f) => f.endsWith('.mp3'))
    .sort()
    .map((f) => path.join(chunksDir, f));

  const chunks: AudioChunk[] = [];
  let offset = 0;
  for (const filePath of finalFiles) {
    const durationSec = await getDurationSec(filePath);
    chunks.push({ filePath, startOffsetSec: offset, durationSec });
    offset += durationSec;
  }
  return chunks;
}

function durationToSec(v: any): number {
  if (!v) return 0;
  const seconds = Number(v.seconds ?? 0);
  const nanos = Number(v.nanos ?? 0);
  const secPart = Number.isFinite(seconds) ? seconds : 0;
  const nanoPart = Number.isFinite(nanos) ? nanos / 1e9 : 0;
  return secPart + nanoPart;
}

function normalizeSpeakerLabel(v: unknown): string {
  if (typeof v === 'string' && v.trim()) {
    const clean = v.trim();
    if (/^SPEAKER_/i.test(clean)) return clean.toUpperCase();
    if (/^\d+$/.test(clean)) return `SPEAKER_${clean}`;
    return `SPEAKER_${clean.replace(/\s+/g, '_').toUpperCase()}`;
  }
  if (typeof v === 'number' && Number.isFinite(v)) return `SPEAKER_${String(v)}`;
  return 'SPEAKER_UNKNOWN';
}

function appendToken(text: string, token: string): string {
  if (!text) return token;
  if (/^[,.;:!?%)]/.test(token)) return `${text}${token}`;
  if (/^['"]/.test(token)) return `${text}${token}`;
  if (/[(]$/.test(text)) return `${text}${token}`;
  return `${text} ${token}`;
}

function wordsToTurns(words: WordTurn[], maxGapSec = 1.2): DiarizedTurn[] {
  if (words.length === 0) return [];
  const sorted = [...words].sort((a, b) => a.start - b.start);
  const out: DiarizedTurn[] = [];

  let cur: DiarizedTurn = {
    start: sorted[0].start,
    end: sorted[0].end,
    speaker: sorted[0].speaker,
    text: sorted[0].word
  };

  for (let i = 1; i < sorted.length; i++) {
    const w = sorted[i];
    const gap = Math.max(0, w.start - cur.end);
    const sameSpeaker = w.speaker === cur.speaker;
    if (sameSpeaker && gap <= maxGapSec) {
      cur.end = Math.max(cur.end, w.end);
      cur.text = appendToken(cur.text, w.word);
      continue;
    }
    out.push(cur);
    cur = {
      start: w.start,
      end: w.end,
      speaker: w.speaker,
      text: w.word
    };
  }

  out.push(cur);
  return out;
}

async function ensureManagedRecognizer(
  speechClient: speechV2.SpeechClient,
  projectId: string,
  location: string,
  recognizerId: string,
  config: {
    model: string;
    language: string;
    minSpeakerCount: number;
    maxSpeakerCount: number;
  }
): Promise<string> {
  const name = `projects/${projectId}/locations/${location}/recognizers/${recognizerId}`;

  try {
    const [existing] = await speechClient.getRecognizer({ name });
    if (existing?.state === 'ACTIVE' || existing?.state === 2) {
      console.log(`Using existing recognizer: ${name}`);
      return name;
    }
  } catch (err) {
    // Not found is expected on first run.
  }

  console.log(`Creating recognizer ${name} with diarization defaults...`);
  const [op] = await speechClient.createRecognizer({
    parent: `projects/${projectId}/locations/${location}`,
    recognizerId,
    recognizer: {
      model: config.model,
      languageCodes: [config.language],
      defaultRecognitionConfig: {
        autoDecodingConfig: {},
        model: config.model,
        languageCodes: [config.language],
        features: {
          enableWordTimeOffsets: true,
          enableAutomaticPunctuation: true,
          diarizationConfig: {
            minSpeakerCount: config.minSpeakerCount,
            maxSpeakerCount: config.maxSpeakerCount
          }
        }
      }
    }
  } as any);

  await op.promise();
  console.log(`Recognizer created: ${name}`);
  return name;
}

function extractWordsFromBatchResponse(
  batch: any,
  uriOffsetSec: Record<string, number>,
  mediaDurationSec?: number
): WordTurn[] {
  const words: WordTurn[] = [];
  const entries = Object.entries(batch?.results ?? {});

  for (const [uri, fileResult] of entries) {
    const offset = Number(uriOffsetSec[uri] ?? 0);
    const transcript =
      (fileResult as any)?.transcript ??
      (fileResult as any)?.inlineResult?.transcript ??
      null;

    const speechResults = transcript?.results;
    if (!Array.isArray(speechResults)) continue;

    for (const result of speechResults) {
      const alt = result?.alternatives?.[0];
      const altWords = alt?.words;
      if (!Array.isArray(altWords)) continue;

      for (const w of altWords) {
        const token = String(w?.word ?? '').trim();
        if (!token) continue;
        const start = durationToSec(w?.startOffset) + offset;
        const end = durationToSec(w?.endOffset) + offset;
        if (!(end > start)) continue;
        if (end - start > 30) continue;
        if (mediaDurationSec && (start > mediaDurationSec + 1 || end > mediaDurationSec + 1)) continue;
        words.push({
          word: token,
          start,
          end,
          speaker: normalizeSpeakerLabel(w?.speakerLabel)
        });
      }
    }
  }

  return words;
}

function collectBatchErrors(batch: any): Array<{ uri: string; code: number; message: string }> {
  const out: Array<{ uri: string; code: number; message: string }> = [];
  const entries = Object.entries(batch?.results ?? {});
  for (const [uri, fileResult] of entries) {
    const err = (fileResult as any)?.error;
    if (err?.code) {
      out.push({
        uri,
        code: Number(err.code),
        message: String(err.message ?? '')
      });
    }
  }
  return out;
}

async function inferSermonBoundsWithLLM(client: OpenAI, turns: DiarizedTurn[]) {
  const timeline = turns
    .slice(0, 2500)
    .map((t) => `[${t.start.toFixed(2)}-${t.end.toFixed(2)}] ${t.speaker}: ${t.text.slice(0, 220)}`)
    .join('\n');

  const prompt = `
You are analyzing a diarized church service transcript.
Find the MAIN SERMON section boundaries and the MAIN SERMON SPEAKER label.

Return strict JSON:
{
  "sermon_speaker": "speaker label exactly as appears in timeline",
  "speaker_start_sec": number,
  "speaker_end_sec": number,
  "confidence": number,
  "notes": "short"
}

Rules:
- Exclude greetings, music, children story, offering, announcements.
- Exclude speakers before and after sermon.
- Use the same speaker label from timeline for sermon_speaker.

Timeline:
${timeline}
`;

  const response = await client.chat.completions.create({
    model: process.env.ANALYZE_OPENAI_MODEL || 'gpt-4o-mini',
    messages: [{ role: 'user', content: prompt }],
    response_format: { type: 'json_object' }
  });

  const content = response.choices[0]?.message?.content;
  if (!content) throw new Error('Empty sermon boundary response from LLM.');
  return JSON.parse(content) as {
    sermon_speaker: string;
    speaker_start_sec: number;
    speaker_end_sec: number;
    confidence: number;
    notes?: string;
  };
}

function nearestSpeakerBounds(
  turns: DiarizedTurn[],
  speaker: string,
  aroundStart: number,
  aroundEnd: number,
  mediaDurationSec?: number
) {
  const same = turns
    .filter((t) => t.speaker === speaker)
    .filter((t) => t.end > t.start && t.end - t.start <= 180)
    .sort((a, b) => a.start - b.start);
  if (same.length === 0) return null;

  const aroundStartTurn =
    same.find((t) => t.start <= aroundStart && t.end >= aroundStart) ??
    same.reduce((best, t) => (Math.abs(t.start - aroundStart) < Math.abs(best.start - aroundStart) ? t : best), same[0]);

  const aroundEndTurn =
    same.find((t) => t.start <= aroundEnd && t.end >= aroundEnd) ??
    same.reduce((best, t) => (Math.abs(t.end - aroundEnd) < Math.abs(best.end - aroundEnd) ? t : best), same[0]);

  const start = aroundStartTurn.start;
  const end = aroundEndTurn.end;

  const prevOther = [...turns]
    .filter((t) => t.speaker !== speaker && t.end <= start)
    .sort((a, b) => b.end - a.end)[0];

  const nextOther = turns
    .filter((t) => t.speaker !== speaker && t.start >= end)
    .sort((a, b) => a.start - b.start)[0];

  const clipStart = Math.max(0, start - Math.min(10, Math.max(0, start - (prevOther?.end ?? 0))));
  const rawClipEnd = nextOther ? Math.min(nextOther.start, end + Math.min(10, Math.max(0, nextOther.start - end))) : end + 10;
  const clipEnd = mediaDurationSec ? Math.min(rawClipEnd, mediaDurationSec) : rawClipEnd;
  const boundedEnd = mediaDurationSec ? Math.min(end, mediaDurationSec) : end;

  return {
    refined_speaker_start_sec: start,
    refined_speaker_end_sec: boundedEnd,
    clip_start_sec: clipStart,
    clip_end_sec: clipEnd,
    prev_other_speaker: prevOther ?? null,
    next_other_speaker: nextOther ?? null
  };
}

async function main() {
  const openAiApiKey = process.env.OPENAI_API_KEY;
  if (!openAiApiKey) throw new Error('OPENAI_API_KEY is required for sermon-bound inference.');
  const openai = new OpenAI({ apiKey: openAiApiKey });

  const projectId = process.env.GOOGLE_CLOUD_PROJECT || process.env.GCLOUD_PROJECT;
  if (!projectId) throw new Error('Set GOOGLE_CLOUD_PROJECT (or GCLOUD_PROJECT).');

  const bucket = process.env.GOOGLE_STT_BUCKET;
  if (!bucket) throw new Error('Set GOOGLE_STT_BUCKET for temporary audio upload.');

  const location = process.env.GOOGLE_STT_LOCATION || 'us-central1';
  const language = process.env.GOOGLE_STT_LANGUAGE || 'es-US';
  const model = process.env.GOOGLE_STT_MODEL || 'chirp_3';
  const minSpeakerCount = Number(process.env.GOOGLE_STT_MIN_SPEAKERS ?? 2);
  const maxSpeakerCount = Number(process.env.GOOGLE_STT_MAX_SPEAKERS ?? 8);
  const segmentSec = Number(process.env.GOOGLE_STT_SEGMENT_SEC ?? 1140);
  const requestedMaxFilesPerBatch = Number(process.env.GOOGLE_STT_MAX_FILES_PER_BATCH ?? 15);
  const maxFilesPerBatch = Math.max(1, Math.min(1, requestedMaxFilesPerBatch));
  if (requestedMaxFilesPerBatch !== maxFilesPerBatch) {
    console.log(
      `GOOGLE_STT_MAX_FILES_PER_BATCH=${requestedMaxFilesPerBatch} adjusted to ${maxFilesPerBatch} (inline output supports one file per batch).`
    );
  }
  const maxBatchRetries = Number(process.env.GOOGLE_STT_BATCH_RETRIES ?? 3);
  const useManagedRecognizer = String(process.env.GOOGLE_STT_USE_MANAGED_RECOGNIZER ?? 'true') === 'true';
  const recognizerId = process.env.GOOGLE_STT_RECOGNIZER_ID || 'cdv-sermon-diarizer';

  const workDirArg = process.argv[2] || 'apps/test_data/e2e_live_20260219_185221';
  const workDir = path.resolve(process.cwd(), workDirArg);
  const inputVideo = process.argv[3] ? path.resolve(process.cwd(), process.argv[3]) : path.join(workDir, 'source.mp4');

  if (!fs.existsSync(inputVideo)) throw new Error(`Video not found: ${inputVideo}`);
  fs.mkdirSync(workDir, { recursive: true });
  const sourceDurationSec = await getDurationSec(inputVideo);

  const inputStem = path.basename(inputVideo, path.extname(inputVideo));
  const chunksDir = path.join(workDir, `${inputStem}.google.chunks_${segmentSec}s`);
  const chunks = await ensureChunkedAudioMp3(inputVideo, chunksDir, segmentSec);
  if (chunks.length === 0) throw new Error('No audio chunks were produced.');

  const runId = new Date().toISOString().replace(/[-:.TZ]/g, '').slice(0, 14);
  const storage = new Storage({ projectId });
  const chunkMeta = [];
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    const gcsObject = `cdv-google-stt/${path.basename(workDir)}/${inputStem}/${runId}/chunk_${String(i).padStart(3, '0')}.mp3`;
    const gcsUri = `gs://${bucket}/${gcsObject}`;
    console.log(`Uploading chunk ${i + 1}/${chunks.length} -> ${gcsUri}`);
    await storage.bucket(bucket).upload(chunk.filePath, {
      destination: gcsObject,
      contentType: 'audio/mpeg'
    });
    chunkMeta.push({ ...chunk, gcsUri });
  }

  const speechClient =
    location === 'global'
      ? new speechV2.SpeechClient({ projectId })
      : new speechV2.SpeechClient({ projectId, apiEndpoint: `${location}-speech.googleapis.com` });
  const recognizer = useManagedRecognizer
    ? await ensureManagedRecognizer(speechClient, projectId, location, recognizerId, {
        model,
        language,
        minSpeakerCount,
        maxSpeakerCount
      })
    : `projects/${projectId}/locations/${location}/recognizers/_`;

  const allWords: WordTurn[] = [];
  const rawResponses: any[] = [];
  const batchCount = Math.ceil(chunkMeta.length / maxFilesPerBatch);

  for (let batchIdx = 0; batchIdx < batchCount; batchIdx++) {
    const from = batchIdx * maxFilesPerBatch;
    const batchChunks = chunkMeta.slice(from, from + maxFilesPerBatch);
    const uriOffsetSec = Object.fromEntries(batchChunks.map((c) => [c.gcsUri, c.startOffsetSec]));

    const request: any = {
      recognizer,
      config: {
        autoDecodingConfig: {},
        model,
        languageCodes: [language],
        features: {
          enableWordTimeOffsets: true,
          enableAutomaticPunctuation: true,
          diarizationConfig: { minSpeakerCount, maxSpeakerCount }
        }
      },
      files: batchChunks.map((c) => ({ uri: c.gcsUri })),
      recognitionOutputConfig: {
        inlineResponseConfig: {}
      },
      processingStrategy: 'DYNAMIC_BATCHING'
    };

    let response: any | null = null;
    for (let attempt = 1; attempt <= maxBatchRetries; attempt++) {
      console.log(
        `Submitting Google STT V2 batch ${batchIdx + 1}/${batchCount} attempt ${attempt}/${maxBatchRetries} (files=${batchChunks.length}, model=${model}) ...`
      );
      const [operation] = await speechClient.batchRecognize(request);
      const [candidate] = await operation.promise();
      const errs = collectBatchErrors(candidate);
      if (errs.length === 0) {
        response = candidate;
        break;
      }
      console.warn(`Batch ${batchIdx + 1} returned ${errs.length} file error(s): ${JSON.stringify(errs)}`);
      if (attempt < maxBatchRetries) {
        await new Promise((r) => setTimeout(r, 2000 * attempt));
      }
    }
    if (!response) {
      throw new Error(`Google STT batch ${batchIdx + 1}/${batchCount} failed after ${maxBatchRetries} attempts.`);
    }
    rawResponses.push(response);
    allWords.push(...extractWordsFromBatchResponse(response, uriOffsetSec, sourceDurationSec));
  }

  const rawOutPath = path.join(workDir, 'transcript.google.batch-responses.json');
  fs.writeFileSync(rawOutPath, JSON.stringify(rawResponses, null, 2));
  console.log(`Wrote raw Google batch responses -> ${rawOutPath}`);

  const words = allWords;
  if (words.length === 0) {
    throw new Error('No diarized words parsed from Google batch response.');
  }

  const turns = wordsToTurns(words);
  const diarizedPath = path.join(workDir, 'transcript.diarized.google.json');
  fs.writeFileSync(diarizedPath, JSON.stringify(turns, null, 2));
  console.log(`Wrote ${turns.length} diarized turns -> ${diarizedPath}`);

  const inferred = await inferSermonBoundsWithLLM(openai, turns);
  const refined = nearestSpeakerBounds(
    turns,
    inferred.sermon_speaker,
    inferred.speaker_start_sec,
    inferred.speaker_end_sec,
    sourceDurationSec
  );
  if (!refined) throw new Error(`Could not locate inferred speaker "${inferred.sermon_speaker}" in Google diarized turns.`);

  const out = {
    model: 'google-speech-to-text-v2',
    config: {
      model,
      dynamic_batching: true,
      language,
      segment_sec: segmentSec,
      chunk_count: chunkMeta.length,
      batch_count: batchCount,
      min_speakers: minSpeakerCount,
      max_speakers: maxSpeakerCount,
      input_gcs_uris: chunkMeta.map((c) => c.gcsUri),
      location
    },
    inferred,
    refined
  };

  const outPath = path.join(workDir, 'sermon.boundaries.google-stt.json');
  fs.writeFileSync(outPath, JSON.stringify(out, null, 2));
  console.log(`Wrote Google STT sermon boundaries -> ${outPath}`);
}

main().catch((err) => {
  const anyErr = err as any;
  const details = anyErr?.statusDetails?.[0]?.fieldViolations;
  if (Array.isArray(details) && details.length) {
    console.error('Google field violations:', JSON.stringify(details, null, 2));
  }
  console.error('test-google-stt-diarization failed:', err);
  process.exit(1);
});
