import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import dotenv from 'dotenv';
import { v1p1beta1 as speechV1p1 } from '@google-cloud/speech';
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

async function ensureCompressedAudioMp3(inputVideo: string, outputAudio: string): Promise<string> {
  if (fs.existsSync(outputAudio) && fs.statSync(outputAudio).size > 0) return outputAudio;
  fs.mkdirSync(path.dirname(outputAudio), { recursive: true });
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
    outputAudio
  ]);
  return outputAudio;
}

function durationToSec(v: any): number {
  if (!v) return 0;
  const seconds = Number(v.seconds ?? 0);
  const nanos = Number(v.nanos ?? 0);
  return (Number.isFinite(seconds) ? seconds : 0) + (Number.isFinite(nanos) ? nanos / 1e9 : 0);
}

function appendToken(text: string, token: string): string {
  if (!text) return token;
  if (/^[,.;:!?%)]/.test(token)) return `${text}${token}`;
  if (/^['\"]/.test(token)) return `${text}${token}`;
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
    if (w.speaker === cur.speaker && gap <= maxGapSec) {
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

function extractWordsFromV1Response(response: any): WordTurn[] {
  const words: WordTurn[] = [];
  const results = Array.isArray(response?.results) ? response.results : [];
  for (const result of results) {
    const alt = result?.alternatives?.[0];
    const altWords = Array.isArray(alt?.words) ? alt.words : [];
    for (const w of altWords) {
      const token = String(w?.word ?? '').trim();
      if (!token) continue;
      const start = durationToSec(w?.startTime);
      const end = durationToSec(w?.endTime);
      if (!(end > start)) continue;
      const speakerTag = Number(w?.speakerTag ?? 0);
      words.push({
        word: token,
        start,
        end,
        speaker: `SPEAKER_${speakerTag || 0}`
      });
    }
  }
  return words;
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

function nearestSpeakerBounds(turns: DiarizedTurn[], speaker: string, aroundStart: number, aroundEnd: number) {
  const same = turns.filter((t) => t.speaker === speaker).sort((a, b) => a.start - b.start);
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
  const clipEnd = nextOther ? Math.min(nextOther.start, end + Math.min(10, Math.max(0, nextOther.start - end))) : end + 10;

  return {
    refined_speaker_start_sec: start,
    refined_speaker_end_sec: end,
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

  const language = process.env.GOOGLE_STT_LANGUAGE || 'es-US';
  const minSpeakerCount = Number(process.env.GOOGLE_STT_MIN_SPEAKERS ?? 2);
  const maxSpeakerCount = Number(process.env.GOOGLE_STT_MAX_SPEAKERS ?? 8);

  const workDirArg = process.argv[2] || 'apps/test_data/e2e_live_20260219_185221';
  const workDir = path.resolve(process.cwd(), workDirArg);
  const inputVideo = process.argv[3] ? path.resolve(process.cwd(), process.argv[3]) : path.join(workDir, 'source.mp4');

  if (!fs.existsSync(inputVideo)) throw new Error(`Video not found: ${inputVideo}`);
  fs.mkdirSync(workDir, { recursive: true });

  const compressedAudio = path.join(workDir, 'source.google-v1.64k.mono.mp3');
  await ensureCompressedAudioMp3(inputVideo, compressedAudio);

  const runId = new Date().toISOString().replace(/[-:.TZ]/g, '').slice(0, 14);
  const gcsObject = `cdv-google-stt/${path.basename(workDir)}/input-v1-${runId}.mp3`;
  const gcsUri = `gs://${bucket}/${gcsObject}`;

  const storage = new Storage({ projectId });
  console.log(`Uploading audio to ${gcsUri} ...`);
  await storage.bucket(bucket).upload(compressedAudio, {
    destination: gcsObject,
    contentType: 'audio/mpeg'
  });

  const speechClient = new speechV1p1.SpeechClient({ projectId });
  const request: any = {
    audio: { uri: gcsUri },
    config: {
      encoding: 'MP3',
      sampleRateHertz: 16000,
      languageCode: language,
      enableAutomaticPunctuation: true,
      enableWordTimeOffsets: true,
      diarizationConfig: {
        enableSpeakerDiarization: true,
        minSpeakerCount,
        maxSpeakerCount
      }
    }
  };

  console.log('Submitting Google STT v1p1beta1 longRunningRecognize with diarization...');
  const [operation] = await speechClient.longRunningRecognize(request);
  const [response] = await operation.promise();

  const rawOutPath = path.join(workDir, 'transcript.google-v1.response.json');
  fs.writeFileSync(rawOutPath, JSON.stringify(response, null, 2));
  console.log(`Wrote raw Google v1 response -> ${rawOutPath}`);

  const words = extractWordsFromV1Response(response);
  if (words.length === 0) throw new Error('No diarized words parsed from Google v1 response.');

  const turns = wordsToTurns(words);
  const diarizedPath = path.join(workDir, 'transcript.diarized.google-v1.json');
  fs.writeFileSync(diarizedPath, JSON.stringify(turns, null, 2));
  console.log(`Wrote ${turns.length} diarized turns -> ${diarizedPath}`);

  const inferred = await inferSermonBoundsWithLLM(openai, turns);
  const refined = nearestSpeakerBounds(turns, inferred.sermon_speaker, inferred.speaker_start_sec, inferred.speaker_end_sec);
  if (!refined) throw new Error(`Could not locate inferred speaker "${inferred.sermon_speaker}" in Google v1 diarized turns.`);

  const out = {
    model: 'google-speech-to-text-v1p1beta1',
    config: {
      language,
      min_speakers: minSpeakerCount,
      max_speakers: maxSpeakerCount,
      input_gcs_uri: gcsUri
    },
    inferred,
    refined
  };

  const outPath = path.join(workDir, 'sermon.boundaries.google-v1.json');
  fs.writeFileSync(outPath, JSON.stringify(out, null, 2));
  console.log(`Wrote Google v1 sermon boundaries -> ${outPath}`);
}

main().catch((err) => {
  console.error('test-google-v1-diarization failed:', err);
  process.exit(1);
});
