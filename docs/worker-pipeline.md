# CDV Worker Pipeline — Technical Reference

Last updated: 2026-03-14

This document describes the internal architecture of the `apps/worker` service: how it processes a job from source URL to rendered clips, what files it writes along the way, and all the environment variables that control its behaviour.

---

## 1. Pipeline Overview

The worker runs a linear pipeline of stages. Each stage is orchestrated from `src/index.ts` → `runPipeline()`.

For specific details on running this on a Mac M1 with hardware acceleration and background persistence, see [Mac M1 Worker Architecture](./mac-m1-worker.md).

```
Job (source_url from Supabase)
  │
  ▼
Stage 1: Ingest ──────────────────── download video + extract audio
  │
  ▼
Stage 2: Transcribe ───────────────── obtain timestamped transcript segments
  │   ┌──────────────────────────────────────────────────────────────────┐
  │   │ Fast-path (YouTube only):  defuddle → captions + music cues     │
  │   │ Fallback 1:                ElevenLabs Scribe v2 (cloud STT)     │
  │   │ Fallback 2:                faster-whisper / local Whisper       │
  │   └──────────────────────────────────────────────────────────────────┘
  │
  ▼
Stage 3: Analyze ──────────────────── find sermon boundaries + select clips
  │
  ▼
Stage 4: Render ───────────────────── ffmpeg: sermon horizontal + vertical clips
  │
  ▼
Stage 5: Store ────────────────────── upload assets, upsert clips in Supabase
  │
  ▼
Stage 6: Blog Artifact ────────────── generate blog post draft (optional)
  │
  ▼
Stage 7: Delivery ─────────────────── optional low-res encode + full-res upload
```

---

## 2. Stage 1 — Ingest

**Source:** `src/pipeline/ingest.ts`

Capabilities:
- YouTube URL: downloads via `yt-dlp` in multiple quality layers.
- Local file: copies / symlinks as needed.

Outputs written to `work_dir/<job_id>/`:

| File | Description |
|---|---|
| `metadata.json` | Source URL + ingest timestamp. **Required by Stage 2 for defuddle.** |
| `source.mp4` | Original download (may be very large) |
| `source.light.mp4` | Lightweight 720p proxy for tracking/analysis |
| `source.hq.mp4` | High-quality re-encode (if `INGEST_MODE=eager`) |
| `audio.wav` | 16kHz mono WAV extracted for STT |

### Ingest env vars

| Variable | Default | Description |
|---|---|---|
| `INGEST_MODE` | `audio_first` | `audio_first` \| `eager` \| `hybrid`. Controls when HQ transcode runs. |
| `RENDER_SOURCE_POLICY` | `original_first` | `original_first` \| `hq_first` \| `force_hq`. Determines which video variant is used for rendering. |
| `YTDLP_RATE_LIMIT` | — | Optional: e.g. `5M` to cap download speed (passed to `yt-dlp`). |
| `YTDLP_PROXY` | — | Optional: proxy URL for yt-dlp. |

---

## 3. Stage 2 — Transcribe

**Source:** `src/pipeline/transcribe.ts`  
**Entry point:** `transcribe(audioPath, options?): Promise<TransientSegment[]>`

### 3.1 Caching

Before attempting any STT, the function checks for a cached `transcript.json` in the same `work_dir`. If found, it is reused. Set `TRANSCRIBE_FORCE_REDO=true` to bypass.

### 3.2 Provider resolution

`TRANSCRIBE_PROVIDER` controls which provider runs:

| Value | Behaviour |
|---|---|
| `auto` (default) | defuddle → ElevenLabs → local Whisper |
| `local` | defuddle (YT only) → local Whisper only |
| `elevenlabs_scribe_v2` | defuddle (YT only) → ElevenLabs, then Whisper fallback (unless `TRANSCRIBE_STRICT_PROVIDER=true`) |

### 3.3 Defuddle Fast-Path (YouTube sources)

When the source URL in `metadata.json` points to `youtube.com` or `youtu.be`, the pipeline first attempts to extract the transcript via `defuddle` — a tool that fetches YouTube's manually curated or auto-generated caption track.

**Why:** manual captions are faster to obtain and often more accurate than STT, and they include music cues already annotated.

**Implementation:** `src/lib/defuddle.ts`

```typescript
fetchYouTubeTranscript(url: string): Promise<DefuddleResult>
```

Where `DefuddleResult` has:
- `segments: DefuddleSegment[]` — transcript with `start`, `end`, `text`
- `audioEvents: DefuddleAudioEvent[]` — detected music cues

Music cues are extracted from caption lines that contain `[música]` or `[music]` (including markdown-escaped variants `\[música\]`). Adjacent events within 1 second are merged.

**Outputs produced by defuddle path (same dir as audioPath):**

| File | Description |
|---|---|
| `transcript.json` | Standard transcript segments |
| `source.srt` | SRT format of the transcript |
| `audio.events.json` | Primary audio events file (music cues from captions) |
| `audio.events.defuddle.json` | Copy of the above for provenance |

If defuddle fails (no captions, network error, etc.) the pipeline falls back to ElevenLabs or Whisper silently.

### 3.4 ElevenLabs Scribe v2

Cloud STT. Requires `ELEVENLABS_API_KEY`. Produces word-level timestamps and speaker diarization.

**ElevenLabs env vars:**

| Variable | Default | Description |
|---|---|---|
| `ELEVENLABS_API_KEY` | — | Required for cloud STT |
| `TRANSCRIBE_ELEVENLABS_MODEL_ID` | `scribe_v2` | ElevenLabs model slug |
| `TRANSCRIBE_ELEVENLABS_DIARIZE` | `true` | Enable speaker diarization |
| `TRANSCRIBE_ELEVENLABS_TAG_AUDIO_EVENTS` | `true` | Tag music/speech/noise events |
| `TRANSCRIBE_ELEVENLABS_INCLUDE_AUDIO_EVENTS_IN_TRANSCRIPT` | `false` | Include event tags inline in transcript text |
| `TRANSCRIBE_ELEVENLABS_TIMESTAMP_GRANULARITY` | `word` | `none` \| `word` \| `character` |
| `TRANSCRIBE_ELEVENLABS_LANGUAGE_CODE` | — | e.g. `es` to force language |
| `TRANSCRIBE_ELEVENLABS_NUM_SPEAKERS` | — | Hint for diarization |
| `TRANSCRIBE_ELEVENLABS_TIMEOUT_MS` | `1800000` | 30 min default timeout |

ElevenLabs outputs (same dir as audioPath):

| File | Description |
|---|---|
| `transcript.elevenlabs.scribe_v2.json` | Raw ElevenLabs response |
| `transcript.diarized.elevenlabs.json` | Speaker-turns format |
| `audio.events.elevenlabs.json` | ElevenLabs-derived music/speech events |
| `source.elevenlabs.srt` | SRT from ElevenLabs additional formats |

### 3.5 Local Whisper

Falls back to `faster-whisper` via a locally-started Python subprocess. Tries `small` model first, then `base`.

### 3.6 Quality Check

All providers run through `isLikelyBrokenTranscript()` which rejects transcripts that are dominated by dots or empty segments. A rejected provider triggers fallback to the next.

### 3.7 Common transcript env vars

| Variable | Default | Description |
|---|---|---|
| `TRANSCRIBE_PROVIDER` | `local` | `auto` \| `local` \| `elevenlabs_scribe_v2` |
| `TRANSCRIBE_FORCE_REDO` | `false` | Ignore cached `transcript.json` and re-run |
| `TRANSCRIBE_STRICT_PROVIDER` | `false` | If `true`, don't fall back to Whisper when ElevenLabs fails |

---

## 4. Stage 3 — Analyze (Boundaries + Clips)

**Source:** `src/pipeline/analyze.ts` → `src/pipeline/boundaries.ts`

Uses the transcript segments plus optional audio/visual signals to:
1. Find sermon start/end (`findSermonBoundaries`)
2. Select highlight clips (`findHighlights`)

### 4.1 Audio Events Signal

`boundaries.ts` reads `audio.events.json` from the work directory. This file is produced by either:
- **defuddle** (if YouTube source with captions)
- **ElevenLabs** (as `audio.events.elevenlabs.json`, then copied if primary is missing)
- **local audio analysis pass** (if `BOUNDARY_ENABLE_AUDIO_SIGNALS=true` and no cloud signal exists)

The format is:
```json
{
  "source": "defuddle-youtube-manual-captions",
  "duration_sec": 5082,
  "step_sec": null,
  "segments": [
    { "label": "music", "start": 29, "end": 61 },
    { "label": "speech", "start": 65, "end": 180 }
  ]
}
```

Labels used:
- `music` — music/worship section
- `speech` / `male` / `female` — spoken content
- `noenergy` — silence / no activity

### 4.2 Boundary pipeline env vars

| Variable | Default | Description |
|---|---|---|
| `BOUNDARY_PIPELINE_PROFILE` | `light` | `light` \| `standard`. Controls depth of diarization scan windows. |
| `BOUNDARY_ENABLE_AUDIO_SIGNALS` | `true` | Whether audio events influence boundary detection |
| `BOUNDARY_ENABLE_SLIDE_OCR_SIGNALS` | `false` | Enable slide OCR as a boundary signal |
| `BOUNDARY_LLM_MODEL` | `gpt-4o-mini` | OpenAI model for LLM boundary stages |
| `BOUNDARY_ALLOW_STALE_TARGETED_CACHE` | `false` | Allow reusing targeted diarization even if cache key mismatches |
| `OPENAI_API_KEY` | — | Required for LLM boundary stages |

---

## 5. Work Directory Layout

For each job, the worker creates:

```
work_dir/<job_id>/
  metadata.json                   ← source URL + ingest timestamp
  source.mp4                      ← original download
  source.light.mp4                ← lightweight proxy
  source.hq.mp4                   ← optional HQ encode
  audio.wav                       ← 16kHz WAV for STT
  transcript.json                 ← canonical transcript (segments)
  source.srt                      ← SRT transcript
  audio.events.json               ← primary audio events (music/speech)
  audio.events.defuddle.json      ← defuddle-specific copy (if YouTube)
  audio.events.elevenlabs.json    ← ElevenLabs-specific copy (if used)
  transcript.elevenlabs.scribe_v2.json
  transcript.diarized.elevenlabs.json
  sermon.boundaries.targeted-diarization.json
  sermon.boundaries.openai.stage1.json
  sermon.boundaries.openai.stage2.json
  analysis.json                   ← highlight clips
  processed/                      ← diarization temp chunks
  sermon_horizontal.mp4           ← rendered sermon cut
  clips/
    clip_<id>.mp4
    ...
  blog.artifact.json              ← generated blog post draft
```

---

## 6. Worker Loop & Job Claiming

**Source:** `src/index.ts`

The worker polls Supabase for jobs with `status = 'queued'` and atomically claims them using a lease mechanism.

| Variable | Default | Description |
|---|---|---|
| `WORKER_ID` | `worker-<pid>` | Unique worker identifier |
| `WORKER_POLL_INTERVAL_MS` | `10000` | How often to poll for new jobs |
| `WORKER_JOB_LEASE_SECONDS` | `120` | Lease duration; job reverts if not renewed |
| `WORKER_RPC_CLAIM_ENABLED` | `true` | Use Supabase RPC for atomic claim |
| `PIPELINE_MODE` | `local` | `local` \| `cloud_limited` \| `prod`. Controls upload/rendering defaults. |

---

## 7. Upload & Storage

| Variable | Default | Description |
|---|---|---|
| `ENABLE_RAILWAY_VERTICAL` | `PIPELINE_MODE != local` | Upload vertical clips to Supabase Storage |
| `ENABLE_RAILWAY_HORIZONTAL_LOWRES` | `PIPELINE_MODE != local` | Encode + upload low-res horizontal cut |
| `ENABLE_PROXMOX_FULLRES` | see code | Upload full-res via SSH to Proxmox |
| `FULLRES_STORAGE_ENABLED` | `false` | Master toggle for full-res storage |
| `FULLRES_STORAGE_PROVIDER` | `proxmox` | `proxmox` \| `none` (s3/supabase stubs) |
| `FULLRES_STORAGE_SSH_HOST` | — | Proxmox SSH host |
| `FULLRES_STORAGE_SSH_USER` | `uploader` | SSH user |
| `FULLRES_STORAGE_SSH_PORT` | `22` | SSH port |
| `FULLRES_STORAGE_SSH_PATH` | `/home/uploader/uploads` | Remote base path |
| `FULLRES_STORAGE_SSH_IDENTITY_FILE` | — | Path to SSH private key |
| `FULLRES_STORAGE_PUBLIC_BASE_URL` | — | Public URL prefix for full-res files |
| `DELIVERY_STAGE_ENABLED` | `true` | Enable delivery encode stage |
| `DELIVERY_ENCODE_HORIZONTAL_LOWRES` | see code | Produce a delivery-grade lowres MP4 |
| `DELIVERY_ENCODE_HQ_PACKAGE` | `false` | Produce full-res HQ package |

---

## 8. Defuddle Integration (added 2026-03-14)

### Why

Manual YouTube captions are faster and more accurate than cloud STT for videos that have them. `defuddle` fetches them without requiring a browser or authentication.

### Key files

| File | Purpose |
|---|---|
| `src/lib/defuddle.ts` | CLI wrapper + markdown parser for defuddle output |
| `src/pipeline/transcribe.ts` | Integration point: tries defuddle before ElevenLabs/Whisper |
| `src/pipeline/ingest.ts` | Writes `metadata.json` with source URL |
| `src/test-defuddle-integration.ts` | Integration test; run with `pnpm exec ts-node-dev src/test-defuddle-integration.ts` |

### How music cues become `audio.events.json`

YouTube captions include lines like `[música]` at moments where music plays. `defuddle.ts`:
1. Parses each timestamped caption line.
2. Checks for music tags using `/\[m[uú]sica\]|\\\[m[uú]sica\\\]/i`.
3. Maps the segment's `start`/`end` to `{ label: "music", start, end }`.
4. Merges adjacent events within < 1s.
5. Writes `audio.events.json` + `audio.events.defuddle.json` in the work dir.

`Stage 3 (analyze/boundaries)` then reads `audio.events.json` as its primary audio signal, so no separate audio analysis pass is needed for YouTube sources.

### Fallback chain

```
YouTube source + captions available  →  defuddle   (fast, free, accurate)
defuddle fails / not YouTube         →  ElevenLabs (cloud STT, best quality)
ElevenLabs fails                     →  faster-whisper small  (local)
faster-whisper small fails           →  faster-whisper base   (local, last resort)
all fail                             →  throw (pipeline error)
```

---

## 9. Test Scripts

All under `apps/worker/src/`, run with `pnpm exec ts-node-dev src/<name>.ts`:

| Script | Purpose |
|---|---|
| `test-ingest.ts` | Test ingestion of a URL/file |
| `test-transcribe.ts` | Test transcription in isolation |
| `test-defuddle-integration.ts` | Test YouTube defuddle fast-path end-to-end |
| `test-analyze.ts` | Test boundary + clip analysis |
| `test-render.ts` | Test rendering |
| `test-e2e.ts` | Full end-to-end pipeline test |
| `monitor.ts` | Watch Supabase for active jobs (TUI) |
