# Current Pipeline Review (Ingestion -> Boundary (Audio + Slide Signals) -> Media + Packaging)

## Scope
This document describes the current pipeline as implemented in:
- `apps/worker/src/pipeline/ingest.ts`
- `apps/worker/src/pipeline/boundaries.ts`
- `apps/worker/src/pipeline/python/slide_ocr_v2.py`
- `apps/worker/src/pipeline/render.ts`

It reflects the current workflow decision:
- boundary scoring uses audio events plus Slide OCR v2 events,
- legacy OCR events (`ocr.events*.json`) are deprecated and disabled in boundary/analysis flows,
- Slide OCR v2 also feeds post-boundary presentation packaging/extraction,
- media generation/extraction runs after boundary selection.
- recent iteration focus has been `slide.events.json` quality + presentation packaging.

### Canonical OCR policy (recommended for smooth runs)
- Keep only one boundary OCR signal family active: Slide OCR v2 (`slide.events.json`) + audio events.
- Disable legacy OCR events in boundary and analysis stages to avoid redundant OCR runtime.
- Use cloud text enrichment inside Slide OCR v2 as the single high-accuracy OCR enrich step.

## 1. Workflow (Execution Order)
1. Ingestion
   - Acquire best source, normalize to `source.mp4`, create `source.light.mp4`, extract audio.
2. Boundary
   - Run multimodal boundary adjudication using face/transcript/diarization plus audio + slide signals.
   - Produce final sermon bounds in `sermon.boundaries.targeted-diarization.json`.
3. Media Generation/Extraction
   - Render final sermon and highlights.
   - Produce extraction/package artifacts from Slide OCR v2 events (with duplicate-aware packaging).

## 2. Detailed Stages

### 2.1 Ingestion Stage (Media Preparation + Tools)
1. Download/acquire original source media.
   - Tools used: `apps/worker/src/pipeline/ingest.ts`, `yt-dlp` (multi-strategy format attempts), optional cookies-from-browser, `ffprobe` quality checks.
2. Keep highest-available original artifact.
   - Output: `source.original.*` (container may vary by source/availability).
3. Normalize to HQ processing source.
   - Tools used: `ffmpeg` (via `fluent-ffmpeg` wrapper in TS).
   - Output: `source.mp4` (HQ normalized mp4).
4. Build lightweight processing source.
   - Tools used: `ffmpeg`.
   - Output: `source.light.mp4` (faster passes that do not need full HQ).
5. Extract audio for transcription/diarization/event passes.
   - Tools used: `ffmpeg`.
   - Output: `audio.wav`.

### 2.2 Inputs
- Audio source: `source.wav`/equivalent service audio.
- Video source (dual path):
  - `source.mp4` (preferred HQ for OCR frame extraction)
  - `source.light.mp4` (lighter pass when useful)
- Transcript/analysis artifacts when available:
  - `transcript.json`, `analysis.doc.json`, `analysis.chapters.llm.json`, etc.

### 2.3 Boundary Pipeline (Single Pipeline + Profiles)
There is one boundary pipeline implementation in `apps/worker/src/pipeline/boundaries.ts`, with profile modes:
- `BOUNDARY_PIPELINE_PROFILE=standard`
- `BOUNDARY_PIPELINE_PROFILE=light`

`light` is the runtime default profile when `BOUNDARY_PIPELINE_PROFILE` is unset.
`standard` is the prior heavier profile (same pipeline, slower/more robust settings).

#### 2.3.1 Boundary Steps + Tools
1. Face pass predicts coarse sermon start/end and candidate ranges.
   - Tools used: `apps/worker/src/pipeline/python/face_sermon_bounds.py`, InsightFace (ArcFace embeddings), OpenCV (`cv2`), NumPy, ONNX Runtime.
2. Transcript cues and rule-based cues add candidate start/end points.
   - Tools used: `apps/worker/src/pipeline/boundaries.ts` transcript heuristics and regex cues, normalized transcript artifacts (`transcript*.json`, `analysis.doc.json`, `analysis.chapters.llm.json`).
3. Targeted diarization windows (pre/post) refine boundaries around speaker transitions.
   - Tools used: `apps/worker/src/pipeline/python/diarize.py`, pyannote `speaker-diarization-3.1` (PyTorch + Hugging Face model/token).
4. Optional signal passes add boundary evidence (current config):
   - Audio events (`audio.events.json`)
     - Tools used: `apps/worker/src/pipeline/python/audio_events.py`, SoundFile, NumPy, SciPy DSP (RMS/ZCR/centroid/flatness/F0 heuristics).
   - Slide OCR v2 events (`slide.events.json`)
     - Tools used: `apps/worker/src/pipeline/python/slide_ocr_v2.py` (OpenCV sampling + text/motion/stillness scoring + OCR + merge + duplicate annotations), consumed by boundary scoring in `apps/worker/src/pipeline/boundaries.ts` (`boundarySlideSignal`).
   - Legacy OCR events (`ocr.events*.json`) from `ocr_events.py`/`ocr_v2.py`/`ocr_v3.py` are deprecated and ignored in standard runs.
5. Candidate scoring combines diarization + transcript + audio + slide signals.
   - Tools used: scoring/adjudication logic in `apps/worker/src/pipeline/boundaries.ts` (`boundaryAudioSignal`, `boundarySlideSignal`, `boundaryTranscriptSignal` + weighted score composition).
6. Final clip bounds are selected and written to `sermon.boundaries.targeted-diarization.json`.
   - Tools used: refinement + boundary adjudication logic in `apps/worker/src/pipeline/boundaries.ts`.

### 2.4 Slide OCR v2 Pipeline (Boundary Evidence + Post-Boundary Extraction)
Slide OCR v2 outputs are used in two places:
- boundary evidence (`boundarySlideSignal`) during boundary scoring, and
- post-boundary presentation packaging/extraction artifacts.

1. Sample frames over time (`--sample-sec`, `--max-sec`) + scene boundaries (fallback + PySceneDetect if enabled).
   - Tools used: OpenCV `VideoCapture` sampling; fallback histogram/Bhattacharyya scene cuts in `ocr_events.detect_scene_cut`; optional PySceneDetect (content/adaptive detectors).
2. Build time segments and compute segment metrics:
   - `avg_text_score`
   - `avg_motion_score`
   - `stillness_score`
   - combined `score`
   - Tools used: OpenCV + NumPy feature extraction (`estimate_text_likelihood`, edge/CC/contrast/line-structure heuristics).
3. Candidate selection with hard gates:
   - min/max duration
   - min text score
   - min stillness
   - min combined score
   - Tools used: threshold gates in `slide_ocr_v2.py` (env/CLI tunables).
4. Text-dominant override (newly active):
   - allows low-stillness segments when text/score are strong.
   - Tools used: override logic in `slide_ocr_v2.py` (`selected_by_text_override`).
5. Budgeted OCR pass:
   - diversity buckets
   - local rescue near selected anchors
   - Tools used: priority scoring + bucket diversity + rescue selector in `slide_ocr_v2.py`.
6. OCR extraction per candidate:
   - EasyOCR primary, Tesseract fallback (if enabled)
   - hard text quality gate
   - Tools used: `run_ocr` from `ocr_events.py`, OpenCV ROI preprocessing, EasyOCR runtime, Tesseract via `pytesseract`.
7. Event merge across nearby similar segments:
   - text similarity + hash distance + merge gap.
   - Tools used: Python `SequenceMatcher`, perceptual hash distance (`avg_hash` + Hamming).
8. Optional fullscreen strict post-filter.
   - Tools used: lexical/quality strict gate + strict rescue rules in `slide_ocr_v2.py`.
9. Global duplicate pass (integrated):
   - runs across the full timeline after final slide events are built.
- Matching features:
  - robust pHash (`frame_phash`) similarity
  - OCR text similarity
  - optional same-type constraint
- Matching rules:
  - strict visual match (very small pHash distance), OR
  - near visual match + high text similarity.
- Tools used: OpenCV DCT-based pHash (`phash64`), Python `SequenceMatcher` text similarity, Hamming distance clustering logic in `slide_ocr_v2.py`.

#### 2.4.1 Important behavior
- **No event is removed from `events`.**
- Events are annotated for packaging/extraction only:
  - `presentation_group_id`
  - `presentation_is_representative`
  - `presentation_duplicate_of`
  - `presentation_match`
  - `extract_for_package`

#### 2.4.2 Output additions in `slide.events.json`
- Top-level `presentation` block with:
  - group counts
  - representative counts
  - duplicate counts
  - group membership map
- `summary` fields:
  - `extract_unique_events`
  - `extract_duplicates`

### 2.5 Pipeline Wiring (TypeScript Layer)
When slide OCR is enabled in the boundary run:
- `slide.events.json` is loaded/generated as before.
- A new artifact is generated automatically:
  - `slide.presentation.package.json`
- Package includes:
  - `events_all` (full timeline)
  - `events_extract` (representatives only)
  - group data and extraction IDs.

Boundary output metadata now also includes:
- `unique_extract_events`
- `duplicates_for_extract`
- `presentation_groups`
- `package_path`
- Tools used: `apps/worker/src/pipeline/boundaries.ts` (`runSlideOcrPass`, `writeSlidePresentationPackage`).

### 2.6 Post-Boundary Media Generation/Extraction (Current Workflow)
This stage runs after final sermon boundary is decided.

1. Render clipped sermon horizontal HQ.
   - Output: `processed/sermon_horizontal.mp4`
   - Tools used: `apps/worker/src/pipeline/render.ts` -> `cutVideo(...)` -> `ffmpeg` (`libx264`, AAC).
2. Generate upload-safe horizontal variant (target: under 50MB for Supabase limits).
   - Output: `*.upload-compressed.mp4` (created when upload of HQ fails for size; target is a sub-50MB deliverable variant)
   - Tools used: `apps/worker/src/index.ts` -> `uploadAssetWithFallback(...)` -> `transcodeForUpload(...)` -> `ffmpeg`.
3. Render 5 vertical clips (top scored highlights).
   - Output: `processed/<clip_id>.mp4` (+ per-clip path debug JSON, + render summary)
   - Tools used: `apps/worker/src/pipeline/render.ts`, `apps/worker/src/pipeline/python/autocrop.py`, `ffmpeg`.
4. Extract highest-resolution still frames from full video (for review/package/UI).
   - Source preference: `source.mp4` (HQ).
   - Tools used: slide OCR event timestamps + `ffmpeg` frame extraction workflow.
5. Presentation package and duplicate-aware extraction list.
   - Output: `slide.presentation.package.json` + optional frame index/review docs.
   - Tools used: `slide_ocr_v2.py` duplicate annotations + packaging in `boundaries.ts`.

### 2.7 Main Artifacts (Current)
- `sermon.boundaries.face-pass.json`
- `sermon.boundaries.targeted-diarization.json`
- `audio.events.json` (if enabled)
- `slide.events.json` (slide OCR v2 events used for boundary evidence + packaging)
- `slide.presentation.package.json` (post-boundary package)
- `processed/sermon_horizontal.mp4` (HQ sermon clip)
- `processed/highlights_vertical.v3.render-summary.json` + `processed/<clip_id>.mp4` (vertical outputs)
- `*.upload-compressed.mp4` (when upload fallback is triggered)

### 2.8 Extraction Review Workflow (Post-run)
For visual QA, optional review artifacts can be generated:
- All frames index: `slide.frames.all.index.md`
- Unique extraction frames: `slide.frames.unique/`
- Duplicate pair review:
  - `slide.duplicates.extraction.review.md`
  - `slide.frames.duplicates.review/`

### 2.9 Practical Result
- Timeline keeps full evidence (all events).
- Extraction set is deduplicated for UI/package efficiency.
- Duplicate handling supports re-appearing slides even when separated by minutes.

## 3. OCR Redundancy Map (Current Code)
Potential OCR execution points:
1. Boundary legacy OCR events pass (`ocr.events*.json`)
   - Status: deprecated/disabled (ignored, warning only)
2. Boundary Slide OCR v2 pass (`slide.events.json`)
   - Trigger: `BOUNDARY_ENABLE_SLIDE_OCR_SIGNALS=true`
   - Consumer: `boundarySlideSignal` in `boundaries.ts`
3. Analysis OCR events pass (artifact generation)
   - Status: deprecated/disabled (ignored, warning only)

### 3.1 Non-redundant run profile (recommended)
Set:
- `BOUNDARY_ENABLE_SLIDE_OCR_SIGNALS=true`
- `BOUNDARY_ENABLE_OCR_SIGNALS` ignored (legacy deprecated)
- `ANALYSIS_ENABLE_OCR_SIGNALS` ignored (legacy deprecated)

Result:
- Boundary uses audio + slide OCR only.
- No extra legacy OCR pass in boundary stage.
- No duplicate OCR pass in analysis stage.

### 3.2 Deprecation guidance to reduce confusion
Deprecate in workflow docs/runbooks:
- “Boundary + analysis OCR-events by default”.
- “Legacy OCR events as primary boundary signal”.

Keep as optional debugging-only tools:
- `BOUNDARY_ENABLE_OCR_SIGNALS=true` only for diagnostics/A-B.
- `ANALYSIS_ENABLE_OCR_SIGNALS=true` only when explicitly auditing OCR-event quality.
