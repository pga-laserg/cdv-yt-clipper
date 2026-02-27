# Sermon Pipeline Capabilities and Gaps

Context: `apps/test_data/e2e_live_wbkSOmlo1fw_light`

## Current state summary

The pipeline is now multimodal and no longer transcript-only. It currently combines:

- Transcript + corrected/polished chapters (LLM-assisted).
- Face-based cues (InsightFace where available; fallback paths exist).
- Targeted diarization around sermon boundary candidates (local + optional cloud fallback).
- Audio-event cues.
- OCR cues (lower-third/slide/full-frame) with scene-cut signals.
- Boundary scoring that directly consumes OCR signals (not only LLM context).

Recent ingest improvement also matters for OCR quality:

- YouTube ingest now uses quality-gated download logic with runtime/cookies support and dual-source outputs (original HQ, normalized HQ mp4, lightweight mp4, wav).

---

## Requested capability list: updated status

### 1) Shot boundaries (PySceneDetect) -> candidate chapter breaks
- Status: **Partially implemented**
- Implemented now:
  - Scene-cut detection exists via histogram/Bhattacharyya diff in:
    - `apps/worker/src/pipeline/python/autocrop.py`
    - `apps/worker/src/pipeline/python/ocr_events.py`
  - Scene-cut information is used in:
    - Vertical clipper reset behavior.
    - OCR segment metadata (`near_scene_cut`).
    - Boundary scoring as a confidence feature.
- Gap:
  - Not using PySceneDetect specifically.
  - Not yet a dedicated global chapter-break generator stage.

### 2) Lower-third OCR (Tesseract/EasyOCR) -> speaker labeling
- Status: **Partially implemented (functional)**
- Implemented now:
  - OCR pass exists and is wired:
    - `apps/worker/src/pipeline/python/ocr_events.py`
    - `apps/worker/src/pipeline/analysis-doc.ts`
    - `apps/worker/src/pipeline/boundaries.ts`
  - Extracted text is classified into cue types including `speaker_name`.
  - OCR events are persisted (`ocr.events.json`) and used by boundary scoring.
- Gap:
  - `easyocr` dependency is not installed in this environment; active backend is mostly `tesseract`.
  - Name normalization/entity resolution is still basic (no robust canonical speaker registry).

### 3) Slide text OCR -> real chapter title extraction
- Status: **Partially implemented**
- Implemented now:
  - OCR scans explicit `slide` region and full frame.
  - OCR text classification includes cues such as:
    - sermon title-like text
    - bible verse cues
    - lyric cues
  - These cues feed both analysis-doc context and boundary scoring.
- Gap:
  - No dedicated slide-title extraction model or structured title timeline yet.
  - Still heuristic+LLM, not deterministic chapter-title extraction.

### 4) Diarization (pyannote) + face tracking -> refined speaker segments
- Status: **Partially implemented**
- Implemented now:
  - Local diarization in Python.
  - Targeted diarization windows around start/end candidates.
  - Face pass integrated in boundary candidate generation.
  - Optional cloud diarization path available as fallback.
- Gap:
  - No fully fused full-service speaker timeline combining face + diarization end-to-end.
  - Reconciliation logic is still boundary-focused, not global speaker graph optimization.

### 5) Topic segmentation (BERTopic) -> labeled chapter hints
- Status: **Not implemented**
- Gap:
  - No BERTopic/topic-embedding segmentation stage in worker pipeline.
  - Chapter semantics still depend on transcript+LLM+signals, not dedicated topic clustering.

### 6) (Optional) Source separation -> cleaner speaker embeddings
- Status: **Not implemented**
- Gap:
  - No Demucs/separation preprocessing before diarization or cue extraction.
  - Worship-heavy overlap cases still depend on current diarization robustness.

---

## Additional changes now in place (important)

### Ingest quality hardening (new)
- Implemented in `apps/worker/src/pipeline/ingest.ts`:
  - Multi-attempt yt-dlp strategy with quality gate.
  - Resolution validation via ffprobe before accepting download.
  - Runtime support (`--js-runtimes`) and optional `--cookies-from-browser`.
  - Env controls:
    - `YTDLP_MIN_PREFERRED_HEIGHT`
    - `YTDLP_ALLOW_LOW_QUALITY_FALLBACK`
    - `YTDLP_COOKIES_FROM_BROWSER`
    - `YTDLP_ENABLE_ANDROID_FALLBACK`
- Benefit:
  - Avoids silently accepting low-res 360p sources when better formats are available.
  - Improves OCR and visual analysis reliability.

### OCR integrated into boundary scoring (new)
- Implemented in `apps/worker/src/pipeline/boundaries.ts`:
  - OCR is not only LLM context anymore.
  - Start/end candidate scores now include OCR-derived transitions:
    - lyric->sermon
    - sermon->lyric
    - speaker-name proximity
    - scene-cut proximity
  - Helps penalize false sermon spans dominated by lyrics/worship overlays.

---

## Practical impact on workflow

- Sermon boundaries:
  - More robust than previous transcript-only and face-only passes.
  - Better handling of lower-thirds/lyrics transitions.
- Chapter quality:
  - OCR cues now provide concrete on-screen anchors.
  - Still dependent on LLM quality and cue interpretation for final labels.
- Performance tradeoff:
  - High-quality AV1 sources greatly improve signal quality but can slow OCR passes.
  - For iterative dev, using HQ H.264 normalized source for OCR is preferred.

---

## Recommended next steps (coherent roadmap)

1. Finalize OCR backend reliability:
   - Install/validate EasyOCR in worker env and benchmark against Tesseract.
   - Add canonical speaker-name post-processing.

2. Promote scene-cuts to first-class chapter candidates:
   - Keep current detector, optionally evaluate PySceneDetect later.

3. Build full-service speaker timeline fusion:
   - Unify face-pass identity tracks + diarization segments globally, not only near boundaries.

4. Add semantic topic segmentation (BERTopic or equivalent):
   - Use as independent chapter prior, then fuse with OCR/audio/face signals.

5. Keep source separation optional:
   - Enable only for difficult worship-overlap profiles to control cost/latency.

