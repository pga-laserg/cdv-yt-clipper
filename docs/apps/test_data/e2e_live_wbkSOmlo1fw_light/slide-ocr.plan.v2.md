# Slide OCR V2 Implementation Plan (from repo research)

## Research Sources Reviewed
- `apps/test_data/slide-ocr-research-repos/slide-extractor`
- `apps/test_data/slide-ocr-research-repos/slide-transition-detector`
- `apps/test_data/slide-ocr-research-repos/PySceneDetect`
- `apps/test_data/slide-ocr-research-repos/TransNetV2_nolfs`

## What We Should Reuse

### 1) From `slide-extractor`
- `pHash`-based near-duplicate filtering (`imagehash.phash`) is simple and effective.
- Keep this for post-OCR event dedupe and re-appearing slide detection.
- Do not copy its frame-skip-only slide detection (too brittle).

### 2) From `slide-transition-detector`
- Histogram/Bhattacharyya-like transition logic and timeline grouping concept are useful.
- Slide grouping/timetable idea is useful for persistent slide IDs with multiple appearances.
- Do not reuse as-is (older architecture, weak transition gate, weak OCR integration).

### 3) From `PySceneDetect`
- Use `AdaptiveDetector` + `ContentDetector` + optional `HashDetector` as the primary shot-boundary engine.
- This gives strong handling of cuts/fades and camera motion robustness.
- Export frame metrics for threshold tuning and audit.

### 4) From `TransNetV2`
- Use as optional high-accuracy shot-boundary lane for difficult streams (gradual transitions).
- Keep optional because model weights are external (LFS/ops dependency).

## Proposed Architecture

### Lane A: Cheap Candidate Generation (low-res source)
- Input: lightweight MP4.
- Run scene/shot boundary detection:
  - Primary: PySceneDetect adaptive/content.
  - Optional: TransNetV2 predictions when enabled.
- Build candidate segments between boundaries.
- Score each segment with:
  - stillness (frame-diff/optical flow magnitude)
  - text-likelihood (edge density + CC stats in slide ROI)
  - no-face penalty (slides tend to have no/low face count)
  - overlay exclusion signal (lower-third likely vs full-screen text)

### Lane B: Candidate Validation + Frame Selection
- Keep segments passing duration + score thresholds.
- Choose representative frame:
  - midpoint first
  - fallback to ±0.5s and ±1.0s if OCR confidence low
  - pick max sharpness among tried frames
- For full-screen slide lane: mask lower 15-25% if lower-third likely.

### Lane C: Heavy OCR (high-res source)
- Input: high-quality MP4 frame snapshots only for selected candidates.
- OCR stack:
  - primary: EasyOCR
  - fallback: Tesseract (language aware)
- Save per-attempt details (frame time, ROI, engine, confidence, text).

### Lane D: Merge + Canonicalization
- Merge adjacent candidates when both are similar by:
  - image similarity (pHash distance / SSIM)
  - text similarity (normalized Levenshtein/Jaccard)
- Produce canonical slide events:
  - `slide_id`, `start`, `end`, `best_text`, `confidence`, `appearances[]`, `evidence[]`

### Lane E: Boundary Integration
- Inject slide cues directly into boundary scoring:
  - boost start candidates around sermon-title / bible-verse slide appearances
  - downscore sermon candidates during persistent lyrics-dominant windows
- Keep cues visible in `analysis.doc.json` and LLM context.

## Data Contracts

### New artifact
- `slide.events.json`
- Minimal schema:
  - `events[]`: `{start,end,type,text,confidence,slide_id,source,appearances,evidence}`
  - `summary`: `{segments_scanned,candidates_selected,ocr_attempts,accepted_events}`

### Optional debug artifacts
- `slide.candidates.json` (pre-OCR candidate list)
- `slide.metrics.csv` (scene detector + stillness/text-likelihood metrics)
- `slide.positive-findings.md` (only high-signal findings)

## Integration Plan (Phased)

### Phase 1 (fast win)
- Implement PySceneDetect-based candidate generator.
- Add segment stillness + text-likelihood scoring.
- Add high-res single-frame OCR + fallback ladder.
- Emit `slide.events.json` and a small summary markdown.

### Phase 2
- Add similarity-based merge (`pHash` + text similarity).
- Add lower-third suppression for full-screen slide lane.
- Wire cues into boundary scorer (boost/downscore).

### Phase 3 (optional advanced)
- Add TransNetV2 lane behind feature flag.
- Add OCR ROI auto-refinement from detected text boxes.
- Add benchmarking harness and A/B tuning profiles.

## Environment/Feature Flags
- `SLIDE_OCR_ENABLED=true`
- `SLIDE_SHOT_DETECTOR=pyscenedetect|transnetv2|hybrid`
- `SLIDE_CANDIDATE_MIN_SEC=2.0`
- `SLIDE_CANDIDATE_TEXT_SCORE_MIN=...`
- `SLIDE_OCR_FALLBACK_FRAMES=mid,minus05,plus05,minus10,plus10`
- `SLIDE_MASK_LOWER_THIRD=true`
- `SLIDE_BOUNDARY_SCORING_ENABLED=true`

## Why this is better than current OCR flow
- Current OCR runs broad and produces noise.
- New flow makes OCR sparse, targeted, and high-confidence.
- It separates detection (cheap) from recognition (expensive), improving both speed and precision.
- It gives deterministic artifacts we can score, debug, and feed to boundary logic directly.

## Repo-Specific Caveats
- `TransNetV2` model weights are not included in this clone (`LFS`/quota issue). Code is available, runtime requires weights.
