# Google Cloud Vision OCR Implementation Notes

Date: 2026-02-27
Scope: Document implemented changes for items 1-5 only.

## Summary
This repo now includes targeted hardening for Google Cloud Vision OCR integration and cloud OCR fallback behavior in the worker pipeline.

## Implemented Changes

### 1) Cloud OCR provenance uses actual backend
Cloud enrichment in slide OCR no longer hardcodes `gcv_text_detection` in `ocr_input` tags.

Implemented behavior:
- If backend is GCV, tags look like:
  - `gcv_text_detection:extract`
  - `gcv_text_detection:group_propagation`
- If backend falls back to OpenAI, tags look like:
  - `openai_text_detection:extract`
  - `openai_text_detection:group_propagation`

File:
- `apps/worker/src/pipeline/python/slide_ocr_v2.py`

### 2) Python OCR module probe includes GCV dependency
Worker OCR python environment detection now treats `google.cloud.vision` as a valid OCR module (in addition to EasyOCR/Tesseract).

File:
- `apps/worker/src/pipeline/boundaries.ts`

### 3) Cloud OCR "required" gate is stricter
When cloud text enrichment is required, the boundary flow now enforces useful cloud output:
- cloud enabled
- attempted > 0
- accepted > 0
- applied + propagated > 0

This prevents success on no-op cloud runs.

File:
- `apps/worker/src/pipeline/boundaries.ts`

### 4) OCR runtime errors are visible and persisted
Runtime OCR failures for GCV/OpenAI are no longer silently swallowed.

Implemented behavior:
- Runtime errors are logged to stderr with `[ocr-events] ...` prefix.
- Runtime errors are deduplicated and appended to `errors` in OCR output JSON.
- For return-meta paths, minimal provider/method/error metadata is returned.

File:
- `apps/worker/src/pipeline/python/ocr_events.py`

### 5) Added dedicated GCV tests
Added focused tests to cover engine fallback and cloud provenance behavior.

New tests:
- `apps/worker/src/test-gcv-ocr-engine.ts`
- `apps/worker/src/test-slide-ocr-cloud-provenance.ts`

NPM scripts added:
- `test:gcv-ocr-engine`
- `test:slide-ocr-cloud-provenance`

File:
- `apps/worker/package.json`

## Validation Commands
Run from `apps/worker`:

```bash
npm run test:gcv-ocr-engine
npm run test:slide-ocr-cloud-provenance
```

Expected outcome:
- both commands exit `0`
- engine test prints PASS for fallback scenarios
- provenance test shows `ocr_input` values using `openai_text_detection:*` when OpenAI fallback is active

## Touched Files
- `apps/worker/src/pipeline/python/slide_ocr_v2.py`
- `apps/worker/src/pipeline/boundaries.ts`
- `apps/worker/src/pipeline/python/ocr_events.py`
- `apps/worker/src/test-gcv-ocr-engine.ts`
- `apps/worker/src/test-slide-ocr-cloud-provenance.ts`
- `apps/worker/package.json`
