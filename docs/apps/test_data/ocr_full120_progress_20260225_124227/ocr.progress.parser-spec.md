# OCR Progress Parser Spec

Use stderr lines prefixed with:
- `[ocr-events-progress] `

Regex:
```regex
^\[ocr-events-progress\]\s+(\{.*\})$
```

Parse group 1 as JSON.

## Event Type
- `type`: always `"ocr_progress"`

## Fields
- `percent` (number): completion percentage [0..100]
- `sampled` (int): processed samples count
- `total_samples` (int): expected samples
- `video_time_sec` (number|null): current sampled video timestamp
- `video_duration_sec` (number|null): max OCR video duration for this run
- `observations` (int): raw OCR observations before merge
- `scene_cuts` (int): detected scene cuts so far
- `ocr_calls` (int): OCR inference calls executed so far
- `roi_candidates` (int): total ROI candidates seen so far
- `elapsed_sec` (number|null): wall-clock elapsed seconds
- `eta_sec` (number|null): estimated time remaining in seconds
- `samples_per_sec` (number): throughput in samples/s
- `frames_per_sec` (number): decoded/processed frames per second
- `video_speed_x` (number): video-seconds processed per wall-second
- `targeted` (bool): whether targeted OCR mode was enabled

## UI Guidance
- Progress bar: `percent`
- Time labels: `elapsed_sec`, `eta_sec`
- Throughput widget: `video_speed_x`, `samples_per_sec`
- OCR load widget: `ocr_calls / roi_candidates`
- Completion: treat `percent >= 100` as done and wait for final process exit code.

## Example
```json
{
  "type": "ocr_progress",
  "percent": 66.667,
  "sampled": 8,
  "total_samples": 12,
  "video_time_sec": 70.0,
  "video_duration_sec": 120.0,
  "observations": 3,
  "scene_cuts": 2,
  "ocr_calls": 13,
  "roi_candidates": 24,
  "elapsed_sec": 97.31,
  "eta_sec": 48.65,
  "samples_per_sec": 0.0822,
  "frames_per_sec": 24.71,
  "video_speed_x": 0.7193,
  "targeted": true
}
```
