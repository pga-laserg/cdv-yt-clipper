-- Add broll_cues JSONB column to store B-roll cue points per clip.
-- Array of {offset_sec, keywords, description} objects.
-- Only populated when HIGHLIGHTS_BROLL_CUES_ENABLED=true in worker env.
ALTER TABLE clips
  ADD COLUMN IF NOT EXISTS broll_cues JSONB;

COMMENT ON COLUMN clips.broll_cues IS
  'Array of B-roll cue points: [{offset_sec: number, keywords: string[], description: string}]. '
  'Populated only when HIGHLIGHTS_BROLL_CUES_ENABLED=true. offset_sec is relative to clip start.';
