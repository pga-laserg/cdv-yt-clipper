-- Migration: Add hook_type and score_breakdown columns to clips table
-- Enables virality sub-score storage and hook-type filtering in the admin UI.

ALTER TABLE clips
  ADD COLUMN IF NOT EXISTS hook_type TEXT,
  ADD COLUMN IF NOT EXISTS score_breakdown JSONB;

-- Optional: constrain hook_type to known enum values.
-- Using a CHECK rather than a Postgres enum so we can add values without a migration.
ALTER TABLE clips
  DROP CONSTRAINT IF EXISTS clips_hook_type_check;

ALTER TABLE clips
  ADD CONSTRAINT clips_hook_type_check
  CHECK (
    hook_type IS NULL OR
    hook_type IN ('question', 'promise', 'scripture', 'story', 'contrast', 'none')
  );

COMMENT ON COLUMN clips.hook_type IS
  'Opening hook style of the clip: question, promise, scripture, story, contrast, or none.';

COMMENT ON COLUMN clips.score_breakdown IS
  'JSON object with keys: hook_strength, spiritual_impact, shareability, ending_completeness, model_confidence (all 0-1 floats). Weights: hook_strength=0.20, spiritual_impact=0.30, shareability=0.15, ending_completeness=0.35.';
