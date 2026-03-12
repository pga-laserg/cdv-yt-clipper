-- Add deterministic key for idempotent draft clip upserts.
alter table if exists public.clips
    add column if not exists source_clip_key text;

-- Generated draft clips are unique per org/job/source key.
create unique index if not exists clips_draft_source_clip_key_unique
    on public.clips (organization_id, job_id, source_clip_key)
    where status = 'draft' and source_clip_key is not null;

create index if not exists clips_org_job_source_clip_key_idx
    on public.clips (organization_id, job_id, source_clip_key);
