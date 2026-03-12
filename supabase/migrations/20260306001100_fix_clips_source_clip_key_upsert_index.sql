-- Ensure upsert on (organization_id, job_id, source_clip_key) can use a unique index.
-- Partial unique indexes are not inferred by ON CONFLICT without matching predicate.
drop index if exists public.clips_draft_source_clip_key_unique;

create unique index if not exists clips_org_job_source_clip_key_unique
    on public.clips (organization_id, job_id, source_clip_key);
