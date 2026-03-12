create table if not exists public.trim_jobs (
  id uuid primary key default gen_random_uuid(),
  clip_id uuid not null references public.clips(id) on delete cascade,
  organization_id uuid not null references public.organizations(id) on delete restrict,
  requested_start_sec numeric not null,
  requested_end_sec numeric not null,
  requested_duration_sec numeric generated always as (requested_end_sec - requested_start_sec) stored,
  status text not null default 'queued', -- queued, processing, completed, failed
  output_bucket text,
  output_path text,
  error_message text,
  created_by uuid,
  created_at timestamptz not null default now(),
  started_at timestamptz,
  completed_at timestamptz
);

create index if not exists trim_jobs_org_idx on public.trim_jobs(organization_id, created_at desc);
create index if not exists trim_jobs_clip_idx on public.trim_jobs(clip_id);
