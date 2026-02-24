-- Ensure LLM cue telemetry table exists in hosted projects.
-- Safe to run multiple times.

create table if not exists public.llm_cue_events (
  id uuid default gen_random_uuid() primary key,
  created_at timestamp with time zone default now(),
  job_id text,
  run_id text,
  source_pass text not null,
  model text,
  language text,
  denomination text,
  service_date text,
  section_type text,
  cue_kind text not null,
  cue_text text not null,
  cue_time_sec float,
  confidence float,
  metadata jsonb default '{}'::jsonb
);

create index if not exists llm_cue_events_job_idx on public.llm_cue_events(job_id);
create index if not exists llm_cue_events_pass_idx on public.llm_cue_events(source_pass);
create index if not exists llm_cue_events_kind_idx on public.llm_cue_events(cue_kind);
create index if not exists llm_cue_events_created_idx on public.llm_cue_events(created_at desc);
