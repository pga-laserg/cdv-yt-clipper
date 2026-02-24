-- Jobs table to track processing status
create table jobs (
  id uuid default gen_random_uuid() primary key,
  created_at timestamp with time zone default now(),
  status text not null default 'pending', -- pending, processing, completed, failed
  youtube_url text,
  original_filename text,
  sermon_start_seconds float,
  sermon_end_seconds float,
  metadata jsonb default '{}'::jsonb
);

-- Clips table for generated shorts
create table clips (
  id uuid default gen_random_uuid() primary key,
  job_id uuid references jobs(id) on delete cascade,
  created_at timestamp with time zone default now(),
  start_seconds float not null,
  end_seconds float not null,
  title text,
  status text not null default 'draft', -- draft, approved, rejected
  transcript_excerpt text,
  video_path text,
  confidence_score float,
  notes text
);

-- Logs for debugging pipeline steps
create table pipeline_logs (
  id uuid default gen_random_uuid() primary key,
  job_id uuid references jobs(id) on delete cascade,
  created_at timestamp with time zone default now(),
  level text not null, -- info, warn, error
  message text not null,
  metadata jsonb
);

-- LLM cue telemetry for rule tuning and chapter/boundary analysis
create table llm_cue_events (
  id uuid default gen_random_uuid() primary key,
  created_at timestamp with time zone default now(),
  job_id text,
  run_id text,
  source_pass text not null, -- boundary_stage1_coarse, analysis_chapters_stage1, ...
  model text,
  language text,
  denomination text,
  service_date text,
  section_type text,
  cue_kind text not null, -- invite_pray, chapter_label, boundary_refined, ...
  cue_text text not null,
  cue_time_sec float,
  confidence float,
  metadata jsonb default '{}'::jsonb
);

create index llm_cue_events_job_idx on llm_cue_events(job_id);
create index llm_cue_events_pass_idx on llm_cue_events(source_pass);
create index llm_cue_events_kind_idx on llm_cue_events(cue_kind);
create index llm_cue_events_created_idx on llm_cue_events(created_at desc);
