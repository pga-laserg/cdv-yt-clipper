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
