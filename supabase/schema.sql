-- Phase 1 SaaS-ready schema reference

create table organizations (
  id uuid default gen_random_uuid() primary key,
  name text not null,
  slug text not null unique,
  created_at timestamp with time zone not null default now()
);

create table organization_memberships (
  id uuid default gen_random_uuid() primary key,
  organization_id uuid not null references organizations(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  role text not null check (role in ('owner', 'admin', 'member')),
  created_at timestamp with time zone not null default now(),
  unique (organization_id, user_id)
);

-- Jobs table to track processing status
create table jobs (
  id uuid default gen_random_uuid() primary key,
  organization_id uuid not null references organizations(id) on delete restrict,
  created_at timestamp with time zone default now(),
  status text not null default 'pending', -- pending, processing:*, completed, failed
  source_url text,
  original_filename text,
  title text,
  video_url text,
  srt_url text,
  sermon_start_seconds float,
  sermon_end_seconds float,
  metadata jsonb default '{}'::jsonb,
  claim_token text,
  claimed_by text,
  claimed_at timestamp with time zone,
  lease_expires_at timestamp with time zone,
  attempt_count integer not null default 0,
  last_error text
);

-- Clips table for generated shorts
create table clips (
  id uuid default gen_random_uuid() primary key,
  organization_id uuid not null references organizations(id) on delete restrict,
  job_id uuid references jobs(id) on delete cascade,
  created_at timestamp with time zone default now(),
  start_seconds float not null,
  end_seconds float not null,
  title text,
  status text not null default 'draft', -- draft, approved, rejected
  transcript_excerpt text,
  video_url text,
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

-- Public posts artifact table consumed by blog renderer/admin
create table posts (
  id text primary key,
  organization_id uuid not null references organizations(id) on delete restrict,
  slug text not null,
  status text not null default 'draft',
  title text not null,
  excerpt text,
  content_markdown text,
  seo_title text,
  seo_description text,
  focus_keyword text,
  tags text[] not null default '{}',
  scripture_refs text[] not null default '{}',
  youtube_url text,
  youtube_video_id text,
  youtube_thumbnail text,
  sermon_date date,
  published_at timestamp with time zone,
  author_name text not null default 'Default Author',
  canonical_url text,
  hero_image_url text,
  created_at timestamp with time zone not null default now(),
  updated_at timestamp with time zone not null default now(),
  unique (organization_id, slug)
);

-- LLM cue telemetry for rule tuning and chapter/boundary analysis
create table llm_cue_events (
  id uuid default gen_random_uuid() primary key,
  organization_id uuid references organizations(id) on delete set null,
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

-- Multi-client blog generation runtime profiles
create table client_blog_profiles (
  organization_id uuid not null references organizations(id) on delete cascade,
  client_id text not null,
  enabled boolean not null default true,
  llm_provider text not null default 'openai',
  llm_model text not null default 'gpt-5-mini',
  prompt_version text not null default 'blog-v1',
  system_prompt text,
  user_prompt_template text,
  default_author_name text not null default 'Default Author',
  default_status text not null default 'draft',
  sync_enabled boolean not null default true,
  sync_endpoint text,
  sync_token text,
  preserve_published_fields boolean not null default true,
  field_rules jsonb not null default '{}'::jsonb,
  created_at timestamp with time zone not null default now(),
  updated_at timestamp with time zone not null default now(),
  primary key (organization_id, client_id)
);

-- Blog generation telemetry events
create table blog_generation_events (
  id uuid default gen_random_uuid() primary key,
  organization_id uuid not null references organizations(id) on delete cascade,
  created_at timestamp with time zone not null default now(),
  job_id text not null,
  client_id text,
  post_id text,
  status text not null,
  stage text not null,
  provider text,
  model text,
  prompt_version text,
  attempt integer not null default 1,
  duration_ms integer,
  error text,
  metadata jsonb not null default '{}'::jsonb
);

-- Publish destinations per client (wordpress, ghost, webhook, ...)
create table client_publish_destinations (
  id uuid default gen_random_uuid() primary key,
  organization_id uuid not null references organizations(id) on delete cascade,
  client_id text not null,
  name text,
  destination_type text not null, -- wordpress, ghost, webhook
  enabled boolean not null default true,
  publish_mode text not null default 'draft', -- draft, publish
  config jsonb not null default '{}'::jsonb,
  field_mapping jsonb not null default '{}'::jsonb,
  created_at timestamp with time zone not null default now(),
  updated_at timestamp with time zone not null default now()
);

-- Publish execution telemetry
create table blog_publish_events (
  id uuid default gen_random_uuid() primary key,
  organization_id uuid not null references organizations(id) on delete cascade,
  created_at timestamp with time zone not null default now(),
  destination_id uuid,
  destination_type text not null,
  job_id text not null,
  client_id text not null,
  post_id text not null,
  status text not null, -- published, failed, skipped
  remote_id text,
  remote_url text,
  attempt integer not null default 1,
  duration_ms integer,
  error text,
  metadata jsonb not null default '{}'::jsonb
);

create index organization_memberships_user_idx on organization_memberships(user_id);
create index organization_memberships_org_idx on organization_memberships(organization_id);

create index jobs_org_created_idx on jobs(organization_id, created_at desc);
create index jobs_claim_queue_idx on jobs(status, lease_expires_at, created_at);
create index clips_org_created_idx on clips(organization_id, created_at desc);
create index posts_org_created_idx on posts(organization_id, created_at desc);

create index llm_cue_events_job_idx on llm_cue_events(job_id);
create index llm_cue_events_pass_idx on llm_cue_events(source_pass);
create index llm_cue_events_kind_idx on llm_cue_events(cue_kind);
create index llm_cue_events_created_idx on llm_cue_events(created_at desc);
create index llm_cue_events_org_idx on llm_cue_events(organization_id, created_at desc);

create index client_blog_profiles_org_idx on client_blog_profiles(organization_id, client_id);
create index blog_generation_events_job_idx on blog_generation_events(job_id);
create index blog_generation_events_status_idx on blog_generation_events(status);
create index blog_generation_events_created_idx on blog_generation_events(created_at desc);
create index blog_generation_events_org_idx on blog_generation_events(organization_id, created_at desc);

create index client_publish_destinations_client_idx on client_publish_destinations(client_id);
create index client_publish_destinations_enabled_idx on client_publish_destinations(enabled);
create index client_publish_destinations_org_idx on client_publish_destinations(organization_id, created_at desc);

create index blog_publish_events_job_idx on blog_publish_events(job_id);
create index blog_publish_events_client_idx on blog_publish_events(client_id);
create index blog_publish_events_status_idx on blog_publish_events(status);
create index blog_publish_events_created_idx on blog_publish_events(created_at desc);
create index blog_publish_events_org_idx on blog_publish_events(organization_id, created_at desc);
