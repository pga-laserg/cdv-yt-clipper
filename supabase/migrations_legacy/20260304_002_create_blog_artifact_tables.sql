-- Blog artifact generation support tables.
-- Safe to run multiple times.

create table if not exists public.client_blog_profiles (
  client_id text primary key,
  enabled boolean not null default true,
  llm_provider text not null default 'openai',
  llm_model text not null default 'gpt-5-mini',
  prompt_version text not null default 'blog-v1',
  system_prompt text,
  user_prompt_template text,
  default_author_name text not null default 'Daniel Orellana',
  default_status text not null default 'draft',
  sync_enabled boolean not null default true,
  sync_endpoint text,
  sync_token text,
  preserve_published_fields boolean not null default true,
  field_rules jsonb not null default '{}'::jsonb,
  created_at timestamp with time zone not null default now(),
  updated_at timestamp with time zone not null default now()
);

insert into public.client_blog_profiles (client_id)
values ('default')
on conflict (client_id) do nothing;

create table if not exists public.blog_generation_events (
  id uuid default gen_random_uuid() primary key,
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

create index if not exists blog_generation_events_job_idx on public.blog_generation_events(job_id);
create index if not exists blog_generation_events_status_idx on public.blog_generation_events(status);
create index if not exists blog_generation_events_created_idx on public.blog_generation_events(created_at desc);
