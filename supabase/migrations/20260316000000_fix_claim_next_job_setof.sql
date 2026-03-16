-- Fix claim_next_job to return SETOF public.jobs so it correctly returns an empty set instead of a null row.
drop function if exists public.claim_next_job(text, integer);

create or replace function public.claim_next_job(
  p_worker_id text,
  p_lease_seconds integer default 120
)
returns setof public.jobs
language plpgsql
security definer
set search_path = public
as $$
declare
  v_job public.jobs;
  v_claim_token text := gen_random_uuid()::text;
  v_lease_seconds integer := greatest(30, coalesce(p_lease_seconds, 120));
begin
  update public.jobs j
  set status = 'processing',
      claim_token = v_claim_token,
      claimed_by = nullif(trim(coalesce(p_worker_id, '')), ''),
      claimed_at = now(),
      lease_expires_at = now() + make_interval(secs => v_lease_seconds),
      attempt_count = coalesce(j.attempt_count, 0) + 1
  where j.id = (
    select j2.id
    from public.jobs j2
    where j2.status = 'pending'
       or (
         (j2.status = 'processing' or j2.status like 'processing:%')
         and j2.lease_expires_at is not null
         and j2.lease_expires_at <= now()
       )
    order by j2.created_at asc
    for update skip locked
    limit 1
  )
  returning j.* into v_job;

  if v_job.id is not null then
    return next v_job;
  end if;

  return;
end;
$$;
