-- Bootstrap memberships for existing auth users in default-org.
-- Grants owner role to existing users so API features are immediately usable.

do $$
declare
  v_default_org uuid;
begin
  select id into v_default_org
  from public.organizations
  where slug = 'default-org'
  limit 1;

  if v_default_org is null then
    raise exception 'default organization missing';
  end if;

  insert into public.organization_memberships (organization_id, user_id, role)
  select v_default_org, u.id, 'owner'
  from auth.users u
  where not exists (
    select 1
    from public.organization_memberships om
    where om.organization_id = v_default_org
      and om.user_id = u.id
  );
end $$;
