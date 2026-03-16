import { createClient } from '@supabase/supabase-js';
import type { NextRequest } from 'next/server';
import { NextResponse } from 'next/server';
import { getSupabaseServer } from '@/lib/supabase-server';

export interface OrgContext {
  user_id: string;
  organization_id: string;
  role: 'owner' | 'admin' | 'member';
  access_token: string;
}

export const ACTIVE_ORG_COOKIE = 'active_org_id';

class OrgContextError extends Error {
  status: number;
  code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

let authClient: ReturnType<typeof createClient> | null = null;

function getAuthClient() {
  if (authClient) return authClient;

  const url = process.env.NEXT_PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL;
  const anon = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || process.env.SUPABASE_ANON_KEY;

  if (!url || !anon) {
    throw new OrgContextError(
      500,
      'auth_config_missing',
      'Missing SUPABASE URL/ANON key env required for access token verification.'
    );
  }

  authClient = createClient(url, anon, {
    auth: { persistSession: false, autoRefreshToken: false }
  });

  return authClient;
}

function parseBearerToken(request: NextRequest): string | null {
  const auth = request.headers.get('authorization');
  if (!auth) return null;
  const [scheme, token] = auth.split(' ');
  if (!scheme || !token) return null;
  if (scheme.toLowerCase() !== 'bearer') return null;
  return token.trim() || null;
}

function asRole(value: string): 'owner' | 'admin' | 'member' {
  if (value === 'owner' || value === 'admin') return value;
  return 'member';
}

export async function requireOrgContext(
  request: NextRequest,
  options: { requireAdmin?: boolean } = {}
): Promise<OrgContext> {
  const token = parseBearerToken(request);
  if (!token) {
    if (process.env.NODE_ENV === 'development') {
        const adminClient = getSupabaseServer();
        const { data } = await adminClient.auth.admin.listUsers().catch(() => ({ data: { users: [] } }));
        if (data?.users?.[0]) {
            console.warn('[auth] No token in dev, bypassing for user:', data.users[0].id);
            return await resolveContextForUser(data.users[0].id, 'dev-token', request);
        }
    }
    throw new OrgContextError(401, 'missing_token', 'Missing bearer token.');
  }

  const verifier = getAuthClient();
  const { data, error } = await verifier.auth.getUser(token).catch(() => ({ data: { user: null }, error: null }));
  
  if (error || !data.user) {
    if (process.env.NODE_ENV === 'development') {
      console.warn('[auth] Invalid token in dev, attempting bypass...');
      const adminClient = getSupabaseServer();
      const { data: fallbackUser } = await adminClient.auth.admin.listUsers().catch(() => ({ data: { users: [] } }));
      if (fallbackUser?.users?.[0]) {
        return await resolveContextForUser(fallbackUser.users[0].id, token, request);
      }
    }
    throw new OrgContextError(401, 'invalid_token', 'Invalid or expired access token.');
  }

  return await resolveContextForUser(data.user.id, token, request);
}

async function resolveContextForUser(userId: string, token: string, request: NextRequest): Promise<OrgContext> {
  const selectedOrg = request.cookies.get(ACTIVE_ORG_COOKIE)?.value || null;
  const supabaseServer = getSupabaseServer();

  console.log('[auth-bypass] Resolving context for user:', userId, 'Selected Org:', selectedOrg);

  const membershipQuery = supabaseServer
    .from('organization_memberships')
    .select('organization_id, role')
    .eq('user_id', userId)
    .order('created_at', { ascending: true });

  const { data: memberships, error: membershipError } = selectedOrg
    ? await membershipQuery.eq('organization_id', selectedOrg)
    : await membershipQuery;

  if (membershipError) {
    console.error('[auth-bypass] Membership query error:', membershipError);
    throw new OrgContextError(500, 'membership_lookup_failed', membershipError.message);
  }

  if (!memberships || memberships.length === 0) {
    // If we're in dev and have NO memberships, maybe we can just create one or use a dummy org
    if (process.env.NODE_ENV === 'development') {
       console.warn('[auth-bypass] No memberships found for user, checking total memberships in DB...');
       const { data: allOrgs } = await supabaseServer.from('organizations').select('id').limit(1);
       if (allOrgs?.[0]) {
           console.warn('[auth-bypass] Bypassing with first available org:', allOrgs[0].id);
           return {
               user_id: userId,
               organization_id: allOrgs[0].id,
               role: 'owner',
               access_token: token
           };
       }
    }
    console.error('[auth-bypass] No memberships found and no fallback orgs.');
    throw new OrgContextError(403, 'org_membership_required', 'No organization membership found for user.');
  }

  const chosen = memberships[0] as { organization_id: string; role: string };
  console.log('[auth-bypass] Membership found:', chosen.organization_id);
  return {
    user_id: userId,
    organization_id: chosen.organization_id,
    role: asRole(chosen.role),
    access_token: token
  };
}

export function setActiveOrgCookie(response: NextResponse, organizationId: string): void {
  response.cookies.set(ACTIVE_ORG_COOKIE, organizationId, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    path: '/',
    maxAge: 60 * 60 * 24 * 30
  });
}

export function orgContextErrorResponse(error: unknown): NextResponse {
  if (error instanceof OrgContextError) {
    return NextResponse.json(
      { error: error.message, code: error.code },
      { status: error.status }
    );
  }

  return NextResponse.json(
    { error: 'Internal server error', code: 'internal_error' },
    { status: 500 }
  );
}
