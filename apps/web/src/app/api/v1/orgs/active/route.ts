import { NextRequest, NextResponse } from 'next/server';
import { getSupabaseServer } from '@/lib/supabase-server';
import { orgContextErrorResponse, requireOrgContext, setActiveOrgCookie } from '@/lib/require-org-context';

export const runtime = 'nodejs';

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export async function POST(request: NextRequest) {
  try {
    const supabaseServer = getSupabaseServer();
    const org = await requireOrgContext(request);
    const body = (await request.json()) as { organization_id?: string };
    const organizationId = String(body?.organization_id || '').trim();

    if (!UUID_RE.test(organizationId)) {
      return NextResponse.json({ error: 'Invalid organization_id' }, { status: 400 });
    }

    const { data, error } = await supabaseServer
      .from('organization_memberships')
      .select('organization_id')
      .eq('user_id', org.user_id)
      .eq('organization_id', organizationId)
      .maybeSingle();

    if (error) {
      return NextResponse.json(
        { error: `Failed to validate org membership: ${error.message}` },
        { status: 500 }
      );
    }

    if (!data) {
      return NextResponse.json({ error: 'Organization membership not found' }, { status: 403 });
    }

    const response = NextResponse.json({ ok: true, organization_id: organizationId });
    setActiveOrgCookie(response, organizationId);
    return response;
  } catch (error) {
    return orgContextErrorResponse(error);
  }
}
