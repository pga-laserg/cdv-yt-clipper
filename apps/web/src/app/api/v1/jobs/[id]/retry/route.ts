import { NextRequest, NextResponse } from 'next/server';
import { getSupabaseServer } from '@/lib/supabase-server';
import { orgContextErrorResponse, requireOrgContext } from '@/lib/require-org-context';

export const runtime = 'nodejs';

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function isApiEnabled(): boolean {
  return String(process.env.API_V1_JOBS_ENABLED ?? 'true').toLowerCase() === 'true';
}

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  if (!isApiEnabled()) {
    return NextResponse.json(
      { error: 'API temporarily disabled', code: 'api_disabled' },
      { status: 503 }
    );
  }

  try {
    const supabaseServer = getSupabaseServer();
    const org = await requireOrgContext(request, { requireAdmin: true });
    const { id } = await context.params;

    if (!UUID_RE.test(id)) {
      return NextResponse.json({ error: 'Invalid job id' }, { status: 400 });
    }

    const { data, error } = await supabaseServer
      .from('jobs')
      .update({
        status: 'pending',
        claim_token: null,
        claimed_by: null,
        claimed_at: null,
        lease_expires_at: null,
        last_error: null
      })
      .eq('id', id)
      .eq('organization_id', org.organization_id)
      .select('id, status, organization_id')
      .maybeSingle();

    if (error) {
      return NextResponse.json(
        { error: `Failed to retry job: ${error.message}` },
        { status: 500 }
      );
    }

    if (!data) {
      return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    }

    return NextResponse.json(data);
  } catch (error) {
    return orgContextErrorResponse(error);
  }
}
