import { NextRequest, NextResponse } from 'next/server';
import { getSupabaseServer } from '@/lib/supabase-server';
import { orgContextErrorResponse, requireOrgContext } from '@/lib/require-org-context';
import type { BlogProfileRecord, CreateBlogProfileRequest } from '@/lib/api-types';

export const runtime = 'nodejs';

function isApiEnabled(): boolean {
  return String(process.env.API_V1_JOBS_ENABLED ?? 'true').toLowerCase() === 'true';
}

function clean(value: unknown): string {
  return String(value ?? '').trim();
}

function isSafeIdentifier(value: string): boolean {
  return /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(value);
}

function toNullableText(value: unknown, max = 4000): string | null {
  const v = clean(value);
  return v ? v.slice(0, max) : null;
}

export async function GET(request: NextRequest) {
  if (!isApiEnabled()) {
    return NextResponse.json({ error: 'API temporarily disabled', code: 'api_disabled' }, { status: 503 });
  }

  try {
    const supabaseServer = getSupabaseServer();
    const org = await requireOrgContext(request);

    const { data, error } = await supabaseServer
      .from('client_blog_profiles')
      .select(
        'organization_id,client_id,enabled,llm_provider,llm_model,prompt_version,system_prompt,user_prompt_template,default_author_name,default_status,sync_enabled,sync_endpoint,preserve_published_fields,field_rules,created_at,updated_at'
      )
      .eq('organization_id', org.organization_id)
      .order('client_id', { ascending: true })
      .returns<BlogProfileRecord[]>();

    if (error) {
      return NextResponse.json({ error: `Failed to list blog profiles: ${error.message}` }, { status: 500 });
    }

    return NextResponse.json({ items: data || [] });
  } catch (error) {
    return orgContextErrorResponse(error);
  }
}

export async function POST(request: NextRequest) {
  if (!isApiEnabled()) {
    return NextResponse.json({ error: 'API temporarily disabled', code: 'api_disabled' }, { status: 503 });
  }

  try {
    const supabaseServer = getSupabaseServer();
    const org = await requireOrgContext(request, { requireAdmin: true });
    const body = (await request.json()) as CreateBlogProfileRequest;

    const clientId = clean(body?.client_id).toLowerCase();
    if (!clientId || !isSafeIdentifier(clientId)) {
      return NextResponse.json({ error: 'client_id is required and must be alphanumeric/underscore/hyphen (max 64)' }, { status: 400 });
    }

    const fieldRules =
      body?.field_rules && typeof body.field_rules === 'object' && !Array.isArray(body.field_rules)
        ? body.field_rules
        : {};

    const payload = {
      organization_id: org.organization_id,
      client_id: clientId,
      enabled: body?.enabled !== false,
      llm_provider: clean(body?.llm_provider || 'openai').toLowerCase().slice(0, 40),
      llm_model: clean(body?.llm_model || process.env.BLOG_OPENAI_MODEL || 'gpt-5-mini').slice(0, 120),
      prompt_version: clean(body?.prompt_version || process.env.BLOG_PROMPT_VERSION || 'blog-v1').slice(0, 60),
      system_prompt: toNullableText(body?.system_prompt, 10_000),
      user_prompt_template: toNullableText(body?.user_prompt_template, 10_000),
      default_author_name: clean(body?.default_author_name || process.env.BLOG_DEFAULT_AUTHOR_NAME || 'Default Author').slice(0, 120),
      default_status: clean(body?.default_status || 'draft').toLowerCase().slice(0, 20),
      sync_enabled: body?.sync_enabled !== false,
      sync_endpoint: toNullableText(body?.sync_endpoint, 1024),
      preserve_published_fields: body?.preserve_published_fields !== false,
      field_rules: fieldRules,
      updated_at: new Date().toISOString()
    };

    const { data, error } = await supabaseServer
      .from('client_blog_profiles')
      .upsert(payload, { onConflict: 'organization_id,client_id' })
      .select(
        'organization_id,client_id,enabled,llm_provider,llm_model,prompt_version,system_prompt,user_prompt_template,default_author_name,default_status,sync_enabled,sync_endpoint,preserve_published_fields,field_rules,created_at,updated_at'
      )
      .single<BlogProfileRecord>();

    if (error) {
      return NextResponse.json({ error: `Failed to save blog profile: ${error.message}` }, { status: 500 });
    }

    return NextResponse.json(data, { status: 201 });
  } catch (error) {
    return orgContextErrorResponse(error);
  }
}

