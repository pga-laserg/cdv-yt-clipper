import crypto from 'crypto';
import { supabase } from '../lib/supabase';
import type { BlogClientProfile, BlogPostDraftRow } from './blog-artifact';

export type PublishDestinationType = 'wordpress' | 'ghost' | 'webhook';

interface PublishDestinationRow {
    id: string;
    organization_id: string;
    client_id: string;
    name: string | null;
    destination_type: PublishDestinationType;
    enabled: boolean;
    publish_mode: 'draft' | 'publish';
    config: Record<string, unknown>;
    field_mapping: Record<string, unknown>;
}

interface PublishDestinationRecordRaw {
    id: string;
    organization_id?: string | null;
    client_id: string;
    name?: string | null;
    destination_type?: string | null;
    enabled?: boolean | null;
    publish_mode?: string | null;
    config?: Record<string, unknown> | null;
    field_mapping?: Record<string, unknown> | null;
}

interface PublishEventPayload {
    organization_id: string;
    destination_id: string;
    destination_type: string;
    job_id: string;
    client_id: string;
    post_id: string;
    status: string;
    remote_id: string | null;
    remote_url: string | null;
    attempt: number;
    duration_ms: number;
    error: string | null;
    metadata: Record<string, unknown>;
}

export interface PublishResult {
    destination_id: string;
    destination_type: string;
    destination_name: string | null;
    status: 'published' | 'failed' | 'skipped';
    attempts: number;
    remote_id: string | null;
    remote_url: string | null;
    error: string | null;
}

interface PublishInput {
    organizationId: string;
    jobId: string;
    clientId: string;
    post: BlogPostDraftRow;
    profile: BlogClientProfile;
}

interface AdapterResult {
    remote_id: string | null;
    remote_url: string | null;
    metadata?: Record<string, unknown>;
}

let destinationTableAvailability: 'unknown' | 'available' | 'missing' = 'unknown';
let publishEventTableAvailability: 'unknown' | 'available' | 'missing' = 'unknown';

export async function publishBlogPostToDestinations(input: PublishInput): Promise<PublishResult[]> {
    const destinations = await resolvePublishDestinations(input.organizationId, input.clientId);
    if (!destinations.length) return [];

    const results: PublishResult[] = [];
    for (const destination of destinations) {
        const startedAt = Date.now();
        const baseResult: PublishResult = {
            destination_id: destination.id,
            destination_type: destination.destination_type,
            destination_name: destination.name,
            status: 'failed',
            attempts: 0,
            remote_id: null,
            remote_url: null,
            error: null
        };

        if (!destination.enabled) {
            const skipped = { ...baseResult, status: 'skipped' as const };
            results.push(skipped);
            continue;
        }

        let lastError: string | null = null;
        let attempts = 0;
        for (const delayMs of [0, 1000, 2500]) {
            attempts += 1;
            if (delayMs > 0) await sleep(delayMs);

            try {
                const mapped = mapPostForDestination(input.post, destination);
                const adapterResult = await publishWithAdapter(destination, mapped, input.jobId, input.clientId);
                const okResult: PublishResult = {
                    ...baseResult,
                    status: 'published',
                    attempts,
                    remote_id: adapterResult.remote_id,
                    remote_url: adapterResult.remote_url,
                    error: null
                };
                results.push(okResult);
                await emitPublishEvent({
                    organization_id: input.organizationId,
                    destination_id: destination.id,
                    destination_type: destination.destination_type,
                    job_id: input.jobId,
                    client_id: input.clientId,
                    post_id: input.post.id,
                    status: 'published',
                    remote_id: adapterResult.remote_id,
                    remote_url: adapterResult.remote_url,
                    attempt: attempts,
                    duration_ms: Date.now() - startedAt,
                    error: null,
                    metadata: adapterResult.metadata ?? {}
                });
                lastError = null;
                break;
            } catch (error) {
                lastError = error instanceof Error ? error.message : String(error);
            }
        }

        if (lastError) {
            const failed = {
                ...baseResult,
                status: 'failed' as const,
                attempts,
                error: lastError
            };
            results.push(failed);
            await emitPublishEvent({
                organization_id: input.organizationId,
                destination_id: destination.id,
                destination_type: destination.destination_type,
                job_id: input.jobId,
                client_id: input.clientId,
                post_id: input.post.id,
                status: 'failed',
                remote_id: null,
                remote_url: null,
                attempt: attempts,
                duration_ms: Date.now() - startedAt,
                error: lastError,
                metadata: {}
            });
        }
    }

    return results;
}

function mapPostForDestination(post: BlogPostDraftRow, destination: PublishDestinationRow): BlogPostDraftRow {
    const mapping = destination.field_mapping || {};
    const titlePrefix = typeof mapping.title_prefix === 'string' ? mapping.title_prefix.trim() : '';
    const titleSuffix = typeof mapping.title_suffix === 'string' ? mapping.title_suffix.trim() : '';
    const mappedTitle = `${titlePrefix}${titlePrefix ? ' ' : ''}${post.title}${titleSuffix ? ` ${titleSuffix}` : ''}`.trim();

    return {
        ...post,
        title: mappedTitle || post.title,
        status: destination.publish_mode === 'publish' ? 'draft' : 'draft'
    };
}

async function publishWithAdapter(
    destination: PublishDestinationRow,
    post: BlogPostDraftRow,
    jobId: string,
    clientId: string
): Promise<AdapterResult> {
    if (destination.destination_type === 'wordpress') {
        return publishToWordpress(destination, post);
    }
    if (destination.destination_type === 'ghost') {
        return publishToGhost(destination, post);
    }
    if (destination.destination_type === 'webhook') {
        return publishToWebhook(destination, post, jobId, clientId);
    }
    throw new Error(`Unsupported publish destination type '${destination.destination_type}'`);
}

async function publishToWordpress(destination: PublishDestinationRow, post: BlogPostDraftRow): Promise<AdapterResult> {
    const cfg = destination.config || {};
    const siteUrl = cleanText(cfg.site_url);
    if (!siteUrl) throw new Error('wordpress config missing site_url');

    const endpointPath = cleanText(cfg.endpoint_path) || '/wp-json/wp/v2/posts';
    const endpoint = `${siteUrl.replace(/\/$/, '')}${endpointPath.startsWith('/') ? '' : '/'}${endpointPath}`;

    const status = destination.publish_mode === 'publish' ? 'publish' : 'draft';
    const body = {
        title: post.title,
        slug: post.slug,
        excerpt: post.excerpt,
        content: post.content_markdown,
        status,
        date_gmt: status === 'publish' && post.published_at ? post.published_at : undefined
    };

    const headers: Record<string, string> = {
        'content-type': 'application/json'
    };

    const bearerToken = cleanText(cfg.auth_token);
    const username = cleanText(cfg.username);
    const appPassword = cleanText(cfg.application_password);

    if (bearerToken) {
        headers.Authorization = `Bearer ${bearerToken}`;
    } else if (username && appPassword) {
        const token = Buffer.from(`${username}:${appPassword}`, 'utf8').toString('base64');
        headers.Authorization = `Basic ${token}`;
    } else {
        throw new Error('wordpress config missing auth_token or username/application_password');
    }

    const response = await fetch(endpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify(body)
    });

    const raw = await response.text();
    if (!response.ok) {
        throw new Error(`wordpress publish failed HTTP ${response.status}: ${raw.slice(0, 400)}`);
    }

    const json = safeJson(raw);
    return {
        remote_id: json && json.id != null ? String(json.id) : null,
        remote_url: json && typeof json.link === 'string' ? json.link : null,
        metadata: {
            endpoint,
            status
        }
    };
}

async function publishToGhost(destination: PublishDestinationRow, post: BlogPostDraftRow): Promise<AdapterResult> {
    const cfg = destination.config || {};
    const apiUrl = cleanText(cfg.api_url);
    const adminApiKey = cleanText(cfg.admin_api_key);
    if (!apiUrl) throw new Error('ghost config missing api_url');
    if (!adminApiKey) throw new Error('ghost config missing admin_api_key');

    const endpoint = `${apiUrl.replace(/\/$/, '')}/ghost/api/admin/posts/`;
    const status = destination.publish_mode === 'publish' ? 'published' : 'draft';

    const jwt = buildGhostAdminJwt(adminApiKey);
    const html = markdownToSimpleHtml(post.content_markdown);

    const payload = {
        posts: [
            {
                title: post.title,
                slug: post.slug,
                excerpt: post.excerpt,
                html,
                status,
                published_at: status === 'published' && post.published_at ? post.published_at : undefined,
                tags: (post.tags || []).map((name) => ({ name }))
            }
        ]
    };

    const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
            'content-type': 'application/json',
            Authorization: `Ghost ${jwt}`
        },
        body: JSON.stringify(payload)
    });

    const raw = await response.text();
    if (!response.ok) {
        throw new Error(`ghost publish failed HTTP ${response.status}: ${raw.slice(0, 400)}`);
    }

    const json = safeJson(raw);
    const first = Array.isArray(json?.posts) ? json.posts[0] : null;

    return {
        remote_id: first?.id ? String(first.id) : null,
        remote_url: first?.url ? String(first.url) : null,
        metadata: {
            endpoint,
            status
        }
    };
}

async function publishToWebhook(
    destination: PublishDestinationRow,
    post: BlogPostDraftRow,
    jobId: string,
    clientId: string
): Promise<AdapterResult> {
    const cfg = destination.config || {};
    const url = cleanText(cfg.url);
    if (!url) throw new Error('webhook config missing url');

    const payload = {
        event: 'blog_post.publish',
        destination_id: destination.id,
        destination_type: destination.destination_type,
        client_id: clientId,
        job_id: jobId,
        post
    };

    const headers: Record<string, string> = {
        'content-type': 'application/json'
    };

    const extraHeaders = cfg.headers;
    if (extraHeaders && typeof extraHeaders === 'object' && !Array.isArray(extraHeaders)) {
        for (const [k, v] of Object.entries(extraHeaders as Record<string, unknown>)) {
            const key = cleanText(k);
            const value = cleanText(v);
            if (key && value) headers[key] = value;
        }
    }

    const signingSecret = cleanText(cfg.signing_secret);
    const body = JSON.stringify(payload);
    if (signingSecret) {
        const signature = crypto.createHmac('sha256', signingSecret).update(body).digest('hex');
        headers['x-blog-signature-sha256'] = signature;
    }

    const response = await fetch(url, {
        method: 'POST',
        headers,
        body
    });

    const raw = await response.text();
    if (!response.ok) {
        throw new Error(`webhook publish failed HTTP ${response.status}: ${raw.slice(0, 400)}`);
    }

    const json = safeJson(raw);
    return {
        remote_id: json && json.remote_id != null ? String(json.remote_id) : null,
        remote_url: json && typeof json.remote_url === 'string' ? json.remote_url : null,
        metadata: {
            endpoint: url
        }
    };
}

function buildGhostAdminJwt(adminApiKey: string): string {
    const parts = adminApiKey.split(':');
    if (parts.length !== 2) {
        throw new Error('ghost admin_api_key must be in "id:secret" format');
    }

    const [id, secretHex] = parts;
    const now = Math.floor(Date.now() / 1000);

    const header = {
        alg: 'HS256',
        typ: 'JWT',
        kid: id
    };

    const payload = {
        iat: now,
        exp: now + 60 * 5,
        aud: '/admin/'
    };

    const encodedHeader = base64UrlEncode(JSON.stringify(header));
    const encodedPayload = base64UrlEncode(JSON.stringify(payload));
    const token = `${encodedHeader}.${encodedPayload}`;
    const signature = crypto
        .createHmac('sha256', Buffer.from(secretHex, 'hex'))
        .update(token)
        .digest('base64')
        .replace(/\+/g, '-')
        .replace(/\//g, '_')
        .replace(/=+$/g, '');

    return `${token}.${signature}`;
}

function base64UrlEncode(input: string): string {
    return Buffer.from(input, 'utf8')
        .toString('base64')
        .replace(/\+/g, '-')
        .replace(/\//g, '_')
        .replace(/=+$/g, '');
}

function markdownToSimpleHtml(markdown: string): string {
    return String(markdown || '')
        .split('\n\n')
        .map((block) => block.trim())
        .filter(Boolean)
        .map((block) => {
            if (block.startsWith('### ')) return `<h3>${escapeHtml(block.slice(4))}</h3>`;
            if (block.startsWith('## ')) return `<h2>${escapeHtml(block.slice(3))}</h2>`;
            if (block.startsWith('# ')) return `<h1>${escapeHtml(block.slice(2))}</h1>`;
            if (block.startsWith('- ')) {
                const items = block
                    .split('\n')
                    .map((line) => line.trim())
                    .filter((line) => line.startsWith('- '))
                    .map((line) => `<li>${escapeHtml(line.slice(2))}</li>`)
                    .join('');
                return `<ul>${items}</ul>`;
            }
            return `<p>${escapeHtml(block).replace(/\n/g, '<br/>')}</p>`;
        })
        .join('\n');
}

function escapeHtml(value: string): string {
    return value
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function safeJson(raw: string): any {
    try {
        return JSON.parse(raw);
    } catch {
        return null;
    }
}

async function resolvePublishDestinations(organizationId: string, clientId: string): Promise<PublishDestinationRow[]> {
    const available = await isDestinationTableAvailable();
    if (!available) return [];

    const { data, error } = await supabase
        .from('client_publish_destinations')
        .select('*')
        .eq('organization_id', organizationId)
        .eq('enabled', true)
        .in('client_id', [clientId, 'default'])
        .returns<PublishDestinationRecordRaw[]>();

    if (error) {
        if ((error as { code?: string }).code === 'PGRST205') {
            destinationTableAvailability = 'missing';
            console.warn('[blog-publish] table public.client_publish_destinations missing. Apply migration to enable destination publishing.');
            return [];
        }
        console.warn(`[blog-publish] destination query failed: ${error.message}`);
        return [];
    }

    const rows = (data || [])
        .map(normalizeDestinationRow)
        .filter((d): d is PublishDestinationRow => Boolean(d));

    const clientRows = rows.filter((r) => r.client_id === clientId);
    if (clientRows.length > 0) return clientRows;

    return rows.filter((r) => r.client_id === 'default');
}

function normalizeDestinationRow(raw: PublishDestinationRecordRaw): PublishDestinationRow | null {
    const typeRaw = cleanText(raw.destination_type).toLowerCase();
    if (!['wordpress', 'ghost', 'webhook'].includes(typeRaw)) return null;
    const organizationId = cleanText(raw.organization_id);
    if (!organizationId) return null;

    return {
        id: raw.id,
        organization_id: organizationId,
        client_id: cleanText(raw.client_id) || 'default',
        name: cleanText(raw.name) || null,
        destination_type: typeRaw as PublishDestinationType,
        enabled: raw.enabled !== false,
        publish_mode: cleanText(raw.publish_mode) === 'publish' ? 'publish' : 'draft',
        config: raw.config && typeof raw.config === 'object' ? raw.config : {},
        field_mapping: raw.field_mapping && typeof raw.field_mapping === 'object' ? raw.field_mapping : {}
    };
}

async function isDestinationTableAvailable(): Promise<boolean> {
    if (destinationTableAvailability === 'available') return true;
    if (destinationTableAvailability === 'missing') return false;

    const { error } = await supabase.from('client_publish_destinations').select('id', { head: true }).limit(1);
    if (!error) {
        destinationTableAvailability = 'available';
        return true;
    }

    if ((error as { code?: string }).code === 'PGRST205') {
        destinationTableAvailability = 'missing';
        console.warn('[blog-publish] table public.client_publish_destinations missing. Apply migration to enable destination publishing.');
        return false;
    }

    return true;
}

async function emitPublishEvent(payload: PublishEventPayload): Promise<void> {
    const available = await isPublishEventTableAvailable();
    if (!available) return;

    const { error } = await supabase.from('blog_publish_events').insert({
        organization_id: payload.organization_id,
        destination_id: payload.destination_id,
        destination_type: payload.destination_type,
        job_id: payload.job_id,
        client_id: payload.client_id,
        post_id: payload.post_id,
        status: payload.status,
        remote_id: payload.remote_id,
        remote_url: payload.remote_url,
        attempt: payload.attempt,
        duration_ms: payload.duration_ms,
        error: payload.error,
        metadata: payload.metadata
    });

    if (error) {
        if ((error as { code?: string }).code === 'PGRST205') {
            publishEventTableAvailability = 'missing';
            console.warn('[blog-publish] table public.blog_publish_events missing. Apply migration to enable publish telemetry.');
            return;
        }
        console.warn(`[blog-publish] failed to insert blog_publish_events row: ${error.message}`);
    }
}

async function isPublishEventTableAvailable(): Promise<boolean> {
    if (publishEventTableAvailability === 'available') return true;
    if (publishEventTableAvailability === 'missing') return false;

    const { error } = await supabase.from('blog_publish_events').select('id', { head: true }).limit(1);
    if (!error) {
        publishEventTableAvailability = 'available';
        return true;
    }

    if ((error as { code?: string }).code === 'PGRST205') {
        publishEventTableAvailability = 'missing';
        console.warn('[blog-publish] table public.blog_publish_events missing. Apply migration to enable publish telemetry.');
        return false;
    }

    return true;
}

function cleanText(value: unknown): string {
    return String(value ?? '').replace(/\s+/g, ' ').trim();
}

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}
