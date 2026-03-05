import fs from 'fs';
import path from 'path';
import { OpenAI } from 'openai';
import { supabase } from '../lib/supabase';
import { publishBlogPostToDestinations, type PublishResult } from './blog-publish';

export interface BlogTranscriptSegment {
    start: number;
    end: number;
    text: string;
}

export interface BlogBoundaryRange {
    start: number;
    end: number;
}

export interface BlogArtifactClipContext {
    start: number;
    end: number;
    title?: string;
    excerpt?: string;
    confidence?: number;
}

export interface BlogArtifactContext {
    jobId: string;
    organizationId?: string;
    workDir: string;
    youtubeUrl: string;
    boundaries: BlogBoundaryRange;
    transcriptSegments: BlogTranscriptSegment[];
    clips?: BlogArtifactClipContext[];
}

export interface BlogClientProfile {
    client_id: string;
    enabled: boolean;
    llm_provider: string;
    llm_model: string;
    prompt_version: string;
    system_prompt: string | null;
    user_prompt_template: string | null;
    default_author_name: string;
    default_status: string;
    sync_enabled: boolean;
    sync_endpoint: string | null;
    sync_token: string | null;
    preserve_published_fields: boolean;
    field_rules: Record<string, unknown>;
}

export interface BlogPostDraftRow {
    id: string;
    organization_id: string;
    slug: string;
    status: 'draft';
    title: string;
    excerpt: string;
    content_markdown: string;
    seo_title: string;
    seo_description: string;
    focus_keyword: string;
    tags: string[];
    scripture_refs: string[];
    youtube_url: string;
    youtube_video_id: string;
    youtube_thumbnail: string;
    sermon_date: string | null;
    published_at: string | null;
    author_name: string;
    canonical_url: string;
    hero_image_url: string;
    created_at: string;
    updated_at: string;
}

export interface BlogArtifactDraft {
    post: BlogPostDraftRow;
    provenance: {
        provider: string;
        model: string;
        prompt_version: string;
        confidence: number;
        generated_at: string;
        client_id: string;
    };
    validation: {
        valid: boolean;
        issues: string[];
    };
}

interface BlogGeneratorInput {
    context: BlogArtifactContext;
    profile: BlogClientProfile;
    clientId: string;
}

interface BlogSyncResult {
    status: 'synced' | 'skipped' | 'failed';
    endpoint: string | null;
    attempts: number;
    error: string | null;
}

interface BlogEventPayload {
    organization_id: string;
    job_id: string;
    client_id: string | null;
    post_id: string | null;
    status: string;
    stage: string;
    provider: string | null;
    model: string | null;
    prompt_version: string | null;
    attempt: number;
    duration_ms: number | null;
    error: string | null;
    metadata: Record<string, unknown>;
}

interface BlogGenerationSummary {
    generation_status: 'generated' | 'failed' | 'skipped';
    post_id?: string;
    slug?: string;
    provider?: string;
    model?: string;
    prompt_version?: string;
    generated_at?: string;
    validation?: { valid: boolean; issues: string[] };
    sync?: BlogSyncResult;
    publish?: {
        attempted: boolean;
        results: PublishResult[];
    };
    reason?: string;
    error?: string;
}

const DEFAULT_MODEL = process.env.BLOG_OPENAI_MODEL || process.env.HIGHLIGHTS_OPENAI_MODEL || process.env.ANALYZE_OPENAI_MODEL || 'gpt-5-mini';
const DEFAULT_PROMPT_VERSION = process.env.BLOG_PROMPT_VERSION || 'blog-v1';
const DEFAULT_AUTHOR = process.env.BLOG_DEFAULT_AUTHOR_NAME || 'Default Author';
const DEFAULT_HERO_IMAGE = process.env.BLOG_DEFAULT_HERO_IMAGE_URL || '';

let profileTableAvailability: 'unknown' | 'available' | 'missing' = 'unknown';
let eventTableAvailability: 'unknown' | 'available' | 'missing' = 'unknown';

export async function runBlogArtifactPostProcess(context: BlogArtifactContext): Promise<void> {
    const startedAt = Date.now();
    const generatedAtIso = new Date().toISOString();
    const artifactDir = path.join(context.workDir, 'blog');
    if (!fs.existsSync(artifactDir)) fs.mkdirSync(artifactDir, { recursive: true });

    const inputPath = path.join(artifactDir, 'blog.artifact.input.json');
    const outputPath = path.join(artifactDir, 'blog.artifact.output.json');
    const eventsPath = path.join(artifactDir, 'blog.artifact.events.jsonl');

    const job = await fetchJobRow(context.jobId);
    const organizationId = cleanText(context.organizationId) || cleanText(job?.organization_id);
    if (!organizationId) {
        throw new Error(`Missing organization_id for blog artifact job ${context.jobId}`);
    }
    const contextWithOrg: BlogArtifactContext = {
        ...context,
        organizationId
    };
    const jobMetadata = parseRecord(job?.metadata);
    const clientId = resolveClientId(jobMetadata);

    writeJsonSafe(inputPath, {
        generated_at: generatedAtIso,
        job_id: context.jobId,
        client_id: clientId,
        youtube_url: context.youtubeUrl,
        boundaries: context.boundaries,
        transcript_count: context.transcriptSegments.length,
        clips_count: context.clips?.length ?? 0
    });

    await appendEventLine(eventsPath, {
        ts: generatedAtIso,
        stage: 'start',
        job_id: context.jobId,
        client_id: clientId
    });

    const profileResolved = await resolveClientProfile(clientId, organizationId);
    if (!profileResolved.profile) {
        const summary: BlogGenerationSummary = {
            generation_status: 'skipped',
            reason: profileResolved.reason ?? undefined
        };
        await writeBlogMetadata(context.jobId, organizationId, jobMetadata, summary, [inputPath]);
        await emitBlogGenerationEvent({
            organization_id: organizationId,
            job_id: context.jobId,
            client_id: clientId,
            post_id: null,
            status: 'skipped',
            stage: 'resolve_profile',
            provider: null,
            model: null,
            prompt_version: null,
            attempt: 1,
            duration_ms: Date.now() - startedAt,
            error: profileResolved.reason,
            metadata: { reason: profileResolved.reason }
        });
        return;
    }

    const profile = profileResolved.profile;

    if (!profile.enabled) {
        const summary: BlogGenerationSummary = {
            generation_status: 'skipped',
            reason: `client profile '${profile.client_id}' disabled`
        };
        await writeBlogMetadata(context.jobId, organizationId, jobMetadata, summary, [inputPath]);
        await emitBlogGenerationEvent({
            organization_id: organizationId,
            job_id: context.jobId,
            client_id: profile.client_id,
            post_id: null,
            status: 'skipped',
            stage: 'profile_disabled',
            provider: profile.llm_provider,
            model: profile.llm_model,
            prompt_version: profile.prompt_version,
            attempt: 1,
            duration_ms: Date.now() - startedAt,
            error: null,
            metadata: { reason: 'profile_disabled' }
        });
        return;
    }

    let artifact: BlogArtifactDraft | null = null;
    let sync: BlogSyncResult = { status: 'skipped', endpoint: null, attempts: 0, error: null };
    let publishResults: PublishResult[] = [];
    let postId: string | null = null;

    try {
        await updateJobStatus(context.jobId, organizationId, 'processing:blog');
        await updateJobStatus(context.jobId, organizationId, 'processing:blog:generate');
        artifact = await generateBlogArtifact({ context: contextWithOrg, profile, clientId: profile.client_id });

        writeJsonSafe(outputPath, artifact);
        await appendEventLine(eventsPath, {
            ts: new Date().toISOString(),
            stage: 'generated',
            post_id: artifact.post.id,
            slug: artifact.post.slug,
            provider: artifact.provenance.provider,
            model: artifact.provenance.model,
            prompt_version: artifact.provenance.prompt_version,
            validation: artifact.validation
        });

        await updateJobStatus(context.jobId, organizationId, 'processing:blog:persist');
        const persisted = await upsertBlogPostDraft(artifact.post, {
            organizationId,
            preservePublishedFields: profile.preserve_published_fields
        });
        postId = persisted.id;

        await emitBlogGenerationEvent({
            organization_id: organizationId,
            job_id: context.jobId,
            client_id: profile.client_id,
            post_id: postId,
            status: 'persisted',
            stage: 'persist',
            provider: artifact.provenance.provider,
            model: artifact.provenance.model,
            prompt_version: artifact.provenance.prompt_version,
            attempt: 1,
            duration_ms: Date.now() - startedAt,
            error: null,
            metadata: { slug: persisted.slug, validation: artifact.validation }
        });

        await updateJobStatus(context.jobId, organizationId, 'processing:blog:sync');
        sync = await syncDraftToSheets(postId, profile);

        await emitBlogGenerationEvent({
            organization_id: organizationId,
            job_id: context.jobId,
            client_id: profile.client_id,
            post_id: postId,
            status: sync.status,
            stage: 'sync',
            provider: artifact.provenance.provider,
            model: artifact.provenance.model,
            prompt_version: artifact.provenance.prompt_version,
            attempt: Math.max(1, sync.attempts),
            duration_ms: Date.now() - startedAt,
            error: sync.error,
            metadata: { endpoint: sync.endpoint }
        });

        await updateJobStatus(context.jobId, organizationId, 'processing:blog:publish');
        publishResults = await publishBlogPostToDestinations({
            organizationId,
            jobId: context.jobId,
            clientId: profile.client_id,
            post: artifact.post,
            profile
        });

        const summary: BlogGenerationSummary = {
            generation_status: 'generated',
            post_id: postId,
            slug: artifact.post.slug,
            provider: artifact.provenance.provider,
            model: artifact.provenance.model,
            prompt_version: artifact.provenance.prompt_version,
            generated_at: artifact.provenance.generated_at,
            validation: artifact.validation,
            sync,
            publish: {
                attempted: true,
                results: publishResults
            }
        };

        await writeBlogMetadata(context.jobId, organizationId, jobMetadata, summary, [inputPath, outputPath, eventsPath]);
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        await appendEventLine(eventsPath, {
            ts: new Date().toISOString(),
            stage: 'failed',
            error: message
        });

        await emitBlogGenerationEvent({
            organization_id: organizationId,
            job_id: context.jobId,
            client_id: profile.client_id,
            post_id: postId,
            status: 'failed',
            stage: 'blog',
            provider: artifact?.provenance.provider ?? profile.llm_provider,
            model: artifact?.provenance.model ?? profile.llm_model,
            prompt_version: artifact?.provenance.prompt_version ?? profile.prompt_version,
            attempt: 1,
            duration_ms: Date.now() - startedAt,
            error: message,
            metadata: {
                sync,
                publish: publishResults,
                slug: artifact?.post.slug ?? null
            }
        });

        const summary: BlogGenerationSummary = {
            generation_status: 'failed',
            post_id: postId ?? undefined,
            slug: artifact?.post.slug,
            provider: artifact?.provenance.provider,
            model: artifact?.provenance.model,
            prompt_version: artifact?.provenance.prompt_version,
            generated_at: artifact?.provenance.generated_at,
            validation: artifact?.validation,
            sync,
            publish: {
                attempted: publishResults.length > 0,
                results: publishResults
            },
            error: message
        };

        await writeBlogMetadata(context.jobId, organizationId, jobMetadata, summary, [inputPath, outputPath, eventsPath]);
        console.warn(`[blog-artifact] post-process failed for job=${context.jobId}: ${message}`);
    }
}

interface ExistingPostPreview {
    id: string;
    status: string;
    published_at: string | null;
    created_at: string;
    canonical_url: string | null;
}

export async function upsertBlogPostDraft(
    post: BlogPostDraftRow,
    options: { organizationId: string; preservePublishedFields: boolean }
): Promise<{ id: string; slug: string }> {
    const nowIso = new Date().toISOString();

    const { data: existing } = await supabase
        .from('posts')
        .select('id,status,published_at,created_at,canonical_url')
        .eq('id', post.id)
        .eq('organization_id', options.organizationId)
        .maybeSingle<ExistingPostPreview>();

    const payload: BlogPostDraftRow = {
        ...post,
        organization_id: options.organizationId,
        status: 'draft',
        created_at: existing?.created_at ?? post.created_at ?? nowIso,
        updated_at: nowIso,
        published_at: options.preservePublishedFields ? existing?.published_at ?? post.published_at : post.published_at,
        canonical_url: options.preservePublishedFields ? existing?.canonical_url ?? post.canonical_url : post.canonical_url
    };

    const { error } = await supabase.from('posts').upsert(payload, { onConflict: 'id' });
    if (error) {
        throw new Error(`Failed to upsert posts row '${post.id}': ${error.message}`);
    }
    return { id: payload.id, slug: payload.slug };
}

function resolveClientId(metadata: Record<string, unknown>): string {
    const clientRaw = metadata.client_id;
    if (typeof clientRaw === 'string' && clientRaw.trim()) return clientRaw.trim();
    return 'default';
}

function extractYoutubeVideoId(url: string): string {
    const trimmed = String(url || '').trim();
    if (!trimmed) return '';

    const regexes = [
        /(?:youtube\.com\/watch\?v=)([A-Za-z0-9_-]{11})/,
        /(?:youtu\.be\/)([A-Za-z0-9_-]{11})/,
        /(?:youtube\.com\/shorts\/)([A-Za-z0-9_-]{11})/
    ];

    for (const re of regexes) {
        const m = trimmed.match(re);
        if (m?.[1]) return m[1];
    }
    return '';
}

function cleanText(value: unknown): string {
    return String(value ?? '').replace(/\s+/g, ' ').trim();
}

function toSentenceCase(value: string): string {
    if (!value) return '';
    return value.charAt(0).toUpperCase() + value.slice(1);
}

function nonEmptyArrayOfStrings(value: unknown): string[] {
    if (Array.isArray(value)) {
        return value
            .map((v) => cleanText(v))
            .filter(Boolean)
            .map((v) => v.slice(0, 80));
    }

    if (typeof value === 'string') {
        return value
            .split(',')
            .map((v) => cleanText(v))
            .filter(Boolean)
            .map((v) => v.slice(0, 80));
    }

    return [];
}

export function slugify(input: string): string {
    const ascii = input
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '')
        .toLowerCase();
    const kebab = ascii
        .replace(/[^a-z0-9\s-]/g, ' ')
        .trim()
        .replace(/[\s_-]+/g, '-')
        .replace(/^-+|-+$/g, '');
    return kebab || 'sermon-post';
}

function summarizeTranscript(segments: BlogTranscriptSegment[], max = 2800): string {
    const sorted = segments
        .filter((s) => Number.isFinite(s.start) && Number.isFinite(s.end) && cleanText(s.text))
        .sort((a, b) => a.start - b.start);
    let out = '';
    for (const s of sorted) {
        const line = `[${s.start.toFixed(1)}-${s.end.toFixed(1)}] ${cleanText(s.text)}`;
        if ((out.length + line.length + 1) > max) break;
        out += `${line}\n`;
    }
    return out.trim();
}

function formatScriptureRefs(raw: unknown): string[] {
    const refs = nonEmptyArrayOfStrings(raw)
        .map((r) => r.replace(/\s+/g, ' ').trim())
        .filter((r) => r.length >= 3);
    return refs.slice(0, 8);
}

function fallbackMarkdownFromTranscript(
    title: string,
    transcriptSegments: BlogTranscriptSegment[],
    clips?: BlogArtifactClipContext[]
): string {
    const lines: string[] = [];
    lines.push(`## ${title}`);
    lines.push('');
    lines.push('### Resumen del mensaje');

    const selectedText = transcriptSegments
        .slice(0, 16)
        .map((s) => cleanText(s.text))
        .filter(Boolean)
        .join(' ')
        .trim();

    lines.push(selectedText || 'Contenido generado automáticamente desde la transcripción del sermón.');

    if (Array.isArray(clips) && clips.length > 0) {
        lines.push('');
        lines.push('### Momentos clave');
        for (const clip of clips.slice(0, 5)) {
            const clipTitle = cleanText(clip.title) || 'Punto principal';
            const excerpt = cleanText(clip.excerpt);
            lines.push(`- **${clipTitle}**${excerpt ? `: ${excerpt}` : ''}`);
        }
    }

    return lines.join('\n').trim();
}

export function normalizeBlogDraft(args: {
    raw: Record<string, unknown>;
    context: BlogArtifactContext;
    profile: BlogClientProfile;
    generatedAt: string;
}): BlogArtifactDraft {
    const { raw, context, profile, generatedAt } = args;
    const nowIso = generatedAt || new Date().toISOString();
    const titleRaw = cleanText(raw.title) || cleanText(raw.seo_title) || 'Sermón de esperanza y fe';
    const title = toSentenceCase(titleRaw.slice(0, 140));

    const excerpt = cleanText(raw.excerpt).slice(0, 320) || `Reflexión basada en el sermón: ${title}`;
    const content = cleanText(raw.content_markdown)
        ? String(raw.content_markdown)
        : fallbackMarkdownFromTranscript(title, context.transcriptSegments, context.clips);

    const youtubeVideoId = extractYoutubeVideoId(context.youtubeUrl);
    const slugBase = cleanText(raw.slug) || title;
    const slugCore = slugify(slugBase);
    const slug = `${slugCore}-${context.jobId.slice(0, 8)}`;

    const tags = nonEmptyArrayOfStrings(raw.tags).slice(0, 10);
    const scriptureRefs = formatScriptureRefs(raw.scripture_refs);

    const seoTitle = cleanText(raw.seo_title).slice(0, 170) || title;
    const seoDescription = cleanText(raw.seo_description).slice(0, 260) || excerpt;
    const focusKeyword = cleanText(raw.focus_keyword).slice(0, 120) || (tags[0] || 'sermón cristiano');

    const author = cleanText(raw.author_name) || cleanText(profile.default_author_name) || DEFAULT_AUTHOR;
    const heroImage = cleanText(raw.hero_image_url) || DEFAULT_HERO_IMAGE;
    const canonical = cleanText(raw.canonical_url);

    const post: BlogPostDraftRow = {
        id: `sermon-${context.jobId}`,
        organization_id: context.organizationId || '',
        slug,
        status: 'draft',
        title,
        excerpt,
        content_markdown: content,
        seo_title: seoTitle,
        seo_description: seoDescription,
        focus_keyword: focusKeyword,
        tags,
        scripture_refs: scriptureRefs,
        youtube_url: cleanText(context.youtubeUrl),
        youtube_video_id: youtubeVideoId,
        youtube_thumbnail: youtubeVideoId ? `https://img.youtube.com/vi/${youtubeVideoId}/maxresdefault.jpg` : '',
        sermon_date: null,
        published_at: null,
        author_name: author,
        canonical_url: canonical,
        hero_image_url: heroImage,
        created_at: nowIso,
        updated_at: nowIso
    };

    const issues: string[] = [];
    if (!post.slug || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(post.slug)) {
        issues.push('invalid_slug');
    }
    if (!post.title) issues.push('missing_title');
    if (!post.content_markdown) issues.push('missing_content_markdown');
    if (!post.author_name) issues.push('missing_author_name');

    return {
        post,
        provenance: {
            provider: cleanText(raw.provider) || profile.llm_provider,
            model: cleanText(raw.model) || profile.llm_model,
            prompt_version: cleanText(raw.prompt_version) || profile.prompt_version,
            confidence: clamp(Number(raw.confidence), 0, 1, 0.65),
            generated_at: nowIso,
            client_id: profile.client_id
        },
        validation: {
            valid: issues.length === 0,
            issues
        }
    };
}

function clamp(value: number, min: number, max: number, fallback: number): number {
    if (!Number.isFinite(value)) return fallback;
    return Math.max(min, Math.min(max, value));
}

interface BlogGeneratorProvider {
    generate(input: BlogGeneratorInput): Promise<Record<string, unknown>>;
}

class OpenAIBlogGenerator implements BlogGeneratorProvider {
    async generate(input: BlogGeneratorInput): Promise<Record<string, unknown>> {
        const apiKey = process.env.OPENAI_API_KEY;
        if (!apiKey) {
            throw new Error('OPENAI_API_KEY missing for blog generation provider=openai');
        }

        const openai = new OpenAI({ apiKey });
        const model = input.profile.llm_model || DEFAULT_MODEL;

        const transcriptSummary = summarizeTranscript(input.context.transcriptSegments);
        const clipSummary = (input.context.clips || [])
            .slice(0, 8)
            .map((c, idx) => `${idx + 1}. ${cleanText(c.title) || 'Momento'} (${c.start.toFixed(1)}-${c.end.toFixed(1)}): ${cleanText(c.excerpt)}`)
            .join('\n');

        const systemPrompt = cleanText(input.profile.system_prompt) || [
            'You write Spanish Christian sermon blog drafts.',
            'Return strict JSON and keep theological tone pastoral, clear, and practical.',
            'Do not invent scripture references unless they are strongly implied by provided transcript content.'
        ].join(' ');

        const prompt = [
            cleanText(input.profile.user_prompt_template) || 'Generate a blog draft artifact from this sermon context.',
            'Return JSON with keys:',
            'title, slug, excerpt, content_markdown, seo_title, seo_description, focus_keyword, tags, scripture_refs, author_name, canonical_url, hero_image_url, confidence',
            `Client: ${input.clientId}`,
            `Prompt version: ${input.profile.prompt_version}`,
            `YouTube URL: ${input.context.youtubeUrl}`,
            `Sermon bounds: ${input.context.boundaries.start.toFixed(2)}-${input.context.boundaries.end.toFixed(2)}`,
            '',
            'Transcript summary:',
            transcriptSummary || '[empty]',
            '',
            'Highlights summary:',
            clipSummary || '[empty]'
        ].join('\n');

        const response = await openai.chat.completions.create({
            model,
            messages: [
                { role: 'system', content: systemPrompt },
                { role: 'user', content: prompt }
            ],
            response_format: { type: 'json_object' }
        });

        const content = response.choices[0]?.message?.content ?? '{}';
        return parseProviderJson(content);
    }
}

const providerRegistry: Record<string, BlogGeneratorProvider> = {
    openai: new OpenAIBlogGenerator()
};

export function parseProviderJson(content: string): Record<string, unknown> {
    try {
        const parsed = JSON.parse(content);
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
            throw new Error('provider response is not a JSON object');
        }
        return parsed as Record<string, unknown>;
    } catch (error) {
        throw new Error(`Invalid provider JSON response: ${error instanceof Error ? error.message : String(error)}`);
    }
}

async function generateBlogArtifact(input: BlogGeneratorInput): Promise<BlogArtifactDraft> {
    const providerName = cleanText(input.profile.llm_provider || 'openai').toLowerCase();
    const provider = providerRegistry[providerName];
    if (!provider) {
        throw new Error(`Unsupported blog provider '${providerName}'`);
    }

    const raw = await provider.generate(input);
    raw.provider = providerName;
    raw.model = input.profile.llm_model;
    raw.prompt_version = input.profile.prompt_version;

    const generatedAt = new Date().toISOString();
    const artifact = normalizeBlogDraft({
        raw,
        context: input.context,
        profile: input.profile,
        generatedAt
    });

    if (!artifact.validation.valid) {
        throw new Error(`Blog artifact validation failed: ${artifact.validation.issues.join(', ')}`);
    }

    return artifact;
}

async function fetchJobRow(jobId: string): Promise<{ metadata: unknown; organization_id: string | null } | null> {
    const { data, error } = await supabase
        .from('jobs')
        .select('metadata,organization_id')
        .eq('id', jobId)
        .maybeSingle();
    if (error) {
        console.warn(`[blog-artifact] failed to fetch job metadata for ${jobId}: ${error.message}`);
        return null;
    }
    return data as { metadata: unknown; organization_id: string | null } | null;
}

function parseRecord(value: unknown): Record<string, unknown> {
    if (value && typeof value === 'object' && !Array.isArray(value)) {
        return value as Record<string, unknown>;
    }
    return {};
}

async function writeBlogMetadata(
    jobId: string,
    organizationId: string,
    currentMetadata: Record<string, unknown>,
    summary: BlogGenerationSummary,
    paths: string[]
): Promise<void> {
    const next = {
        ...currentMetadata,
        blog_artifact: {
            ...(parseRecord(currentMetadata.blog_artifact)),
            ...summary
        },
        artifacts: {
            ...parseRecord(currentMetadata.artifacts),
            blog: {
                ...(parseRecord(parseRecord(currentMetadata.artifacts).blog)),
                paths
            }
        }
    };

    const { error } = await supabase
        .from('jobs')
        .update({ metadata: next })
        .eq('id', jobId)
        .eq('organization_id', organizationId);
    if (error) {
        console.warn(`[blog-artifact] failed to update jobs.metadata for ${jobId}: ${error.message}`);
    }
}

async function updateJobStatus(jobId: string, organizationId: string, status: string): Promise<void> {
    const { error } = await supabase
        .from('jobs')
        .update({ status })
        .eq('id', jobId)
        .eq('organization_id', organizationId);
    if (error) {
        console.warn(`[blog-artifact] failed to update job status ${jobId}=${status}: ${error.message}`);
    }
}

interface ClientProfileRow {
    client_id: string;
    enabled?: boolean | null;
    llm_provider?: string | null;
    llm_model?: string | null;
    prompt_version?: string | null;
    system_prompt?: string | null;
    user_prompt_template?: string | null;
    default_author_name?: string | null;
    default_status?: string | null;
    sync_enabled?: boolean | null;
    sync_endpoint?: string | null;
    sync_token?: string | null;
    preserve_published_fields?: boolean | null;
    field_rules?: Record<string, unknown> | null;
}

export function selectProfileCandidate(clientId: string, rows: ClientProfileRow[]): BlogClientProfile | null {
    const pick = rows.find((r) => r.client_id === clientId) || rows.find((r) => r.client_id === 'default');
    if (!pick) return null;

    return {
        client_id: pick.client_id,
        enabled: pick.enabled !== false,
        llm_provider: cleanText(pick.llm_provider) || 'openai',
        llm_model: cleanText(pick.llm_model) || DEFAULT_MODEL,
        prompt_version: cleanText(pick.prompt_version) || DEFAULT_PROMPT_VERSION,
        system_prompt: cleanText(pick.system_prompt) || null,
        user_prompt_template: cleanText(pick.user_prompt_template) || null,
        default_author_name: cleanText(pick.default_author_name) || DEFAULT_AUTHOR,
        default_status: cleanText(pick.default_status) || 'draft',
        sync_enabled: pick.sync_enabled !== false,
        sync_endpoint: cleanText(pick.sync_endpoint) || null,
        sync_token: cleanText(pick.sync_token) || null,
        preserve_published_fields: pick.preserve_published_fields !== false,
        field_rules: pick.field_rules && typeof pick.field_rules === 'object' ? pick.field_rules : {}
    };
}

function defaultProfile(clientId: string): BlogClientProfile {
    return {
        client_id: clientId === 'default' ? 'default' : clientId,
        enabled: true,
        llm_provider: 'openai',
        llm_model: DEFAULT_MODEL,
        prompt_version: DEFAULT_PROMPT_VERSION,
        system_prompt: null,
        user_prompt_template: null,
        default_author_name: DEFAULT_AUTHOR,
        default_status: 'draft',
        sync_enabled: true,
        sync_endpoint: null,
        sync_token: null,
        preserve_published_fields: true,
        field_rules: {}
    };
}

async function resolveClientProfile(
    clientId: string,
    organizationId: string
): Promise<{ profile: BlogClientProfile | null; reason: string | null }> {
    const tableAvailable = await isProfileTableAvailable();
    if (!tableAvailable) {
        if (clientId === 'default') {
            return { profile: defaultProfile('default'), reason: null };
        }
        return { profile: null, reason: `missing client profile table and no profile for '${clientId}'` };
    }

    const { data, error } = await supabase
        .from('client_blog_profiles')
        .select('*')
        .eq('organization_id', organizationId)
        .in('client_id', [clientId, 'default'])
        .returns<ClientProfileRow[]>();

    if (error) {
        if ((error as { code?: string }).code === 'PGRST205') {
            profileTableAvailability = 'missing';
            if (clientId === 'default') {
                return { profile: defaultProfile('default'), reason: null };
            }
            return { profile: null, reason: `client profile table missing for '${clientId}'` };
        }
        if (clientId === 'default') {
            console.warn(`[blog-artifact] client profile query failed, using default fallback: ${error.message}`);
            return { profile: defaultProfile('default'), reason: null };
        }
        return { profile: null, reason: `profile query failed: ${error.message}` };
    }

    const rows = Array.isArray(data) ? data : [];
    const selected = selectProfileCandidate(clientId, rows);
    if (!selected) {
        return { profile: null, reason: `no client_blog_profiles row for '${clientId}' and no default row` };
    }

    return { profile: selected, reason: null };
}

async function isProfileTableAvailable(): Promise<boolean> {
    if (profileTableAvailability === 'available') return true;
    if (profileTableAvailability === 'missing') return false;

    const { error } = await supabase.from('client_blog_profiles').select('client_id', { head: true }).limit(1);
    if (!error) {
        profileTableAvailability = 'available';
        return true;
    }

    if ((error as { code?: string }).code === 'PGRST205') {
        profileTableAvailability = 'missing';
        console.warn("[blog-artifact] table public.client_blog_profiles missing. Apply migration to enable profile-based generation.");
        return false;
    }

    return true;
}

async function syncDraftToSheets(postId: string, profile: BlogClientProfile): Promise<BlogSyncResult> {
    if (!profile.sync_enabled) {
        return { status: 'skipped', endpoint: null, attempts: 0, error: null };
    }

    const endpoint = resolveSyncEndpoint(profile);
    if (!endpoint) {
        return { status: 'skipped', endpoint: null, attempts: 0, error: null };
    }

    const token = profile.sync_token || cleanText(process.env.BLOG_SYNC_TOKEN);
    let attempts = 0;
    let lastError: string | null = null;

    for (const delayMs of [0, 1000, 2500]) {
        attempts += 1;
        if (delayMs > 0) await sleep(delayMs);

        try {
            const url = new URL(endpoint);
            if (!url.searchParams.get('ids')) {
                url.searchParams.set('ids', postId);
            }

            const headers: Record<string, string> = { 'content-type': 'application/json' };
            if (token) {
                headers['x-sync-token'] = token;
                headers.Authorization = `Bearer ${token}`;
            }

            const response = await fetch(url.toString(), {
                method: 'POST',
                headers,
                body: JSON.stringify({ ids: [postId] })
            });

            if (response.ok) {
                return { status: 'synced', endpoint: url.toString(), attempts, error: null };
            }

            const body = await response.text();
            lastError = `HTTP ${response.status}: ${body.slice(0, 320)}`;
        } catch (error) {
            lastError = error instanceof Error ? error.message : String(error);
        }
    }

    return {
        status: 'failed',
        endpoint,
        attempts,
        error: lastError
    };
}

function resolveSyncEndpoint(profile: BlogClientProfile): string | null {
    const fromProfile = cleanText(profile.sync_endpoint);
    if (fromProfile) return fromProfile;

    const direct = cleanText(process.env.BLOG_SYNC_ENDPOINT);
    if (direct) return direct;

    const baseUrl = cleanText(process.env.BLOG_SYNC_BASE_URL);
    if (!baseUrl) return null;

    return `${baseUrl.replace(/\/$/, '')}/api/blog/sync-supabase-to-sheets`;
}

async function emitBlogGenerationEvent(payload: BlogEventPayload): Promise<void> {
    const available = await isEventTableAvailable();
    if (!available) return;

    const { error } = await supabase.from('blog_generation_events').insert({
        organization_id: payload.organization_id,
        job_id: payload.job_id,
        client_id: payload.client_id,
        post_id: payload.post_id,
        status: payload.status,
        stage: payload.stage,
        provider: payload.provider,
        model: payload.model,
        prompt_version: payload.prompt_version,
        attempt: payload.attempt,
        duration_ms: payload.duration_ms,
        error: payload.error,
        metadata: payload.metadata
    });

    if (error) {
        if ((error as { code?: string }).code === 'PGRST205') {
            eventTableAvailability = 'missing';
            console.warn('[blog-artifact] table public.blog_generation_events missing. Apply migration to enable telemetry.');
            return;
        }
        console.warn(`[blog-artifact] failed to insert blog_generation_events row: ${error.message}`);
    }
}

async function isEventTableAvailable(): Promise<boolean> {
    if (eventTableAvailability === 'available') return true;
    if (eventTableAvailability === 'missing') return false;

    const { error } = await supabase.from('blog_generation_events').select('id', { head: true }).limit(1);
    if (!error) {
        eventTableAvailability = 'available';
        return true;
    }

    if ((error as { code?: string }).code === 'PGRST205') {
        eventTableAvailability = 'missing';
        console.warn('[blog-artifact] table public.blog_generation_events missing. Apply migration to enable telemetry.');
        return false;
    }

    return true;
}

async function appendEventLine(filePath: string, payload: Record<string, unknown>): Promise<void> {
    const line = `${JSON.stringify(payload)}\n`;
    await fs.promises.appendFile(filePath, line, 'utf8');
}

function writeJsonSafe(filePath: string, payload: unknown): void {
    try {
        fs.writeFileSync(filePath, JSON.stringify(payload, null, 2));
    } catch (error) {
        console.warn(`[blog-artifact] failed writing ${path.basename(filePath)}: ${error instanceof Error ? error.message : String(error)}`);
    }
}

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}
