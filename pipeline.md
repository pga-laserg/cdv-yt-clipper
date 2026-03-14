# Blog + Admin Implementation Reference (Pipeline Integration)

Last updated: 2026-03-04

This document describes the current production implementation in this repository so the video/transcript pipeline can reliably produce blog artifacts that end up live on `/blog`.

## 1) Architecture At A Glance

Current topology:

1. Public site (`/blog`, `/blog/[slug]`) reads from **Google Sheets** only.
2. Admin dashboard (`/admin/...`) reads/writes **Supabase**.
3. On manual save in admin, the saved Supabase row is synced to Google Sheets for the same `id`.
4. Autosave writes to Supabase only (no sheet sync).

Implication:

- Supabase is operational source for editing.
- Google Sheets is publishing source for the public blog renderer.

## 2) Route Surface

Admin routes:

- `/admin/login`
- `/admin`
- `/admin/blog`
- `/admin/blog/new/edit`
- `/admin/blog/[slug]` (internal preview)
- `/admin/blog/[slug]/edit`
- Legacy redirects:
  - `/admin/posts` -> `/admin/blog`
  - `/admin/posts/[id]` -> `/admin/blog/[slug]`

Public routes:

- `/blog` (index)
- `/blog/[slug]` (article detail)

## 3) Data Contracts

### 3.1 Google Sheet schema (required)

Tab columns expected by the app:

1. `id`
2. `slug`
3. `status`
4. `title`
5. `excerpt`
6. `content_markdown`
7. `seo_title`
8. `seo_description`
9. `focus_keyword`
10. `tags`
11. `scripture_refs`
12. `source_url`
13. `youtube_video_id`
14. `youtube_thumbnail`
15. `sermon_date`
16. `published_at`
17. `author_name`
18. `canonical_url`
19. `hero_image_url`

Source: `lib/blog/sheets.ts` and `lib/blog/sync-supabase-to-sheets.ts`.

### 3.2 App model

Public model types are defined in `lib/blog/types.ts`:

- `BlogPostIndexItem`
- `BlogPostDetail`
- `BlogStatus = "draft" | "published" | "archived"`

### 3.3 Supabase row shape (editor/sync side)

`SupabasePost` in `lib/blog/supabase-posts.ts`:

- `id`, `slug`, `status`, `title`, `excerpt`, `content_markdown`
- `seo_title`, `seo_description`, `focus_keyword`
- `tags` (`string[]`)
- `scripture_refs` (`string[]`)
- `youtube_url`, `youtube_video_id`, `youtube_thumbnail`
- `sermon_date`, `published_at`
- `author_name`, `canonical_url`, `hero_image_url`
- optional `updated_at`, `created_at`

## 4) Publish Gating Rules (What Actually Renders Live)

Public blog will include only rows that pass parser validation and have `status=published`.

Hard requirements for a published row (from `rowToPost` in `lib/blog/sheets.ts`):

1. valid `slug` (kebab-case)
2. non-empty `title`
3. non-empty `content_markdown`
4. valid ISO `published_at`
5. valid YouTube payload:
   - `source_url` non-empty
   - `youtube_video_id` resolved and 11 chars

If a published row fails validation, it is skipped (logged as warning).

## 5) Public SEO Behavior

From `app/blog/[slug]/page.tsx`:

1. Metadata title: `seoTitle || title`
2. Metadata description: `seoDescription || excerpt`
3. Canonical: `canonicalUrl || /blog/[slug]`
4. OG/Twitter image: `heroImageUrl || youtubeThumbnail || /images/hombros-delgados-wide2k.jpg`
5. JSON-LD `Article` includes:
   - `headline`, `description`, `datePublished`, `author`, `mainEntityOfPage`
   - `keywords` from `focusKeyword + tags`
   - `isBasedOn` set to YouTube URL
6. Bottom CTA card links to original YouTube video.

## 6) Editor + Content Contract

Editor implementation is Tiptap-based and markdown-canonical:

1. In-memory editing: Tiptap JSON doc.
2. Persisted value: `content_markdown`.
3. Conversion boundary in `lib/blog/editor-content.ts`:
   - `markdownToEditorDoc(markdown)`
   - `editorDocToMarkdown(doc)`
   - `normalizeMarkdown(markdown)`

Current supported fidelity focus:

1. headings H2/H3
2. lists, blockquote, inline marks, link, code
3. fenced code blocks with language
4. table conversion to markdown table format
5. heading/block spacing normalization

Editor UX features (current):

1. Visual mode + Markdown mode toggle
2. Side-by-side preview toggle
3. keyboard shortcuts (bold/italic/list/heading/undo/redo)
4. bubble/floating menus
5. image/youtube/table nodes enabled
6. URL validation + auto-detect helper in embed modals

## 7) Admin Save And Autosave Behavior

### 7.1 Manual save

Path:

- Server action `savePostAction` (`app/admin/posts/actions.ts`)

Flow:

1. session check
2. upsert Supabase row
3. sync same `id` to Google Sheets (`syncSupabaseToGoogleSheets({ ids: [saved.id] })`)
4. redirect to `/admin/blog/[slug]/edit?saved=1`

### 7.2 Autosave

Path:

- `POST /api/admin/blog/autosave`

Flow:

1. session check
2. stale-write check using `expected_updated_at`
3. upsert Supabase row
4. return `{ ok, id, slug, updatedAt, savedAt }`
5. does **not** sync Google Sheets

Conflict handling:

- returns `409` when `expected_updated_at` does not match current row.

Client behavior:

- `AutoSaveDraft` interval default `15s`
- autosaves only when `status === "draft"`

## 8) Pipeline-Facing APIs

### 8.1 Sync Supabase -> Google Sheets

Endpoint:

- `GET|POST /api/blog/sync-supabase-to-sheets`

Optional auth:

- `BLOG_SYNC_TOKEN` env
- send either:
  - `x-sync-token: <token>`
  - `Authorization: Bearer <token>`

Query params:

1. `limit` (default 200, max 1000)
2. `status` (`draft|published|archived|all`)
3. `table` (override table name)
4. `id` (single id)
5. `ids` (comma-separated ids)
6. `dryRun=1` (no writes)

Example:

```bash
curl -X POST "http://localhost:3000/api/blog/sync-supabase-to-sheets?ids=post_123&dryRun=1" \
  -H "x-sync-token: $BLOG_SYNC_TOKEN"
```

### 8.2 Validate draft readiness from Sheet

Endpoint:

- `GET /api/blog/validate-draft?slug=<slug>`
- or `GET /api/blog/validate-draft?id=<id>`

Returns validation report with:

- `validForPublish`
- `missingFields`
- `warnings`

### 8.3 Blog health check

Endpoint:

- `GET /api/blog/health`

Returns:

- source (`google-sheets`)
- published count
- latest slug

## 9) Environment Variables

### 9.1 Required for public blog (Sheets read)

1. `GOOGLE_SERVICE_ACCOUNT_EMAIL`
2. `GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY`
3. `GOOGLE_SHEETS_ID`
4. `GOOGLE_SHEETS_TAB_NAME` (must match your tab, typically `blog`)

### 9.2 Required for admin auth (Supabase OAuth)

1. `SUPABASE_URL` or `NEXT_PUBLIC_SUPABASE_URL`
2. `SUPABASE_ANON_KEY` or `NEXT_PUBLIC_SUPABASE_ANON_KEY`
3. optional: `ADMIN_ALLOWED_EMAILS` (comma-separated allowlist)

### 9.3 Required for admin data writes/sync

1. `SUPABASE_SERVICE_ROLE_KEY`
2. optional: `SUPABASE_BLOG_TABLE` (default `posts`)
3. optional: `BLOG_SYNC_TOKEN` (protect sync endpoint)

Important:

- Keep `GOOGLE_SHEETS_TAB_NAME` identical for both read and sync paths.

## 10) Recommended Pipeline Integration Flow

Use this as the operational sequence from video/transcript to live post:

1. Generate article artifact (JSON) using `docs/blog-generation-prompt.md`.
2. Upsert artifact into Supabase table as `status=draft`.
3. Call sync endpoint for that `id` (or batch `ids`) to mirror draft into Sheet.
4. Optional CI validation: call `/api/blog/validate-draft` before switching status.
5. Publish by setting:
   - `status=published`
   - `published_at=<ISO datetime>`
6. Sync published row to Sheet.
7. Verify with:
   - `/api/blog/health`
   - `/blog/[slug]`

## 11) Caching Notes

Public list/detail are cached with `revalidate: 3600`.

Effects:

1. a newly published row in Sheets can take up to ~1 hour to appear without cache invalidation
2. local dev may still show stale data if cache is warm

If you need faster freshness for pipeline QA, lower revalidate for test environments.

## 12) File Map (Implementation Pointers)

Core blog read/validation:

- `lib/blog/sheets.ts`
- `lib/blog/types.ts`

Supabase CRUD/sync:

- `lib/blog/supabase-posts.ts`
- `lib/blog/sync-supabase-to-sheets.ts`

Public pages:

- `app/blog/page.tsx`
- `app/blog/[slug]/page.tsx`
- `components/BlogPostCard.tsx`
- `components/MarkdownContent.tsx`

Admin pages:

- `app/admin/login/page.tsx`
- `app/admin/page.tsx`
- `app/admin/blog/page.tsx`
- `app/admin/blog/[slug]/page.tsx`
- `app/admin/blog/[slug]/edit/page.tsx`
- `app/admin/blog/new/edit/page.tsx`
- `app/admin/blog/EditPostForm.tsx`

Editor internals:

- `app/admin/blog/editor/BlogTiptapEditor.tsx`
- `app/admin/blog/editor/extensions.ts`
- `lib/blog/editor-content.ts`

Pipeline APIs:

- `app/api/blog/sync-supabase-to-sheets/route.ts`
- `app/api/blog/validate-draft/route.ts`
- `app/api/blog/health/route.ts`
- `app/api/admin/blog/autosave/route.ts`
