# SupoClip vs CDV Pipeline – Comparison & Improvement Map

> Analyzed: [FujiwaraChoki/supoclip](https://github.com/FujiwaraChoki/supoclip) vs CDV `/apps/worker/src/pipeline/`

---

## TL;DR

SupoClip is a general-purpose video clipper (any genre). CDV is a specialized sermon pipeline. The two share ~60% of the same conceptual stages but diverge sharply on scoring philosophy and rendering. SupoClip's **virality scoring model** is the most transferable idea.

---

## Architecture Comparison

| Dimension | SupoClip (Python, FastAPI) | CDV (TypeScript, Node.js) |
|---|---|---|
| Language | Python 3.12 + `pydantic-ai` | TypeScript / Node.js |
| LLM Provider | Pluggable (Google, OpenAI, Anthropic, Ollama) | OpenAI only (`gpt-5-mini` or `ANALYZE_OPENAI_MODEL`) |
| Transcript source | YouTube captions / Whisper | Google STT / AssemblyAI / Defuddle (in plan) |
| Clip selection | Single LLM call → pass1 → validate → sort | 2-pass OpenAI calls → local re-scoring |
| Score storage | SQLAlchemy ORM → Postgres (4 int columns) | Supabase `clips.confidence_score` (single float) |
| Rendering | Python `moviepy` + FFmpeg | TypeScript `fluent-ffmpeg` + FFmpeg |
| Delivery format | Local files only | Multi-storage (Railway, Proxmox/SSH, Supabase) |
| Content type | Generic / social media | Sermon / religious |
| B-roll detection | ✅ Yes (LLM identifies cue points) | ❌ No |
| Captions / subtitles | ✅ ASS/SSA burned-in captions | ✅ SRT generated but not burned-in |
| Blog artifact stage | ❌ No | ✅ Yes (Stage 6) |
| Domain specificity | None | Sermon boundary detection, bilingual amen/amén |

---

## 🔑 Virality Scoring — The Key Differentiator

### SupoClip's Virality System (`ai.py`)

SupoClip uses an **LLM-driven 4-subscore system on a 0–100 scale**. The LLM itself produces all four numbers; the backend validates they sum correctly.

```
total_score = hook_score + engagement_score + value_score + shareability_score
              (0-25)       (0-25)             (0-25)        (0-25)
```

**Four subscores defined in the system prompt:**

| Subscore | 0–25 Range | What it captures |
|---|---|---|
| `hook_score` | Attention-grabbing opening. 20-25 = surprising fact/bold claim/question | First-second retention |
| `engagement_score` | Entertaining, emotional, or dramatic content | Watch-through rate |
| `value_score` | Actionable insights or unique knowledge | Save/bookmark rate |
| `shareability_score` | "I need to send this to someone" content | Share/repost rate |

On top of this, SupoClip asks the LLM to also classify the **hook type**: `question`, `statement`, `statistic`, `story`, `contrast`, or `none`.

**Sort order:** by `(virality.total_score, relevance_score)` descending.

---

### CDV's Score System (`highlights.ts`)

CDV uses a **hybrid model**: LLM outputs `score` and `confidence` as raw floats, then locally computes a weighted composite:

```typescript
final_score =
  virality       * 0.50   // model_virality (LLM's "score" = viral potential)
  + confidence   * 0.10   // model_confidence (LLM's self-assessed coherence)
  + durScore     * 0.05   // duration_preference (smooth curve, prefers ~90s)
  + endScore     * 0.35   // ending_completeness (regex + keyword heuristic)
```

**Ending completeness heuristic** (local, no LLM):
```typescript
if (/[.!?]["')]?\s*$/.test(tail))         → 1.0  (complete sentence)
if (/\b(amén|amen|gracias|oramos)\b/i...) → 0.9  (sermon closure)
if (/[,;:]\s*$/)                          → 0.45 (incomplete)
else                                       → 0.65
```

**Score breakdown is stored** in `score_breakdown: { model_virality, model_confidence, duration_preference, ending_completeness }`.

---

### Scoring Comparison

| | SupoClip | CDV |
|---|---|---|
| Score scale | 0–100 (composite int) | 0–1.0 (composite float) |
| Scoring agent | LLM produces all sub-scores | LLM gives 2 floats, local code adds 2 more |
| Sub-dimensions | Hook, Engagement, Value, Shareability | VIrality, Confidence, Duration, EndingComplete |
| Hook typed | ✅ Yes (6 types) | ❌ No |
| Objective | Generic platform virality | Sermon completeness & spiritual resonance |
| Sort input | virality score primary | composite score primary |

---

## LLM Prompt Comparison

### SupoClip system prompt (key excerpt)

```
You are an expert at analyzing video transcripts to find the most engaging
segments for short-form content creation with viral potential.

VIRALITY SCORING (0-100 total, from four 0-25 subscores):
1. HOOK STRENGTH (0-25): Immediately grabs attention (surprising fact, bold claim...)
2. ENGAGEMENT (0-25): Highly entertaining, emotional, or dramatic
3. VALUE (0-25): Actionable insights, unique knowledge, or transformative ideas
4. SHAREABILITY (0-25): "I need to send this to someone" content

B-ROLL OPPORTUNITIES: Identify 2-4 moments per segment...

TIMING GUIDELINES: Segments MUST be 10-45 seconds for optimal engagement
```

**Format:** Pydantic-AI structured output (`TranscriptAnalysis` model with typed fields).  
**Provider:** Pluggable (Google/OpenAI/Anthropic/Ollama via `config.llm`).

---

### CDV user prompt (from `highlights.ts`, `askHighlights()`)

```
Select social-ready sermon highlights from this sermon-only payload.
Output JSON only in this schema:
{"highlights":[{"start":number,"end":number,"title":string,"excerpt":string,
"hook":string,"confidence":number,"score":number,"why_end_complete":"short"}]}

Constraints:
- return exactly N highlight(s)
- "score" must mean viral potential of the idea itself on a 0..1 scale
- "confidence" must mean your confidence that the selected clip is coherent, complete, and correctly bounded
- each duration must be between {min}s and {max}s
- avoid clipping before the core idea lands; ending completeness is critical
- spread clips across different sermon moments
- selection profile: {profile}  ← standard | dense | arc

Sermon bounds: {start}-{end}
Already selected ranges (avoid overlap): ...
Chapter hints: ...
Paragraphs (preferred source): ...
Transcript fallback: ...
```

**Format:** JSON mode (`response_format: { type: 'json_object' }`).  
**Provider:** OpenAI only.  
**Key difference:** CDV runs **2 passes** (pass1 fills N slots, pass2 fills remaining), SupoClip does one shot.

---

## What SupoClip Does Better

### 1. ✅ Structured virality subscores exposed to users
SupoClip stores and **surfaces** 4 integer sub-scores per clip in the DB. Users can see exactly why a clip was ranked high. CDV stores a breakdown but it's buried in metadata JSON.

### 2. ✅ Hook type classification
SupoClip asks the LLM to label whether a clip opens with a `question`, `statement`, `statistic`, `story`, `contrast`, or none. This is useful for A/B testing which hook types perform best for a given channel.

### 3. ✅ Multi-provider LLM support
SupoClip can swap OpenAI → Google → Anthropic → Ollama via `LLM=` env var. CDV is currently OpenAI-only. This matters for cost and latency tuning.

### 4. ✅ B-roll detection
SupoClip's LLM identifies 2–4 moments per clip where stock footage would work, with searchable keywords (e.g., `"coffee shop"`, `"money stack"`). CDV has no equivalent.

### 5. ✅ Processing cache
SupoClip has a `ProcessingCache` model that caches transcript + analysis by source URL. CDV re-runs the full ingest+transcribe pipeline on every job re-submission.

### 6. ✅ Shorter clip target (10–45s)
SupoClip targets the TikTok/Reels sweet spot. CDV targets 30–180s, which is more appropriate for sermon content but may not be optimized for social.

---

## What CDV Does Better

### 1. ✅ Sermon-domain expertise
CDV's scoring intentionally weights **ending completeness** at 35% (very high). For sermons, a clip that ends mid-thought is worthless. SupoClip's generic model doesn't know about amen endings, spiritual closure, or bilingual sermons.

### 2. ✅ 2-pass highlight selection with used-range tracking
CDV's two LLM passes with explicit `usedRanges` context prevents duplicates even when the LLM hallucinates. SupoClip's single-shot approach may return overlapping or redundant segments.

### 3. ✅ Profile system (standard / dense / arc)
CDV's `HIGHLIGHTS_PROFILE` env var changes both the scoring weights and the prompt guidance. SupoClip has no equivalent concept.

### 4. ✅ Chapter + paragraph context injection
CDV passes the polished `transcript.polished.json` (with chapters and paragraphs) as context to the LLM. SupoClip sends raw transcript text only.

### 5. ✅ End-to-end pipeline with blog artifact
CDV generates a blog post from the sermon in Stage 6. SupoClip is clip-only.

### 6. ✅ Production delivery infrastructure
CDV routes to Railway (vertical), Proxmox (full-res), and Supabase Storage. SupoClip delivers only to local disk.

---

## How SupoClip Can Improve CDV

### 🔴 High Priority: Expand virality scoring with hook taxonomy

**Current state:** CDV asks the LLM for a single `score` float (0–1) labeled "viral potential".

**Improvement:** Split this into 4 CDV-specific subscores, still weighted by the composite formula:

| New subscore | Replaces | CDV mapping |
|---|---|---|
| `hook_strength` | `model_virality` (partial) | Opening line grab: question, bold claim, scripture citation |
| `spiritual_impact` | `model_virality` (partial) | Theological depth, conviction moments |
| `shareability` | `model_confidence` (repurpose) | "I'm sending this to someone" relevance |
| `ending_completeness` | Keep existing local heuristic, but also ask LLM | Validate against heuristic |

Add `hook_type` classification to the prompt schema:
```
"hook_type": "question" | "promise" | "scripture" | "story" | "contrast" | "none"
```

This gives the admin UI meaningful filters like "show only clips with hook_type=promise".

---

### 🟡 Medium Priority: Add B-roll cue detection to the LLM prompt

**Current state:** CDV has no B-roll stage.

**Improvement:** Add an optional `include_broll: boolean` flag to `findHighlights()`. When enabled, extend the LLM prompt to request 2–3 moments per clip suitable for cutaway footage, with searchable keywords. These can be stored in clip metadata and shown in the admin trimmer (`/admin/clips/[clipId]`).

Practical use: the admin could pull Getty/Pexels images or short stock clips at the identified cue points.

---

### 🟡 Medium Priority: Multi-provider LLM support

**Current state:** OpenAI hardcoded throughout `highlights.ts`.

**Improvement:** Create a provider abstraction (similar to SupoClip's `config.llm` pattern) so `HIGHLIGHTS_LLM_PROVIDER` can switch between `openai`, `google`, or `anthropic`. This is particularly relevant now that Gemini 2.0 Flash is dramatically cheaper than GPT-5-mini for transcript workloads.

---

### 🟢 Low Priority: Processing cache for re-submissions

**Current state:** Job re-submission reruns all stages.

**Improvement:** Cache `(source_url, transcript_hash)` → `{ segments, analysis_json }` in Supabase. On re-submit, skip ingest + transcribe if cache hit exists. This mirrors SupoClip's `ProcessingCache` table.

---

### 🟢 Low Priority: Surface score breakdown in admin UI

**Current state:** `score_breakdown` is stored in `clips.confidence_score` and metadata JSON, not visible in the admin.

**Improvement:** Expose it in the `/admin/clips/[clipId]` trimmer as a small scorecard showing virality, confidence, duration preference, and ending completeness bars. This helps users understand why a clip was ranked high/low and enables smarter manual curation.

---

## Virality Scoring — Recommended Formula Merge

Here is a concrete proposed upgrade to CDV's `scoreClip()` in `highlights.ts`:

```typescript
// PROPOSED: expanded score breakdown aligned with SupoClip's 4-factor taxonomy
score_breakdown: {
  hook_strength:        // 0..1 – LLM rated (new subscore in prompt)
  spiritual_impact:     // 0..1 – LLM rated (replaces flat "virality" ask)
  shareability:         // 0..1 – LLM rated (new subscore in prompt)
  ending_completeness:  // 0..1 – keep local heuristic (as strong signal)
}

// Weights (still configurable):
final_score =
  hook_strength      * 0.20
  spiritual_impact   * 0.30
  shareability       * 0.15
  ending_completeness * 0.35   // keep highest – sermon-critical
```

This preserves CDV's ending completeness dominance while adding the richer signal structure from SupoClip.

---

## What NOT to Take from SupoClip

| Feature | Why to skip |
|---|---|
| 10–45s clip range | Too short for sermon content; 60–120s is right for the medium |
| Generic hook taxonomy | "statistic" and "money stack" B-roll are not relevant to religious content |
| Pydantic-AI agent structure | CDV is TypeScript; would require a full Python service rewrite |
| Single-shot LLM call | CDV's 2-pass approach with used-range tracking is strictly better |
| `ProcessingCache` (for now) | Adds DB complexity; only worth it once job volume justifies it |
