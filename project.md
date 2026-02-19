# Project Purpose: “Sermon-to-Shorts Automation Pipeline”

## One-liner

Turn a weekly church livestream recording into (1) a clean sermon-only video and (2) several draft vertical short clips, then present them in a review dashboard so a volunteer can approve and publish them.

⸻

## Why this exists (the problem)

The church already livestreams weekly, but:
	•	No consistent media volunteer
	•	Full livestreams include announcements, music, dead time
	•	Editing into “sermon-only” + “short clips” is time-consuming
	•	Social posting is fragmented across platforms

So the pipeline’s job is to reduce the weekly workload from “hours of editing” to “review and click approve”.

⸻

## Desired weekly workflow (from the church’s perspective)
	1.	Church finishes streaming on YouTube
	2.	The system automatically prepares:
	•	Sermon-only cut (horizontal)
	•	6 vertical clips (<60 seconds), with titles + transcript snippets
	3.	A volunteer opens a dashboard:
	•	previews each clip
	•	edits the title if needed
	•	approves/rejects
	•	downloads clips or sends them to a scheduler later (Postiz integration is optional/stub)

⸻

## What the pipeline must produce (outputs)

For each “job” (one livestream or one video):

### Must-have outputs
	•	sermon_horizontal.mp4 (sermon only, no announcements)
	•	shorts/clip_01.mp4 … clip_06.mp4 (vertical 9:16, each < 60s)
	•	manifest.json containing:
	•	sermon start/end timestamps
	•	clip timestamp ranges
	•	suggested titles
	•	transcript excerpts (“quote”)
	•	confidence values + notes

### Must-have UI

A “Review Queue” web page that shows:
	•	job status and logs
	•	preview players for each clip
	•	approve/reject buttons
	•	title editing
	•	download buttons
	•	export approved manifest

⸻

## High-level system components

This is a Node-first system with three major components:
	1.	Worker (automation engine)
	•	downloads or loads video
	•	runs transcription
	•	analyzes transcript for sermon boundaries and highlight moments
	•	renders clips with ffmpeg
	•	stores results + metadata
	2.	Web App (Next.js)
	•	dashboard for jobs/clips
	•	review & approval UI
	•	triggers job runs (manual button)
	•	later can integrate Postiz, but not required now
	3.	Database (Supabase)
	•	stores job state + clip metadata
	•	optionally stores files in Supabase Storage

⸻

## Pipeline stages (what the worker does)

### Stage 1: Ingest

Input can be:
	•	YouTube URL (typical) OR
	•	local file (for testing)

The system saves:
	•	source.mp4
	•	extracted audio.wav

### Stage 2: Transcribe locally

Run local speech-to-text (Whisper / faster-whisper):
	•	output timestamped segments like:
	•	{start_sec, end_sec, text}

This transcript becomes the “source of truth” for decisions.

### Stage 3: Find sermon boundaries

Goal: identify sermon start/end and ignore:
	•	announcements (“anuncios”)
	•	welcome section
	•	music/cantos
	•	offering / church housekeeping

This can be:
	•	heuristic first (keywords)
	•	optionally refined by an LLM

Output:
	•	sermon_start_sec, sermon_end_sec

### Stage 4: Select highlight clips (<60s)

Goal: pick 4–8 moments that are:
	•	meaningful out of context
	•	short enough for Reels/Shorts
	•	not cut mid-thought
	•	spread across the sermon (not all from one part)

There are multiple “strategies” to test:
	•	DeepSeek LLM analysis
	•	OpenAI LLM analysis
	•	Cohere embeddings + clustering (no generative model for selection)

Output:
	•	list of {start_sec, end_sec, title, excerpt, hook, confidence}



### Stage 5: Render videos with ffmpeg

Create:
	•	sermon-only horizontal mp4
	•	vertical clips (9:16)

For v1, captions can be optional (not required).
The key is correct cutting.

### Stage 6: Store + queue for review

Write:
	•	job status + logs
	•	clip records
	•	asset paths/URLs
into Supabase.

Dashboard reads from Supabase to show review queue.

⸻

## Important constraints (product decisions)
	•	Draft-first: never auto-publish clips.
	•	Non-technical users: volunteer should only review and click approve.
	•	Low recurring costs: designed to run with minimal paid services.
	•	Portability: must run in 3 environments:
	•	Hetzner VPS
	•	Proxmox server (Chile)
	•	MacBook Air M1

⸻

## Success criteria

A prototype is “successful” if:
	•	Given a YouTube livestream URL, it produces:
	•	sermon-only cut
	•	6 shorts under 60 seconds
	•	manifest.json
	•	review dashboard entries
	•	Works end-to-end at least on one environment
	•	The three “clip selection” strategies can be swapped via config
	•	Produces results without manual editing in Premiere/Resolve