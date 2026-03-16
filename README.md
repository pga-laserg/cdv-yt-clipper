# Sermon-to-Shorts Automation Pipeline

A modern, automated pipeline designed to transform church livestreams into polished sermon videos and engaging social media shorts.

## 🚀 The Mission

Many churches lack consistent media volunteers. This project serves as a "force multiplier," reducing the weekly workload from hours of manual editing to a simple "review and approve" workflow.

## 🔄 How it Works

1.  **Ingest**: YouTube livestreams or local uploads are pulled into the system.
2.  **Transcription**:
    - **Defuddle Fast-Path**: (YouTube only) Fetches existing captions and music cues immediately.
    - **ElevenLabs Scribe v2**: Cloud STT fallback for high-quality word-level timestamps.
    - **Local STT**: `faster-whisper` serves as a reliable local fallback.
3.  **Analysis**: AI-powered detection identifies sermon boundaries and selects high-impact highlight moments.
    - **Virality Scoring**: 4-factor taxonomy (Hook, Impact, Shareability, Completeness).
    - **B-roll Cues**: AI identifies moments for stock footage insertion.
4.  **Rendering**: `ffmpeg` automatically cuts the horizontal sermon-only video and vertical 9:16 shorts.
5.  **Review**: A volunteer uses the Next.js dashboard to preview, edit titles, and approve clips for publishing.

## 🛠 Tech Stack

-   **Monorepo**: Managed with [Turborepo](https://turbo.build/repo).
-   **Web App**: [Next.js](https://nextjs.org/) (Frontend & Dashboard).
-   **Worker**: Node.js engine for heavy lifting (processing & rendering).
-   **Transcription**: [Defuddle](src/lib/defuddle.ts), [ElevenLabs Scribe v2](src/pipeline/transcribe.ts), and local Whisper models.
-   **Database**: [Supabase](https://supabase.com/) (Auth, Metadata, and Processing Cache).
-   **Processing**: `ffmpeg` for video manipulation and local/cloud STT providers.

## 📁 Repository Structure

-   `apps/web`: The Next.js dashboard for reviewing and managing jobs.
-   `apps/worker`: The core logic for ingestion, transcription, and video rendering.
-   `docs/worker-pipeline.md`: Technical deep-dive into the processing pipeline.

## ⚡️ Getting Started

### Prerequisites

-   Node.js (v18+)
-   `ffmpeg` installed on your system.
-   A Supabase project.

### Setup

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pnpm install
    ```
3.  Configure environment variables:
    -   See `apps/web/.env.example`
    -   See `apps/worker/.env.example`
4.  Run the development server:
    ```bash
    pnpm dev
    ```

## 📈 Roadmap

-   [x] Speaker diarization to isolate the preacher.
-   [x] Processing Cache to skip redundant work.
-   [ ] Integrated captioning for vertical shorts.
-   [ ] Direct social media publishing (Postiz integration).


