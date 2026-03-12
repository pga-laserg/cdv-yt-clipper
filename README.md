# Sermon-to-Shorts Automation Pipeline

A modern, automated pipeline designed to transform church livestreams into polished sermon videos and engaging social media shorts.

## 🚀 The Mission

Many churches lack consistent media volunteers. This project serves as a "force multiplier," reducing the weekly workload from hours of manual editing to a simple "review and approve" workflow.

## 🔄 How it Works

1.  **Ingest**: YouTube livestreams or local uploads are pulled into the system.
2.  **Transcription**: Local STT (Whisper/faster-whisper) creates a timestamped transcript.
3.  **Analysis**: AI-powered detection identifies sermon boundaries and selects high-impact highlight moments (under 60s).
4.  **Rendering**: `ffmpeg` automatically cuts the horizontal sermon-only video and vertical 9:16 shorts.
5.  **Review**: A volunteer uses the Next.js dashboard to preview, edit titles, and approve clips for publishing.

## 🛠 Tech Stack

-   **Monorepo**: Managed with [Turborepo](https://turbo.build/repo).
-   **Web App**: [Next.js](https://nextjs.org/) (Frontend & Dashboard).
-   **Worker**: Node.js engine for heavy lifting (processing & rendering).
-   **Database**: [Supabase](https://supabase.com/) (Auth, Metadata, and optional Storage).
-   **Processing**: `ffmpeg` for video manipulation and local Whisper models for transcription.

## 📁 Repository Structure

-   `apps/web`: The Next.js dashboard for reviewing and managing jobs.
-   `apps/worker`: The core logic for ingestion, transcription, and video rendering.
-   `packages/*`: Shared utilities and configurations.

## ⚡️ Getting Started

### Prerequisites

-   Node.js (v18+)
-   `ffmpeg` installed on your system.
-   A Supabase project.

### Setup

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Configure environment variables:
    -   See `apps/web/.env.example`
    -   See `apps/worker/.env.example`
4.  Run the development server:
    ```bash
    npm run dev
    ```

## 📈 Roadmap

-   [ ] Speaker diarization to better isolate the preacher.
-   [ ] Integrated captioning for vertical shorts.
-   [ ] Direct social media publishing (Postiz integration).
