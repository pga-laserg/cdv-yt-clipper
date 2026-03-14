-- Rename youtube_url to source_url in jobs table to support diverse ingestion sources
-- (local files, YouTube, Google Drive, direct HTTP links, etc.)
ALTER TABLE jobs RENAME COLUMN youtube_url TO source_url;
