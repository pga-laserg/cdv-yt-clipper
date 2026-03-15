-- Create processing_cache table to store results for source URLs to skip redundant stages on re-submission.
CREATE TABLE IF NOT EXISTS public.processing_cache (
    source_url TEXT PRIMARY KEY,
    transcript_hash TEXT, -- MD5 of segments to validate analysis cache
    segments JSONB NOT NULL,
    analysis_json JSONB, -- boundaries and clips
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Enable RLS
ALTER TABLE public.processing_cache ENABLE ROW LEVEL SECURITY;

-- Allow worker to read/write (assuming service_role or authenticated with proper policies)
-- For now, adding a simple policy if not exists
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE tablename = 'processing_cache' AND policyname = 'Allow service role all'
    ) THEN
        CREATE POLICY "Allow service role all" ON public.processing_cache
            FOR ALL USING (true) WITH CHECK (true);
    END IF;
END $$;

-- Add comment
COMMENT ON TABLE public.processing_cache IS 'Stores cached results (segments, analysis) for source URLs to avoid redundant processing.';
