export interface CreateJobRequest {
  source_url: string;
  title?: string;
  client_id?: string;
  content_type?: 'sermon' | 'podcast' | 'interview' | 'talk' | 'generic';
  batch_id?: string;
  monitor_id?: string;
  monitor_rule_id?: string;
  metadata?: Record<string, unknown>;
}

export interface JobRecord {
  id: string;
  organization_id: string;
  status: string;
  created_at: string;
  source_url: string;
  title?: string | null;
  video_url?: string | null;
  srt_url?: string | null;
  sermon_start_seconds?: number | null;
  sermon_end_seconds?: number | null;
  metadata?: Record<string, unknown> | null;
}

export interface ClipRecord {
  id: string;
  job_id: string;
  organization_id: string;
  start_seconds: number;
  end_seconds: number;
  title?: string | null;
  transcript_excerpt?: string | null;
  status: string;
  video_url?: string | null;
}

export interface JobListResponse {
  items: JobRecord[];
  count: number;
  limit: number;
  offset: number;
}

export interface JobDetailResponse {
  job: JobRecord;
  clips: ClipRecord[];
}

export interface BlogProfileRecord {
  organization_id: string;
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
  preserve_published_fields: boolean;
  field_rules: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface CreateBlogProfileRequest {
  client_id: string;
  enabled?: boolean;
  llm_provider?: string;
  llm_model?: string;
  prompt_version?: string;
  system_prompt?: string | null;
  user_prompt_template?: string | null;
  default_author_name?: string;
  default_status?: string;
  sync_enabled?: boolean;
  sync_endpoint?: string | null;
  preserve_published_fields?: boolean;
  field_rules?: Record<string, unknown>;
}
