'use client';

import { useCallback, useEffect, useState } from 'react';
import Link from 'next/link';
import { Play, Clock, CheckCircle, AlertCircle, Trash2, X, MoreVertical, RotateCcw, Copy, Trash, Ban, MessageSquare, ExternalLink } from 'lucide-react';
import { supabase } from '@/lib/supabase';
import type { JobListResponse, JobRecord } from '@/lib/api-types';
import { JobProgress } from '@/components/JobProgress';

type Job = JobRecord;

function getThumbnailUrl(job: Job): string | null {
  if (!job.video_url) return null;
  return job.video_url.replace(/\/sermon_horizontal\.mp4(\?.*)?$/, '/thumbnail.jpg');
}

async function buildAuthHeaders(includeJson = false): Promise<HeadersInit> {
  const headers: Record<string, string> = {};
  if (includeJson) headers['content-type'] = 'application/json';

  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  if (token) headers.Authorization = `Bearer ${token}`;

  return headers;
}

export default function Home() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [apiError, setApiError] = useState<string>('');
  const [newJobUrl, setNewJobUrl] = useState('');
  const [isMounted, setIsMounted] = useState(false);
  const [openMenuId, setOpenMenuId] = useState<string|null>(null);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  const fetchJobs = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);

    try {
      const headers = await buildAuthHeaders();
      const response = await fetch('/api/v1/jobs?limit=100', {
        headers,
        cache: 'no-store'
      });

      if (response.status === 401) {
        setApiError('Authentication required. Sign in/check session.');
        setJobs([]);
        setLoading(false);
        return;
      }

      if (response.status === 403) {
        setApiError('Forbidden: You do not have access to this organization.');
        setJobs([]);
        setLoading(false);
        return;
      }

      if (!response.ok) {
        const body = await response.json().catch(() => ({ error: 'Unknown API error' }));
        setApiError(body.error || 'Failed to load jobs.');
        setJobs([]);
        setLoading(false);
        return;
      }

      const data = (await response.json()) as JobListResponse;
      console.log('Fetched jobs:', data.items?.length, data);
      setJobs(data.items || []);
      setApiError('');
      setLoading(false);
    } catch (error) {
      setApiError(error instanceof Error ? error.message : 'Failed to load jobs.');
      setJobs([]);
      setLoading(false);
    }
  }, []);

  const handleRetryJob = async (e: React.MouseEvent, id: string) => {
    e.preventDefault();
    e.stopPropagation();
    setOpenMenuId(null);
    
    try {
      const headers = await buildAuthHeaders();
      const response = await fetch(`/api/v1/jobs/${id}`, {
        method: 'PATCH',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'retry' })
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({ error: 'Failed' }));
        alert(body.error);
      } else {
        await fetchJobs(true);
      }
    } catch (err) {
      alert('Network error');
    }
  };

  const handleDeleteJob = async (e: React.MouseEvent, id: string) => {
    e.preventDefault();
    e.stopPropagation();
    setOpenMenuId(null);
    
    if (!confirm('Delete this job permanently? This cannot be undone.')) return;

    try {
      const headers = await buildAuthHeaders();
      const response = await fetch(`/api/v1/jobs/${id}`, {
        method: 'DELETE',
        headers
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({ error: 'Failed' }));
        alert(body.error);
      } else {
        await fetchJobs(true);
      }
    } catch (err) {
      alert('Network error');
    }
  };

  const handleAbortJob = async (e: React.MouseEvent, id: string) => {
    e.preventDefault();
    e.stopPropagation();
    setOpenMenuId(null);
    
    if (!confirm('Abort this job? This will stop processing.')) return;

    try {
      const headers = await buildAuthHeaders();
      const response = await fetch(`/api/v1/jobs/${id}`, {
        method: 'PATCH',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'abort' })
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({ error: 'Failed' }));
        alert(body.error);
      } else {
        await fetchJobs(true);
      }
    } catch (err) {
      alert('Network error');
    }
  };

  const handleAddJob = async () => {
    if (!newJobUrl) return;
    setLoading(true);

    try {
      const headers = await buildAuthHeaders(true);
      const response = await fetch('/api/v1/jobs', {
        method: 'POST',
        headers,
        body: JSON.stringify({ source_url: newJobUrl, title: 'New Sermon Job' })
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({ error: 'Failed to create job' }));
        alert(body.error || 'Failed to create job');
      } else {
        setNewJobUrl('');
        await fetchJobs(true);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const initial = setTimeout(() => {
      void fetchJobs();
    }, 0);
    const interval = setInterval(() => fetchJobs(true), 5000);
    return () => {
      clearTimeout(initial);
      clearInterval(interval);
    };
  }, [fetchJobs]);

  useEffect(() => {
    const handleClickOutside = () => setOpenMenuId(null);
    window.addEventListener('click', handleClickOutside);
    return () => window.removeEventListener('click', handleClickOutside);
  }, []);

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-900 selection:bg-indigo-100 selection:text-indigo-700" suppressHydrationWarning>
      <main className="max-w-5xl mx-auto px-4 py-12 sm:px-6 lg:px-8">
        
        {/* Header & Submission Section */}
        <div className="relative mb-16 text-center">
          <div className="mb-4 inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-50 border border-indigo-100 text-indigo-600 text-xs font-bold uppercase tracking-widest animate-fade-in">
            <Play size={10} fill="currentColor" />
            Live Pipeline
          </div>
          <h1 className="text-4xl sm:text-5xl font-black tracking-tight text-slate-900 mb-4">
            Sermon <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-violet-600">Shorts</span>
          </h1>
          <p className="text-slate-500 text-lg max-w-2xl mx-auto mb-10">
            Automated video intelligence for modern ministry. Drop a URL and let AI do the heavy lifting.
          </p>

          <div className="max-w-xl mx-auto relative group">
            <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-2xl blur opacity-20 group-hover:opacity-40 transition duration-1000 group-hover:duration-200"></div>
            <div className="relative flex flex-col sm:flex-row gap-3 p-2 bg-white/80 backdrop-blur-xl border border-slate-200 rounded-2xl shadow-xl">
              <input
                type="text"
                value={newJobUrl}
                onChange={(e) => setNewJobUrl(e.target.value)}
                placeholder="Paste YouTube URL or local path..."
                className="flex-1 px-5 py-3 bg-transparent border-none focus:ring-0 text-slate-700 placeholder:text-slate-400 font-medium"
                onKeyDown={(e) => e.key === 'Enter' && handleAddJob()}
              />
              <button
                onClick={handleAddJob}
                disabled={loading || !newJobUrl}
                className="px-8 py-3 bg-slate-900 text-white font-bold rounded-xl hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg active:scale-95"
              >
                {loading ? 'Starting...' : 'Process Video'}
              </button>
            </div>
          </div>
        </div>

        {apiError && (
          <div className="mb-8 p-4 rounded-2xl border border-red-100 bg-red-50/50 backdrop-blur-sm text-red-600 text-sm flex items-center gap-3 animate-shake">
            <AlertCircle size={18} />
            {apiError}
          </div>
        )}

        {/* Jobs List Section */}
        <section>
          <div className="flex items-center justify-between mb-8">
            <h2 className="text-xl font-bold flex items-center gap-2 text-slate-800">
              <Clock size={20} className="text-indigo-600" />
              Active Pipeline
            </h2>
            <div className="text-xs font-medium text-slate-400 flex items-center gap-2">
              Auto-refreshing 5s
              <button 
                onClick={() => void fetchJobs()}
                className="p-1 hover:bg-slate-100 rounded-md transition-colors"
                title="Refresh now"
              >
                <Clock size={12} />
              </button>
            </div>
          </div>

          <div className="grid gap-4">
            {jobs.map((job) => (
              <Link
                key={job.id}
                href={`/jobs/${job.id}`}
                className="group relative overflow-hidden bg-white/60 backdrop-blur-md border border-slate-200/60 rounded-3xl p-4 sm:p-5 shadow-sm hover:shadow-xl hover:border-indigo-200/50 transition-all duration-300 transform hover:-translate-y-1"
              >
                {/* Background Decor */}
                <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-50/50 rounded-full blur-3xl -mr-16 -mt-16 group-hover:bg-indigo-100/50 transition-colors" />

                <div className="relative flex flex-col sm:flex-row gap-5 items-start sm:items-center">
                  {/* Thumbnail / Icon */}
                  <div className="relative w-full sm:w-40 aspect-video bg-slate-100 rounded-2xl overflow-hidden shadow-inner shrink-0 group-hover:ring-4 ring-indigo-50 transition-all">
                    {getThumbnailUrl(job) ? (
                      <img
                        src={getThumbnailUrl(job)!}
                        alt="Thumbnail"
                        className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                        loading="lazy"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-slate-400">
                        <Play size={32} />
                      </div>
                    )}
                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center">
                      <Play size={24} className="text-white opacity-0 group-hover:opacity-100 transform scale-50 group-hover:scale-100 transition-all" />
                    </div>
                  </div>

                  {/* Info & Progress */}
                  <div className="flex-1 min-w-0 w-full space-y-4">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                      <div className="flex items-center gap-2 min-w-0 flex-1">
                        <h3 className="text-lg font-bold text-slate-900 group-hover:text-indigo-600 transition-colors truncate">
                          {job.title || 'Untitled Sermon'}
                        </h3>
                        
                        <div className="relative">
                          <button
                            onClick={(e) => {
                              e.preventDefault();
                              e.stopPropagation();
                              setOpenMenuId(openMenuId === job.id ? null : job.id);
                            }}
                            className="p-1 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition-all"
                            title="More options"
                          >
                            <MoreVertical size={18} />
                          </button>

                          {openMenuId === job.id && (
                            <div className="absolute left-0 top-full mt-1 w-48 bg-white border border-slate-200 rounded-xl shadow-xl z-50 py-1 overflow-hidden animate-in fade-in slide-in-from-top-1 duration-200">
                              {job.status !== 'completed' && job.status !== 'failed' && (
                                <button
                                  onClick={(e) => handleAbortJob(e, job.id)}
                                  className="w-full text-left px-4 py-2 text-xs font-bold text-rose-600 hover:bg-rose-50 flex items-center gap-2"
                                >
                                  <Ban size={14} /> ABORT JOB
                                </button>
                              )}
                              
                              {(job.status === 'failed' || job.status === 'completed') && (
                                <button
                                  onClick={(e) => handleRetryJob(e, job.id)}
                                  className="w-full text-left px-4 py-2 text-xs font-bold text-indigo-600 hover:bg-indigo-50 flex items-center gap-2"
                                >
                                  <RotateCcw size={14} /> RETRY JOB
                                </button>
                              )}

                              <button
                                onClick={(e) => {
                                  e.preventDefault(); e.stopPropagation();
                                  navigator.clipboard.writeText(job.id);
                                  setOpenMenuId(null);
                                }}
                                className="w-full text-left px-4 py-2 text-xs font-bold text-slate-600 hover:bg-slate-50 flex items-center gap-2"
                              >
                                <Copy size={14} /> COPY ID
                              </button>

                              <button
                                onClick={(e) => {
                                  e.preventDefault(); e.stopPropagation();
                                  window.open(job.source_url, '_blank');
                                  setOpenMenuId(null);
                                }}
                                className="w-full text-left px-4 py-2 text-xs font-bold text-slate-600 hover:bg-slate-50 flex items-center gap-2"
                              >
                                <ExternalLink size={14} /> VIEW SOURCE
                              </button>

                              <button
                                onClick={(e) => {
                                  e.preventDefault(); e.stopPropagation();
                                  alert('Comments feature coming soon!');
                                  setOpenMenuId(null);
                                }}
                                className="w-full text-left px-4 py-2 text-xs font-bold text-slate-600 hover:bg-slate-50 flex items-center gap-2"
                              >
                                <MessageSquare size={14} /> SEND COMMENTS
                              </button>

                              <div className="border-t border-slate-100 my-1"></div>

                              <button
                                onClick={(e) => handleDeleteJob(e, job.id)}
                                className="w-full text-left px-4 py-2 text-xs font-bold text-rose-600 hover:bg-rose-50 flex items-center gap-2"
                              >
                                <Trash size={14} /> DELETE PERMANENTLY
                              </button>
                            </div>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                         <span className={`px-2.5 py-1 rounded-lg text-[10px] font-black uppercase tracking-tighter ${
                          job.status === 'completed' ? 'bg-emerald-100 text-emerald-700' :
                          job.status === 'failed' ? 'bg-rose-100 text-rose-700' :
                          'bg-indigo-100 text-indigo-700'
                        }`}>
                          {job.status.replace('processing:', '').split(':')[0] || job.status}
                        </span>
                        <time className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">
                          {isMounted 
                            ? new Date(job.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
                            : '...'
                          }
                        </time>
                      </div>
                    </div>
                    
                    <JobProgress job={job} />
                  </div>
                </div>
              </Link>
            ))}

            {loading && jobs.length === 0 && (
              <div className="py-20 text-center space-y-4">
                <div className="inline-block w-8 h-8 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin" />
                <p className="text-slate-400 font-medium">Syncing pipeline status...</p>
              </div>
            )}
            
            {!loading && jobs.length === 0 && !apiError && (
              <div className="py-20 text-center bg-white/40 backdrop-blur-sm border-2 border-dashed border-slate-200 rounded-3xl">
                <p className="text-slate-400 font-medium">No jobs in the queue. Processing is idle.</p>
              </div>
            )}
          </div>
        </section>
        </main>
    </div>
  );
}

