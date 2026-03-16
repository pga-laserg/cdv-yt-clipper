'use client';

import { useCallback, useEffect, useState, use } from 'react';
import { supabase } from '@/lib/supabase';
import type { ClipRecord, JobDetailResponse, JobRecord } from '@/lib/api-types';
import { ChevronLeft, Play, CheckCircle, XCircle, Download, Scissors, AlertCircle, X, MoreVertical, RotateCcw, Copy, Trash, Ban, MessageSquare, ExternalLink } from 'lucide-react';
import Link from 'next/link';
import { JobProgress } from '@/components/JobProgress';

type Clip = ClipRecord;
type Job = JobRecord;

async function buildAuthHeaders(includeJson = false): Promise<HeadersInit> {
    const headers: Record<string, string> = {};
    if (includeJson) headers['content-type'] = 'application/json';

    const { data } = await supabase.auth.getSession();
    const token = data.session?.access_token;
    if (token) headers.Authorization = `Bearer ${token}`;

    return headers;
}

function parseFilenameFromDisposition(disposition: string | null): string | null {
    if (!disposition) return null;
    const match = disposition.match(/filename="([^"]+)"/i) || disposition.match(/filename=([^;]+)/i);
    if (!match?.[1]) return null;
    return match[1].trim();
}

export default function JobDetails({ params: paramsPromise }: { params: Promise<{ id: string }> }) {
    const params = use(paramsPromise);
    const [job, setJob] = useState<Job | null>(null);
    const [clips, setClips] = useState<Clip[]>([]);
    const [loading, setLoading] = useState(true);
    const [apiError, setApiError] = useState('');
    const [downloading, setDownloading] = useState(false);
    const [isMounted, setIsMounted] = useState(false);
    const [openMenu, setOpenMenu] = useState(false);

    useEffect(() => {
        setIsMounted(true);
    }, []);

    const fetchJobAndClips = useCallback(async (silent = false) => {
        if (!silent) setLoading(true);
        try {
            const headers = await buildAuthHeaders();
            const response = await fetch(`/api/v1/jobs/${params.id}`, {
                headers,
                cache: 'no-store'
            });

            if (response.status === 401) {
                setApiError('Authentication required. Sign in to view this job.');
                setJob(null);
                setClips([]);
                setLoading(false);
                return;
            }

            if (!response.ok) {
                const body = await response.json().catch(() => ({ error: 'Failed to fetch job.' }));
                setApiError(body.error || 'Failed to fetch job.');
                setJob(null);
                setClips([]);
                setLoading(false);
                return;
            }

            const data = (await response.json()) as JobDetailResponse;
            setJob(data.job);
            setClips(data.clips || []);
            setApiError('');
        } catch (error) {
            setApiError(error instanceof Error ? error.message : 'Failed to fetch job.');
            setJob(null);
            setClips([]);
        } finally {
            setLoading(false);
        }
    }, [params.id]);

    useEffect(() => {
        const initial = setTimeout(() => {
            void fetchJobAndClips();
        }, 0);
        const interval = setInterval(() => fetchJobAndClips(true), 5000);
        return () => {
            clearTimeout(initial);
            clearInterval(interval);
        };
    }, [fetchJobAndClips]);

    const downloadFullRes = useCallback(async () => {
        if (!job || downloading) return;
        setDownloading(true);

        try {
            const headers = await buildAuthHeaders();
            const response = await fetch(`/api/jobs/${job.id}/download`, { headers });
            if (!response.ok) {
                const text = await response.text();
                throw new Error(text || 'Download failed.');
            }

            const blob = await response.blob();
            const objectUrl = URL.createObjectURL(blob);
            const filename =
                parseFilenameFromDisposition(response.headers.get('content-disposition')) ||
                `job-${job.id}-fullres.mp4`;

            const anchor = document.createElement('a');
            anchor.href = objectUrl;
            anchor.download = filename;
            document.body.appendChild(anchor);
            anchor.click();
            anchor.remove();
            setTimeout(() => URL.revokeObjectURL(objectUrl), 30_000);
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Download failed.';
            alert(message);
        } finally {
            setDownloading(false);
        }
    }, [downloading, job]);

    const handleAbortJob = async () => {
        if (!job) return;
        if (!confirm('Abort this job? This will stop processing.')) return;

        try {
            const headers = await buildAuthHeaders();
            const response = await fetch(`/api/v1/jobs/${job.id}`, {
                method: 'PATCH',
                headers: { ...headers, 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: 'abort' })
            });

            if (!response.ok) {
                const body = await response.json().catch(() => ({ error: 'Failed' }));
                alert(body.error);
            } else {
                void fetchJobAndClips(true);
            }
        } catch (err) {
            alert('Network error');
        }
    };

    const handleRetryJob = async () => {
        if (!job) return;
        try {
            const headers = await buildAuthHeaders();
            const response = await fetch(`/api/v1/jobs/${job.id}`, {
                method: 'PATCH',
                headers: { ...headers, 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: 'retry' })
            });

            if (!response.ok) {
                const body = await response.json().catch(() => ({ error: 'Failed' }));
                alert(body.error);
            } else {
                void fetchJobAndClips(true);
            }
        } catch (err) {
            alert('Network error');
        }
    };

    const handleDeleteJob = async () => {
        if (!job) return;
        if (!confirm('Delete this job permanently? This cannot be undone.')) return;

        try {
            const headers = await buildAuthHeaders();
            const response = await fetch(`/api/v1/jobs/${job.id}`, {
                method: 'DELETE',
                headers
            });

            if (!response.ok) {
                const body = await response.json().catch(() => ({ error: 'Failed' }));
                alert(body.error);
            } else {
                window.location.href = '/';
            }
        } catch (err) {
            alert('Network error');
        }
    };

    return (
        <div className="min-h-screen bg-[#f8fafc] p-4 sm:p-12" suppressHydrationWarning>
            <header className="mb-12 flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-center gap-4 min-w-0">
                    <Link href="/" className="p-3 bg-white shadow-sm border border-slate-200 rounded-2xl hover:bg-slate-50 transition-colors">
                        <ChevronLeft size={20} />
                    </Link>
                    <div>
                        <span className="text-[10px] font-black uppercase tracking-widest text-indigo-600 mb-1 block">Detail View</span>
                        <h1 className="text-2xl sm:text-4xl font-black text-slate-900 truncate tracking-tight">{job?.title || 'Processing Video'}</h1>
                    </div>
                </div>

                <div className="flex flex-col sm:flex-row gap-3">
                    {job?.srt_url && (
                        <button
                            className="flex items-center justify-center gap-2 px-6 py-3 bg-indigo-50 text-indigo-700 font-bold rounded-2xl hover:bg-indigo-100 transition-colors border border-indigo-100 w-full sm:w-auto"
                            onClick={() => {
                                if (job.srt_url) window.open(job.srt_url);
                            }}
                        >
                            <Download size={18} />
                            SRT
                        </button>
                    )}
                    {job && (
                        <button
                            className="flex items-center justify-center gap-2 px-6 py-3 bg-slate-900 text-white font-bold rounded-2xl hover:bg-slate-800 transition-all shadow-lg active:scale-95 w-full sm:w-auto disabled:opacity-50"
                            disabled={downloading}
                            onClick={() => void downloadFullRes()}
                        >
                            <Download size={18} />
                            {downloading ? 'Packing...' : 'Full Resolution'}
                        </button>
                    )}
                    {job && (
                        <div className="relative">
                            <button
                                className="flex items-center justify-center gap-2 p-3 bg-white border border-slate-200 text-slate-600 font-bold rounded-2xl hover:bg-slate-50 transition-all shadow-sm active:scale-95"
                                onClick={() => setOpenMenu(!openMenu)}
                            >
                                <MoreVertical size={20} />
                            </button>

                            {openMenu && (
                                <div className="absolute right-0 top-full mt-2 w-56 bg-white border border-slate-200 rounded-2xl shadow-2xl z-50 py-2 animate-in fade-in zoom-in-95 duration-200">
                                    {job.status !== 'completed' && job.status !== 'failed' && (
                                        <button
                                            onClick={() => { setOpenMenu(false); void handleAbortJob(); }}
                                            className="w-full text-left px-4 py-3 text-sm font-bold text-rose-600 hover:bg-rose-50 flex items-center gap-3"
                                        >
                                            <Ban size={18} /> Abort Pipeline
                                        </button>
                                    )}

                                    { (job.status === 'failed' || job.status === 'completed') && (
                                        <button
                                            onClick={() => { setOpenMenu(false); void handleRetryJob(); }}
                                            className="w-full text-left px-4 py-3 text-sm font-bold text-indigo-600 hover:bg-indigo-50 flex items-center gap-3"
                                        >
                                            <RotateCcw size={18} /> Restart Process
                                        </button>
                                    )}

                                    <button
                                        onClick={() => {
                                            setOpenMenu(false);
                                            navigator.clipboard.writeText(window.location.href);
                                            alert('Link copied to clipboard!');
                                        }}
                                        className="w-full text-left px-4 py-3 text-sm font-bold text-slate-700 hover:bg-slate-50 flex items-center gap-3"
                                    >
                                        <Copy size={18} /> Copy Share Link
                                    </button>

                                    <button
                                        onClick={() => { setOpenMenu(false); alert('Comments feature coming soon!'); }}
                                        className="w-full text-left px-4 py-3 text-sm font-bold text-slate-700 hover:bg-slate-50 flex items-center gap-3"
                                    >
                                        <MessageSquare size={18} /> Feedback / Comments
                                    </button>

                                    <div className="mx-2 my-2 border-t border-slate-100"></div>

                                    <button
                                        onClick={() => { setOpenMenu(false); void handleDeleteJob(); }}
                                        className="w-full text-left px-4 py-3 text-sm font-bold text-rose-600 hover:bg-rose-50 flex items-center gap-3"
                                    >
                                        <Trash size={18} /> Delete Permanently
                                    </button>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </header>

            {apiError && (
                <div className="mb-8 p-4 rounded-2xl border border-red-100 bg-red-50 text-red-600 text-sm flex items-center gap-3">
                    <AlertCircle size={18} />
                    {apiError}
                </div>
            )}

            {job && (
                <section className="mb-12 bg-white/60 backdrop-blur-md p-8 rounded-[2.5rem] border border-slate-200 shadow-xl shadow-slate-200/50">
                    <JobProgress job={job} />
                </section>
            )}

            {job?.video_url && (
                <section className="mb-12 bg-white p-6 rounded-2xl border border-gray-200 shadow-sm">
                    <h2 className="text-xl font-bold mb-4">Full Sermon (Horizontal)</h2>
                    <div className="aspect-video bg-black rounded-xl overflow-hidden shadow-inner max-w-4xl mx-auto">
                        <video src={job.video_url} controls className="w-full h-full" />
                    </div>
                </section>
            )}

            <h2 className="text-lg sm:text-xl font-bold mb-6">Generated Short Clips (Vertical)</h2>

            <main className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {clips.map((clip) => (
                    <div key={clip.id} className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden hover:shadow-md transition-shadow">
                        <div className="aspect-[9/16] bg-black flex items-center justify-center relative group">
                            {clip.video_url ? (
                                <video
                                    src={clip.video_url}
                                    controls
                                    className="w-full h-full object-cover"
                                />
                            ) : (
                                <>
                                    <Play size={48} className="text-white opacity-50 group-hover:opacity-100 transition-opacity" />
                                    <div className="absolute bottom-4 left-4 right-4 bg-black/60 p-2 rounded text-xs text-white">
                                        {Math.floor(clip.start_seconds)}s - {Math.floor(clip.end_seconds)}s
                                    </div>
                                </>
                            )}
                        </div>

                        <div className="p-6">
                            <h3 className="font-bold text-lg mb-2">{clip.title}</h3>
                            <p className="text-sm text-gray-600 line-clamp-3 mb-6 italic">
                                &quot;{clip.transcript_excerpt}&quot;
                            </p>

                            <div className="flex flex-col sm:flex-row gap-2">
                                <Link
                                    href={`/admin/clips/${clip.id}`}
                                    className="flex-1 flex items-center justify-center gap-2 py-2 bg-blue-50 text-blue-600 rounded-xl font-medium hover:bg-blue-100 transition-colors"
                                >
                                    <Scissors size={18} />
                                    Trim
                                </Link>
                                <button className="flex-1 flex items-center justify-center gap-2 py-2 bg-green-600 text-white rounded-xl font-medium hover:bg-green-700 transition-colors">
                                    <CheckCircle size={18} />
                                    Approve
                                </button>
                                <button className="flex-1 flex items-center justify-center gap-2 py-2 bg-red-50 text-red-600 rounded-xl font-medium hover:bg-red-100 transition-colors">
                                    <XCircle size={18} />
                                    Reject
                                </button>
                            </div>
                        </div>
                    </div>
                ))}

                {clips.length === 0 && !loading && (
                    <div className="col-span-full p-16 text-center text-gray-400 bg-white rounded-3xl border-2 border-dashed border-gray-100">
                        No clips generated yet. Wait for the processing to finish.
                    </div>
                )}
                </main>
        </div>
    );
}
