/**
 * CDV Worker Monitor TUI
 * Run: pnpm run monitor
 *
 * Polls Supabase every 3s and renders a live terminal dashboard showing
 * all workers (local + Colab) and their current jobs.
 */

import path from 'path';
import dotenv from 'dotenv';
dotenv.config({ path: path.resolve(__dirname, '../.env.local') });

import { createClient } from '@supabase/supabase-js';
import * as readline from 'readline';

// ── Supabase ────────────────────────────────────────────────────────────────

const supabase = createClient(
    process.env.SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
);

// ── ANSI helpers ─────────────────────────────────────────────────────────────

const C = {
    reset:   '\x1b[0m',
    bold:    '\x1b[1m',
    dim:     '\x1b[2m',
    red:     '\x1b[31m',
    green:   '\x1b[32m',
    yellow:  '\x1b[33m',
    blue:    '\x1b[34m',
    magenta: '\x1b[35m',
    cyan:    '\x1b[36m',
    white:   '\x1b[37m',
    gray:    '\x1b[90m',
    bgBlue:  '\x1b[44m',
    bgGray:  '\x1b[100m',
} as const;

const W = Math.max(process.stdout.columns || 80, 60);

function clr(text: string, ...codes: string[]): string {
    return codes.join('') + text + C.reset;
}

function pad(s: string, len: number): string {
    const plain = s.replace(/\x1b\[[0-9;]*m/g, '');
    const diff = len - plain.length;
    return diff > 0 ? s + ' '.repeat(diff) : s;
}

function trunc(s: string, len: number): string {
    if (!s) return '';
    return s.length > len ? s.slice(0, len - 1) + '…' : s;
}

function hr(char = '─'): string {
    return clr(char.repeat(W), C.gray);
}

function box(title: string): string {
    const inner = ` ${title} `;
    const left = Math.floor((W - inner.length) / 2);
    const right = W - inner.length - left;
    return clr('═'.repeat(left) + inner + '═'.repeat(right), C.bgBlue, C.bold, C.white);
}

// ── Status rendering ─────────────────────────────────────────────────────────

interface StatusInfo {
    icon: string;
    color: string;
    label: string;
}

function parseStatus(status: string): StatusInfo {
    if (status === 'pending')   return { icon: '·', color: C.gray,    label: 'pending' };
    if (status === 'completed') return { icon: '✓', color: C.green,   label: 'completed' };
    if (status === 'failed')    return { icon: '✗', color: C.red,     label: 'failed' };

    if (status.startsWith('processing:ingest'))     return { icon: '↓', color: C.yellow,  label: 'ingesting' };
    if (status.startsWith('processing:transcribe')) return { icon: '◎', color: C.cyan,    label: parseTranscribeProgress(status) };
    if (status.startsWith('processing:analyze'))    return { icon: '◈', color: C.blue,    label: 'analyzing' };
    if (status.startsWith('processing:render'))     return { icon: '▶', color: C.magenta, label: 'rendering' };
    if (status.startsWith('processing:store'))      return { icon: '↑', color: C.green,   label: parseStoreProgress(status) };
    if (status.startsWith('processing:blog'))       return { icon: '✍', color: C.yellow,  label: 'blog' };
    if (status.startsWith('processing'))            return { icon: '●', color: C.yellow,  label: 'processing' };

    return { icon: '?', color: C.gray, label: status };
}

function parseTranscribeProgress(status: string): string {
    const m = status.match(/processing:transcribe:(\d+)\/(\d+)/);
    if (!m) return 'transcribing';
    const cur = Number(m[1]);
    const tot = Number(m[2]);
    return `${fmtSecs(cur)} / ${fmtSecs(tot)}`;
}

function parseStoreProgress(status: string): string {
    const m = status.match(/processing:store:(\d+)\/(\d+)/);
    if (!m) return 'storing clips';
    return `saving ${m[1]}/${m[2]} clips`;
}

function fmtSecs(s: number): string {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
    return `${m}:${String(sec).padStart(2, '0')}`;
}

function progressBar(status: string, width = 20): string {
    let pct = 0;
    if (status === 'completed') pct = 100;
    else if (status === 'failed') pct = 100;
    else if (status.startsWith('processing:transcribe:')) {
        const m = status.match(/processing:transcribe:(\d+)\/(\d+)/);
        if (m) pct = Math.round((Number(m[1]) / Number(m[2])) * 60) + 10;
    } else if (status.startsWith('processing:store:')) {
        const m = status.match(/processing:store:(\d+)\/(\d+)/);
        if (m) pct = Math.round((Number(m[1]) / Number(m[2])) * 20) + 75;
    } else if (status.startsWith('processing:ingest'))     pct = 5;
    else if (status.startsWith('processing:analyze'))      pct = 70;
    else if (status.startsWith('processing:render'))       pct = 72;
    else if (status.startsWith('processing:blog'))         pct = 90;
    else if (status.startsWith('processing'))              pct = 40;

    const filled = Math.round((pct / 100) * width);
    const empty  = width - filled;
    const barColor = status === 'failed' ? C.red : status === 'completed' ? C.green : C.cyan;
    return clr('█'.repeat(filled), barColor) + clr('░'.repeat(empty), C.gray);
}

// ── Data types ───────────────────────────────────────────────────────────────

interface Job {
    id: string;
    title: string | null;
    source_url: string | null;
    status: string;
    claimed_by: string | null;
    created_at: string;
    attempt_count: number;
    last_error: string | null;
}

// ── Rendering ────────────────────────────────────────────────────────────────

function renderHeader(lastRefresh: Date): void {
    const time = lastRefresh.toLocaleTimeString('es-AR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const title = 'CDV Worker Monitor';
    const right = `${time}  ${clr('q', C.bold, C.yellow)}=quit  ${clr('r', C.bold, C.yellow)}=refresh`;
    const rightPlain = time + '  q=quit  r=refresh';
    const gap = W - title.length - rightPlain.length;
    console.log(box(' CDV Worker Monitor '));
    console.log(clr(' ' + title + ' '.repeat(Math.max(1, gap)) + rightPlain + ' ', C.bgGray, C.white));
    console.log('');
}

function renderWorkers(jobs: Job[]): void {
    // Derive active workers from claimed_by in recent jobs
    const workerMap = new Map<string, Job | null>();
    for (const job of jobs) {
        const w = job.claimed_by || '(unclaimed)';
        if (!workerMap.has(w)) {
            const isActive = job.status.startsWith('processing');
            workerMap.set(w, isActive ? job : null);
        } else if (job.status.startsWith('processing')) {
            workerMap.set(w, job); // update to active job
        }
    }

    // Also ensure known unclaimed shows up if there are pending jobs
    if (!workerMap.has('(pending)') && jobs.some(j => j.status === 'pending')) {
        workerMap.set('(pending)', null);
    }

    if (workerMap.size === 0) {
        console.log(clr('  No workers seen yet.', C.gray));
        console.log('');
        return;
    }

    console.log(clr('  WORKERS', C.bold, C.white));
    console.log(hr());

    for (const [workerId, activeJob] of workerMap) {
        if (workerId === '(pending)') continue; // skip virtual entry
        const isColab = workerId.startsWith('colab');
        const icon = isColab ? '☁' : '🖥';
        const workerTag = clr(workerId, C.bold, isColab ? C.cyan : C.green);

        if (activeJob) {
            const { icon: statusIcon, color, label } = parseStatus(activeJob.status);
            const jobTitle = trunc(activeJob.title || activeJob.source_url || 'Untitled', 30);
            console.log(`  ${icon}  ${pad(workerTag, 24)} ${clr(statusIcon + ' ' + label, color)}  ${clr(jobTitle, C.dim)}`);
        } else {
            console.log(`  ${icon}  ${pad(workerTag, 24)} ${clr('idle', C.gray)}`);
        }
    }
    console.log('');
}

function renderJobs(jobs: Job[]): void {
    const display = jobs.slice(0, 25);

    console.log(clr(`  JOBS  ${clr('(' + jobs.length + ' total)', C.gray)}`, C.bold, C.white));
    console.log(hr());

    if (display.length === 0) {
        console.log(clr('  No jobs found.', C.gray));
        console.log('');
        return;
    }

    const BAR_W = 16;
    const TITLE_W = W - 54;

    for (const job of display) {
        const { icon, color, label } = parseStatus(job.status);
        const statusMark = clr(pad(icon + ' ' + label, 16), color);
        const workerTag  = clr(trunc(job.claimed_by || '—', 16), C.gray);
        const title      = clr(trunc(job.title || job.source_url || 'Untitled', TITLE_W), C.white);
        const bar        = progressBar(job.status, BAR_W);

        console.log(`  ${statusMark}  ${pad(title, TITLE_W)}  ${workerTag}`);
        
        // Show sub-info for active / failed jobs
        if (job.status.startsWith('processing:transcribe')) {
            const extra = parseTranscribeProgress(job.status);
            console.log(`  ${bar}  ${clr(extra, C.cyan)}`);
        } else if (job.status.startsWith('processing:store')) {
            const extra = parseStoreProgress(job.status);
            console.log(`  ${bar}  ${clr(extra, C.green)}`);
        } else if (job.status === 'failed' && job.last_error) {
            const errTrunc = trunc(job.last_error, W - 6);
            console.log(`  ${clr('└─ ' + errTrunc, C.red, C.dim)}`);
        } else if (job.status.startsWith('processing')) {
            console.log(`  ${bar}`);
        }
    }
    console.log('');
}

function renderFooter(): void {
    console.log(hr());
    console.log(clr('  Updates every 3s  │  [r] refresh now  │  [q] quit', C.gray));
}

async function fetchJobs(): Promise<Job[]> {
    const { data, error } = await supabase
        .from('jobs')
        .select('id, title, source_url, status, claimed_by, created_at, attempt_count, last_error')
        .order('created_at', { ascending: false })
        .limit(50);

    if (error) throw error;
    return (data || []) as Job[];
}

// ── Main loop ────────────────────────────────────────────────────────────────

let lastError: string | null = null;

async function render(): Promise<void> {
    let jobs: Job[] = [];
    try {
        jobs = await fetchJobs();
        lastError = null;
    } catch (err: any) {
        lastError = err?.message || 'Unknown fetch error';
    }

    process.stdout.write('\x1b[2J\x1b[H'); // clear screen

    const now = new Date();
    renderHeader(now);
    
    if (lastError) {
        console.log(clr(`  ⚠ Fetch error: ${lastError}`, C.red));
        console.log('');
    }

    renderWorkers(jobs);
    renderJobs(jobs);
    renderFooter();
}

async function main() {
    // Hide cursor
    process.stdout.write('\x1b[?25l');

    readline.emitKeypressEvents(process.stdin);
    if (process.stdin.isTTY) process.stdin.setRawMode(true);

    process.stdin.on('keypress', async (_str, key) => {
        if (!key) return;
        const k = key.name?.toLowerCase();
        if (k === 'q' || (key.ctrl && k === 'c')) {
            process.stdout.write('\x1b[?25h'); // restore cursor
            process.stdout.write('\x1b[2J\x1b[H');
            console.log(clr('  CDV Monitor stopped.', C.gray));
            process.exit(0);
        }
        if (k === 'r') {
            await render();
        }
    });

    // Initial render
    await render();

    // Poll every 3s
    setInterval(() => { void render(); }, 3000);
}

main().catch((err) => {
    process.stdout.write('\x1b[?25h');
    console.error('Monitor crashed:', err);
    process.exit(1);
});
