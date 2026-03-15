import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';

export interface DefuddleSegment {
    start: number;
    end: number;
    text: string;
}

export interface DefuddleAudioEvent {
    label: string;
    start: number;
    end: number;
}

export interface DefuddleResult {
    segments: DefuddleSegment[];
    audioEvents: DefuddleAudioEvent[];
}

export async function fetchYouTubeTranscript(url: string): Promise<DefuddleResult> {
    return new Promise((resolve, reject) => {
        const args = ['defuddle', 'parse', url, '--markdown'];
        console.log(`Running: npx ${args.join(' ')}`);
        
        const child = spawn('npx', args);
        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        child.stderr.on('data', (data) => {
            stderr += data.toString();
        });

        child.on('close', (code) => {
            if (code !== 0) {
                return reject(new Error(`defuddle failed with code ${code}. stderr: ${stderr}`));
            }

            try {
                const result = parseDefuddleMarkdown(stdout);
                resolve(result);
            } catch (error) {
                reject(error);
            }
        });
    });
}

function parseDefuddleMarkdown(md: string): DefuddleResult {
    const lines = md.split('\n');
    const segments: DefuddleSegment[] = [];
    const audioEvents: DefuddleAudioEvent[] = [];

    // Regex for: **0:29** · [música] ... or **1:23:37** · ...
    const lineRegex = /^\*\*([\d:]+)\*\*\s*·\s*(.*)$/;

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;

        const match = line.match(lineRegex);
        if (match) {
            const timestampStr = match[1];
            const text = match[2];
            const start = parseTimestamp(timestampStr);
            
            // For now, we don't have the 'end' time for each segment from defuddle directly,
            // so we'll approximate it using the next segment's start or a default duration.
            segments.push({
                start,
                end: start + 2, // Placeholder, will be fixed in post-processing
                text
            });
        }
    }

    // Post-process segments to fix 'end' times
    for (let i = 0; i < segments.length; i++) {
        if (i < segments.length - 1) {
            segments[i].end = segments[i + 1].start;
        } else {
            // Last segment, use a small default
            segments[i].end = segments[i].start + 5;
        }
    }

    // Extract music events
    // We'll look for segments that contain [música], [music], or their escaped versions \[música\]
    const musicRegex = /\[m[uú]sica\]|\\\[m[uú]sica\\\]/i;
    for (const seg of segments) {
        if (musicRegex.test(seg.text)) {
            audioEvents.push({
                label: 'music',
                start: seg.start,
                end: seg.end
            });
        }
    }

    // Merge adjacent music events
    const mergedEvents: DefuddleAudioEvent[] = [];
    if (audioEvents.length > 0) {
        let current = { ...audioEvents[0] };
        for (let i = 1; i < audioEvents.length; i++) {
            if (audioEvents[i].start - current.end < 1.0) { // small gap
                current.end = audioEvents[i].end;
            } else {
                mergedEvents.push(current);
                current = { ...audioEvents[i] };
            }
        }
        mergedEvents.push(current);
    }

    return {
        segments,
        audioEvents: mergedEvents
    };
}

function parseTimestamp(ts: string): number {
    const parts = ts.split(':').map(Number);
    if (parts.length === 3) {
        // H:MM:SS
        return parts[0] * 3600 + parts[1] * 60 + parts[2];
    } else if (parts.length === 2) {
        // MM:SS
        return parts[PartIndex(parts, 0)] * 60 + parts[parts.length - 1]; // Wait, parts[0]*60 + parts[1]
    }
    return 0;
}

// Helper to avoid index issues if I ever change parts
function PartIndex(arr: any[], i: number) { return i }

function parseTimestampFixed(ts: string): number {
    const parts = ts.split(':').map(Number);
    if (parts.length === 3) {
        return parts[0] * 3600 + parts[1] * 60 + parts[2];
    }
    if (parts.length === 2) {
        return parts[0] * 60 + parts[1];
    }
    return 0;
}
