#!/usr/bin/env tsx
/**
 * Standalone script to clip a sermon video with fade in and fade out effects.
 * 
 * Usage:
 *   tsx apps/worker/src/clip-sermon.ts <input> <start> <end> <output> [fadeIn] [fadeOut]
 * 
 * Example:
 *   tsx apps/worker/src/clip-sermon.ts source.mp4 2324.96 5310.2 output.mp4 2 2
 */

import ffmpeg from 'fluent-ffmpeg';
import path from 'path';

type FadeOpts = { fadeInSec?: number; fadeOutSec?: number };

/**
 * Cut a video segment with optional fade in/out effects.
 * 
 * @param input - Path to the input video file
 * @param start - Start time in seconds
 * @param end - End time in seconds
 * @param output - Path to the output video file
 * @param fades - Optional fade in/out durations in seconds
 */
function cutVideo(input: string, start: number, end: number, output: string, fades: FadeOpts = {}): Promise<void> {
    return new Promise((resolve, reject) => {
        console.log(`Cutting sermon: ${start}s to ${end}s`);
        const duration = end - start;
        const fadeIn = Math.max(0, Math.min(fades.fadeInSec ?? 0, duration / 2));
        const fadeOut = Math.max(0, Math.min(fades.fadeOutSec ?? 0, duration / 2));

        const vf: string[] = [];
        const af: string[] = [];

        if (fadeIn > 0) {
            vf.push(`fade=t=in:st=0:d=${fadeIn}`);
            af.push(`afade=t=in:st=0:d=${fadeIn}`);
        }
        if (fadeOut > 0) {
            vf.push(`fade=t=out:st=${Math.max(0, duration - fadeOut)}:d=${fadeOut}`);
            af.push(`afade=t=out:st=${Math.max(0, duration - fadeOut)}:d=${fadeOut}`);
        }

        const cmd = ffmpeg(input)
            .setStartTime(start)
            .setDuration(duration)
            .videoCodec('libx264')
            .audioCodec('aac')
            .outputOptions([
                '-preset veryfast',
                '-crf 20',
                '-movflags +faststart',
                '-b:a 192k'
            ])
            .output(output)
            .on('end', () => {
                console.log(`Successfully created: ${output}`);
                resolve();
            })
            .on('error', (err) => {
                console.error(`Error cutting video: ${err.message}`);
                reject(err);
            });

        if (vf.length) cmd.videoFilters(vf.join(','));
        if (af.length) cmd.audioFilters(af.join(','));

        console.log(`Starting FFmpeg process...`);
        console.log(`Input: ${input}`);
        console.log(`Output: ${output}`);
        console.log(`Duration: ${duration.toFixed(2)}s (${start.toFixed(2)}s - ${end.toFixed(2)}s)`);
        console.log(`Fade In: ${fadeIn.toFixed(2)}s, Fade Out: ${fadeOut.toFixed(2)}s`);

        cmd.run();
    });
}

/**
 * Main execution function.
 */
async function main() {
    const args = process.argv.slice(2);

    if (args.length < 4) {
        console.error('Usage: tsx clip-sermon.ts <input> <start> <end> <output> [fadeIn] [fadeOut]');
        console.error('Example: tsx clip-sermon.ts source.mp4 2324.96 5310.2 output.mp4 2 2');
        process.exit(1);
    }

    const [input, startStr, endStr, output, fadeInStr, fadeOutStr] = args;

    const start = parseFloat(startStr);
    const end = parseFloat(endStr);
    const fadeIn = fadeInStr ? parseFloat(fadeInStr) : 2; // Default 2 seconds
    const fadeOut = fadeOutStr ? parseFloat(fadeOutStr) : 2; // Default 2 seconds

    if (isNaN(start) || isNaN(end) || isNaN(fadeIn) || isNaN(fadeOut)) {
        console.error('Error: start, end, fadeIn, and fadeOut must be valid numbers');
        process.exit(1);
    }

    if (end <= start) {
        console.error('Error: end time must be greater than start time');
        process.exit(1);
    }

    try {
        await cutVideo(input, start, end, output, { fadeInSec: fadeIn, fadeOutSec: fadeOut });
        console.log('✓ Video clipping completed successfully');
    } catch (error) {
        console.error('✗ Video clipping failed:', error);
        process.exit(1);
    }
}

// Run the script
main().catch((error) => {
    console.error('Unhandled error:', error);
    process.exit(1);
});
