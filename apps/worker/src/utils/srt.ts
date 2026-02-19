export function jsonToSrt(segments: { start: number; end: number; text: string }[]): string {
    return segments.map((seg, index) => {
        const start = formatTimestamp(seg.start);
        const end = formatTimestamp(seg.end);
        return `${index + 1}\n${start} --> ${end}\n${seg.text.trim()}`;
    }).join('\n\n');
}

function formatTimestamp(seconds: number): string {
    const date = new Date(0);
    date.setUTCMilliseconds(seconds * 1000);

    const hours = String(date.getUTCHours()).padStart(2, '0');
    const minutes = String(date.getUTCMinutes()).padStart(2, '0');
    const secs = String(date.getUTCSeconds()).padStart(2, '0');
    const ms = String(date.getUTCMilliseconds()).padStart(3, '0');

    return `${hours}:${minutes}:${secs},${ms}`;
}
