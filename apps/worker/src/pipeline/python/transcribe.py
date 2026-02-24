import sys
import json
import os
import argparse
import re
from typing import Any
from faster_whisper import WhisperModel

def _normalize_words(segment: Any):
    words = getattr(segment, "words", None) or []
    normalized = []
    for w in words:
        ws = getattr(w, "start", None)
        we = getattr(w, "end", None)
        ww = str(getattr(w, "word", ""))
        wp = getattr(w, "probability", None)
        if ws is None or we is None:
            continue
        item = {
            "start": float(ws),
            "end": float(we),
            "word": ww
        }
        if wp is not None:
            try:
                item["probability"] = float(wp)
            except Exception:
                pass
        normalized.append(item)
    return normalized

def _split_segment_by_word_gaps(segment: Any, words: list[dict[str, Any]], max_gap_sec: float):
    if not words or max_gap_sec <= 0:
        return [{
            "start": float(segment.start),
            "end": float(segment.end),
            "text": str(segment.text or "").strip()
        }]

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = [words[0]]

    for w in words[1:]:
        prev = current[-1]
        gap = float(w["start"]) - float(prev["end"])
        if gap > max_gap_sec:
            groups.append(current)
            current = [w]
        else:
            current.append(w)
    groups.append(current)

    out = []
    for g in groups:
        start = float(g[0]["start"])
        end = float(g[-1]["end"])
        raw_text = "".join(str(w.get("word", "")) for w in g).strip()
        text = re.sub(r"\s+", " ", raw_text).strip()
        if not text:
            continue
        out.append({
            "start": start,
            "end": end,
            "text": text
        })

    if out:
        return out

    return [{
        "start": float(words[0]["start"]),
        "end": float(words[-1]["end"]),
        "text": str(segment.text or "").strip()
    }]

def transcribe_audio(file_path, model_size, beam_size, word_timestamps=True, word_gap_split_sec=0.55):
    # Reduce parallel tokenizer worker churn that can lead to leaked semaphore warnings.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Check if CUDA is available, otherwise allow CPU
    device = "cpu"
    compute_type = "int8"
    
    print(f"Loading Whisper model '{model_size}' on {device}...", file=sys.stderr)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"Transcribing {file_path} (Forcing Spanish)...", file=sys.stderr)
    # Force language to Spanish to avoid English hallucinations/translations
    segments, info = model.transcribe(
        file_path, 
        beam_size=beam_size,
        language="es", 
        task="transcribe",
        word_timestamps=word_timestamps,
        vad_filter=True,
        condition_on_previous_text=False,
        temperature=0.0,
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.6
    )

    print(f"Detected language '{info.language}' with probability {info.language_probability}", file=sys.stderr)

    result_segments = []
    for segment in segments:
        words = _normalize_words(segment) if word_timestamps else []
        if words:
            split_segments = _split_segment_by_word_gaps(segment, words, word_gap_split_sec)
            for ss in split_segments:
                start = float(ss["start"])
                end = float(ss["end"])
                if end <= start:
                    continue
                text = str(ss["text"]).strip()
                if not text:
                    continue
                result_segments.append({
                    "start": start,
                    "end": end,
                    "text": text
                })
                print("[%.2fs -> %.2fs] %s" % (start, end, text), file=sys.stderr)
        else:
            start = float(segment.start)
            end = float(segment.end)
            if end <= start:
                continue
            text = str(segment.text or "").strip()
            if not text:
                continue
            result_segments.append({
                "start": start,
                "end": end,
                "text": text
            })
            print("[%.2fs -> %.2fs] %s" % (start, end, text), file=sys.stderr)

    result_segments.sort(key=lambda s: (float(s["start"]), float(s["end"])))
    cleaned = []
    for seg in result_segments:
        s = float(seg["start"])
        e = float(seg["end"])
        t = str(seg["text"]).strip()
        if not t:
            continue
        if cleaned:
            prev = cleaned[-1]
            if s < prev["end"]:
                s = prev["end"]
        if e - s < 0.05:
            continue
        cleaned.append({"start": s, "end": e, "text": t})

    return cleaned

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio using faster-whisper.")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--model", default="small", help="Whisper model size (e.g. base, small)")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding")
    parser.add_argument(
        "--no-word-timestamps",
        action="store_true",
        help="Disable word-level timestamp tightening (legacy behavior)."
    )
    parser.add_argument(
        "--word-gap-split-sec",
        type=float,
        default=float(os.getenv("TRANSCRIBE_WORD_GAP_SPLIT_SEC", "0.55")),
        help="Split segment on inter-word pauses longer than this value in seconds."
    )
    args = parser.parse_args()

    audio_file = args.audio_file
    
    if not os.path.exists(audio_file):
        print(f"Error: File {audio_file} not found", file=sys.stderr)
        sys.exit(1)

    try:
        result = transcribe_audio(
            audio_file,
            args.model,
            args.beam_size,
            word_timestamps=not args.no_word_timestamps,
            word_gap_split_sec=args.word_gap_split_sec
        )
        print(json.dumps(result))
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        sys.exit(1)
