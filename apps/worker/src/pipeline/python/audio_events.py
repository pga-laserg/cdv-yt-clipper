import argparse
import json
import math
import os
import sys
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal


def to_mono_16k(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        g = math.gcd(sr, 16000)
        up = 16000 // g
        down = sr // g
        audio = signal.resample_poly(audio, up, down)
        sr = 16000
    return audio.astype(np.float32), sr


def estimate_f0(x: np.ndarray, sr: int) -> float | None:
    # Basic autocorrelation pitch estimate; robust enough for coarse gender cues.
    if len(x) < int(0.08 * sr):
        return None
    x = x - np.mean(x)
    den = np.max(np.abs(x)) + 1e-9
    x = x / den
    corr = signal.correlate(x, x, mode="full")
    corr = corr[len(corr) // 2 :]
    min_lag = int(sr / 320.0)
    max_lag = int(sr / 70.0)
    if max_lag <= min_lag or max_lag >= len(corr):
        return None
    search = corr[min_lag:max_lag]
    if search.size == 0:
        return None
    peak_idx = int(np.argmax(search))
    peak_val = float(search[peak_idx] / (corr[0] + 1e-9))
    if peak_val < 0.18:
        return None
    lag = peak_idx + min_lag
    if lag <= 0:
        return None
    f0 = float(sr / lag)
    if 70.0 <= f0 <= 320.0:
        return f0
    return None


def classify_window(x: np.ndarray, sr: int) -> tuple[str, dict[str, Any]]:
    if x.size == 0:
        return "noenergy", {"rms": 0.0}
    rms = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
    if rms < 0.006:
        return "noenergy", {"rms": rms}

    zcr = float(np.mean(np.abs(np.diff(np.signbit(x).astype(np.int8)))))
    freqs, pxx = signal.welch(x, fs=sr, nperseg=min(1024, len(x)))
    pxx = np.maximum(pxx, 1e-12)
    pxx_sum = float(np.sum(pxx))
    centroid = float(np.sum(freqs * pxx) / pxx_sum)
    flatness = float(np.exp(np.mean(np.log(pxx))) / np.mean(pxx))
    f0 = estimate_f0(x, sr)

    speech_like = (0.02 <= zcr <= 0.20) and (150 <= centroid <= 3800)
    tonal_like = (flatness < 0.22) and (zcr < 0.10)
    music_like = tonal_like and not speech_like

    if music_like:
        return "music", {"rms": rms, "zcr": zcr, "centroid": centroid, "flatness": flatness, "f0": f0}
    if speech_like:
        if f0 is not None:
            if f0 < 165:
                return "male", {"rms": rms, "zcr": zcr, "centroid": centroid, "flatness": flatness, "f0": f0}
            return "female", {"rms": rms, "zcr": zcr, "centroid": centroid, "flatness": flatness, "f0": f0}
        return "speech", {"rms": rms, "zcr": zcr, "centroid": centroid, "flatness": flatness, "f0": f0}
    return "noise", {"rms": rms, "zcr": zcr, "centroid": centroid, "flatness": flatness, "f0": f0}


def merge_segments(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not windows:
        return []
    out: list[dict[str, Any]] = [windows[0].copy()]
    for w in windows[1:]:
        last = out[-1]
        if w["label"] == last["label"] and abs(float(w["start"]) - float(last["end"])) <= 0.05:
            last["end"] = w["end"]
        else:
            out.append(w.copy())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Local audio event pass (music/speech/gender) using lightweight DSP.")
    parser.add_argument("audio_path", help="Path to WAV audio input.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--step-sec", type=float, default=float(os.getenv("AUDIO_EVENT_STEP_SEC", "2.0")))
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        raise FileNotFoundError(f"Audio not found: {args.audio_path}")
    if args.step_sec <= 0:
        raise ValueError("step-sec must be > 0")

    audio, sr = sf.read(args.audio_path)
    audio, sr = to_mono_16k(audio, int(sr))
    duration = float(len(audio) / sr) if sr > 0 else 0.0
    win = int(round(args.step_sec * sr))
    total_windows = max(1, int(math.ceil(len(audio) / max(1, win))))
    print(
        f"[audio-events] source={args.audio_path} duration={duration:.1f}s sr={sr} step={args.step_sec}s windows={total_windows}",
        file=sys.stderr,
    )

    windows: list[dict[str, Any]] = []
    i = 0
    window_idx = 0
    while i < len(audio):
        j = min(len(audio), i + win)
        x = audio[i:j]
        label, features = classify_window(x, sr)
        start = float(i / sr)
        end = float(j / sr)
        windows.append({"label": label, "start": start, "end": end, "features": features})
        i = j
        window_idx += 1
        if window_idx % 200 == 0:
            print(
                f"[audio-events] processed {window_idx}/{total_windows} windows (~{(window_idx/max(1,total_windows))*100:.1f}%)",
                file=sys.stderr,
            )

    segments = merge_segments(windows)
    out = {
        "source": "local-dsp-heuristic-v1",
        "duration_sec": duration,
        "step_sec": args.step_sec,
        "segments": [{"label": s["label"], "start": s["start"], "end": s["end"]} for s in segments],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    label_counts: dict[str, int] = {}
    for s in out["segments"]:
        label = str(s["label"])
        label_counts[label] = label_counts.get(label, 0) + 1
    print(
        f"[audio-events] wrote {len(out['segments'])} merged segments -> {args.out} labels={label_counts}",
        file=sys.stderr,
    )

    print(json.dumps({"ok": True, "segments": len(out["segments"]), "duration_sec": duration}))


if __name__ == "__main__":
    main()
