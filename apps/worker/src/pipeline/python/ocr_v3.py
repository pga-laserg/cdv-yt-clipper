from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from difflib import SequenceMatcher
from typing import Any, Callable

import cv2

from ocr_events import (
    classify_text,
    clean_text as clean_text_base,
    detect_scene_cut,
    estimate_text_likelihood,
    init_ocr_engine,
    parse_bool,
    preprocess_roi,
    run_ocr,
)
from ocr_v2_detector import detect_text_boxes, load_east_model
from ocr_v2_east import DEFAULT_EAST_URLS, resolve_east_model_path
from ocr_v2_fusion import OcrObservation, fuse_observations
from ocr_v2_tracker import SimpleBoxTracker

LYRIC_KEYWORDS = {
    "dios",
    "jesus",
    "jesús",
    "cristo",
    "señor",
    "senor",
    "alaba",
    "alabanza",
    "adorar",
    "adoración",
    "adoracion",
    "gloria",
    "santo",
    "santa",
    "espiritu",
    "espíritu",
    "gracia",
    "amor",
    "salvador",
    "aleluya",
    "amen",
    "amén",
    "quiero",
    "eres",
    "tu",
    "tú",
    "mi",
    "alma",
    "corazón",
    "corazon",
    "cantar",
    "cantare",
    "cantaré",
    "alabarte",
    "adorarte",
    "sublime",
    "nombre",
}

NON_LYRIC_OVERLAY_RX = re.compile(
    r"\b(pr\.?|pastor|bienvenida|iglesia|ministerio|predicador|hno\.?|hna\.?|worship|"
    r"adoraci[oó]n infantil|vers[ií]culo b[ií]blico|escuela sab[aá]tica)\b",
    re.IGNORECASE,
)

SLIDE_VERSE_CUE_RX = re.compile(
    r"\b(vers[ií]culo b[ií]blico|texto b[ií]blico|lectura b[ií]blica|santa biblia)\b",
    re.IGNORECASE,
)

SLIDE_SERMON_CUE_RX = re.compile(
    r"\b(tema|t[ií]tulo|mensaje|predicaci[oó]n|serm[oó]n|sermon)\b",
    re.IGNORECASE,
)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def format_duration(sec: float) -> str:
    s = int(max(0.0, sec))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    return f"{h:02d}:{m:02d}:{ss:02d}"


def clean_text(text: str) -> str:
    return clean_text_base(str(text or ""))


def normalize_text(text: str) -> str:
    t = clean_text(text).lower()
    t = re.sub(r"[^a-z0-9áéíóúñü ]+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def text_similarity(a: str, b: str) -> float:
    na = normalize_text(a)
    nb = normalize_text(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def text_quality_metrics(text: str) -> dict[str, float]:
    s = str(text or "").strip()
    n = len(s)
    if n == 0:
        return {"alpha_ratio": 0.0, "alnum_ratio": 0.0, "symbol_ratio": 1.0}
    alpha = sum(1 for ch in s if ch.isalpha())
    alnum = sum(1 for ch in s if ch.isalnum())
    symbols = sum(1 for ch in s if (not ch.isalnum()) and (not ch.isspace()))
    return {
        "alpha_ratio": float(alpha) / float(n),
        "alnum_ratio": float(alnum) / float(n),
        "symbol_ratio": float(symbols) / float(n),
    }


def keep_text(
    text: str,
    confidence: float,
    *,
    min_len: int,
    min_conf: float,
    min_alpha_ratio: float,
    min_alnum_ratio: float,
    max_symbol_ratio: float,
) -> bool:
    t = clean_text(text)
    if len(t) < max(1, int(min_len)):
        return False
    if not re.search(r"[A-Za-zÁÉÍÓÚáéíóúÑñ0-9]", t):
        return False
    q = text_quality_metrics(t)
    if q["alpha_ratio"] < float(min_alpha_ratio):
        return False
    if q["alnum_ratio"] < float(min_alnum_ratio):
        return False
    if q["symbol_ratio"] > float(max_symbol_ratio):
        return False
    if float(confidence) < float(min_conf):
        return False
    return True


def crop_norm(frame, y1: float, y2: float, x1: float, x2: float):
    h, w = frame.shape[:2]
    yy1 = int(clamp(y1, 0.0, 1.0) * h)
    yy2 = int(clamp(y2, 0.0, 1.0) * h)
    xx1 = int(clamp(x1, 0.0, 1.0) * w)
    xx2 = int(clamp(x2, 0.0, 1.0) * w)
    if yy2 <= yy1 or xx2 <= xx1:
        return None, 0, 0
    roi = frame[yy1:yy2, xx1:xx2]
    return roi, xx1, yy1


def avg_hash(img) -> int:
    if img is None or img.size == 0:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    m = float(small.mean())
    bits = (small > m).astype("uint8").flatten().tolist()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return int(out)


def hamming_distance(a: int, b: int) -> int:
    return int((int(a) ^ int(b)).bit_count())


def looks_like_lyric_text(text: str, *, min_words: int = 4) -> bool:
    t = clean_text(text)
    if not t:
        return False
    if NON_LYRIC_OVERLAY_RX.search(t):
        return False
    words = [w for w in re.split(r"\s+", t) if w]
    if len(words) < max(1, int(min_words)):
        return False
    lowered = [w.strip(".,;:!?¡¿()[]{}\"'").lower() for w in words]
    keyword_hits = sum(1 for w in lowered if w in LYRIC_KEYWORDS)
    if keyword_hits >= 2:
        return True
    if keyword_hits >= 1 and len(words) >= max(4, int(min_words)):
        return True
    return False


def classify_slide_text(text: str, region: str) -> str:
    t = clean_text(text)
    if not t:
        return "on_screen_text"
    if SLIDE_VERSE_CUE_RX.search(t):
        return "bible_verse"
    if SLIDE_SERMON_CUE_RX.search(t):
        return "sermon_title"
    return classify_text(t, region)


def classify_lyrics_text(text: str, region: str, *, min_words: int = 4) -> str:
    if looks_like_lyric_text(text, min_words=min_words):
        return "song_lyric"
    return classify_text(text, region)


def merge_lane_events(
    events: list[dict[str, Any]],
    *,
    merge_gap_sec: float,
    sim_threshold: float,
    lane: str,
    classify_fn: Callable[[str, str], str] | None = None,
    force_type: str | None = None,
) -> list[dict[str, Any]]:
    if not events:
        return []
    items = sorted(events, key=lambda e: float(e["start"]))
    merged: list[dict[str, Any]] = []
    curr = {
        "start": float(items[0]["start"]),
        "end": float(items[0]["end"]),
        "text": clean_text(items[0]["text"]),
        "region": str(items[0]["region"]),
        "conf_sum": float(items[0]["confidence"]),
        "samples": 1,
        "best_conf": float(items[0]["confidence"]),
        "best_text": clean_text(items[0]["text"]),
    }

    def flush_current() -> None:
        if not curr["text"]:
            return
        rep_text = clean_text(curr["best_text"] if curr["best_text"] else curr["text"])
        region = str(curr["region"])
        if force_type:
            inferred_type = force_type
        elif classify_fn is not None:
            inferred_type = classify_fn(rep_text, region)
        else:
            inferred_type = classify_text(rep_text, region)
        merged.append(
            {
                "start": float(curr["start"]),
                "end": float(curr["end"]),
                "text": rep_text,
                "type": inferred_type,
                "region": region,
                "confidence": round(float(curr["conf_sum"]) / max(1, int(curr["samples"])), 3),
                "lane": lane,
            }
        )

    for ev in items[1:]:
        text = clean_text(ev["text"])
        if not text:
            continue
        gap = float(ev["start"]) - float(curr["end"])
        sim = text_similarity(curr["text"], text)
        same_region = str(ev["region"]) == str(curr["region"])
        if gap <= float(merge_gap_sec) and sim >= float(sim_threshold) and same_region:
            curr["end"] = float(max(float(curr["end"]), float(ev["end"])))
            curr["conf_sum"] += float(ev["confidence"])
            curr["samples"] += 1
            if float(ev["confidence"]) >= float(curr["best_conf"]):
                curr["best_conf"] = float(ev["confidence"])
                curr["best_text"] = text
            if len(text) > len(curr["text"]):
                curr["text"] = text
        else:
            flush_current()
            curr = {
                "start": float(ev["start"]),
                "end": float(ev["end"]),
                "text": text,
                "region": str(ev["region"]),
                "conf_sum": float(ev["confidence"]),
                "samples": 1,
                "best_conf": float(ev["confidence"]),
                "best_text": text,
            }
    flush_current()
    return merged


def pick_best_ocr_attempt(
    backend: str,
    engine: Any,
    raw_img,
    preproc_img,
    lang: str,
    *,
    psm: int,
    upscale: float,
    min_len: int,
    min_conf: float,
    min_alpha_ratio: float,
    min_alnum_ratio: float,
    max_symbol_ratio: float,
    allow_preproc_fallback: bool,
) -> tuple[str, float]:
    raw_first = backend in ("easyocr", "gcv_text_detection")
    attempts = [("raw", raw_img)] if raw_first else [("preproc", preproc_img)]
    if allow_preproc_fallback:
        attempts = [("raw", raw_img), ("preproc", preproc_img)] if raw_first else [("preproc", preproc_img), ("raw", raw_img)]

    best_text = ""
    best_conf = 0.0
    best_quality = -1.0
    for _kind, img in attempts:
        text, conf = run_ocr(backend, engine, img, lang, psm=psm, upscale=upscale)
        text = clean_text(text)
        conf = float(max(0.0, min(1.0, conf)))
        if not keep_text(
            text,
            conf,
            min_len=min_len,
            min_conf=min_conf,
            min_alpha_ratio=min_alpha_ratio,
            min_alnum_ratio=min_alnum_ratio,
            max_symbol_ratio=max_symbol_ratio,
        ):
            continue
        quality = conf + min(1.0, len(text) / 200.0) * 0.2
        if quality > best_quality:
            best_quality = quality
            best_text = text
            best_conf = conf
    return best_text, best_conf


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR v3 lane pipeline: slides + lower-third + lyrics")
    parser.add_argument("video_path", help="Input video path")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--sample-sec", type=float, default=float(os.getenv("OCR_V3_SAMPLE_SEC", "1.0")))
    parser.add_argument("--max-sec", type=float, default=float(os.getenv("OCR_V3_MAX_DURATION_SEC", "0")))
    parser.add_argument("--backend", type=str, default=os.getenv("OCR_V3_BACKEND", os.getenv("OCR_V2_BACKEND", os.getenv("OCR_BACKEND", "tesseract"))))
    parser.add_argument("--lang", type=str, default=os.getenv("OCR_LANG", "es,en"))
    parser.add_argument("--scene-cut-threshold", type=float, default=float(os.getenv("OCR_SCENE_CUT_THRESHOLD", "0.32")))
    parser.add_argument("--progress-every-samples", type=int, default=int(os.getenv("OCR_PROGRESS_EVERY_SAMPLES", "0")))
    parser.add_argument("--progress-json", type=str, default=os.getenv("OCR_PROGRESS_JSON", "true"))
    parser.add_argument("--preproc-fallback", type=str, default=os.getenv("OCR_PREPROC_FALLBACK", "true"))

    parser.add_argument("--east-enabled", type=str, default=os.getenv("OCR_V3_EAST_ENABLED", os.getenv("OCR_V2_EAST_ENABLED", "true")))
    parser.add_argument("--east-model", type=str, default=os.getenv("OCR_V3_EAST_MODEL", os.getenv("OCR_V2_EAST_MODEL", "")))
    parser.add_argument("--east-model-dir", type=str, default=os.getenv("OCR_V3_EAST_MODEL_DIR", os.getenv("OCR_V2_EAST_MODEL_DIR", "")))
    parser.add_argument(
        "--east-auto-download",
        type=str,
        default=os.getenv("OCR_V3_EAST_AUTO_DOWNLOAD", os.getenv("OCR_V2_EAST_AUTO_DOWNLOAD", "true")),
    )
    parser.add_argument("--east-url", type=str, default=os.getenv("OCR_V3_EAST_URL", os.getenv("OCR_V2_EAST_URL", "")))
    parser.add_argument("--east-sha256", type=str, default=os.getenv("OCR_V3_EAST_SHA256", os.getenv("OCR_V2_EAST_SHA256", "")))

    parser.add_argument("--detector-mode", type=str, default=os.getenv("OCR_V3_DETECTOR_MODE", os.getenv("OCR_V2_DETECTOR_MODE", "hybrid")))
    parser.add_argument("--detector-min-conf", type=float, default=float(os.getenv("OCR_V3_DETECTOR_MIN_CONF", os.getenv("OCR_V2_DETECTOR_MIN_CONF", "0.30"))))
    parser.add_argument("--detector-east-min-conf", type=float, default=float(os.getenv("OCR_V3_EAST_MIN_CONF", os.getenv("OCR_V2_EAST_MIN_CONF", "0.55"))))
    parser.add_argument("--detector-mser-min-conf", type=float, default=float(os.getenv("OCR_V3_MSER_MIN_CONF", os.getenv("OCR_V2_MSER_MIN_CONF", "0.35"))))
    parser.add_argument("--detector-max-boxes", type=int, default=int(os.getenv("OCR_V3_DETECTOR_MAX_BOXES", os.getenv("OCR_V2_DETECTOR_MAX_BOXES", "6"))))
    parser.add_argument("--detector-min-w", type=int, default=int(os.getenv("OCR_V3_DETECTOR_MIN_W", os.getenv("OCR_V2_DETECTOR_MIN_W", "24"))))
    parser.add_argument("--detector-min-h", type=int, default=int(os.getenv("OCR_V3_DETECTOR_MIN_H", os.getenv("OCR_V2_DETECTOR_MIN_H", "12"))))
    parser.add_argument("--detector-min-aspect", type=float, default=float(os.getenv("OCR_V3_DETECTOR_MIN_ASPECT", os.getenv("OCR_V2_DETECTOR_MIN_ASPECT", "1.2"))))
    parser.add_argument("--detector-max-aspect", type=float, default=float(os.getenv("OCR_V3_DETECTOR_MAX_ASPECT", os.getenv("OCR_V2_DETECTOR_MAX_ASPECT", "16.0"))))

    parser.add_argument("--min-text-len", type=int, default=int(os.getenv("OCR_V3_MIN_TEXT_LEN", os.getenv("OCR_V2_MIN_TEXT_LEN", "5"))))
    parser.add_argument("--min-text-confidence", type=float, default=float(os.getenv("OCR_V3_MIN_TEXT_CONFIDENCE", os.getenv("OCR_V2_MIN_TEXT_CONFIDENCE", "0.45"))))
    parser.add_argument("--min-text-alpha-ratio", type=float, default=float(os.getenv("OCR_V3_MIN_TEXT_ALPHA_RATIO", os.getenv("OCR_V2_MIN_TEXT_ALPHA_RATIO", "0.55"))))
    parser.add_argument("--min-text-alnum-ratio", type=float, default=float(os.getenv("OCR_V3_MIN_TEXT_ALNUM_RATIO", os.getenv("OCR_V2_MIN_TEXT_ALNUM_RATIO", "0.75"))))
    parser.add_argument("--max-text-symbol-ratio", type=float, default=float(os.getenv("OCR_V3_MAX_TEXT_SYMBOL_RATIO", os.getenv("OCR_V2_MAX_TEXT_SYMBOL_RATIO", "0.20"))))
    parser.add_argument("--small-box-upscale", type=float, default=float(os.getenv("OCR_V3_SMALL_BOX_UPSCALE", os.getenv("OCR_V2_SMALL_BOX_UPSCALE", "2.0"))))
    parser.add_argument("--small-box-threshold-w", type=int, default=int(os.getenv("OCR_V3_SMALL_BOX_THRESHOLD_W", os.getenv("OCR_V2_SMALL_BOX_THRESHOLD_W", "120"))))
    parser.add_argument("--small-box-threshold-h", type=int, default=int(os.getenv("OCR_V3_SMALL_BOX_THRESHOLD_H", os.getenv("OCR_V2_SMALL_BOX_THRESHOLD_H", "44"))))

    parser.add_argument("--lane-slides-enabled", type=str, default=os.getenv("OCR_V3_LANE_SLIDES_ENABLED", "true"))
    parser.add_argument("--slides-y1", type=float, default=float(os.getenv("OCR_V3_SLIDES_Y1", "0.06")))
    parser.add_argument("--slides-y2", type=float, default=float(os.getenv("OCR_V3_SLIDES_Y2", "0.62")))
    parser.add_argument("--slides-x1", type=float, default=float(os.getenv("OCR_V3_SLIDES_X1", "0.02")))
    parser.add_argument("--slides-x2", type=float, default=float(os.getenv("OCR_V3_SLIDES_X2", "0.98")))
    parser.add_argument("--slides-force-every-samples", type=int, default=int(os.getenv("OCR_V3_SLIDES_FORCE_EVERY_SAMPLES", "10")))
    parser.add_argument("--slides-presence-threshold", type=float, default=float(os.getenv("OCR_V3_SLIDES_PRESENCE_THRESHOLD", "0.26")))
    parser.add_argument("--slides-merge-gap-sec", type=float, default=float(os.getenv("OCR_V3_SLIDES_MERGE_GAP_SEC", "2.5")))
    parser.add_argument("--slides-sim-threshold", type=float, default=float(os.getenv("OCR_V3_SLIDES_SIM_THRESHOLD", "0.88")))

    parser.add_argument("--lane-lower-third-enabled", type=str, default=os.getenv("OCR_V3_LANE_LOWER_THIRD_ENABLED", "true"))
    parser.add_argument("--lower-third-y1", type=float, default=float(os.getenv("OCR_LOWER_THIRD_Y1", "0.68")))
    parser.add_argument("--lower-third-y2", type=float, default=float(os.getenv("OCR_LOWER_THIRD_Y2", "0.92")))
    parser.add_argument("--lower-third-x1", type=float, default=float(os.getenv("OCR_LOWER_THIRD_X1", "0.06")))
    parser.add_argument("--lower-third-x2", type=float, default=float(os.getenv("OCR_LOWER_THIRD_X2", "0.94")))
    parser.add_argument("--lower-third-detect-every-samples", type=int, default=int(os.getenv("OCR_V3_LOWER_THIRD_DETECT_EVERY_SAMPLES", "2")))
    parser.add_argument("--lower-third-presence-threshold", type=float, default=float(os.getenv("OCR_V3_LOWER_THIRD_PRESENCE_THRESHOLD", "0.22")))
    parser.add_argument("--lower-third-track-iou-threshold", type=float, default=float(os.getenv("OCR_V3_LOWER_THIRD_TRACK_IOU_THRESHOLD", os.getenv("OCR_V2_TRACK_IOU_THRESHOLD", "0.35"))))
    parser.add_argument("--lower-third-track-max-misses", type=int, default=int(os.getenv("OCR_V3_LOWER_THIRD_TRACK_MAX_MISSES", "6")))
    parser.add_argument("--lower-third-min-track-observations", type=int, default=int(os.getenv("OCR_V3_LOWER_THIRD_MIN_TRACK_OBSERVATIONS", "2")))
    parser.add_argument("--lower-third-ocr-hash-distance", type=int, default=int(os.getenv("OCR_V3_LOWER_THIRD_OCR_HASH_DISTANCE", "6")))
    parser.add_argument("--lower-third-fuse-min-samples", type=int, default=int(os.getenv("OCR_V3_LOWER_THIRD_FUSE_MIN_SAMPLES", "3")))
    parser.add_argument("--lower-third-fuse-sim-threshold", type=float, default=float(os.getenv("OCR_V3_LOWER_THIRD_FUSE_SIM_THRESHOLD", os.getenv("OCR_V2_FUSE_SIM_THRESHOLD", "0.84"))))
    parser.add_argument("--lower-third-segment-min-confidence", type=float, default=float(os.getenv("OCR_V3_LOWER_THIRD_SEGMENT_MIN_CONFIDENCE", "0.45")))
    parser.add_argument("--lower-third-segment-min-samples", type=int, default=int(os.getenv("OCR_V3_LOWER_THIRD_SEGMENT_MIN_SAMPLES", "3")))
    parser.add_argument("--lower-third-segment-high-conf-override", type=float, default=float(os.getenv("OCR_V3_LOWER_THIRD_SEGMENT_HIGH_CONF_OVERRIDE", "0.85")))

    parser.add_argument("--lane-lyrics-enabled", type=str, default=os.getenv("OCR_V3_LANE_LYRICS_ENABLED", "true"))
    parser.add_argument("--lyrics-y1", type=float, default=float(os.getenv("OCR_V3_LYRICS_Y1", "0.58")))
    parser.add_argument("--lyrics-y2", type=float, default=float(os.getenv("OCR_V3_LYRICS_Y2", "0.94")))
    parser.add_argument("--lyrics-x1", type=float, default=float(os.getenv("OCR_V3_LYRICS_X1", "0.12")))
    parser.add_argument("--lyrics-x2", type=float, default=float(os.getenv("OCR_V3_LYRICS_X2", "0.88")))
    parser.add_argument("--lyrics-presence-threshold", type=float, default=float(os.getenv("OCR_V3_LYRICS_PRESENCE_THRESHOLD", "0.24")))
    parser.add_argument("--lyrics-force-every-samples", type=int, default=int(os.getenv("OCR_V3_LYRICS_FORCE_EVERY_SAMPLES", "6")))
    parser.add_argument("--lyrics-merge-gap-sec", type=float, default=float(os.getenv("OCR_V3_LYRICS_MERGE_GAP_SEC", "2.0")))
    parser.add_argument("--lyrics-sim-threshold", type=float, default=float(os.getenv("OCR_V3_LYRICS_SIM_THRESHOLD", "0.88")))
    parser.add_argument("--lyrics-min-confidence", type=float, default=float(os.getenv("OCR_V3_LYRICS_MIN_CONFIDENCE", "0.50")))
    parser.add_argument("--lyrics-min-words", type=int, default=int(os.getenv("OCR_V3_LYRICS_MIN_WORDS", "4")))
    parser.add_argument("--lyrics-strict-classifier", type=str, default=os.getenv("OCR_V3_LYRICS_STRICT_CLASSIFIER", "true"))
    parser.add_argument("--lyrics-drop-non-lyric", type=str, default=os.getenv("OCR_V3_LYRICS_DROP_NON_LYRIC", "true"))
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")
    if args.sample_sec <= 0:
        raise ValueError("sample-sec must be > 0")

    lane_slides_enabled = parse_bool(args.lane_slides_enabled, default=True)
    lane_lt_enabled = parse_bool(args.lane_lower_third_enabled, default=True)
    lane_lyrics_enabled = parse_bool(args.lane_lyrics_enabled, default=True)
    lyrics_strict_classifier = parse_bool(args.lyrics_strict_classifier, default=True)
    lyrics_drop_non_lyric = parse_bool(args.lyrics_drop_non_lyric, default=True)
    allow_preproc_fallback = parse_bool(args.preproc_fallback, default=True)

    min_text_len = max(1, int(args.min_text_len))
    min_text_conf = clamp(float(args.min_text_confidence), 0.0, 1.0)
    min_text_alpha_ratio = clamp(float(args.min_text_alpha_ratio), 0.0, 1.0)
    min_text_alnum_ratio = clamp(float(args.min_text_alnum_ratio), 0.0, 1.0)
    max_text_symbol_ratio = clamp(float(args.max_text_symbol_ratio), 0.0, 1.0)
    small_box_upscale = max(1.0, float(args.small_box_upscale))
    small_box_threshold_w = max(1, int(args.small_box_threshold_w))
    small_box_threshold_h = max(1, int(args.small_box_threshold_h))

    backend, engine, engine_errors = init_ocr_engine(str(args.backend).strip().lower(), str(args.lang))
    if backend == "none":
        out_obj = {
            "source": "ocr-v3-none",
            "duration_sec": 0.0,
            "sample_sec": float(args.sample_sec),
            "scene_cuts_sec": [],
            "lanes": {"slides": {}, "lower_third": {}, "lyrics": {}},
            "tracks": [],
            "segments": [],
            "errors": engine_errors,
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        print(json.dumps({"ok": True, "source": out_obj["source"], "segments": 0}))
        return

    east_enabled = parse_bool(args.east_enabled, default=True)
    east_model_path: str | None = None
    east_model_downloaded = False
    east_message = "disabled"
    east_net = None
    if east_enabled:
        east_urls = [u.strip() for u in str(args.east_url or "").split(",") if u.strip()] or DEFAULT_EAST_URLS
        east_result = resolve_east_model_path(
            explicit_model_path=str(args.east_model).strip() or None,
            model_dir=str(args.east_model_dir).strip() or None,
            auto_download=parse_bool(args.east_auto_download, default=True),
            model_urls=east_urls,
            sha256=str(args.east_sha256).strip() or None,
        )
        east_model_path = east_result.model_path
        east_model_downloaded = bool(east_result.downloaded)
        east_message = east_result.message
        if east_model_path:
            east_net = load_east_model(east_model_path)
            if east_net is None:
                east_message = f"failed to load EAST model at {east_model_path}"

    cap = cv2.VideoCapture(args.video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = float(total_frames / fps) if fps > 0 and total_frames > 0 else 0.0
    max_sec = duration_sec if float(args.max_sec) <= 0 else min(duration_sec, float(args.max_sec))
    if max_sec <= 0:
        raise RuntimeError("Could not determine video duration for OCR v3.")

    sample_every_frames = max(1, int(round(float(args.sample_sec) * fps))) if fps > 0 else 1
    max_frame_idx = int(max_sec * fps) if fps > 0 else total_frames
    total_samples = max(1, int(math.floor(max_sec / float(args.sample_sec) + 1e-9)) + 1)

    progress_every_samples = int(args.progress_every_samples)
    progress_json = parse_bool(args.progress_json, default=True)
    if progress_every_samples <= 0:
        progress_every_samples = max(1, int(math.ceil(total_samples * 0.05)))

    stats: dict[str, Any] = {
        "samples": 0,
        "scene_cut_samples": 0,
        "slides_ocr_calls": 0,
        "slides_events": 0,
        "lower_detector_calls": 0,
        "lower_detected_boxes": 0,
        "lower_track_assignments": 0,
        "lower_ocr_calls": 0,
        "lower_observations": 0,
        "lyrics_ocr_calls": 0,
        "lyrics_events": 0,
        "lyrics_filtered_non_lyric": 0,
    }
    started_at = time.monotonic()

    slides_events: list[dict[str, Any]] = []
    lyrics_events: list[dict[str, Any]] = []
    lower_observations: list[OcrObservation] = []
    lower_track_hash: dict[str, int] = {}
    lower_track_hits: dict[str, int] = {}

    lower_tracker = SimpleBoxTracker(
        region="lower_third",
        iou_threshold=float(args.lower_third_track_iou_threshold),
        max_misses=int(args.lower_third_track_max_misses),
    )

    prev_frame = None
    scene_cuts: list[float] = []
    frame_idx = 0
    sample_idx = 0
    next_sample_frame = 0

    print(
        f"[ocr-v3] source={args.video_path} backend={backend} duration={duration_sec:.1f}s "
        f"max_sec={max_sec:.1f}s sample={args.sample_sec}s video={frame_width}x{frame_height} "
        f"lanes={{slides:{lane_slides_enabled},lower_third:{lane_lt_enabled},lyrics:{lane_lyrics_enabled}}} "
        f"detector_mode={args.detector_mode} east_model={'loaded' if east_net is not None else 'none'} "
        f"east_status=\"{east_message}\"",
        file=sys.stderr,
    )

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if fps > 0 and frame_idx > max_frame_idx:
            break
        if frame_idx < next_sample_frame:
            frame_idx += 1
            continue

        t = float(frame_idx / fps) if fps > 0 else float(sample_idx * float(args.sample_sec))
        next_sample_frame = frame_idx + sample_every_frames
        stats["samples"] += 1

        scene_cut_now = False
        if prev_frame is not None and detect_scene_cut(prev_frame, frame, float(args.scene_cut_threshold)):
            if len(scene_cuts) == 0 or abs(float(t) - scene_cuts[-1]) >= 0.25:
                scene_cuts.append(float(t))
            scene_cut_now = True
        prev_frame = frame.copy()
        if scene_cut_now:
            stats["scene_cut_samples"] += 1

        # Lane 1: slides
        if lane_slides_enabled:
            slide_roi, _sx, _sy = crop_norm(frame, float(args.slides_y1), float(args.slides_y2), float(args.slides_x1), float(args.slides_x2))
            if slide_roi is not None and slide_roi.size > 0:
                slide_force = int(args.slides_force_every_samples) > 0 and (sample_idx % max(1, int(args.slides_force_every_samples)) == 0)
                slide_score, _ = estimate_text_likelihood(slide_roi, region="slide", resize_max_width=int(os.getenv("OCR_V3_TARGET_RESIZE_MAX_WIDTH", "960")))
                slide_should_ocr = scene_cut_now or sample_idx == 0 or slide_force
                if slide_should_ocr and slide_score >= float(args.slides_presence_threshold):
                    preproc = preprocess_roi(slide_roi)
                    stats["slides_ocr_calls"] += 1
                    s_text, s_conf = pick_best_ocr_attempt(
                        backend,
                        engine,
                        slide_roi,
                        preproc,
                        str(args.lang),
                        psm=6,
                        upscale=1.2,
                        min_len=max(min_text_len, 6),
                        min_conf=min_text_conf,
                        min_alpha_ratio=min_text_alpha_ratio,
                        min_alnum_ratio=min_text_alnum_ratio,
                        max_symbol_ratio=max_text_symbol_ratio,
                        allow_preproc_fallback=allow_preproc_fallback,
                    )
                    if s_text:
                        slides_events.append(
                            {
                                "start": float(t),
                                "end": float(t + float(args.sample_sec)),
                                "text": s_text,
                                "confidence": float(s_conf),
                                "region": "slide",
                            }
                        )
                        stats["slides_events"] += 1

        # Lane 2: lower-third track lane
        if lane_lt_enabled:
            lt_roi, lox, loy = crop_norm(
                frame,
                float(args.lower_third_y1),
                float(args.lower_third_y2),
                float(args.lower_third_x1),
                float(args.lower_third_x2),
            )
            detections: list[tuple[int, int, int, int, float]] = []
            if lt_roi is not None and lt_roi.size > 0:
                lt_score, _ = estimate_text_likelihood(lt_roi, region="lower_third", resize_max_width=int(os.getenv("OCR_V3_TARGET_RESIZE_MAX_WIDTH", "960")))
                detect_every = max(1, int(args.lower_third_detect_every_samples))
                detect_now = scene_cut_now or (sample_idx % detect_every == 0) or (lt_score >= float(args.lower_third_presence_threshold))
                if detect_now:
                    stats["lower_detector_calls"] += 1
                    lt_boxes = detect_text_boxes(
                        lt_roi,
                        mode=str(args.detector_mode),
                        min_conf=float(args.detector_min_conf),
                        east_min_conf=float(args.detector_east_min_conf),
                        mser_min_conf=float(args.detector_mser_min_conf),
                        max_boxes=int(args.detector_max_boxes),
                        min_w=int(args.detector_min_w),
                        min_h=int(args.detector_min_h),
                        min_aspect=float(args.detector_min_aspect),
                        max_aspect=float(args.detector_max_aspect),
                        east_net=east_net,
                    )
                    stats["lower_detected_boxes"] += len(lt_boxes)
                    for b in lt_boxes:
                        detections.append((int(b.x1 + lox), int(b.y1 + loy), int(b.x2 + lox), int(b.y2 + loy), float(b.score)))

            assignments = lower_tracker.update(detections, t_sec=float(t))
            stats["lower_track_assignments"] += len(assignments)
            for track_id, (x1, y1, x2, y2), _ in assignments:
                track_key = f"lower_third:{track_id}"
                lower_track_hits[track_key] = int(lower_track_hits.get(track_key, 0)) + 1
                if lower_track_hits[track_key] < max(1, int(args.lower_third_min_track_observations)):
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop is None or crop.size == 0:
                    continue
                h = avg_hash(crop)
                prev_h = lower_track_hash.get(track_key)
                hash_dist = hamming_distance(prev_h, h) if prev_h is not None else 999
                changed = prev_h is None or hash_dist >= int(args.lower_third_ocr_hash_distance) or scene_cut_now
                lower_track_hash[track_key] = h
                if not changed:
                    continue

                preproc = preprocess_roi(crop)
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                upscale = small_box_upscale if (bw <= small_box_threshold_w or bh <= small_box_threshold_h) else 1.0
                stats["lower_ocr_calls"] += 1
                lt_text, lt_conf = pick_best_ocr_attempt(
                    backend,
                    engine,
                    crop,
                    preproc,
                    str(args.lang),
                    psm=7,
                    upscale=upscale,
                    min_len=min_text_len,
                    min_conf=min_text_conf,
                    min_alpha_ratio=min_text_alpha_ratio,
                    min_alnum_ratio=min_text_alnum_ratio,
                    max_symbol_ratio=max_text_symbol_ratio,
                    allow_preproc_fallback=allow_preproc_fallback,
                )
                if lt_text:
                    lower_observations.append(
                        OcrObservation(
                            track_key=track_key,
                            start=float(t),
                            end=float(t + float(args.sample_sec)),
                            text=lt_text,
                            confidence=float(lt_conf),
                            region="lower_third",
                        )
                    )
                    stats["lower_observations"] += 1

        # Lane 3: lyrics
        if lane_lyrics_enabled:
            ly_roi, _lyx, _lyy = crop_norm(frame, float(args.lyrics_y1), float(args.lyrics_y2), float(args.lyrics_x1), float(args.lyrics_x2))
            if ly_roi is not None and ly_roi.size > 0:
                ly_score, _ = estimate_text_likelihood(ly_roi, region="lower_third", resize_max_width=int(os.getenv("OCR_V3_TARGET_RESIZE_MAX_WIDTH", "960")))
                ly_force = int(args.lyrics_force_every_samples) > 0 and (sample_idx % max(1, int(args.lyrics_force_every_samples)) == 0)
                ly_should_ocr = (ly_score >= float(args.lyrics_presence_threshold)) or scene_cut_now or ly_force
                if ly_should_ocr:
                    preproc = preprocess_roi(ly_roi)
                    stats["lyrics_ocr_calls"] += 1
                    ly_text, ly_conf = pick_best_ocr_attempt(
                        backend,
                        engine,
                        ly_roi,
                        preproc,
                        str(args.lang),
                        psm=6,
                        upscale=1.3,
                        min_len=max(min_text_len, 6),
                        min_conf=max(0.30, min_text_conf - 0.05),
                        min_alpha_ratio=max(0.45, min_text_alpha_ratio - 0.1),
                        min_alnum_ratio=max(0.60, min_text_alnum_ratio - 0.1),
                        max_symbol_ratio=max_text_symbol_ratio,
                        allow_preproc_fallback=allow_preproc_fallback,
                    )
                    if ly_text:
                        if float(ly_conf) < float(args.lyrics_min_confidence):
                            stats["lyrics_filtered_non_lyric"] += 1
                            continue
                        if lyrics_strict_classifier and not looks_like_lyric_text(ly_text, min_words=int(args.lyrics_min_words)):
                            stats["lyrics_filtered_non_lyric"] += 1
                            continue
                        lyrics_events.append(
                            {
                                "start": float(t),
                                "end": float(t + float(args.sample_sec)),
                                "text": ly_text,
                                "confidence": float(ly_conf),
                                "region": "lower_third",
                            }
                        )
                        stats["lyrics_events"] += 1

        sample_idx += 1
        if sample_idx == 1 or sample_idx == total_samples or sample_idx % progress_every_samples == 0:
            pct = (sample_idx / max(1, total_samples)) * 100.0
            elapsed_sec = time.monotonic() - started_at
            speed = sample_idx / elapsed_sec if elapsed_sec > 0 else 0.0
            eta_sec = ((total_samples - sample_idx) / speed) if speed > 0 else 0.0
            print(
                f"[ocr-v3] progress={pct:.1f}% sampled={sample_idx}/{total_samples} "
                f"video_t={format_duration(t)}/{format_duration(max_sec)} scene_cuts={len(scene_cuts)} "
                f"slides={len(slides_events)} lower_obs={len(lower_observations)} lyrics={len(lyrics_events)} "
                f"elapsed={format_duration(elapsed_sec)} eta={format_duration(eta_sec)}",
                file=sys.stderr,
            )
            if progress_json:
                payload = {
                    "type": "ocr_progress",
                    "pipeline": "v3",
                    "percent": round(pct, 3),
                    "sampled": int(sample_idx),
                    "total_samples": int(total_samples),
                    "video_time_sec": round(float(t), 3),
                    "video_duration_sec": round(float(max_sec), 3),
                    "scene_cuts": int(len(scene_cuts)),
                    "slides_events": int(len(slides_events)),
                    "lower_observations": int(len(lower_observations)),
                    "lyrics_events": int(len(lyrics_events)),
                    "elapsed_sec": round(float(elapsed_sec), 3),
                    "eta_sec": round(float(eta_sec), 3),
                }
                print(f"[ocr-events-progress] {json.dumps(payload, ensure_ascii=False)}", file=sys.stderr)

        frame_idx += 1

    cap.release()

    # Lane outputs
    slides_segments = merge_lane_events(
        slides_events,
        merge_gap_sec=float(args.slides_merge_gap_sec),
        sim_threshold=clamp(float(args.slides_sim_threshold), 0.5, 0.99),
        lane="slides",
        classify_fn=classify_slide_text,
        force_type=None,
    )

    fused_lower = fuse_observations(
        lower_observations,
        min_samples=max(1, int(args.lower_third_fuse_min_samples)),
        similarity_threshold=clamp(float(args.lower_third_fuse_sim_threshold), 0.5, 0.99),
    )
    lower_tracks: list[dict[str, Any]] = []
    lower_segments: list[dict[str, Any]] = []
    for tr in lower_tracker.finalize():
        key = f"lower_third:{tr.track_id}"
        fused = fused_lower.get(key)
        lower_tracks.append(
            {
                "track_key": key,
                "track_id": int(tr.track_id),
                "region": "lower_third",
                "start": float(tr.start_sec),
                "end": float(tr.end_sec),
                "bbox": {"x1": int(tr.x1), "y1": int(tr.y1), "x2": int(tr.x2), "y2": int(tr.y2)},
                "hits": int(tr.hits),
                "misses": int(tr.misses),
                "score": round(float(tr.score), 3),
                "fused_text": fused["text"] if fused else "",
                "fused_confidence": float(fused["confidence"]) if fused else 0.0,
                "fused_samples": int(fused["samples"]) if fused else 0,
            }
        )
        if not fused or not fused.get("text"):
            continue
        text = clean_text(str(fused["text"]))
        conf = float(fused["confidence"])
        samples = int(fused["samples"])
        q = text_quality_metrics(text)
        quality_ok = (
            q["alpha_ratio"] >= min_text_alpha_ratio
            and q["alnum_ratio"] >= min_text_alnum_ratio
            and q["symbol_ratio"] <= max_text_symbol_ratio
        )
        support_ok = samples >= int(args.lower_third_segment_min_samples) or conf >= float(args.lower_third_segment_high_conf_override)
        confidence_ok = conf >= float(args.lower_third_segment_min_confidence) or conf >= float(args.lower_third_segment_high_conf_override)
        if not (text and quality_ok and support_ok and confidence_ok):
            continue
        lower_segments.append(
            {
                "start": float(max(tr.start_sec, float(fused["start"]))),
                "end": float(min(tr.end_sec, float(fused["end"]))),
                "text": text,
                "type": classify_text(text, "lower_third"),
                "region": "lower_third",
                "confidence": conf,
                "track_key": key,
                "lane": "lower_third",
            }
        )

    lyrics_segments = merge_lane_events(
        lyrics_events,
        merge_gap_sec=float(args.lyrics_merge_gap_sec),
        sim_threshold=clamp(float(args.lyrics_sim_threshold), 0.5, 0.99),
        lane="lyrics",
        classify_fn=lambda text, region: classify_lyrics_text(text, region, min_words=int(args.lyrics_min_words)),
        force_type=None,
    )
    if lyrics_drop_non_lyric:
        lyrics_segments = [seg for seg in lyrics_segments if str(seg.get("type")) == "song_lyric"]

    all_segments = sorted(
        [*slides_segments, *lower_segments, *lyrics_segments],
        key=lambda s: (float(s["start"]), str(s.get("lane", ""))),
    )

    out_obj = {
        "source": "ocr-v3-lanes",
        "duration_sec": float(duration_sec),
        "sample_sec": float(args.sample_sec),
        "scene_cuts_sec": scene_cuts,
        "config": {
            "backend": str(backend),
            "lanes": {
                "slides": bool(lane_slides_enabled),
                "lower_third": bool(lane_lt_enabled),
                "lyrics": bool(lane_lyrics_enabled),
            },
            "detector_mode": str(args.detector_mode),
            "detector_min_conf": float(args.detector_min_conf),
            "detector_east_min_conf": float(args.detector_east_min_conf),
            "detector_mser_min_conf": float(args.detector_mser_min_conf),
            "detector_max_boxes": int(args.detector_max_boxes),
            "detector_min_w": int(args.detector_min_w),
            "detector_min_h": int(args.detector_min_h),
            "detector_min_aspect": float(args.detector_min_aspect),
            "detector_max_aspect": float(args.detector_max_aspect),
            "min_text_len": int(min_text_len),
            "min_text_confidence": float(min_text_conf),
            "min_text_alpha_ratio": float(min_text_alpha_ratio),
            "min_text_alnum_ratio": float(min_text_alnum_ratio),
            "max_text_symbol_ratio": float(max_text_symbol_ratio),
            "small_box_upscale": float(small_box_upscale),
            "small_box_threshold_w": int(small_box_threshold_w),
            "small_box_threshold_h": int(small_box_threshold_h),
            "east_enabled": bool(east_enabled),
            "east_model": str(east_model_path or args.east_model or ""),
            "east_model_downloaded": bool(east_model_downloaded),
            "east_status": str(east_message),
            "east_loaded": bool(east_net is not None),
            "video_width": int(frame_width),
            "video_height": int(frame_height),
        },
        "lanes": {
            "slides": {"events": slides_events, "segments": slides_segments},
            "lower_third": {"tracks": lower_tracks, "segments": lower_segments},
            "lyrics": {"events": lyrics_events, "segments": lyrics_segments},
        },
        "tracks": lower_tracks,
        "segments": all_segments,
        "stats": stats,
        "errors": engine_errors,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(
        f"[ocr-v3] wrote tracks={len(lower_tracks)} segments={len(all_segments)} "
        f"(slides={len(slides_segments)} lower_third={len(lower_segments)} lyrics={len(lyrics_segments)}) "
        f"scene_cuts={len(scene_cuts)} -> {args.out}",
        file=sys.stderr,
    )
    print(json.dumps({"ok": True, "source": out_obj["source"], "tracks": len(lower_tracks), "segments": len(all_segments)}))


if __name__ == "__main__":
    main()
