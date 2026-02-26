from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from typing import Any

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


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def format_duration(sec: float) -> str:
    s = int(max(0.0, sec))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    return f"{h:02d}:{m:02d}:{ss:02d}"


def region_rois(
    frame,
    enabled_regions: set[str],
    lower_third_y1: float,
    lower_third_y2: float,
    lower_third_x1: float,
    lower_third_x2: float,
    slide_y1: float,
    slide_y2: float,
    slide_x1: float,
    slide_x2: float,
) -> list[tuple[str, Any, int, int]]:
    h, w = frame.shape[:2]
    rois: list[tuple[str, Any, int, int]] = []
    if "lower_third" in enabled_regions:
        y1 = int(lower_third_y1 * h)
        y2 = int(lower_third_y2 * h)
        x1 = int(lower_third_x1 * w)
        x2 = int(lower_third_x2 * w)
        rois.append(("lower_third", frame[y1:y2, x1:x2], x1, y1))
    if "slide" in enabled_regions:
        sy1 = int(slide_y1 * h)
        sy2 = int(slide_y2 * h)
        sx1 = int(slide_x1 * w)
        sx2 = int(slide_x2 * w)
        rois.append(("slide", frame[sy1:sy2, sx1:sx2], sx1, sy1))
    if "full" in enabled_regions:
        rois.append(("full", frame, 0, 0))
    return rois


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


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR v2 scaffold: detect -> track -> OCR -> temporal fuse")
    parser.add_argument("video_path", help="Input video path")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--sample-sec", type=float, default=float(os.getenv("OCR_V2_SAMPLE_SEC", "1.0")))
    parser.add_argument("--max-sec", type=float, default=float(os.getenv("OCR_V2_MAX_DURATION_SEC", "0")))
    parser.add_argument("--backend", type=str, default=os.getenv("OCR_V2_BACKEND", os.getenv("OCR_BACKEND", "tesseract")))
    parser.add_argument("--lang", type=str, default=os.getenv("OCR_LANG", "es,en"))
    parser.add_argument("--regions", type=str, default=os.getenv("OCR_V2_REGIONS", "lower_third,slide"))
    parser.add_argument("--lower-third-y1", type=float, default=float(os.getenv("OCR_LOWER_THIRD_Y1", "0.68")))
    parser.add_argument("--lower-third-y2", type=float, default=float(os.getenv("OCR_LOWER_THIRD_Y2", "0.92")))
    parser.add_argument("--lower-third-x1", type=float, default=float(os.getenv("OCR_LOWER_THIRD_X1", "0.06")))
    parser.add_argument("--lower-third-x2", type=float, default=float(os.getenv("OCR_LOWER_THIRD_X2", "0.94")))
    parser.add_argument("--slide-y1", type=float, default=float(os.getenv("OCR_V2_SLIDE_Y1", "0.08")))
    parser.add_argument("--slide-y2", type=float, default=float(os.getenv("OCR_V2_SLIDE_Y2", "0.48")))
    parser.add_argument("--slide-x1", type=float, default=float(os.getenv("OCR_V2_SLIDE_X1", "0.08")))
    parser.add_argument("--slide-x2", type=float, default=float(os.getenv("OCR_V2_SLIDE_X2", "0.92")))
    parser.add_argument("--east-enabled", type=str, default=os.getenv("OCR_V2_EAST_ENABLED", "true"))
    parser.add_argument("--east-model", type=str, default=os.getenv("OCR_V2_EAST_MODEL", ""))
    parser.add_argument("--east-model-dir", type=str, default=os.getenv("OCR_V2_EAST_MODEL_DIR", ""))
    parser.add_argument("--east-auto-download", type=str, default=os.getenv("OCR_V2_EAST_AUTO_DOWNLOAD", "true"))
    parser.add_argument("--east-url", type=str, default=os.getenv("OCR_V2_EAST_URL", ""))
    parser.add_argument("--east-sha256", type=str, default=os.getenv("OCR_V2_EAST_SHA256", ""))
    parser.add_argument("--detector-mode", type=str, default=os.getenv("OCR_V2_DETECTOR_MODE", "hybrid"))
    parser.add_argument("--detector-min-conf", type=float, default=float(os.getenv("OCR_V2_DETECTOR_MIN_CONF", "0.30")))
    parser.add_argument("--detector-east-min-conf", type=float, default=float(os.getenv("OCR_V2_EAST_MIN_CONF", "0.55")))
    parser.add_argument("--detector-mser-min-conf", type=float, default=float(os.getenv("OCR_V2_MSER_MIN_CONF", "0.35")))
    parser.add_argument("--detector-max-boxes", type=int, default=int(os.getenv("OCR_V2_DETECTOR_MAX_BOXES", "6")))
    parser.add_argument("--detector-min-w", type=int, default=int(os.getenv("OCR_V2_DETECTOR_MIN_W", "24")))
    parser.add_argument("--detector-min-h", type=int, default=int(os.getenv("OCR_V2_DETECTOR_MIN_H", "12")))
    parser.add_argument("--detector-min-aspect", type=float, default=float(os.getenv("OCR_V2_DETECTOR_MIN_ASPECT", "1.2")))
    parser.add_argument("--detector-max-aspect", type=float, default=float(os.getenv("OCR_V2_DETECTOR_MAX_ASPECT", "16.0")))
    parser.add_argument("--track-iou-threshold", type=float, default=float(os.getenv("OCR_V2_TRACK_IOU_THRESHOLD", "0.35")))
    parser.add_argument("--track-max-misses", type=int, default=int(os.getenv("OCR_V2_TRACK_MAX_MISSES", "3")))
    parser.add_argument("--ocr-min-text-confidence", type=float, default=float(os.getenv("OCR_V2_MIN_TEXT_CONFIDENCE", "0.45")))
    parser.add_argument("--ocr-preproc-fallback", type=str, default=os.getenv("OCR_PREPROC_FALLBACK", "true"))
    parser.add_argument("--min-text-len", type=int, default=int(os.getenv("OCR_V2_MIN_TEXT_LEN", "5")))
    parser.add_argument("--min-text-alpha-ratio", type=float, default=float(os.getenv("OCR_V2_MIN_TEXT_ALPHA_RATIO", "0.55")))
    parser.add_argument("--min-text-alnum-ratio", type=float, default=float(os.getenv("OCR_V2_MIN_TEXT_ALNUM_RATIO", "0.75")))
    parser.add_argument("--max-text-symbol-ratio", type=float, default=float(os.getenv("OCR_V2_MAX_TEXT_SYMBOL_RATIO", "0.20")))
    parser.add_argument("--min-track-observations", type=int, default=int(os.getenv("OCR_V2_MIN_TRACK_OBSERVATIONS", "2")))
    parser.add_argument("--fuse-min-samples", type=int, default=int(os.getenv("OCR_V2_FUSE_MIN_SAMPLES", "3")))
    parser.add_argument("--fuse-similarity-threshold", type=float, default=float(os.getenv("OCR_V2_FUSE_SIM_THRESHOLD", "0.84")))
    parser.add_argument("--segment-min-confidence", type=float, default=float(os.getenv("OCR_V2_SEGMENT_MIN_CONFIDENCE", "0.45")))
    parser.add_argument("--segment-min-samples", type=int, default=int(os.getenv("OCR_V2_SEGMENT_MIN_SAMPLES", "3")))
    parser.add_argument("--segment-high-confidence-override", type=float, default=float(os.getenv("OCR_V2_SEGMENT_HIGH_CONF_OVERRIDE", "0.85")))
    parser.add_argument("--small-box-upscale", type=float, default=float(os.getenv("OCR_V2_SMALL_BOX_UPSCALE", "2.0")))
    parser.add_argument("--small-box-threshold-w", type=int, default=int(os.getenv("OCR_V2_SMALL_BOX_THRESHOLD_W", "120")))
    parser.add_argument("--small-box-threshold-h", type=int, default=int(os.getenv("OCR_V2_SMALL_BOX_THRESHOLD_H", "44")))
    parser.add_argument("--targeted", type=str, default=os.getenv("OCR_V2_TARGETED", "true"))
    parser.add_argument("--target-threshold-lower-third", type=float, default=float(os.getenv("OCR_V2_TARGET_THRESHOLD_LOWER_THIRD", "0.22")))
    parser.add_argument("--target-threshold-slide", type=float, default=float(os.getenv("OCR_V2_TARGET_THRESHOLD_SLIDE", "0.26")))
    parser.add_argument("--target-threshold-full", type=float, default=float(os.getenv("OCR_V2_TARGET_THRESHOLD_FULL", "0.30")))
    parser.add_argument("--target-persist-relax", type=float, default=float(os.getenv("OCR_V2_TARGET_PERSIST_RELAX", "0.82")))
    parser.add_argument("--target-force-every-samples", type=int, default=int(os.getenv("OCR_V2_TARGET_FORCE_EVERY_SAMPLES", "8")))
    parser.add_argument("--target-scene-cut-boost", type=float, default=float(os.getenv("OCR_V2_TARGET_SCENE_CUT_BOOST", "0.03")))
    parser.add_argument("--target-resize-max-width", type=int, default=int(os.getenv("OCR_V2_TARGET_RESIZE_MAX_WIDTH", "960")))
    parser.add_argument("--min-video-width", type=int, default=int(os.getenv("OCR_V2_MIN_VIDEO_WIDTH", "0")))
    parser.add_argument("--require-min-video-width", type=str, default=os.getenv("OCR_V2_REQUIRE_MIN_VIDEO_WIDTH", "false"))
    parser.add_argument("--scene-cut-threshold", type=float, default=float(os.getenv("OCR_SCENE_CUT_THRESHOLD", "0.32")))
    parser.add_argument("--progress-every-samples", type=int, default=int(os.getenv("OCR_PROGRESS_EVERY_SAMPLES", "0")))
    parser.add_argument("--progress-json", type=str, default=os.getenv("OCR_PROGRESS_JSON", "true"))
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")
    if args.sample_sec <= 0:
        raise ValueError("sample-sec must be > 0")

    enabled_regions = {x.strip().lower() for x in str(args.regions).split(",") if x.strip()}
    valid_regions = {"lower_third", "slide", "full"}
    enabled_regions = enabled_regions.intersection(valid_regions)
    if not enabled_regions:
        enabled_regions = {"lower_third"}

    lower_third_y1 = clamp(float(args.lower_third_y1), 0.0, 1.0)
    lower_third_y2 = clamp(float(args.lower_third_y2), 0.0, 1.0)
    lower_third_x1 = clamp(float(args.lower_third_x1), 0.0, 1.0)
    lower_third_x2 = clamp(float(args.lower_third_x2), 0.0, 1.0)
    slide_y1 = clamp(float(args.slide_y1), 0.0, 1.0)
    slide_y2 = clamp(float(args.slide_y2), 0.0, 1.0)
    slide_x1 = clamp(float(args.slide_x1), 0.0, 1.0)
    slide_x2 = clamp(float(args.slide_x2), 0.0, 1.0)
    detector_mode = str(args.detector_mode or "hybrid").strip().lower()
    targeted_enabled = parse_bool(args.targeted, default=True)
    min_track_observations = max(1, int(args.min_track_observations))
    min_text_len = max(1, int(args.min_text_len))
    min_text_alpha_ratio = clamp(float(args.min_text_alpha_ratio), 0.0, 1.0)
    min_text_alnum_ratio = clamp(float(args.min_text_alnum_ratio), 0.0, 1.0)
    max_text_symbol_ratio = clamp(float(args.max_text_symbol_ratio), 0.0, 1.0)
    segment_min_confidence = clamp(float(args.segment_min_confidence), 0.0, 1.0)
    segment_min_samples = max(1, int(args.segment_min_samples))
    segment_high_conf_override = clamp(float(args.segment_high_confidence_override), 0.0, 1.0)
    small_box_upscale = max(1.0, float(args.small_box_upscale))
    small_box_threshold_w = max(1, int(args.small_box_threshold_w))
    small_box_threshold_h = max(1, int(args.small_box_threshold_h))
    target_thresholds = {
        "lower_third": clamp(float(args.target_threshold_lower_third), 0.0, 1.0),
        "slide": clamp(float(args.target_threshold_slide), 0.0, 1.0),
        "full": clamp(float(args.target_threshold_full), 0.0, 1.0),
    }
    target_persist_relax = clamp(float(args.target_persist_relax), 0.0, 1.0)
    target_force_every_samples = max(1, int(args.target_force_every_samples))
    target_scene_cut_boost = max(0.0, float(args.target_scene_cut_boost))
    target_resize_max_width = max(0, int(args.target_resize_max_width))
    require_min_video_width = parse_bool(args.require_min_video_width, default=False)
    min_video_width = max(0, int(args.min_video_width))

    backend, engine, engine_errors = init_ocr_engine(args.backend.strip().lower(), args.lang)
    if backend == "none":
        out_obj = {
            "source": "ocr-v2-none",
            "duration_sec": 0.0,
            "sample_sec": float(args.sample_sec),
            "scene_cuts_sec": [],
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
            explicit_model_path=args.east_model.strip() or None,
            model_dir=args.east_model_dir.strip() or None,
            auto_download=parse_bool(args.east_auto_download, default=True),
            model_urls=east_urls,
            sha256=args.east_sha256.strip() or None,
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
    if require_min_video_width and min_video_width > 0 and frame_width > 0 and frame_width < min_video_width:
        raise RuntimeError(
            f"OCR v2 requires min width {min_video_width}px, got {frame_width}px. Use HQ source.mp4/source.original.*."
        )
    max_sec = duration_sec if args.max_sec <= 0 else min(duration_sec, float(args.max_sec))
    if max_sec <= 0:
        raise RuntimeError("Could not determine video duration for OCR v2.")

    trackers: dict[str, SimpleBoxTracker] = {
        region: SimpleBoxTracker(region=region, iou_threshold=float(args.track_iou_threshold), max_misses=int(args.track_max_misses))
        for region in sorted(enabled_regions)
    }

    observations: list[OcrObservation] = []
    scene_cuts: list[float] = []
    prev_frame = None

    total_samples = max(1, int(math.floor(max_sec / args.sample_sec + 1e-9)) + 1)
    progress_every_samples = int(args.progress_every_samples)
    progress_json = parse_bool(args.progress_json, default=True)
    if progress_every_samples <= 0:
        progress_every_samples = max(1, int(math.ceil(total_samples * 0.05)))

    sample_every_frames = max(1, int(round(args.sample_sec * fps))) if fps > 0 else 1
    max_frame_idx = int(max_sec * fps) if fps > 0 else total_frames
    next_sample_frame = 0
    frame_idx = 0
    sample_idx = 0
    started_at = time.monotonic()

    stats: dict[str, Any] = {
        "samples": 0,
        "detector_calls": 0,
        "detected_boxes": 0,
        "track_assignments": 0,
        "ocr_calls": 0,
        "ocr_calls_raw": 0,
        "ocr_calls_preproc": 0,
        "ocr_observations": 0,
        "scene_cut_samples": 0,
        "regions_skipped_low_text": 0,
        "tracks_skipped_unstable": 0,
        "segments_filtered": 0,
    }
    region_likely_previous: dict[str, bool] = {}
    track_observations: dict[str, int] = {}

    print(
        f"[ocr-v2] source={args.video_path} backend={backend} duration={duration_sec:.1f}s "
        f"max_sec={max_sec:.1f}s sample={args.sample_sec}s regions={sorted(enabled_regions)} "
        f"video={frame_width}x{frame_height} detector_mode={detector_mode} "
        f"east_model={'loaded' if east_net is not None else 'none'} "
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

        t = float(frame_idx / fps) if fps > 0 else float(sample_idx * args.sample_sec)
        next_sample_frame = frame_idx + sample_every_frames

        scene_cut_now = False
        if prev_frame is not None and detect_scene_cut(prev_frame, frame, float(args.scene_cut_threshold)):
            if len(scene_cuts) == 0 or abs(float(t) - scene_cuts[-1]) >= 0.25:
                scene_cuts.append(float(t))
            scene_cut_now = True
        prev_frame = frame.copy()
        if scene_cut_now:
            stats["scene_cut_samples"] += 1

        rois = region_rois(
            frame,
            enabled_regions=enabled_regions,
            lower_third_y1=lower_third_y1,
            lower_third_y2=lower_third_y2,
            lower_third_x1=lower_third_x1,
            lower_third_x2=lower_third_x2,
            slide_y1=slide_y1,
            slide_y2=slide_y2,
            slide_x1=slide_x1,
            slide_x2=slide_x2,
        )

        for region, roi, ox, oy in rois:
            if roi is None or roi.size == 0:
                continue
            if targeted_enabled:
                threshold = float(target_thresholds.get(region, 0.24))
                score, _metrics = estimate_text_likelihood(
                    roi,
                    region=region,
                    resize_max_width=target_resize_max_width,
                )
                effective_threshold = max(
                    0.0,
                    threshold - (target_scene_cut_boost if scene_cut_now else 0.0),
                )
                previously_likely = bool(region_likely_previous.get(region, False))
                persistent_threshold = effective_threshold * target_persist_relax if previously_likely else effective_threshold
                force_sample = (sample_idx % target_force_every_samples) == 0
                likely = score >= persistent_threshold
                region_likely_previous[region] = likely
                if (not likely) and (not force_sample):
                    stats["regions_skipped_low_text"] += 1
                    continue
            stats["detector_calls"] += 1
            boxes = detect_text_boxes(
                roi,
                mode=detector_mode,
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
            stats["detected_boxes"] += len(boxes)
            dets: list[tuple[int, int, int, int, float]] = []
            for b in boxes:
                dets.append((int(b.x1 + ox), int(b.y1 + oy), int(b.x2 + ox), int(b.y2 + oy), float(b.score)))

            assignments = trackers[region].update(dets, t_sec=float(t))
            stats["track_assignments"] += len(assignments)

            for track_id, (x1, y1, x2, y2), _ in assignments:
                crop = frame[y1:y2, x1:x2]
                if crop is None or crop.size == 0:
                    continue
                track_key = f"{region}:{track_id}"
                track_observations[track_key] = int(track_observations.get(track_key, 0)) + 1
                if track_observations[track_key] < min_track_observations:
                    stats["tracks_skipped_unstable"] += 1
                    continue
                proc = preprocess_roi(crop)
                if backend == "easyocr":
                    attempts = [("raw", crop), ("preproc", proc)] if parse_bool(args.ocr_preproc_fallback, default=True) else [("raw", crop)]
                else:
                    attempts = [("preproc", proc), ("raw", crop)] if parse_bool(args.ocr_preproc_fallback, default=True) else [("preproc", proc)]

                best_text = ""
                best_conf = 0.0
                best_quality = -1.0
                psm = 7 if region == "lower_third" else 6
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                use_upscale = small_box_upscale if (bw <= small_box_threshold_w or bh <= small_box_threshold_h) else 1.0
                for input_kind, input_img in attempts:
                    stats["ocr_calls"] += 1
                    if input_kind == "raw":
                        stats["ocr_calls_raw"] += 1
                    else:
                        stats["ocr_calls_preproc"] += 1
                    cand_text, cand_conf = run_ocr(
                        backend,
                        engine,
                        input_img,
                        args.lang,
                        psm=psm,
                        upscale=use_upscale,
                    )
                    cand_text = clean_text_base(cand_text)
                    if len(cand_text) < min_text_len:
                        continue
                    if not re.search(r"[A-Za-zÁÉÍÓÚáéíóúÑñ0-9]", cand_text):
                        continue
                    quality_metrics = text_quality_metrics(cand_text)
                    if quality_metrics["alpha_ratio"] < min_text_alpha_ratio:
                        continue
                    if quality_metrics["alnum_ratio"] < min_text_alnum_ratio:
                        continue
                    if quality_metrics["symbol_ratio"] > max_text_symbol_ratio:
                        continue
                    cand_conf = float(max(0.0, min(1.0, cand_conf)))
                    quality = cand_conf + min(1.0, len(cand_text) / 180.0) * 0.25
                    if cand_conf >= float(args.ocr_min_text_confidence):
                        if quality > best_quality:
                            best_quality = quality
                            best_text = cand_text
                            best_conf = cand_conf
                if best_text:
                    stats["ocr_observations"] += 1
                    observations.append(
                        OcrObservation(
                            track_key=track_key,
                            start=float(t),
                            end=float(t + args.sample_sec),
                            text=best_text,
                            confidence=float(best_conf),
                            region=str(region),
                        )
                    )

        sample_idx += 1
        stats["samples"] += 1
        if sample_idx == 1 or sample_idx == total_samples or sample_idx % progress_every_samples == 0:
            pct = (sample_idx / max(1, total_samples)) * 100.0
            elapsed_sec = time.monotonic() - started_at
            speed = sample_idx / elapsed_sec if elapsed_sec > 0 else 0.0
            eta_sec = ((total_samples - sample_idx) / speed) if speed > 0 else 0.0
            print(
                f"[ocr-v2] progress={pct:.1f}% sampled={sample_idx}/{total_samples} "
                f"video_t={format_duration(t)}/{format_duration(max_sec)} "
                f"tracks_obs={len(observations)} scene_cuts={len(scene_cuts)} "
                f"det_boxes={stats['detected_boxes']} ocr_calls={stats['ocr_calls']} "
                f"skipped_roi={stats['regions_skipped_low_text']} skipped_tracks={stats['tracks_skipped_unstable']} "
                f"elapsed={format_duration(elapsed_sec)} eta={format_duration(eta_sec)}",
                file=sys.stderr,
            )
            if progress_json:
                payload = {
                    "type": "ocr_progress",
                    "pipeline": "v2",
                    "percent": round(pct, 3),
                    "sampled": int(sample_idx),
                    "total_samples": int(total_samples),
                    "video_time_sec": round(float(t), 3),
                    "video_duration_sec": round(float(max_sec), 3),
                    "scene_cuts": int(len(scene_cuts)),
                    "detected_boxes": int(stats["detected_boxes"]),
                    "ocr_calls": int(stats["ocr_calls"]),
                    "ocr_observations": int(stats["ocr_observations"]),
                    "regions_skipped_low_text": int(stats["regions_skipped_low_text"]),
                    "tracks_skipped_unstable": int(stats["tracks_skipped_unstable"]),
                    "elapsed_sec": round(float(elapsed_sec), 3),
                    "eta_sec": round(float(eta_sec), 3),
                }
                print(f"[ocr-events-progress] {json.dumps(payload, ensure_ascii=False)}", file=sys.stderr)
        frame_idx += 1

    cap.release()

    fused = fuse_observations(
        observations,
        min_samples=max(1, int(args.fuse_min_samples)),
        similarity_threshold=clamp(float(args.fuse_similarity_threshold), 0.5, 0.99),
    )

    tracks: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    for region, tracker in trackers.items():
        for tr in tracker.finalize():
            key = f"{region}:{tr.track_id}"
            fused_text = fused.get(key)
            track_obj: dict[str, Any] = {
                "track_key": key,
                "track_id": int(tr.track_id),
                "region": str(region),
                "start": float(tr.start_sec),
                "end": float(tr.end_sec),
                "bbox": {"x1": int(tr.x1), "y1": int(tr.y1), "x2": int(tr.x2), "y2": int(tr.y2)},
                "hits": int(tr.hits),
                "misses": int(tr.misses),
                "score": round(float(tr.score), 3),
                "fused_text": fused_text["text"] if fused_text else "",
                "fused_confidence": float(fused_text["confidence"]) if fused_text else 0.0,
                "fused_samples": int(fused_text["samples"]) if fused_text else 0,
            }
            tracks.append(track_obj)
            if fused_text and fused_text.get("text"):
                ctext = clean_text_base(str(fused_text["text"]))
                cmetrics = text_quality_metrics(ctext)
                cconf = float(fused_text["confidence"])
                csamples = int(fused_text["samples"])
                quality_ok = (
                    cmetrics["alpha_ratio"] >= min_text_alpha_ratio
                    and cmetrics["alnum_ratio"] >= min_text_alnum_ratio
                    and cmetrics["symbol_ratio"] <= max_text_symbol_ratio
                )
                support_ok = csamples >= segment_min_samples or cconf >= segment_high_conf_override
                confidence_ok = cconf >= segment_min_confidence or cconf >= segment_high_conf_override
                if ctext and quality_ok and support_ok and confidence_ok:
                    segments.append(
                        {
                            "start": float(max(tr.start_sec, float(fused_text["start"]))),
                            "end": float(min(tr.end_sec, float(fused_text["end"]))),
                            "text": ctext,
                            "type": classify_text(ctext, str(region)),
                            "region": str(region),
                            "confidence": cconf,
                            "track_key": key,
                        }
                    )
                elif ctext:
                    stats["segments_filtered"] += 1

    tracks = sorted(tracks, key=lambda x: (float(x["start"]), str(x["region"]), int(x["track_id"])))
    segments = sorted(segments, key=lambda x: (float(x["start"]), str(x["region"])))

    out_obj = {
        "source": "ocr-v2-scaffold",
        "duration_sec": float(duration_sec),
        "sample_sec": float(args.sample_sec),
        "scene_cuts_sec": scene_cuts,
        "config": {
            "backend": str(backend),
            "regions": sorted(enabled_regions),
            "detector_mode": detector_mode,
            "detector_min_conf": float(args.detector_min_conf),
            "detector_east_min_conf": float(args.detector_east_min_conf),
            "detector_mser_min_conf": float(args.detector_mser_min_conf),
            "detector_max_boxes": int(args.detector_max_boxes),
            "detector_min_w": int(args.detector_min_w),
            "detector_min_h": int(args.detector_min_h),
            "detector_min_aspect": float(args.detector_min_aspect),
            "detector_max_aspect": float(args.detector_max_aspect),
            "track_iou_threshold": float(args.track_iou_threshold),
            "track_max_misses": int(args.track_max_misses),
            "ocr_min_text_confidence": float(args.ocr_min_text_confidence),
            "min_text_len": min_text_len,
            "min_text_alpha_ratio": min_text_alpha_ratio,
            "min_text_alnum_ratio": min_text_alnum_ratio,
            "max_text_symbol_ratio": max_text_symbol_ratio,
            "min_track_observations": min_track_observations,
            "fuse_min_samples": int(args.fuse_min_samples),
            "fuse_similarity_threshold": float(args.fuse_similarity_threshold),
            "segment_min_confidence": segment_min_confidence,
            "segment_min_samples": segment_min_samples,
            "segment_high_confidence_override": segment_high_conf_override,
            "small_box_upscale": small_box_upscale,
            "small_box_threshold_w": small_box_threshold_w,
            "small_box_threshold_h": small_box_threshold_h,
            "targeted_enabled": targeted_enabled,
            "target_thresholds": target_thresholds,
            "target_persist_relax": target_persist_relax,
            "target_force_every_samples": target_force_every_samples,
            "target_scene_cut_boost": target_scene_cut_boost,
            "target_resize_max_width": target_resize_max_width,
            "video_width": frame_width,
            "video_height": frame_height,
            "min_video_width": min_video_width,
            "require_min_video_width": require_min_video_width,
            "east_enabled": bool(east_enabled),
            "east_model": str(east_model_path or args.east_model or ""),
            "east_model_downloaded": bool(east_model_downloaded),
            "east_status": str(east_message),
            "east_loaded": bool(east_net is not None),
        },
        "stats": stats,
        "tracks": tracks,
        "segments": segments,
        "errors": engine_errors,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(
        f"[ocr-v2] wrote tracks={len(tracks)} fused_segments={len(segments)} scene_cuts={len(scene_cuts)} -> {args.out}",
        file=sys.stderr,
    )
    print(json.dumps({"ok": True, "source": out_obj["source"], "tracks": len(tracks), "segments": len(segments)}))


if __name__ == "__main__":
    main()
