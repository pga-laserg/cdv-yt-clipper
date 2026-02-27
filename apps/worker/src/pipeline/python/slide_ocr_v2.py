from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import cv2
import numpy as np

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


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def clean_text(text: str) -> str:
    return clean_text_base(str(text or ""))


def format_duration(sec: float) -> str:
    s = int(max(0.0, sec))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    return f"{h:02d}:{m:02d}:{ss:02d}"


def normalize_text(text: str) -> str:
    t = clean_text(text).lower()
    t = re.sub(r"[^a-z0-9áéíóúñü ]+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    raw = str(raw).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def text_similarity(a: str, b: str) -> float:
    na = normalize_text(a)
    nb = normalize_text(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


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


def phash64(img) -> int:
    if img is None or getattr(img, "size", 0) == 0:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    low = dct[:8, :8].flatten()
    # Drop DC component to reduce brightness sensitivity.
    coeff = low[1:] if low.size > 1 else low
    if coeff.size == 0:
        return 0
    med = float(np.median(coeff))
    out = 0
    for v in low:
        out = (out << 1) | int(float(v) > med)
    return int(out)


def _read_json(path_value: str) -> dict[str, Any] | None:
    try:
        if not path_value or not os.path.exists(path_value):
            return None
        with open(path_value, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _extract_boundary_pair(payload: dict[str, Any]) -> tuple[float | None, float | None]:
    # Supports targeted diarization and boundary adjudicator layouts.
    for key in ("final_clip_bounds", "boundaries", "refined_speaker_bounds"):
        block = payload.get(key)
        if isinstance(block, dict):
            s = block.get("clip_start_sec", block.get("start_sec", block.get("start")))
            e = block.get("clip_end_sec", block.get("end_sec", block.get("end")))
            try:
                start = float(s) if s is not None else None
                end = float(e) if e is not None else None
                return start, end
            except Exception:
                pass
    for s_key, e_key in (
        ("clip_start_sec", "clip_end_sec"),
        ("start_sec", "end_sec"),
        ("start", "end"),
    ):
        try:
            s = payload.get(s_key)
            e = payload.get(e_key)
            start = float(s) if s is not None else None
            end = float(e) if e is not None else None
            return start, end
        except Exception:
            continue
    return None, None


def estimate_sermon_window(
    *,
    sermon_start_sec: float,
    sermon_end_sec: float,
    out_path: str,
    max_sec: float,
) -> tuple[float | None, float | None]:
    # 1) Explicit values win.
    start = float(sermon_start_sec) if float(sermon_start_sec) > 0 else None
    end = float(sermon_end_sec) if float(sermon_end_sec) > 0 else None
    if start is not None:
        if end is not None and end <= start:
            end = None
        return start, end

    # 2) Auto-discover from nearby boundary artifacts.
    work_dir = os.path.dirname(out_path)
    parent_dir = os.path.dirname(work_dir)
    candidates = [
        os.path.join(work_dir, "sermon.boundaries.targeted-diarization.json"),
        os.path.join(work_dir, "sermon.boundaries.face-pass.json"),
        os.path.join(parent_dir, "sermon.boundaries.targeted-diarization.json"),
        os.path.join(parent_dir, "sermon.boundaries.face-pass.json"),
    ]
    for p in candidates:
        payload = _read_json(p)
        if not payload:
            continue
        s, e = _extract_boundary_pair(payload)
        if s is None:
            continue
        if e is not None and e <= s:
            e = None
        return float(max(0.0, s)), (float(max(s, e)) if e is not None else None)

    # 3) If unavailable, no sermon-body boost.
    return None, None


def crop_norm(frame, y1: float, y2: float, x1: float, x2: float):
    h, w = frame.shape[:2]
    yy1 = int(clamp(y1, 0.0, 1.0) * h)
    yy2 = int(clamp(y2, 0.0, 1.0) * h)
    xx1 = int(clamp(x1, 0.0, 1.0) * w)
    xx2 = int(clamp(x2, 0.0, 1.0) * w)
    if yy2 <= yy1 or xx2 <= xx1:
        return None
    return frame[yy1:yy2, xx1:xx2]


def get_timecode_sec(tc: Any) -> float:
    if tc is None:
        return 0.0
    getter = getattr(tc, "get_seconds", None)
    if callable(getter):
        try:
            return float(getter())
        except Exception:
            pass
    try:
        return float(tc)
    except Exception:
        return 0.0


def dedupe_boundaries(points: list[float], duration_sec: float) -> list[float]:
    out: list[float] = []
    seen = set()
    for p in sorted(points):
        v = clamp(float(p), 0.0, duration_sec)
        k = int(round(v * 100))
        if k in seen:
            continue
        seen.add(k)
        out.append(v)
    if not out:
        return [0.0, duration_sec]
    if out[0] > 0.001:
        out = [0.0, *out]
    if out[-1] < duration_sec - 0.001:
        out.append(duration_sec)
    return out


def detect_boundaries_pyscenedetect(
    video_path: str,
    duration_sec: float,
    detector_mode: str,
    content_threshold: float,
    adaptive_threshold: float,
    adaptive_min_content: float,
    min_scene_len_frames: int,
) -> tuple[list[float], str | None]:
    try:
        from scenedetect import detect as sd_detect  # type: ignore
        from scenedetect.detectors import AdaptiveDetector, ContentDetector, HashDetector  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        return [], f"pyscenedetect unavailable: {exc}"

    try:
        mode = str(detector_mode or "adaptive").strip().lower()
        if mode == "content":
            detector = ContentDetector(
                threshold=float(content_threshold),
                min_scene_len=max(1, int(min_scene_len_frames)),
            )
        elif mode == "hash":
            detector = HashDetector(
                threshold=0.395,
                min_scene_len=max(1, int(min_scene_len_frames)),
            )
        else:
            detector = AdaptiveDetector(
                adaptive_threshold=float(adaptive_threshold),
                min_scene_len=max(1, int(min_scene_len_frames)),
                min_content_val=float(adaptive_min_content),
            )
        scenes = sd_detect(
            video_path,
            detector,
            start_time=0.0,
            end_time=float(duration_sec),
            start_in_scene=True,
            show_progress=False,
        )
        points: list[float] = [0.0, float(duration_sec)]
        for start_tc, end_tc in scenes:
            points.append(get_timecode_sec(start_tc))
            points.append(get_timecode_sec(end_tc))
        return dedupe_boundaries(points, duration_sec), None
    except Exception as exc:
        return [], f"pyscenedetect detection failed: {exc}"


@dataclass
class SampleRow:
    t: float
    text_score: float
    motion_score: float
    scene_cut: bool


def collect_samples(
    video_path: str,
    max_sec: float,
    sample_sec: float,
    scene_cut_threshold: float,
    roi_y1: float,
    roi_y2: float,
    roi_x1: float,
    roi_x2: float,
    resize_max_width: int,
    progress_every_samples: int,
    progress_json: bool,
) -> tuple[list[SampleRow], list[float], dict[str, Any], int, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for sampling: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    sample_every_frames = max(1, int(round(sample_sec * fps))) if fps > 0 else 1
    max_frame_idx = int(max_sec * fps) if fps > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_samples = max(1, int(math.floor(max_sec / sample_sec + 1e-9)) + 1)

    rows: list[SampleRow] = []
    fallback_scene_cuts: list[float] = []
    prev_frame = None
    prev_roi_gray = None
    frame_idx = 0
    sample_idx = 0
    next_sample_frame = 0
    started_at = time.monotonic()
    stats = {
        "samples": 0,
        "scene_cut_samples": 0,
        "avg_text_score": 0.0,
        "avg_motion_score": 0.0,
    }

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if fps > 0 and frame_idx > max_frame_idx:
            break
        if frame_idx < next_sample_frame:
            frame_idx += 1
            continue
        t = float(frame_idx / fps) if fps > 0 else float(sample_idx * sample_sec)
        next_sample_frame = frame_idx + sample_every_frames

        scene_cut_now = False
        if prev_frame is not None and detect_scene_cut(prev_frame, frame, scene_cut_threshold):
            if len(fallback_scene_cuts) == 0 or abs(float(t) - fallback_scene_cuts[-1]) >= 0.25:
                fallback_scene_cuts.append(float(t))
            scene_cut_now = True
        prev_frame = frame.copy()

        roi = crop_norm(frame, roi_y1, roi_y2, roi_x1, roi_x2)
        if roi is None or roi.size == 0:
            frame_idx += 1
            continue

        text_score, _likelihood = estimate_text_likelihood(roi, region="slide", resize_max_width=resize_max_width)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        motion_score = 0.0
        if prev_roi_gray is not None:
            if prev_roi_gray.shape != gray.shape:
                prev_resized = cv2.resize(prev_roi_gray, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)
            else:
                prev_resized = prev_roi_gray
            diff = cv2.absdiff(gray, prev_resized)
            motion_score = float(np.mean(diff) / 255.0)
        prev_roi_gray = gray

        rows.append(SampleRow(t=float(t), text_score=float(text_score), motion_score=float(motion_score), scene_cut=bool(scene_cut_now)))
        stats["samples"] += 1
        stats["scene_cut_samples"] += 1 if scene_cut_now else 0
        stats["avg_text_score"] += float(text_score)
        stats["avg_motion_score"] += float(motion_score)

        sample_idx += 1
        if sample_idx == 1 or sample_idx == total_samples or sample_idx % max(1, progress_every_samples) == 0:
            pct = (sample_idx / max(1, total_samples)) * 100.0
            elapsed_sec = time.monotonic() - started_at
            speed = sample_idx / elapsed_sec if elapsed_sec > 0 else 0.0
            eta_sec = ((total_samples - sample_idx) / speed) if speed > 0 else 0.0
            print(
                f"[slide-ocr] sample-progress={pct:.1f}% sampled={sample_idx}/{total_samples} "
                f"video_t={format_duration(t)}/{format_duration(max_sec)} "
                f"scene_cuts={len(fallback_scene_cuts)} elapsed={format_duration(elapsed_sec)} eta={format_duration(eta_sec)}",
                file=sys.stderr,
            )
            if progress_json:
                payload = {
                    "type": "slide_ocr_sample_progress",
                    "percent": round(pct, 3),
                    "sampled": int(sample_idx),
                    "total_samples": int(total_samples),
                    "video_time_sec": round(float(t), 3),
                    "video_duration_sec": round(float(max_sec), 3),
                    "scene_cuts": int(len(fallback_scene_cuts)),
                    "elapsed_sec": round(float(elapsed_sec), 3),
                    "eta_sec": round(float(eta_sec), 3),
                }
                print(f"[slide-ocr-progress] {json.dumps(payload, ensure_ascii=False)}", file=sys.stderr)

        frame_idx += 1

    cap.release()
    if stats["samples"] > 0:
        stats["avg_text_score"] = float(stats["avg_text_score"]) / float(stats["samples"])
        stats["avg_motion_score"] = float(stats["avg_motion_score"]) / float(stats["samples"])
    return rows, fallback_scene_cuts, stats, frame_w, frame_h, fps


def segment_metrics(rows: list[SampleRow], start: float, end: float) -> dict[str, float]:
    selected = [r for r in rows if r.t >= start and r.t <= end]
    if not selected:
        return {"avg_text_score": 0.0, "avg_motion_score": 1.0, "stillness_score": 0.0, "score": 0.0}
    avg_text = float(sum(r.text_score for r in selected)) / float(len(selected))
    avg_motion = float(sum(r.motion_score for r in selected)) / float(len(selected))
    # Motion around ~0.12+ is highly dynamic for these overlays.
    stillness = clamp(1.0 - (avg_motion / 0.12), 0.0, 1.0)
    score = clamp(0.62 * avg_text + 0.38 * stillness, 0.0, 1.0)
    return {
        "avg_text_score": round(avg_text, 4),
        "avg_motion_score": round(avg_motion, 4),
        "stillness_score": round(stillness, 4),
        "score": round(score, 4),
    }


def build_segments(boundaries: list[float], duration_sec: float) -> list[dict[str, float]]:
    points = dedupe_boundaries(boundaries, duration_sec)
    segs: list[dict[str, float]] = []
    for i in range(1, len(points)):
        s = float(points[i - 1])
        e = float(points[i])
        if e <= s + 1e-6:
            continue
        segs.append({"start": s, "end": e, "duration": e - s})
    return segs


def read_frame_at_sec(cap: cv2.VideoCapture, sec: float):
    cap.set(cv2.CAP_PROP_POS_MSEC, float(sec) * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def mask_lower_band(roi, mask_start_y_norm: float):
    if roi is None or roi.size == 0:
        return roi
    out = roi.copy()
    h = out.shape[0]
    y = int(clamp(mask_start_y_norm, 0.0, 1.0) * h)
    if 0 <= y < h:
        out[y:, :] = 0
    return out


def keep_text(text: str, confidence: float, min_len: int, min_conf: float) -> bool:
    t = clean_text(text)
    if len(t) < max(1, int(min_len)):
        return False
    if not re.search(r"[A-Za-zÁÉÍÓÚáéíóúÑñ0-9]", t):
        return False
    if float(confidence) < float(min_conf):
        return False
    return True


def merge_slide_events(
    events: list[dict[str, Any]],
    merge_gap_sec: float,
    text_sim_threshold: float,
    hash_dist_threshold: int,
) -> list[dict[str, Any]]:
    if not events:
        return []
    items = sorted(events, key=lambda e: float(e["start"]))
    out: list[dict[str, Any]] = []
    current = dict(items[0])
    current["appearances"] = [float(current["start"])]
    current["samples"] = 1

    for ev in items[1:]:
        gap = float(ev["start"]) - float(current["end"])
        sim = text_similarity(str(current.get("text", "")), str(ev.get("text", "")))
        hash_dist = hamming_distance(int(current.get("hash", 0)), int(ev.get("hash", 0)))
        similar = sim >= float(text_sim_threshold) or hash_dist <= int(hash_dist_threshold)
        if gap <= float(merge_gap_sec) and similar:
            current["end"] = max(float(current["end"]), float(ev["end"]))
            current["confidence"] = max(float(current["confidence"]), float(ev["confidence"]))
            if len(str(ev.get("text", ""))) >= len(str(current.get("text", ""))):
                current["text"] = str(ev.get("text", ""))
                current["type"] = str(ev.get("type", "on_screen_text"))
            current["appearances"].append(float(ev["start"]))
            current["samples"] = int(current["samples"]) + 1
        else:
            out.append(current)
            current = dict(ev)
            current["appearances"] = [float(current["start"])]
            current["samples"] = 1
    out.append(current)

    for idx, ev in enumerate(out, start=1):
        ev["slide_id"] = f"slide_{idx:03d}"
        ev["confidence"] = round(float(ev.get("confidence", 0.0)), 3)
        ev["appearances"] = [round(float(t), 3) for t in ev.get("appearances", [])]
    return out


def annotate_global_duplicates_for_extraction(
    events: list[dict[str, Any]],
    *,
    enabled: bool,
    require_same_type: bool,
    text_sim_threshold: float,
    phash_near_threshold: int,
    phash_strict_threshold: int,
) -> dict[str, Any]:
    if not events:
        return {
            "enabled": bool(enabled),
            "groups_total": 0,
            "representatives_total": 0,
            "duplicates_total": 0,
            "extract_slide_ids": [],
            "groups": [],
        }

    # Initialize extraction fields (do not remove any event from list).
    for ev in events:
        ev["presentation_group_id"] = ""
        ev["presentation_is_representative"] = False
        ev["presentation_duplicate_of"] = None
        ev["presentation_match"] = None
        ev["extract_for_package"] = False

    if not enabled:
        groups = []
        for idx, ev in enumerate(events, start=1):
            gid = f"group_{idx:03d}"
            ev["presentation_group_id"] = gid
            ev["presentation_is_representative"] = True
            ev["extract_for_package"] = True
            groups.append(
                {
                    "group_id": gid,
                    "representative_slide_id": str(ev.get("slide_id", f"slide_{idx:03d}")),
                    "member_slide_ids": [str(ev.get("slide_id", f"slide_{idx:03d}"))],
                    "member_count": 1,
                }
            )
        return {
            "enabled": False,
            "groups_total": len(groups),
            "representatives_total": len(groups),
            "duplicates_total": 0,
            "extract_slide_ids": [g["representative_slide_id"] for g in groups],
            "groups": groups,
        }

    text_sim_threshold = clamp(float(text_sim_threshold), 0.5, 0.99)
    phash_near_threshold = max(0, int(phash_near_threshold))
    phash_strict_threshold = max(0, int(phash_strict_threshold))

    def _event_hash(ev: dict[str, Any]) -> int:
        for key in ("frame_phash", "hash"):
            val = ev.get(key)
            if val is None:
                continue
            try:
                return int(val)
            except Exception:
                continue
        return 0

    groups: list[dict[str, Any]] = []
    # Keep a stable order by start time to preserve timeline intuition.
    ordered: list[tuple[int, dict[str, Any]]] = list(enumerate(events))
    ordered.sort(key=lambda x: float(x[1].get("start", 0.0)))

    for idx, ev in ordered:
        ev_type = str(ev.get("type", "")).strip().lower()
        ev_text = str(ev.get("text", ""))
        ev_hash = _event_hash(ev)
        best_match: dict[str, Any] | None = None

        for g in groups:
            rep_idx = int(g["rep_idx"])
            rep = events[rep_idx]
            rep_type = str(rep.get("type", "")).strip().lower()
            if require_same_type and ev_type != rep_type:
                continue
            rep_text = str(rep.get("text", ""))
            rep_hash = _event_hash(rep)
            hash_dist = hamming_distance(ev_hash, rep_hash)
            sim = text_similarity(ev_text, rep_text)
            is_dup = False
            rule = ""
            if hash_dist <= phash_strict_threshold:
                is_dup = True
                rule = "strict_hash"
            elif hash_dist <= phash_near_threshold and sim >= text_sim_threshold:
                is_dup = True
                rule = "near_hash_plus_text"
            if not is_dup:
                continue
            # Prefer best match by tighter hash, then stronger text.
            score = float((64 - min(hash_dist, 64))) + (sim * 20.0)
            if best_match is None or score > float(best_match["score"]):
                best_match = {
                    "group_idx": int(g["group_idx"]),
                    "rep_idx": rep_idx,
                    "hash_dist": int(hash_dist),
                    "text_sim": float(sim),
                    "rule": rule,
                    "score": score,
                }

        if best_match is None:
            groups.append(
                {
                    "group_idx": len(groups),
                    "rep_idx": idx,
                    "member_indices": [idx],
                    "match_meta": {},
                }
            )
        else:
            g = groups[int(best_match["group_idx"])]
            g["member_indices"].append(idx)
            g["match_meta"][idx] = {
                "hash_dist": int(best_match["hash_dist"]),
                "text_sim": round(float(best_match["text_sim"]), 4),
                "rule": str(best_match["rule"]),
            }

    # Choose representative per group by confidence/text length and earliest tie-break.
    out_groups: list[dict[str, Any]] = []
    extract_slide_ids: list[str] = []
    for g_idx, g in enumerate(groups, start=1):
        member_indices: list[int] = list(g["member_indices"])
        rep_idx = max(
            member_indices,
            key=lambda i: (
                float(events[i].get("confidence", 0.0)),
                len(str(events[i].get("text", ""))),
                -float(events[i].get("start", 0.0)),
            ),
        )
        g["rep_idx"] = rep_idx
        gid = f"group_{g_idx:03d}"
        rep_slide_id = str(events[rep_idx].get("slide_id", f"slide_{rep_idx+1:03d}"))
        extract_slide_ids.append(rep_slide_id)

        for i in member_indices:
            ev = events[i]
            ev["presentation_group_id"] = gid
            is_rep = i == rep_idx
            ev["presentation_is_representative"] = bool(is_rep)
            ev["extract_for_package"] = bool(is_rep)
            if is_rep:
                ev["presentation_duplicate_of"] = None
                ev["presentation_match"] = None
            else:
                ev["presentation_duplicate_of"] = rep_slide_id
                ev["presentation_match"] = g.get("match_meta", {}).get(i, None)

        out_groups.append(
            {
                "group_id": gid,
                "representative_slide_id": rep_slide_id,
                "member_slide_ids": [str(events[i].get("slide_id", f"slide_{i+1:03d}")) for i in member_indices],
                "member_count": len(member_indices),
            }
        )

    return {
        "enabled": True,
        "require_same_type": bool(require_same_type),
        "text_sim_threshold": float(text_sim_threshold),
        "phash_near_threshold": int(phash_near_threshold),
        "phash_strict_threshold": int(phash_strict_threshold),
        "groups_total": len(out_groups),
        "representatives_total": len(out_groups),
        "duplicates_total": max(0, len(events) - len(out_groups)),
        "extract_slide_ids": extract_slide_ids,
        "groups": out_groups,
    }


def _safe_token(value: str, fallback: str = "item") -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    token = token.strip("._-")
    return token or fallback


def _selected_event_indices_for_extraction(events: list[dict[str, Any]], mode: str) -> list[int]:
    if not events:
        return []
    mode_norm = str(mode or "unique").strip().lower()
    if mode_norm == "all":
        return list(range(len(events)))
    selected = [i for i, ev in enumerate(events) if bool(ev.get("extract_for_package"))]
    if not selected:
        selected = [i for i, ev in enumerate(events) if bool(ev.get("presentation_is_representative"))]
    if not selected:
        selected = list(range(len(events)))
    return selected


def extract_frames_for_events(
    *,
    hq_video: str,
    events: list[dict[str, Any]],
    out_dir: str,
    mode: str,
    jpeg_quality: int,
    overwrite: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "enabled": True,
        "mode": str(mode or "unique"),
        "output_dir": out_dir,
        "selected_events": 0,
        "written_frames": 0,
        "failed_events": 0,
        "skipped_events": 0,
        "files": [],
        "errors": [],
    }

    for ev in events:
        ev["frame_extract_selected"] = False
        ev["frame_extract_status"] = "not_selected"
        ev["frame_extract_path"] = None
        ev["frame_extract_file"] = None
        ev["frame_extract_time_sec"] = None

    selected_indices = _selected_event_indices_for_extraction(events, mode)
    result["selected_events"] = int(len(selected_indices))
    result["skipped_events"] = int(max(0, len(events) - len(selected_indices)))
    if not selected_indices:
        return result

    if not os.path.exists(hq_video):
        result["errors"].append(f"HQ video not found: {hq_video}")
        for idx in selected_indices:
            events[idx]["frame_extract_selected"] = True
            events[idx]["frame_extract_status"] = "failed_missing_video"
        result["failed_events"] = int(len(selected_indices))
        return result

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(hq_video)
    if not cap.isOpened():
        result["errors"].append(f"Cannot open HQ video for extraction: {hq_video}")
        for idx in selected_indices:
            events[idx]["frame_extract_selected"] = True
            events[idx]["frame_extract_status"] = "failed_open_video"
        result["failed_events"] = int(len(selected_indices))
        return result

    quality = int(clamp(float(jpeg_quality), 40.0, 100.0))
    rows: list[dict[str, Any]] = []
    for order, idx in enumerate(selected_indices, start=1):
        ev = events[idx]
        ev["frame_extract_selected"] = True

        t = ev.get("sample_time_sec")
        if not isinstance(t, (int, float)) or not math.isfinite(float(t)):
            start = float(ev.get("start", 0.0))
            end = float(ev.get("end", start))
            t = 0.5 * (start + end)
        t = max(0.0, float(t))

        frame = read_frame_at_sec(cap, t)
        if frame is None:
            ev["frame_extract_status"] = "failed_read_frame"
            result["failed_events"] = int(result["failed_events"]) + 1
            continue

        slide_id = _safe_token(str(ev.get("slide_id", f"slide_{idx + 1:03d}")), fallback=f"slide_{idx + 1:03d}")
        file_name = f"frame_{order:03d}_{slide_id}_{t:09.3f}s.jpg"
        abs_path = os.path.join(out_dir, file_name)
        rel_path = os.path.relpath(abs_path, os.path.dirname(out_dir))

        if overwrite or (not os.path.exists(abs_path)):
            ok = cv2.imwrite(abs_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok:
                ev["frame_extract_status"] = "failed_write"
                result["failed_events"] = int(result["failed_events"]) + 1
                continue

        ev["frame_extract_status"] = "written"
        ev["frame_extract_path"] = rel_path
        ev["frame_extract_file"] = file_name
        ev["frame_extract_time_sec"] = round(float(t), 3)
        result["written_frames"] = int(result["written_frames"]) + 1
        rows.append(
            {
                "order": int(order),
                "event_index": int(idx),
                "slide_id": str(ev.get("slide_id", "")),
                "start": round(float(ev.get("start", 0.0)), 3),
                "end": round(float(ev.get("end", 0.0)), 3),
                "sample_time_sec": round(float(t), 3),
                "type": str(ev.get("type", "")),
                "confidence": round(float(ev.get("confidence", 0.0)), 3),
                "file": file_name,
                "path": rel_path,
            }
        )

    cap.release()
    result["files"] = rows
    return result


def write_extraction_index_md(out_path: str, extraction: dict[str, Any]) -> None:
    rows = extraction.get("files")
    if not isinstance(rows, list):
        rows = []

    lines: list[str] = []
    lines.append("# Slide Frame Extraction Index")
    lines.append("")
    lines.append(f"- Mode: `{extraction.get('mode', 'unknown')}`")
    lines.append(f"- Selected events: `{extraction.get('selected_events', 0)}`")
    lines.append(f"- Written frames: `{extraction.get('written_frames', 0)}`")
    lines.append(f"- Failed events: `{extraction.get('failed_events', 0)}`")
    lines.append("")

    if not rows:
        lines.append("- No extracted frames.")
    else:
        lines.append("| # | Slide ID | Start | End | Sample | Type | Conf | File |")
        lines.append("|---|---|---:|---:|---:|---|---:|---|")
        for row in rows:
            lines.append(
                f"| {int(row.get('order', 0))} | {row.get('slide_id', '')} | "
                f"{float(row.get('start', 0.0)):.3f} | {float(row.get('end', 0.0)):.3f} | "
                f"{float(row.get('sample_time_sec', 0.0)):.3f} | {row.get('type', '')} | "
                f"{float(row.get('confidence', 0.0)):.3f} | `{row.get('file', '')}` |"
            )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def positive_findings_markdown(events: list[dict[str, Any]]) -> str:
    title_events = [e for e in events if str(e.get("type")) == "sermon_title"]
    verse_events = [e for e in events if str(e.get("type")) == "bible_verse"]
    speaker_events = [e for e in events if str(e.get("type")) == "speaker_name"]
    lines: list[str] = []
    lines.append("# Slide OCR Positive Findings")
    lines.append("")
    lines.append(f"- Total events: {len(events)}")
    lines.append(f"- Sermon title slides: {len(title_events)}")
    lines.append(f"- Bible verse slides: {len(verse_events)}")
    lines.append(f"- Speaker-name slides: {len(speaker_events)}")
    lines.append("")
    if len(events) == 0:
        lines.append("- No positive findings.")
        return "\n".join(lines).strip() + "\n"
    lines.append("## Findings")
    for ev in events:
        lines.append(
            f"- [{format_duration(float(ev['start']))} - {format_duration(float(ev['end']))}] "
            f"type={ev.get('type')} conf={float(ev.get('confidence', 0.0)):.2f} text={clean_text(str(ev.get('text', '')))}"
        )
    return "\n".join(lines).strip() + "\n"


def _alnum_ratio(text: str) -> float:
    t = str(text or "")
    if not t:
        return 0.0
    return float(sum(1 for ch in t if ch.isalnum())) / float(max(1, len(t)))


def _symbol_noise_ratio(text: str) -> float:
    t = str(text or "")
    if not t:
        return 1.0
    noisy = set("|[]{}<>~_=")
    bad = sum(1 for ch in t if ch in noisy)
    return float(bad) / float(max(1, len(t)))


def _word_shape_counts(text: str) -> tuple[int, int, int]:
    words = [w for w in str(text or "").split() if w]
    normalized = [re.sub(r"[^A-Za-zÁÉÍÓÚáéíóúÑñÜü0-9]", "", w) for w in words]
    long_words = sum(1 for w in normalized if len(w) >= 3)
    short_words = sum(1 for w in normalized if len(w) <= 1)
    return len(words), long_words, short_words


def keep_text_hard_quality(
    text: str,
    *,
    min_alnum_ratio: float,
    max_symbol_noise_ratio: float,
    min_words: int,
    min_alpha_chars: int,
) -> bool:
    t = clean_text(str(text or ""))
    if not t:
        return False
    if _alnum_ratio(t) < float(min_alnum_ratio):
        return False
    if _symbol_noise_ratio(t) > float(max_symbol_noise_ratio):
        return False

    words, long_words, _short_words = _word_shape_counts(t)
    alpha_chars = sum(1 for ch in t if ch.isalpha())
    if words < int(min_words) and alpha_chars < int(min_alpha_chars):
        return False
    if len(t) >= 18 and words < 2:
        return False
    if long_words == 0 and alpha_chars < int(min_alpha_chars) + 2:
        return False
    return True


def _alpha_chars_count(text: str) -> int:
    return sum(1 for ch in str(text or "") if ch.isalpha())


def lexical_quality_score(text: str) -> float:
    t = clean_text(str(text or ""))
    if not t:
        return 0.0
    words, long_words, _short_words = _word_shape_counts(t)
    alnum = _alnum_ratio(t)
    noise = _symbol_noise_ratio(t)
    length_norm = clamp(len(t) / 120.0, 0.0, 1.0)
    word_norm = clamp(words / 10.0, 0.0, 1.0)
    long_norm = clamp(long_words / 6.0, 0.0, 1.0)
    alpha_norm = clamp(_alpha_chars_count(t) / 60.0, 0.0, 1.0)
    score = (
        0.30 * alnum
        + 0.22 * (1.0 - noise)
        + 0.16 * length_norm
        + 0.12 * word_norm
        + 0.10 * long_norm
        + 0.10 * alpha_norm
    )
    return round(float(clamp(score, 0.0, 1.0)), 6)


def should_apply_cloud_override(
    *,
    local_text: str,
    local_confidence: float,
    cloud_text: str,
    cloud_confidence: float,
    min_gain: float,
) -> bool:
    cloud_clean = clean_text(cloud_text)
    if not cloud_clean:
        return False
    local_clean = clean_text(local_text)
    local_quality = lexical_quality_score(local_clean)
    cloud_quality = lexical_quality_score(cloud_clean)

    if not local_clean:
        return True
    if cloud_quality >= local_quality + max(0.0, float(min_gain)):
        return True
    if cloud_confidence >= local_confidence + 0.14 and cloud_quality >= local_quality - 0.02:
        return True
    if len(cloud_clean) >= int(len(local_clean) * 1.35) and cloud_quality >= local_quality:
        return True
    return False


def enrich_events_with_cloud_text(
    *,
    events: list[dict[str, Any]],
    extraction: dict[str, Any],
    lang_hint: str,
    enabled: bool,
    min_confidence: float,
    min_chars: int,
    min_quality: float,
    max_images: int,
    min_apply_gain: float,
    propagate_duplicates: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "enabled": bool(enabled),
        "backend": "none",
        "attempted": 0,
        "accepted": 0,
        "applied": 0,
        "propagated": 0,
        "max_images": int(max_images),
        "elapsed_sec": 0.0,
        "errors": [],
    }
    if not enabled:
        return summary

    files = extraction.get("files") if isinstance(extraction, dict) else None
    out_dir = str(extraction.get("output_dir", "")) if isinstance(extraction, dict) else ""
    if not isinstance(files, list) or not files:
        summary["errors"].append("cloud_enrichment_skipped:no_extracted_frames")
        return summary

    backend, engine, engine_errors = init_ocr_engine("gcv_text_detection", str(lang_hint))
    cloud_backend = str(backend)
    summary["backend"] = cloud_backend
    if backend not in ("gcv_text_detection", "openai_text_detection") or engine is None:
        if engine_errors:
            summary["errors"].extend([f"cloud_enrichment_unavailable:{e}" for e in engine_errors])
        else:
            summary["errors"].append("cloud_enrichment_unavailable:no_cloud_engine")
        return summary

    started_at = time.monotonic()
    valid_rows: list[dict[str, Any]] = []
    for row in files:
        if not isinstance(row, dict):
            continue
        idx_raw = row.get("event_index")
        try:
            idx = int(idx_raw)
        except Exception:
            continue
        if idx < 0 or idx >= len(events):
            continue
        valid_rows.append(row)
    if max_images > 0:
        valid_rows = valid_rows[: max(1, int(max_images))]

    for row in valid_rows:
        idx = int(row["event_index"])
        ev = events[idx]
        file_name = str(row.get("file", "")).strip()
        row_path = str(row.get("path", "")).strip()
        candidate_paths = []
        if file_name and out_dir:
            candidate_paths.append(os.path.join(out_dir, file_name))
        if row_path:
            if os.path.isabs(row_path):
                candidate_paths.append(row_path)
            elif out_dir:
                candidate_paths.append(os.path.join(os.path.dirname(out_dir), row_path))
        frame_path = next((p for p in candidate_paths if p and os.path.exists(p)), "")
        if not frame_path:
            continue
        img = cv2.imread(frame_path)
        if img is None or img.size == 0:
            continue

        summary["attempted"] = int(summary["attempted"]) + 1
        text, conf, meta = run_ocr(
            str(backend),
            engine,
            img,
            str(lang_hint),
            psm=6,
            upscale=1.0,
            return_meta=True,
        )
        text = clean_text(text)
        conf = float(max(0.0, min(1.0, conf)))
        quality = lexical_quality_score(text)
        accepted = (
            len(text) >= int(max(1, min_chars))
            and conf >= float(min_confidence)
            and quality >= float(min_quality)
        )

        cloud_meta = dict(meta if isinstance(meta, dict) else {})
        cloud_meta.update(
            {
                "text": text,
                "confidence": round(conf, 4),
                "quality": round(quality, 4),
                "accepted": bool(accepted),
                "path": frame_path,
                "file": file_name or os.path.basename(frame_path),
            }
        )
        ev["ocr_cloud"] = cloud_meta
        ev["ocr_cloud_quality"] = round(quality, 4)
        ev["ocr_cloud_applied"] = False

        if not accepted:
            continue
        summary["accepted"] = int(summary["accepted"]) + 1

        local_text = str(ev.get("text", ""))
        local_conf = float(ev.get("confidence", 0.0))
        apply_cloud = should_apply_cloud_override(
            local_text=local_text,
            local_confidence=local_conf,
            cloud_text=text,
            cloud_confidence=conf,
            min_gain=float(min_apply_gain),
        )
        if not apply_cloud:
            continue

        ev["text_local"] = local_text
        ev["confidence_local"] = round(local_conf, 4)
        ev["text"] = text
        ev["confidence"] = round(max(local_conf, conf), 3)
        ev["type"] = classify_text(text, "slide")
        prev_ocr_input = str(ev.get("ocr_input", "")).strip()
        source_tag = f"{cloud_backend}:extract"
        ev["ocr_input"] = f"{prev_ocr_input}+{source_tag}" if prev_ocr_input else source_tag
        ev["ocr_cloud_applied"] = True
        ev["ocr_cloud"]["applied"] = True
        summary["applied"] = int(summary["applied"]) + 1

    if propagate_duplicates:
        groups: dict[str, list[dict[str, Any]]] = {}
        for ev in events:
            gid = str(ev.get("presentation_group_id", "")).strip()
            if not gid:
                continue
            groups.setdefault(gid, []).append(ev)
        for gid, members in groups.items():
            rep = next((ev for ev in members if bool(ev.get("presentation_is_representative"))), members[0])
            rep_cloud = rep.get("ocr_cloud")
            rep_applied = bool(rep.get("ocr_cloud_applied"))
            rep_text = clean_text(str(rep.get("text", "")))
            rep_conf = float(rep.get("confidence", 0.0))
            if not rep_applied or not rep_text:
                continue
            rep_meta = dict(rep_cloud if isinstance(rep_cloud, dict) else {})
            rep_slide_id = str(rep.get("slide_id", "")).strip()
            for ev in members:
                if ev is rep or bool(ev.get("ocr_cloud_applied")):
                    continue
                local_text = str(ev.get("text", ""))
                local_conf = float(ev.get("confidence", 0.0))
                if not should_apply_cloud_override(
                    local_text=local_text,
                    local_confidence=local_conf,
                    cloud_text=rep_text,
                    cloud_confidence=rep_conf,
                    min_gain=max(0.0, float(min_apply_gain) * 0.6),
                ):
                    continue
                ev["text_local"] = local_text
                ev["confidence_local"] = round(local_conf, 4)
                ev["text"] = rep_text
                ev["confidence"] = round(max(local_conf, rep_conf), 3)
                ev["type"] = classify_text(rep_text, "slide")
                prev_ocr_input = str(ev.get("ocr_input", "")).strip()
                source_tag = f"{cloud_backend}:group_propagation"
                ev["ocr_input"] = f"{prev_ocr_input}+{source_tag}" if prev_ocr_input else source_tag
                ev["ocr_cloud"] = {
                    **rep_meta,
                    "applied": True,
                    "propagated": True,
                    "propagated_from_group_id": gid,
                    "propagated_from_slide_id": rep_slide_id or None,
                }
                ev["ocr_cloud_applied"] = True
                summary["propagated"] = int(summary["propagated"]) + 1

    summary["elapsed_sec"] = round(float(time.monotonic() - started_at), 3)
    return summary


def candidate_priority_score(
    candidate: dict[str, Any],
    *,
    sermon_start_sec: float | None = None,
    sermon_end_sec: float | None = None,
) -> float:
    dur = float(candidate.get("duration", 0.0))
    dur_score = clamp(min(max(0.0, dur), 45.0) / 45.0, 0.0, 1.0)
    score = (
        0.55 * float(candidate.get("score", 0.0))
        + 0.20 * float(candidate.get("avg_text_score", 0.0))
        + 0.15 * float(candidate.get("stillness_score", 0.0))
        + 0.10 * float(dur_score)
    )
    start = float(candidate.get("start", 0.0))
    end = float(candidate.get("end", start))
    center = 0.5 * (start + end)
    if sermon_start_sec is not None and sermon_start_sec >= 0:
        # Strong boost for sermon-body material, mild penalties for early pre-sermon noise.
        if center >= sermon_start_sec:
            score += 0.12
            # Extra preference as we move into body while avoiding runaway weight.
            body_depth = clamp((center - sermon_start_sec) / 1800.0, 0.0, 1.0)
            score += 0.08 * body_depth
        else:
            lead = sermon_start_sec - center
            if lead > 120:
                score -= 0.10
            elif lead > 45:
                score -= 0.05
    if sermon_end_sec is not None and sermon_end_sec > 0:
        # Tiny penalty for far post-sermon tail windows (closing logistics noise).
        trail = center - sermon_end_sec
        if trail > 300:
            score -= 0.05
    return round(float(clamp(score, 0.0, 1.0)), 6)


def select_budgeted_candidates(
    selected_candidates: list[dict[str, Any]],
    *,
    max_candidates: int,
    diversity_enabled: bool,
    diversity_bucket_sec: float,
    diversity_min_per_bucket: int,
    rescue_enabled: bool,
    rescue_max_extra: int,
    rescue_per_anchor: int,
    rescue_window_sec: float,
    rescue_min_score: float,
    rescue_min_text_score: float,
) -> list[dict[str, Any]]:
    ranked = sorted(selected_candidates, key=lambda c: float(c.get("budget_priority", 0.0)), reverse=True)
    if max_candidates <= 0:
        return []

    bucket_sec = max(30.0, float(diversity_bucket_sec))
    min_per_bucket = max(1, int(diversity_min_per_bucket))
    max_candidates = max(1, int(max_candidates))

    buckets: dict[int, list[dict[str, Any]]] = {}
    for candidate in ranked:
        start_sec = float(candidate.get("start", 0.0))
        bucket_id = int(max(0.0, math.floor(start_sec / bucket_sec)))
        buckets.setdefault(bucket_id, []).append(candidate)

    # Strongest buckets get priority when seed-filling.
    bucket_order = sorted(
        buckets.keys(),
        key=lambda bid: float(buckets[bid][0].get("budget_priority", 0.0)),
        reverse=True,
    )

    picked: list[dict[str, Any]] = []
    picked_keys: set[tuple[float, float]] = set()

    if diversity_enabled:
        # Pass A: guarantee temporal spread.
        saturated = False
        for bucket_id in bucket_order:
            for candidate in buckets[bucket_id][:min_per_bucket]:
                key = (float(candidate.get("start", 0.0)), float(candidate.get("end", 0.0)))
                if key in picked_keys:
                    continue
                picked.append(candidate)
                picked_keys.add(key)
                if len(picked) >= max_candidates:
                    saturated = True
                    break
            if saturated:
                break

        # Pass B: fill remaining budget with global best.
        if len(picked) < max_candidates:
            for candidate in ranked:
                key = (float(candidate.get("start", 0.0)), float(candidate.get("end", 0.0)))
                if key in picked_keys:
                    continue
                picked.append(candidate)
                picked_keys.add(key)
                if len(picked) >= max_candidates:
                    break
    else:
        for candidate in ranked:
            key = (float(candidate.get("start", 0.0)), float(candidate.get("end", 0.0)))
            if key in picked_keys:
                continue
            picked.append(candidate)
            picked_keys.add(key)
            if len(picked) >= max_candidates:
                break

    # Pass C: local rescue for near-neighbor candidates in the same temporal bucket.
    rescue_enabled = bool(rescue_enabled)
    rescue_max_extra = max(0, int(rescue_max_extra))
    rescue_per_anchor = max(0, int(rescue_per_anchor))
    rescue_window_sec = max(5.0, float(rescue_window_sec))
    rescue_min_score = float(rescue_min_score)
    rescue_min_text_score = float(rescue_min_text_score)
    if not rescue_enabled or rescue_max_extra <= 0 or rescue_per_anchor <= 0:
        return picked

    buckets_by_start: dict[int, list[dict[str, Any]]] = {}
    for bid, arr in buckets.items():
        buckets_by_start[bid] = sorted(arr, key=lambda c: float(c.get("start", 0.0)))

    rescue_pool: list[dict[str, Any]] = []
    rescue_keys: set[tuple[float, float]] = set()
    anchors = sorted(picked, key=lambda c: float(c.get("start", 0.0)))
    for anchor in anchors:
        a_start = float(anchor.get("start", 0.0))
        a_end = float(anchor.get("end", a_start))
        bucket_id = int(max(0.0, math.floor(a_start / bucket_sec)))
        candidates_in_bucket = buckets_by_start.get(bucket_id, [])
        local: list[dict[str, Any]] = []
        for candidate in candidates_in_bucket:
            key = (float(candidate.get("start", 0.0)), float(candidate.get("end", 0.0)))
            if key in picked_keys or key in rescue_keys:
                continue
            c_start = float(candidate.get("start", 0.0))
            if c_start < a_end:
                continue
            if c_start - a_end > rescue_window_sec:
                break
            if float(candidate.get("score", 0.0)) < rescue_min_score:
                continue
            if float(candidate.get("avg_text_score", 0.0)) < rescue_min_text_score:
                continue
            local.append(candidate)
        if not local:
            continue
        local.sort(
            key=lambda c: (
                float(c.get("avg_text_score", 0.0)),
                float(c.get("score", 0.0)),
                float(c.get("budget_priority", 0.0)),
                -float(c.get("start", 0.0)),
            ),
            reverse=True,
        )
        for candidate in local[:rescue_per_anchor]:
            key = (float(candidate.get("start", 0.0)), float(candidate.get("end", 0.0)))
            if key in picked_keys or key in rescue_keys:
                continue
            rescue_pool.append(candidate)
            rescue_keys.add(key)

    if not rescue_pool:
        return picked

    # Keep local-anchor order to avoid global-priority domination and preserve nearby context.
    rescue_extra = rescue_pool[:rescue_max_extra]
    for candidate in rescue_extra:
        candidate["budget_rescued"] = True
        key = (float(candidate.get("start", 0.0)), float(candidate.get("end", 0.0)))
        if key in picked_keys:
            continue
        picked.append(candidate)
        picked_keys.add(key)
    return picked


def _parse_csv_set(raw: str) -> set[str]:
    return {part.strip().lower() for part in str(raw or "").split(",") if part.strip()}


def keep_event_by_strict_rescue(
    event: dict[str, Any],
    *,
    min_confidence: float,
    min_chars: int,
    allowed_types: set[str],
    keyword_regex: str,
) -> bool:
    text = clean_text(str(event.get("text", "")))
    if len(text) < int(min_chars):
        return False
    if float(event.get("confidence", 0.0)) < float(min_confidence):
        return False

    ev_type = str(event.get("type", "")).strip().lower()
    if ev_type in allowed_types:
        return True

    pattern = str(keyword_regex or "").strip()
    if pattern:
        try:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return True
        except re.error:
            return False
    return False


def keep_fullscreen_strict(
    event: dict[str, Any],
    *,
    min_duration_sec: float,
    min_confidence: float,
    min_chars: int,
    min_words: int,
    min_long_words: int,
    min_alnum_ratio: float,
    max_symbol_noise_ratio: float,
) -> bool:
    start = float(event.get("start", 0.0))
    end = float(event.get("end", 0.0))
    duration = float(max(0.0, end - start))
    if duration < float(min_duration_sec):
        return False

    conf = float(event.get("confidence", 0.0))
    if conf < float(min_confidence):
        return False

    text = clean_text(str(event.get("text", "")))
    if len(text) < int(min_chars):
        return False

    words, long_words, short_words = _word_shape_counts(text)
    if words < int(min_words):
        return False
    if long_words < int(min_long_words):
        return False
    if short_words > max(2, words // 3):
        return False

    if _alnum_ratio(text) < float(min_alnum_ratio):
        return False
    if _symbol_noise_ratio(text) > float(max_symbol_noise_ratio):
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Slide OCR v2: robust slide-segment candidate lane + targeted HQ OCR.")
    parser.add_argument("video_path", help="Low-res/working video path for cheap candidate generation.")
    parser.add_argument("--hq-video", type=str, default="", help="HQ video path for OCR frame extraction.")
    parser.add_argument("--out", required=True, help="Output JSON path (slide.events.json).")
    parser.add_argument("--summary-md", type=str, default="", help="Optional markdown summary output path.")
    parser.add_argument("--sample-sec", type=float, default=float(os.getenv("SLIDE_OCR_SAMPLE_SEC", "1.0")))
    parser.add_argument("--max-sec", type=float, default=float(os.getenv("SLIDE_OCR_MAX_DURATION_SEC", "0")))
    parser.add_argument("--scene-cut-threshold", type=float, default=float(os.getenv("SLIDE_OCR_SCENE_CUT_THRESHOLD", "0.32")))
    parser.add_argument("--roi-y1", type=float, default=float(os.getenv("SLIDE_OCR_ROI_Y1", "0.05")))
    parser.add_argument("--roi-y2", type=float, default=float(os.getenv("SLIDE_OCR_ROI_Y2", "0.70")))
    parser.add_argument("--roi-x1", type=float, default=float(os.getenv("SLIDE_OCR_ROI_X1", "0.02")))
    parser.add_argument("--roi-x2", type=float, default=float(os.getenv("SLIDE_OCR_ROI_X2", "0.98")))
    parser.add_argument("--resize-max-width", type=int, default=int(os.getenv("SLIDE_OCR_RESIZE_MAX_WIDTH", "640")))
    parser.add_argument("--candidate-min-sec", type=float, default=float(os.getenv("SLIDE_OCR_CANDIDATE_MIN_SEC", "2.5")))
    parser.add_argument("--candidate-max-sec", type=float, default=float(os.getenv("SLIDE_OCR_CANDIDATE_MAX_SEC", "220.0")))
    parser.add_argument("--candidate-min-text", type=float, default=float(os.getenv("SLIDE_OCR_CANDIDATE_MIN_TEXT", "0.18")))
    parser.add_argument("--candidate-min-stillness", type=float, default=float(os.getenv("SLIDE_OCR_CANDIDATE_MIN_STILLNESS", "0.28")))
    parser.add_argument("--candidate-min-score", type=float, default=float(os.getenv("SLIDE_OCR_CANDIDATE_MIN_SCORE", "0.32")))
    parser.add_argument(
        "--candidate-text-dominant-override",
        type=str,
        default=os.getenv("SLIDE_OCR_CANDIDATE_TEXT_DOMINANT_OVERRIDE", "true"),
        help="Allow low-stillness candidates when text signal is strong (helps short transition-heavy slides).",
    )
    parser.add_argument(
        "--candidate-text-dominant-min-sec",
        type=float,
        default=float(os.getenv("SLIDE_OCR_CANDIDATE_TEXT_DOMINANT_MIN_SEC", "5.0")),
    )
    parser.add_argument(
        "--candidate-text-dominant-min-text",
        type=float,
        default=float(os.getenv("SLIDE_OCR_CANDIDATE_TEXT_DOMINANT_MIN_TEXT", "0.58")),
    )
    parser.add_argument(
        "--candidate-text-dominant-min-score",
        type=float,
        default=float(os.getenv("SLIDE_OCR_CANDIDATE_TEXT_DOMINANT_MIN_SCORE", "0.42")),
    )
    parser.add_argument("--two-pass-budgeted", type=str, default=os.getenv("SLIDE_OCR_TWO_PASS_BUDGETED", "true"))
    parser.add_argument("--ocr-budget-max-candidates", type=int, default=int(os.getenv("SLIDE_OCR_BUDGET_MAX_CANDIDATES", "120")))
    parser.add_argument("--ocr-budget-tries-per-candidate", type=int, default=int(os.getenv("SLIDE_OCR_BUDGET_TRIES_PER_CANDIDATE", "2")))
    parser.add_argument("--ocr-budget-diversity-enabled", type=str, default=os.getenv("SLIDE_OCR_BUDGET_DIVERSITY_ENABLED", "true"))
    parser.add_argument("--ocr-budget-diversity-bucket-sec", type=float, default=float(os.getenv("SLIDE_OCR_BUDGET_DIVERSITY_BUCKET_SEC", "300")))
    parser.add_argument("--ocr-budget-diversity-min-per-bucket", type=int, default=int(os.getenv("SLIDE_OCR_BUDGET_DIVERSITY_MIN_PER_BUCKET", "4")))
    parser.add_argument("--ocr-budget-rescue-enabled", type=str, default=os.getenv("SLIDE_OCR_BUDGET_RESCUE_ENABLED", "true"))
    parser.add_argument("--ocr-budget-rescue-max-extra", type=int, default=int(os.getenv("SLIDE_OCR_BUDGET_RESCUE_MAX_EXTRA", "30")))
    parser.add_argument("--ocr-budget-rescue-per-anchor", type=int, default=int(os.getenv("SLIDE_OCR_BUDGET_RESCUE_PER_ANCHOR", "2")))
    parser.add_argument("--ocr-budget-rescue-window-sec", type=float, default=float(os.getenv("SLIDE_OCR_BUDGET_RESCUE_WINDOW_SEC", "60")))
    parser.add_argument("--ocr-budget-rescue-min-score", type=float, default=float(os.getenv("SLIDE_OCR_BUDGET_RESCUE_MIN_SCORE", "0.44")))
    parser.add_argument("--ocr-budget-rescue-min-text-score", type=float, default=float(os.getenv("SLIDE_OCR_BUDGET_RESCUE_MIN_TEXT_SCORE", "0.42")))
    parser.add_argument("--pyscenedetect-enabled", type=str, default=os.getenv("SLIDE_OCR_PYSCENEDETECT_ENABLED", "true"))
    parser.add_argument("--pyscenedetect-mode", type=str, default=os.getenv("SLIDE_OCR_PYSCENEDETECT_MODE", "adaptive"))
    parser.add_argument("--pyscenedetect-content-threshold", type=float, default=float(os.getenv("SLIDE_OCR_PYSCENEDETECT_CONTENT_THRESHOLD", "27.0")))
    parser.add_argument("--pyscenedetect-adaptive-threshold", type=float, default=float(os.getenv("SLIDE_OCR_PYSCENEDETECT_ADAPTIVE_THRESHOLD", "3.0")))
    parser.add_argument("--pyscenedetect-adaptive-min-content", type=float, default=float(os.getenv("SLIDE_OCR_PYSCENEDETECT_ADAPTIVE_MIN_CONTENT", "15.0")))
    parser.add_argument("--pyscenedetect-min-scene-len-frames", type=int, default=int(os.getenv("SLIDE_OCR_PYSCENEDETECT_MIN_SCENE_LEN_FRAMES", "15")))
    parser.add_argument("--backend", type=str, default=os.getenv("SLIDE_OCR_BACKEND", os.getenv("OCR_BACKEND", "auto")))
    parser.add_argument("--lang", type=str, default=os.getenv("SLIDE_OCR_LANG", os.getenv("OCR_LANG", "es,en")))
    parser.add_argument(
        "--tesseract-fallback",
        type=str,
        default=os.getenv("SLIDE_OCR_TESSERACT_FALLBACK", "true"),
        help="If primary backend is EasyOCR, fallback to Tesseract when no valid text is produced.",
    )
    parser.add_argument("--min-text-len", type=int, default=int(os.getenv("SLIDE_OCR_MIN_TEXT_LEN", "6")))
    parser.add_argument("--min-text-confidence", type=float, default=float(os.getenv("SLIDE_OCR_MIN_TEXT_CONFIDENCE", "0.40")))
    parser.add_argument("--hard-text-gate", type=str, default=os.getenv("SLIDE_OCR_HARD_TEXT_GATE", "true"))
    parser.add_argument(
        "--hard-text-min-alnum-ratio",
        type=float,
        default=float(os.getenv("SLIDE_OCR_HARD_TEXT_MIN_ALNUM_RATIO", "0.48")),
    )
    parser.add_argument(
        "--hard-text-max-symbol-noise-ratio",
        type=float,
        default=float(os.getenv("SLIDE_OCR_HARD_TEXT_MAX_SYMBOL_NOISE_RATIO", "0.14")),
    )
    parser.add_argument("--hard-text-min-words", type=int, default=int(os.getenv("SLIDE_OCR_HARD_TEXT_MIN_WORDS", "1")))
    parser.add_argument(
        "--hard-text-min-alpha-chars",
        type=int,
        default=int(os.getenv("SLIDE_OCR_HARD_TEXT_MIN_ALPHA_CHARS", "5")),
    )
    parser.add_argument("--mask-lower-third", type=str, default=os.getenv("SLIDE_OCR_MASK_LOWER_THIRD", "true"))
    parser.add_argument("--mask-lower-third-start-y", type=float, default=float(os.getenv("SLIDE_OCR_MASK_LOWER_THIRD_START_Y", "0.82")))
    parser.add_argument("--merge-gap-sec", type=float, default=float(os.getenv("SLIDE_OCR_MERGE_GAP_SEC", "6.0")))
    parser.add_argument("--merge-text-sim-threshold", type=float, default=float(os.getenv("SLIDE_OCR_MERGE_TEXT_SIM_THRESHOLD", "0.88")))
    parser.add_argument("--merge-hash-dist-threshold", type=int, default=int(os.getenv("SLIDE_OCR_MERGE_HASH_DIST_THRESHOLD", "4")))
    parser.add_argument(
        "--global-dup-enabled",
        type=str,
        default=os.getenv("SLIDE_OCR_GLOBAL_DUP_ENABLED", "true"),
        help="Annotate global duplicate groups for extraction package (keeps full event list).",
    )
    parser.add_argument(
        "--global-dup-require-same-type",
        type=str,
        default=os.getenv("SLIDE_OCR_GLOBAL_DUP_REQUIRE_SAME_TYPE", "true"),
    )
    parser.add_argument(
        "--global-dup-text-sim-threshold",
        type=float,
        default=float(os.getenv("SLIDE_OCR_GLOBAL_DUP_TEXT_SIM_THRESHOLD", "0.88")),
    )
    parser.add_argument(
        "--global-dup-phash-near-threshold",
        type=int,
        default=int(os.getenv("SLIDE_OCR_GLOBAL_DUP_PHASH_NEAR_THRESHOLD", "5")),
    )
    parser.add_argument(
        "--global-dup-phash-strict-threshold",
        type=int,
        default=int(os.getenv("SLIDE_OCR_GLOBAL_DUP_PHASH_STRICT_THRESHOLD", "2")),
    )
    parser.add_argument(
        "--extract-frames",
        type=str,
        default=os.getenv("SLIDE_OCR_EXTRACT_FRAMES", "true"),
        help="Extract still frames from HQ video for selected slide events.",
    )
    parser.add_argument(
        "--extract-mode",
        type=str,
        default=os.getenv("SLIDE_OCR_EXTRACT_MODE", "unique"),
        help="Frame extraction mode: unique (representatives only) or all.",
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        default=os.getenv("SLIDE_OCR_EXTRACT_DIR", "slide.frames.extracted"),
        help="Output directory for extracted frames (relative to output directory unless absolute).",
    )
    parser.add_argument(
        "--extract-jpeg-quality",
        type=int,
        default=int(os.getenv("SLIDE_OCR_EXTRACT_JPEG_QUALITY", "95")),
    )
    parser.add_argument(
        "--extract-overwrite",
        type=str,
        default=os.getenv("SLIDE_OCR_EXTRACT_OVERWRITE", "true"),
    )
    parser.add_argument(
        "--extract-index-md",
        type=str,
        default=os.getenv("SLIDE_OCR_EXTRACT_INDEX_MD", ""),
        help="Optional markdown index path for extracted frames.",
    )
    parser.add_argument(
        "--cloud-text-enrich",
        type=str,
        default=os.getenv("SLIDE_OCR_CLOUD_TEXT_ENRICH", "false"),
        help="Run Google Cloud Vision TEXT_DETECTION on extracted slide frames and enrich event text/types.",
    )
    parser.add_argument(
        "--cloud-text-min-confidence",
        type=float,
        default=float(os.getenv("SLIDE_OCR_CLOUD_TEXT_MIN_CONFIDENCE", "0.40")),
    )
    parser.add_argument(
        "--cloud-text-min-chars",
        type=int,
        default=int(os.getenv("SLIDE_OCR_CLOUD_TEXT_MIN_CHARS", "8")),
    )
    parser.add_argument(
        "--cloud-text-min-quality",
        type=float,
        default=float(os.getenv("SLIDE_OCR_CLOUD_TEXT_MIN_QUALITY", "0.48")),
    )
    parser.add_argument(
        "--cloud-text-max-images",
        type=int,
        default=int(os.getenv("SLIDE_OCR_CLOUD_TEXT_MAX_IMAGES", "0")),
    )
    parser.add_argument(
        "--cloud-text-min-apply-gain",
        type=float,
        default=float(os.getenv("SLIDE_OCR_CLOUD_TEXT_MIN_APPLY_GAIN", "0.06")),
    )
    parser.add_argument(
        "--cloud-text-propagate-duplicates",
        type=str,
        default=os.getenv("SLIDE_OCR_CLOUD_TEXT_PROPAGATE_DUPLICATES", "true"),
    )
    parser.add_argument(
        "--fullscreen-strict",
        type=str,
        default=os.getenv("SLIDE_OCR_FULLSCREEN_STRICT", "false"),
        help="Keep only high-signal, full-screen-like slide events (aggressive noise suppression).",
    )
    parser.add_argument(
        "--fullscreen-strict-min-duration-sec",
        type=float,
        default=float(os.getenv("SLIDE_OCR_FULLSCREEN_STRICT_MIN_DURATION_SEC", "8.0")),
    )
    parser.add_argument(
        "--fullscreen-strict-min-confidence",
        type=float,
        default=float(os.getenv("SLIDE_OCR_FULLSCREEN_STRICT_MIN_CONFIDENCE", "0.50")),
    )
    parser.add_argument(
        "--fullscreen-strict-min-chars",
        type=int,
        default=int(os.getenv("SLIDE_OCR_FULLSCREEN_STRICT_MIN_CHARS", "12")),
    )
    parser.add_argument(
        "--fullscreen-strict-min-words",
        type=int,
        default=int(os.getenv("SLIDE_OCR_FULLSCREEN_STRICT_MIN_WORDS", "2")),
    )
    parser.add_argument(
        "--fullscreen-strict-min-long-words",
        type=int,
        default=int(os.getenv("SLIDE_OCR_FULLSCREEN_STRICT_MIN_LONG_WORDS", "1")),
    )
    parser.add_argument(
        "--fullscreen-strict-min-alnum-ratio",
        type=float,
        default=float(os.getenv("SLIDE_OCR_FULLSCREEN_STRICT_MIN_ALNUM_RATIO", "0.55")),
    )
    parser.add_argument(
        "--fullscreen-strict-max-symbol-noise-ratio",
        type=float,
        default=float(os.getenv("SLIDE_OCR_FULLSCREEN_STRICT_MAX_SYMBOL_NOISE_RATIO", "0.12")),
    )
    parser.add_argument(
        "--fullscreen-strict-rescue",
        type=str,
        default=os.getenv("SLIDE_OCR_FULLSCREEN_STRICT_RESCUE", "true"),
    )
    parser.add_argument(
        "--fullscreen-strict-rescue-min-confidence",
        type=float,
        default=float(os.getenv("SLIDE_OCR_FULLSCREEN_STRICT_RESCUE_MIN_CONFIDENCE", "0.42")),
    )
    parser.add_argument(
        "--fullscreen-strict-rescue-min-chars",
        type=int,
        default=int(os.getenv("SLIDE_OCR_FULLSCREEN_STRICT_RESCUE_MIN_CHARS", "6")),
    )
    parser.add_argument(
        "--fullscreen-strict-rescue-types",
        type=str,
        default=os.getenv("SLIDE_OCR_FULLSCREEN_STRICT_RESCUE_TYPES", "sermon_title,bible_verse,speaker_name,song_lyric,on_screen_text"),
    )
    parser.add_argument(
        "--fullscreen-strict-rescue-keywords",
        type=str,
        default=os.getenv(
            "SLIDE_OCR_FULLSCREEN_STRICT_RESCUE_KEYWORDS",
            r"\b(pr\.?|pastor|predicaci[oó]n|serm[oó]n|tema|t[ií]tulo|mensaje|[1-3]?\s?[a-záéíóúñü]+\s+\d{1,3}[:.]\d{1,3})\b",
        ),
    )
    parser.add_argument("--progress-every-samples", type=int, default=int(os.getenv("SLIDE_OCR_PROGRESS_EVERY_SAMPLES", "0")))
    parser.add_argument("--progress-json", type=str, default=os.getenv("SLIDE_OCR_PROGRESS_JSON", "true"))
    parser.add_argument(
        "--sermon-start-sec",
        type=float,
        default=env_float("SLIDE_OCR_SERMON_START_SEC", 0.0),
        help="Optional sermon start hint (seconds) used to boost mid/late sermon candidates.",
    )
    parser.add_argument(
        "--sermon-end-sec",
        type=float,
        default=env_float("SLIDE_OCR_SERMON_END_SEC", 0.0),
        help="Optional sermon end hint (seconds) used to slightly penalize far post-sermon candidates.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")
    if args.sample_sec <= 0:
        raise ValueError("sample-sec must be > 0")

    progress_json = parse_bool(args.progress_json, default=True)
    progress_every_samples = int(args.progress_every_samples)
    if progress_every_samples <= 0:
        progress_every_samples = 10

    probe = cv2.VideoCapture(args.video_path)
    if not probe.isOpened():
        raise RuntimeError(f"Cannot open input video: {args.video_path}")
    fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = float(total_frames / fps) if fps > 0 and total_frames > 0 else 0.0
    probe.release()
    max_sec = duration_sec if float(args.max_sec) <= 0 else min(duration_sec, float(args.max_sec))
    if max_sec <= 0:
        raise RuntimeError("Could not determine video duration for slide OCR pass.")

    hq_video = str(args.hq_video).strip() or str(args.video_path)
    if not os.path.exists(hq_video):
        hq_video = str(args.video_path)

    print(
        f"[slide-ocr] source_low={args.video_path} source_hq={hq_video} duration={duration_sec:.1f}s "
        f"max_sec={max_sec:.1f}s sample={args.sample_sec}s",
        file=sys.stderr,
    )

    started_at = time.monotonic()
    rows, fallback_scene_cuts, sample_stats, frame_w, frame_h, sampled_fps = collect_samples(
        args.video_path,
        max_sec=float(max_sec),
        sample_sec=float(args.sample_sec),
        scene_cut_threshold=float(args.scene_cut_threshold),
        roi_y1=float(args.roi_y1),
        roi_y2=float(args.roi_y2),
        roi_x1=float(args.roi_x1),
        roi_x2=float(args.roi_x2),
        resize_max_width=max(160, int(args.resize_max_width)),
        progress_every_samples=max(1, int(progress_every_samples)),
        progress_json=bool(progress_json),
    )

    boundaries: list[float] = [0.0, float(max_sec), *fallback_scene_cuts]
    errors: list[str] = []
    if parse_bool(args.pyscenedetect_enabled, default=True):
        py_boundaries, py_error = detect_boundaries_pyscenedetect(
            args.video_path,
            float(max_sec),
            detector_mode=str(args.pyscenedetect_mode),
            content_threshold=float(args.pyscenedetect_content_threshold),
            adaptive_threshold=float(args.pyscenedetect_adaptive_threshold),
            adaptive_min_content=float(args.pyscenedetect_adaptive_min_content),
            min_scene_len_frames=max(1, int(args.pyscenedetect_min_scene_len_frames)),
        )
        if py_error:
            errors.append(py_error)
        if py_boundaries:
            boundaries.extend(py_boundaries)
            print(f"[slide-ocr] pyscenedetect boundaries={len(py_boundaries)}", file=sys.stderr)
    boundaries = dedupe_boundaries(boundaries, float(max_sec))

    segments = build_segments(boundaries, float(max_sec))
    candidates: list[dict[str, Any]] = []
    text_override_enabled = parse_bool(args.candidate_text_dominant_override, default=True)
    text_override_count = 0
    for seg in segments:
        s = float(seg["start"])
        e = float(seg["end"])
        dur = e - s
        m = segment_metrics(rows, s, e)
        selected_base = (
            dur >= float(args.candidate_min_sec)
            and dur <= float(args.candidate_max_sec)
            and float(m["avg_text_score"]) >= float(args.candidate_min_text)
            and float(m["stillness_score"]) >= float(args.candidate_min_stillness)
            and float(m["score"]) >= float(args.candidate_min_score)
        )
        selected_text_override = (
            bool(text_override_enabled)
            and dur >= float(args.candidate_text_dominant_min_sec)
            and float(m["avg_text_score"]) >= float(args.candidate_text_dominant_min_text)
            and float(m["score"]) >= float(args.candidate_text_dominant_min_score)
        )
        selected = bool(selected_base or selected_text_override)
        if selected_text_override and not selected_base:
            text_override_count += 1
        candidates.append(
            {
                "start": round(s, 3),
                "end": round(e, 3),
                "duration": round(dur, 3),
                **m,
                "selected": bool(selected),
                "selected_by_text_override": bool(selected_text_override and not selected_base),
            }
        )
    if text_override_enabled:
        print(f"[slide-ocr] text-dominant candidate overrides={text_override_count}", file=sys.stderr)

    sermon_start_hint, sermon_end_hint = estimate_sermon_window(
        sermon_start_sec=float(args.sermon_start_sec),
        sermon_end_sec=float(args.sermon_end_sec),
        out_path=str(args.out),
        max_sec=float(max_sec),
    )
    if sermon_start_hint is not None:
        hint_msg = f"[slide-ocr] sermon-window start={sermon_start_hint:.1f}s"
        if sermon_end_hint is not None:
            hint_msg += f" end={sermon_end_hint:.1f}s"
        print(hint_msg, file=sys.stderr)
    else:
        print("[slide-ocr] sermon-window unavailable (no boost)", file=sys.stderr)

    for candidate in candidates:
        candidate["budget_priority"] = candidate_priority_score(
            candidate,
            sermon_start_sec=sermon_start_hint,
            sermon_end_sec=sermon_end_hint,
        )
        candidate["budget_rescued"] = False

    selected_candidates = [c for c in candidates if bool(c.get("selected"))]
    budget_enabled = parse_bool(args.two_pass_budgeted, default=True)
    budget_max_candidates = max(1, int(args.ocr_budget_max_candidates))
    diversity_enabled = parse_bool(args.ocr_budget_diversity_enabled, default=True)
    diversity_bucket_sec = max(30.0, float(args.ocr_budget_diversity_bucket_sec))
    diversity_min_per_bucket = max(1, int(args.ocr_budget_diversity_min_per_bucket))
    rescue_enabled = parse_bool(args.ocr_budget_rescue_enabled, default=True)
    rescue_max_extra = max(0, int(args.ocr_budget_rescue_max_extra))
    rescue_per_anchor = max(0, int(args.ocr_budget_rescue_per_anchor))
    rescue_window_sec = max(5.0, float(args.ocr_budget_rescue_window_sec))
    rescue_min_score = float(args.ocr_budget_rescue_min_score)
    rescue_min_text_score = float(args.ocr_budget_rescue_min_text_score)
    if budget_enabled:
        budgeted = select_budgeted_candidates(
            selected_candidates,
            max_candidates=budget_max_candidates,
            diversity_enabled=diversity_enabled,
            diversity_bucket_sec=diversity_bucket_sec,
            diversity_min_per_bucket=diversity_min_per_bucket,
            rescue_enabled=rescue_enabled,
            rescue_max_extra=rescue_max_extra,
            rescue_per_anchor=rescue_per_anchor,
            rescue_window_sec=rescue_window_sec,
            rescue_min_score=rescue_min_score,
            rescue_min_text_score=rescue_min_text_score,
        )
    else:
        budgeted = sorted(selected_candidates, key=lambda c: float(c.get("budget_priority", 0.0)), reverse=True)
    budgeted_key = {(float(c["start"]), float(c["end"])) for c in budgeted}
    for candidate in candidates:
        key = (float(candidate["start"]), float(candidate["end"]))
        candidate["budget_selected"] = bool(candidate.get("selected")) and key in budgeted_key
    selected_candidates = sorted(budgeted, key=lambda c: float(c.get("start", 0.0)))
    budget_rescued_count = sum(1 for c in selected_candidates if bool(c.get("budget_rescued")))
    print(
        f"[slide-ocr] segments={len(segments)} selected_candidates={len([c for c in candidates if c.get('selected')])} "
        f"budgeted_candidates={len(selected_candidates)} budget_enabled={budget_enabled} "
        f"diversity_enabled={diversity_enabled} bucket_sec={diversity_bucket_sec:.0f} min_per_bucket={diversity_min_per_bucket} "
        f"rescue_enabled={rescue_enabled} rescued={budget_rescued_count} "
        f"sample_stats={sample_stats}",
        file=sys.stderr,
    )

    backend, engine, engine_errors = init_ocr_engine(str(args.backend).strip().lower(), str(args.lang))
    fallback_backend = "none"
    fallback_engine: Any = None
    use_tesseract_fallback = parse_bool(args.tesseract_fallback, default=True)
    if backend in ("easyocr", "gcv_text_detection") and use_tesseract_fallback:
        fb_backend, fb_engine, fb_errors = init_ocr_engine("tesseract", str(args.lang))
        if fb_backend == "tesseract":
            fallback_backend = fb_backend
            fallback_engine = fb_engine
            print("[slide-ocr] tesseract fallback enabled", file=sys.stderr)
        elif fb_errors:
            errors.extend([f"tesseract fallback unavailable: {err}" for err in fb_errors])
    if backend == "none":
        errors.extend(engine_errors)
        out_obj = {
            "source": "slide-ocr-v2-none",
            "low_video_path": str(args.video_path),
            "hq_video_path": str(hq_video),
            "duration_sec": float(duration_sec),
            "sample_sec": float(args.sample_sec),
            "shot_boundaries_sec": boundaries,
            "segments": segments,
            "candidates": candidates,
            "events": [],
            "summary": {
                "segments_total": len(segments),
                "candidates_selected": len([c for c in candidates if c.get("selected")]),
                "candidates_budgeted": len(selected_candidates),
                "budget_enabled": bool(budget_enabled),
                "budget_max_candidates": int(budget_max_candidates),
                "budget_diversity_enabled": bool(diversity_enabled),
                "budget_diversity_bucket_sec": float(diversity_bucket_sec),
                "budget_diversity_min_per_bucket": int(diversity_min_per_bucket),
                "budget_rescue_enabled": bool(rescue_enabled),
                "budget_rescue_max_extra": int(rescue_max_extra),
                "budget_rescue_per_anchor": int(rescue_per_anchor),
                "budget_rescue_window_sec": float(rescue_window_sec),
                "budget_rescue_min_score": float(rescue_min_score),
                "budget_rescue_min_text_score": float(rescue_min_text_score),
                "budget_rescued_candidates": int(budget_rescued_count),
                "ocr_attempts": 0,
                "accepted_events": 0,
                "elapsed_sec": round(float(time.monotonic() - started_at), 3),
            },
            "errors": errors,
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        print(json.dumps({"ok": True, "source": out_obj["source"], "events": 0}))
        return

    cap_low = cv2.VideoCapture(str(args.video_path))
    if not cap_low.isOpened():
        raise RuntimeError(f"Cannot open low video for OCR: {args.video_path}")

    cap_hq = cap_low
    if str(hq_video) != str(args.video_path):
        cap_hq = cv2.VideoCapture(hq_video)
        if not cap_hq.isOpened():
            print(
                f"[slide-ocr] warn: cannot open HQ video for OCR path={hq_video}; falling back to low source",
                file=sys.stderr,
            )
            cap_hq = cap_low

    use_mask = parse_bool(args.mask_lower_third, default=True)
    events_raw: list[dict[str, Any]] = []
    ocr_attempts = 0
    ocr_success = 0
    ocr_attempts_primary = 0
    ocr_attempts_fallback = 0
    fallback_hits = 0
    hard_text_gate = parse_bool(args.hard_text_gate, default=True)
    hard_rejects = 0
    tries_per_candidate = max(1, int(args.ocr_budget_tries_per_candidate))
    primary_uses_hq = backend == "gcv_text_detection"
    fallback_uses_hq = fallback_backend == "gcv_text_detection"
    local_ocr_video_path = str(args.video_path)
    cloud_ocr_video_path = str(hq_video if cap_hq is not cap_low else args.video_path)
    print(
        f"[slide-ocr] ocr-source primary={backend}:{'hq' if primary_uses_hq else 'low'} "
        f"fallback={fallback_backend}:{'hq' if fallback_uses_hq else 'low'}",
        file=sys.stderr,
    )
    for idx, cand in enumerate(selected_candidates, start=1):
        s = float(cand["start"])
        e = float(cand["end"])
        mid = 0.5 * (s + e)
        tries = [mid, mid - 0.5, mid + 0.5, mid - 1.0, mid + 1.0]
        tries = [clamp(float(t), s, e) for t in tries]
        # de-duplicate keep order
        uniq_tries: list[float] = []
        seen = set()
        for t in tries:
            k = int(round(t * 100))
            if k in seen:
                continue
            seen.add(k)
            uniq_tries.append(t)
        uniq_tries = uniq_tries[:tries_per_candidate]

        best: dict[str, Any] | None = None
        for t in uniq_tries:
            roi_cache: dict[str, dict[str, Any] | None] = {"low": None, "hq": None}

            def load_source_payload(use_hq: bool) -> dict[str, Any] | None:
                key = "hq" if use_hq else "low"
                cached = roi_cache.get(key)
                if cached is not None:
                    return cached
                cap = cap_hq if use_hq else cap_low
                frame = read_frame_at_sec(cap, float(t))
                if frame is None:
                    roi_cache[key] = None
                    return None
                roi = crop_norm(frame, float(args.roi_y1), float(args.roi_y2), float(args.roi_x1), float(args.roi_x2))
                if roi is None or roi.size == 0:
                    roi_cache[key] = None
                    return None
                if use_mask:
                    roi = mask_lower_band(roi, float(args.mask_lower_third_start_y))
                payload = {"frame": frame, "roi": roi, "preproc": preprocess_roi(roi)}
                roi_cache[key] = payload
                return payload

            primary_payload = load_source_payload(primary_uses_hq)
            if primary_payload is None:
                continue
            frame = primary_payload["frame"]
            roi = primary_payload["roi"]
            preproc = primary_payload["preproc"]
            ocr_attempts += 1

            attempts: list[tuple[str, Any]]
            if backend in ("easyocr", "gcv_text_detection"):
                attempts = [("raw", roi), ("preproc", preproc)]
            else:
                attempts = [("preproc", preproc), ("raw", roi)]

            local_best = {"text": "", "conf": 0.0, "kind": "", "score": -1.0, "meta": {}}
            for kind, img in attempts:
                text, conf, meta = run_ocr(
                    backend,
                    engine,
                    img,
                    str(args.lang),
                    psm=6,
                    upscale=1.2,
                    return_meta=True,
                )
                ocr_attempts_primary += 1
                text = clean_text(text)
                conf = float(max(0.0, min(1.0, conf)))
                if not keep_text(text, conf, min_len=int(args.min_text_len), min_conf=float(args.min_text_confidence)):
                    print(
                        f"[slide-ocr] reject-keep_text cand={idx} t={t:.1f} conf={conf:.2f} len={len(text)} text={text[:60]!r}",
                        file=sys.stderr,
                    )
                    continue
                if hard_text_gate and not keep_text_hard_quality(
                    text,
                    min_alnum_ratio=clamp(float(args.hard_text_min_alnum_ratio), 0.0, 1.0),
                    max_symbol_noise_ratio=clamp(float(args.hard_text_max_symbol_noise_ratio), 0.0, 1.0),
                    min_words=max(1, int(args.hard_text_min_words)),
                    min_alpha_chars=max(1, int(args.hard_text_min_alpha_chars)),
                ):
                    hard_rejects += 1
                    print(
                        f"[slide-ocr] reject-hard_text cand={idx} t={t:.1f} alnum={_alnum_ratio(text):.2f} "
                        f"noise={_symbol_noise_ratio(text):.2f} words={_word_shape_counts(text)} text={text[:60]!r}",
                        file=sys.stderr,
                    )
                    continue
                quality = conf + min(1.0, len(text) / 200.0) * 0.2
                if quality > float(local_best["score"]):
                    local_best = {
                        "text": text,
                        "conf": conf,
                        "kind": f"{backend}:{kind}",
                        "score": quality,
                        "meta": meta if isinstance(meta, dict) else {},
                    }

            if not local_best["text"] and fallback_backend == "tesseract" and fallback_engine is not None:
                fallback_payload = load_source_payload(fallback_uses_hq)
                if fallback_payload is None:
                    continue
                fallback_roi = fallback_payload["roi"]
                fallback_preproc = fallback_payload["preproc"]
                fallback_attempts: list[tuple[str, Any]] = [("preproc", fallback_preproc), ("raw", fallback_roi)]
                for kind, img in fallback_attempts:
                    text, conf, meta = run_ocr(
                        fallback_backend,
                        fallback_engine,
                        img,
                        str(args.lang),
                        psm=6,
                        upscale=1.2,
                        return_meta=True,
                    )
                    ocr_attempts_fallback += 1
                    text = clean_text(text)
                    conf = float(max(0.0, min(1.0, conf)))
                    if not keep_text(text, conf, min_len=int(args.min_text_len), min_conf=float(args.min_text_confidence)):
                        print(
                            f"[slide-ocr] reject-keep_text(fb) cand={idx} t={t:.1f} conf={conf:.2f} len={len(text)} text={text[:60]!r}",
                            file=sys.stderr,
                        )
                        continue
                    if hard_text_gate and not keep_text_hard_quality(
                        text,
                        min_alnum_ratio=clamp(float(args.hard_text_min_alnum_ratio), 0.0, 1.0),
                        max_symbol_noise_ratio=clamp(float(args.hard_text_max_symbol_noise_ratio), 0.0, 1.0),
                        min_words=max(1, int(args.hard_text_min_words)),
                        min_alpha_chars=max(1, int(args.hard_text_min_alpha_chars)),
                    ):
                        hard_rejects += 1
                        print(
                            f"[slide-ocr] reject-hard_text(fb) cand={idx} t={t:.1f} alnum={_alnum_ratio(text):.2f} "
                            f"noise={_symbol_noise_ratio(text):.2f} words={_word_shape_counts(text)} text={text[:60]!r}",
                            file=sys.stderr,
                        )
                        continue
                    quality = conf + min(1.0, len(text) / 200.0) * 0.2
                    if quality > float(local_best["score"]):
                        local_best = {
                            "text": text,
                            "conf": conf,
                            "kind": f"{fallback_backend}:{kind}",
                            "score": quality,
                            "meta": meta if isinstance(meta, dict) else {},
                        }
                if local_best["text"]:
                    fallback_hits += 1

            if not local_best["text"]:
                continue
            item = {
                "start": s,
                "end": e,
                "text": str(local_best["text"]),
                "type": classify_text(str(local_best["text"]), "slide"),
                "confidence": float(local_best["conf"]),
                "sample_time_sec": float(t),
                "ocr_input": str(local_best["kind"]),
                "ocr_meta": dict(local_best.get("meta") or {}),
                "hash": int(avg_hash(roi)),
                "frame_phash": int(phash64(frame)),
                "evidence": {
                    "candidate_score": float(cand["score"]),
                    "avg_text_score": float(cand["avg_text_score"]),
                    "avg_motion_score": float(cand["avg_motion_score"]),
                    "stillness_score": float(cand["stillness_score"]),
                },
            }
            if best is None or float(item["confidence"]) >= float(best["confidence"]):
                best = item

        if best is not None:
            events_raw.append(best)
            ocr_success += 1

        print(
            f"[slide-ocr] ocr-progress={idx}/{len(selected_candidates)} "
            f"accepted={ocr_success} attempts={ocr_attempts} fb_hits={fallback_hits}",
            file=sys.stderr,
        )

    cap_low.release()
    if cap_hq is not cap_low:
        cap_hq.release()
    merged = merge_slide_events(
        events_raw,
        merge_gap_sec=float(args.merge_gap_sec),
        text_sim_threshold=clamp(float(args.merge_text_sim_threshold), 0.5, 0.99),
        hash_dist_threshold=max(0, int(args.merge_hash_dist_threshold)),
    )
    strict_before = len(merged)
    strict_enabled = parse_bool(args.fullscreen_strict, default=False)
    strict_rescue_enabled = parse_bool(args.fullscreen_strict_rescue, default=True)
    strict_rescued = 0
    if strict_enabled:
        strict_out: list[dict[str, Any]] = []
        allowed_types = _parse_csv_set(args.fullscreen_strict_rescue_types)
        for ev in merged:
            if keep_fullscreen_strict(
                ev,
                min_duration_sec=float(args.fullscreen_strict_min_duration_sec),
                min_confidence=float(args.fullscreen_strict_min_confidence),
                min_chars=max(1, int(args.fullscreen_strict_min_chars)),
                min_words=max(1, int(args.fullscreen_strict_min_words)),
                min_long_words=max(1, int(args.fullscreen_strict_min_long_words)),
                min_alnum_ratio=clamp(float(args.fullscreen_strict_min_alnum_ratio), 0.0, 1.0),
                max_symbol_noise_ratio=clamp(float(args.fullscreen_strict_max_symbol_noise_ratio), 0.0, 1.0),
            ):
                strict_out.append(ev)
                continue
            if strict_rescue_enabled and keep_event_by_strict_rescue(
                ev,
                min_confidence=float(args.fullscreen_strict_rescue_min_confidence),
                min_chars=max(1, int(args.fullscreen_strict_rescue_min_chars)),
                allowed_types=allowed_types,
                keyword_regex=str(args.fullscreen_strict_rescue_keywords),
            ):
                strict_out.append(ev)
                strict_rescued += 1
                continue
            # Log why this event was dropped by strict filter
            ev_dur = float(ev.get("end", 0.0)) - float(ev.get("start", 0.0))
            ev_text = clean_text(str(ev.get("text", "")))
            print(
                f"[slide-ocr] reject-strict slide={ev.get('slide_id')} type={ev.get('type')} "
                f"dur={ev_dur:.1f}s conf={float(ev.get('confidence', 0.0)):.2f} "
                f"text={ev_text[:60]!r}",
                file=sys.stderr,
            )
        merged = strict_out
        print(
            f"[slide-ocr] fullscreen-strict kept={len(merged)}/{strict_before} rescued={strict_rescued}",
            file=sys.stderr,
        )

    presentation = annotate_global_duplicates_for_extraction(
        merged,
        enabled=parse_bool(args.global_dup_enabled, default=True),
        require_same_type=parse_bool(args.global_dup_require_same_type, default=True),
        text_sim_threshold=float(args.global_dup_text_sim_threshold),
        phash_near_threshold=max(0, int(args.global_dup_phash_near_threshold)),
        phash_strict_threshold=max(0, int(args.global_dup_phash_strict_threshold)),
    )
    print(
        f"[slide-ocr] global-dup groups={presentation.get('groups_total', 0)} "
        f"duplicates={presentation.get('duplicates_total', 0)} "
        f"extract={presentation.get('representatives_total', 0)}",
        file=sys.stderr,
    )

    extraction_enabled = parse_bool(args.extract_frames, default=True)
    extraction_mode = str(args.extract_mode or "unique").strip().lower()
    if extraction_mode not in {"unique", "all"}:
        extraction_mode = "unique"
    extraction: dict[str, Any] = {
        "enabled": bool(extraction_enabled),
        "mode": extraction_mode,
        "output_dir": None,
        "selected_events": 0,
        "written_frames": 0,
        "failed_events": 0,
        "skipped_events": 0,
        "files": [],
        "errors": [],
        "index_md": None,
    }
    if extraction_enabled:
        extract_dir_raw = str(args.extract_dir).strip() or "slide.frames.extracted"
        extract_dir = (
            extract_dir_raw
            if os.path.isabs(extract_dir_raw)
            else os.path.join(os.path.dirname(args.out), extract_dir_raw)
        )
        extraction = extract_frames_for_events(
            hq_video=str(hq_video),
            events=merged,
            out_dir=extract_dir,
            mode=extraction_mode,
            jpeg_quality=int(args.extract_jpeg_quality),
            overwrite=parse_bool(args.extract_overwrite, default=True),
        )
        index_md_path = str(args.extract_index_md).strip()
        if not index_md_path:
            index_md_path = os.path.join(os.path.dirname(args.out), "slide.frames.extracted.index.md")
        elif not os.path.isabs(index_md_path):
            index_md_path = os.path.join(os.path.dirname(args.out), index_md_path)
        try:
            write_extraction_index_md(index_md_path, extraction)
            extraction["index_md"] = index_md_path
        except Exception as exc:
            extraction.setdefault("errors", []).append(f"index_md_failed: {exc}")
        print(
            f"[slide-ocr] extraction mode={extraction_mode} selected={extraction.get('selected_events', 0)} "
            f"written={extraction.get('written_frames', 0)} failed={extraction.get('failed_events', 0)}",
            file=sys.stderr,
        )

    cloud_enrichment = enrich_events_with_cloud_text(
        events=merged,
        extraction=extraction,
        lang_hint=str(args.lang),
        enabled=parse_bool(args.cloud_text_enrich, default=False),
        min_confidence=float(args.cloud_text_min_confidence),
        min_chars=max(1, int(args.cloud_text_min_chars)),
        min_quality=clamp(float(args.cloud_text_min_quality), 0.0, 1.0),
        max_images=max(0, int(args.cloud_text_max_images)),
        min_apply_gain=max(0.0, float(args.cloud_text_min_apply_gain)),
        propagate_duplicates=parse_bool(args.cloud_text_propagate_duplicates, default=True),
    )
    if cloud_enrichment.get("enabled"):
        print(
            f"[slide-ocr] cloud-enrich attempted={cloud_enrichment.get('attempted', 0)} "
            f"accepted={cloud_enrichment.get('accepted', 0)} applied={cloud_enrichment.get('applied', 0)} "
            f"propagated={cloud_enrichment.get('propagated', 0)} elapsed={cloud_enrichment.get('elapsed_sec', 0.0)}s",
            file=sys.stderr,
        )

    # Keep payload clean while preserving extraction annotations.
    for ev in merged:
        ev.pop("hash", None)
        ev.pop("frame_phash", None)

    elapsed_sec = float(time.monotonic() - started_at)
    out_obj = {
        "source": "slide-ocr-v2",
        "low_video_path": str(args.video_path),
        "hq_video_path": str(hq_video),
        "duration_sec": float(duration_sec),
        "sample_sec": float(args.sample_sec),
        "ocr_local_video_path": local_ocr_video_path,
        "ocr_cloud_video_path": cloud_ocr_video_path,
        "ocr_primary_source": "hq" if primary_uses_hq else "low",
        "ocr_fallback_source": "hq" if fallback_uses_hq else "low",
        "video_width": int(frame_w),
        "video_height": int(frame_h),
        "fps": float(sampled_fps),
        "shot_boundaries_sec": [round(float(v), 3) for v in boundaries],
        "segments": segments,
        "candidates": candidates,
        "events": merged,
        "presentation": presentation,
        "extraction": extraction,
        "cloud_text_enrichment": cloud_enrichment,
        "summary": {
            "segments_total": len(segments),
            "candidates_selected": len([c for c in candidates if c.get("selected")]),
            "candidates_budgeted": len(selected_candidates),
            "budget_enabled": bool(budget_enabled),
            "budget_max_candidates": int(budget_max_candidates),
            "budget_diversity_enabled": bool(diversity_enabled),
            "budget_diversity_bucket_sec": float(diversity_bucket_sec),
            "budget_diversity_min_per_bucket": int(diversity_min_per_bucket),
            "budget_rescue_enabled": bool(rescue_enabled),
            "budget_rescue_max_extra": int(rescue_max_extra),
            "budget_rescue_per_anchor": int(rescue_per_anchor),
            "budget_rescue_window_sec": float(rescue_window_sec),
            "budget_rescue_min_score": float(rescue_min_score),
            "budget_rescue_min_text_score": float(rescue_min_text_score),
            "budget_rescued_candidates": int(budget_rescued_count),
            "tries_per_candidate": int(tries_per_candidate),
            "ocr_attempts": int(ocr_attempts),
            "ocr_attempts_primary": int(ocr_attempts_primary),
            "ocr_attempts_fallback": int(ocr_attempts_fallback),
            "fallback_hits": int(fallback_hits),
            "hard_text_gate_enabled": bool(hard_text_gate),
            "hard_text_rejects": int(hard_rejects),
            "accepted_events_raw": int(len(events_raw)),
            "accepted_events_before_strict": int(strict_before),
            "accepted_events": int(len(merged)),
            "extract_unique_events": int(presentation.get("representatives_total", len(merged))),
            "extract_duplicates": int(presentation.get("duplicates_total", 0)),
            "extract_frames_enabled": bool(extraction.get("enabled", False)),
            "extract_frames_mode": str(extraction.get("mode", "")),
            "extract_frames_written": int(extraction.get("written_frames", 0)),
            "extract_frames_failed": int(extraction.get("failed_events", 0)),
            "cloud_text_enrich_enabled": bool(cloud_enrichment.get("enabled", False)),
            "cloud_text_attempted": int(cloud_enrichment.get("attempted", 0)),
            "cloud_text_accepted": int(cloud_enrichment.get("accepted", 0)),
            "cloud_text_applied": int(cloud_enrichment.get("applied", 0)),
            "cloud_text_propagated": int(cloud_enrichment.get("propagated", 0)),
            "fullscreen_strict_enabled": bool(strict_enabled),
            "fullscreen_strict_rescue_enabled": bool(strict_rescue_enabled),
            "fullscreen_strict_rescued": int(strict_rescued),
            "elapsed_sec": round(elapsed_sec, 3),
        },
        "stats": sample_stats,
        "errors": [*errors, *engine_errors],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    summary_md = str(args.summary_md).strip()
    if not summary_md:
        summary_md = os.path.join(os.path.dirname(args.out), "slide.ocr.positive-findings.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(positive_findings_markdown(merged))

    print(
        f"[slide-ocr] wrote events={len(merged)} candidates={len(selected_candidates)} "
        f"ocr_attempts={ocr_attempts} extracted={int(extraction.get('written_frames', 0))} "
        f"elapsed={elapsed_sec:.1f}s -> {args.out}",
        file=sys.stderr,
    )
    print(json.dumps({"ok": True, "source": out_obj["source"], "events": len(merged), "candidates": len(selected_candidates)}))


if __name__ == "__main__":
    main()
