import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any


def _safe_imports() -> tuple[Any, Any]:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        return cv2, np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependencies for face pass. Install with: "
            "pip install insightface opencv-python numpy onnxruntime"
        ) from exc


@dataclass
class Hit:
    t: float
    area_ratio: float
    center_score: float
    det_score: float
    emb: list[float]


@dataclass
class Cluster:
    id: int
    centroid: list[float]
    hits: list[Hit] = field(default_factory=list)

    def update(self, emb, hit: Hit, np):
        n = len(self.hits)
        c = np.asarray(self.centroid, dtype=np.float32)
        e = np.asarray(emb, dtype=np.float32)
        c = (c * n + e) / float(n + 1)
        norm = float(np.linalg.norm(c))
        if norm > 0:
            c = c / norm
        self.centroid = c.tolist()
        self.hits.append(hit)


def cosine(a, b, np) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da == 0 or db == 0:
        return -1.0
    return float(np.dot(a / da, b / db))


def build_segments(times: list[float], step_sec: float, bridge_gap_sec: float) -> list[dict[str, float]]:
    if not times:
        return []
    out: list[dict[str, float]] = []
    s = times[0]
    prev = times[0]
    for t in times[1:]:
        if t - prev <= bridge_gap_sec:
            prev = t
            continue
        out.append({"start": s, "end": prev + step_sec, "duration": (prev + step_sec) - s})
        s = t
        prev = t
    out.append({"start": s, "end": prev + step_sec, "duration": (prev + step_sec) - s})
    return out


def scan_faces(
    video_path: str,
    app,
    sample_step_sec: float,
    similarity_threshold: float,
    det_score_threshold: float,
    min_face_area_ratio: float,
    max_faces_per_frame: int,
):
    cv2, np = _safe_imports()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError("Could not read fps/frame_count from video.")
    duration_sec = frame_count / fps

    clusters: list[Cluster] = []
    next_cluster_id = 0
    processed_samples = 0
    total_samples_est = int(math.ceil(duration_sec / sample_step_sec))
    print(
        f"[face-pass] duration={duration_sec:.1f}s fps={fps:.2f} estimated_samples={total_samples_est} step_sec={sample_step_sec}",
        file=sys.stderr,
    )
    sample_t = 0.0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        t = float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        if t + 1e-3 < sample_t:
            continue

        sample_t += sample_step_sec
        processed_samples += 1
        if processed_samples % 50 == 0:
            print(
                f"[face-pass] processed {processed_samples}/{total_samples_est} samples (~{(processed_samples / max(1,total_samples_est))*100:.1f}%)",
                file=sys.stderr,
            )

        h, w = frame.shape[:2]
        frame_area = float(max(1, w * h))

        faces = app.get(frame)
        if not faces:
            continue

        faces = sorted(
            faces,
            key=lambda f: float(getattr(f, "det_score", 0.0)),
            reverse=True,
        )[:max_faces_per_frame]

        for face in faces:
            det_score = float(getattr(face, "det_score", 0.0))
            if det_score < det_score_threshold:
                continue

            bbox = getattr(face, "bbox", None)
            emb = getattr(face, "embedding", None)
            if bbox is None or emb is None:
                continue

            x1, y1, x2, y2 = [float(v) for v in bbox]
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            area_ratio = (bw * bh) / frame_area
            if area_ratio < min_face_area_ratio:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            nx = (cx / max(1.0, w)) - 0.5
            ny = (cy / max(1.0, h)) - 0.5
            dist = math.sqrt(nx * nx + ny * ny)
            center_score = max(0.0, 1.0 - (dist / 0.75))

            emb_arr = np.asarray(emb, dtype=np.float32)
            norm = float(np.linalg.norm(emb_arr))
            if norm == 0:
                continue
            emb_arr = emb_arr / norm
            emb_list = emb_arr.tolist()

            best_cluster = None
            best_sim = -1.0
            for c in clusters:
                sim = cosine(c.centroid, emb_list, np)
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = c

            hit = Hit(
                t=float(t),
                area_ratio=float(area_ratio),
                center_score=float(center_score),
                det_score=float(det_score),
                emb=emb_list,
            )

            if best_cluster is not None and best_sim >= similarity_threshold:
                best_cluster.update(emb_list, hit, np)
            else:
                c = Cluster(id=next_cluster_id, centroid=emb_list, hits=[hit])
                clusters.append(c)
                next_cluster_id += 1

    cap.release()
    return duration_sec, clusters


def score_clusters(clusters: list[Cluster], duration_sec: float, sample_step_sec: float, bridge_gap_sec: float):
    scored = []
    for c in clusters:
        if not c.hits:
            continue
        times = sorted(h.t for h in c.hits)
        segments = build_segments(times, sample_step_sec, bridge_gap_sec)
        total_presence_sec = float(len(times) * sample_step_sec)
        longest_run_sec = max((s["duration"] for s in segments), default=0.0)
        avg_center = sum(h.center_score for h in c.hits) / len(c.hits)
        avg_area = sum(h.area_ratio for h in c.hits) / len(c.hits)
        mid_hits = sum(1 for h in c.hits if (0.2 * duration_sec) <= h.t <= (0.8 * duration_sec))
        mid_ratio = float(mid_hits / len(c.hits))

        norm_total = min(1.0, total_presence_sec / max(1.0, duration_sec))
        norm_longest = min(1.0, longest_run_sec / max(1.0, duration_sec))
        norm_area = min(1.0, avg_area * 12.0)

        score = (
            0.45 * norm_longest
            + 0.30 * norm_total
            + 0.10 * avg_center
            + 0.10 * norm_area
            + 0.05 * mid_ratio
        )

        scored.append(
            {
                "id": c.id,
                "score": score,
                "total_presence_sec": total_presence_sec,
                "longest_run_sec": longest_run_sec,
                "avg_center_score": avg_center,
                "avg_area_ratio": avg_area,
                "mid_ratio": mid_ratio,
                "segments": sorted(segments, key=lambda x: x["duration"], reverse=True),
                "hit_count": len(c.hits),
                "centroid": c.centroid,
            }
        )

    return sorted(scored, key=lambda x: x["score"], reverse=True)


def refine_bounds_from_dense_scan(
    video_path: str,
    app,
    selected_centroid: list[float],
    coarse_start: float,
    coarse_end: float,
    duration_sec: float,
    coarse_step_sec: float,
    fine_step_sec: float,
    similarity_threshold: float,
    dense_window_sec: float,
):
    cv2, np = _safe_imports()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return coarse_start, coarse_end

    def collect_hits(win_start: float, win_end: float) -> list[float]:
        out: list[float] = []
        t = max(0.0, win_start)
        while t <= min(duration_sec, win_end):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                t += fine_step_sec
                continue
            faces = app.get(frame) or []
            matched = False
            for f in faces:
                emb = getattr(f, "embedding", None)
                if emb is None:
                    continue
                sim = cosine(selected_centroid, emb, np)
                if sim >= similarity_threshold:
                    matched = True
                    break
            if matched:
                out.append(float(t))
            t += fine_step_sec
        return out

    start_hits = collect_hits(coarse_start - dense_window_sec, coarse_start + dense_window_sec)
    end_hits = collect_hits(coarse_end - dense_window_sec, coarse_end + dense_window_sec)
    cap.release()

    refined_start = coarse_start
    refined_end = coarse_end

    if start_hits:
        segments = build_segments(sorted(start_hits), fine_step_sec, max(8.0, coarse_step_sec * 3))
        candidate = min(segments, key=lambda s: abs(s["start"] - coarse_start))
        refined_start = candidate["start"]
    if end_hits:
        segments = build_segments(sorted(end_hits), fine_step_sec, max(8.0, coarse_step_sec * 3))
        candidate = min(segments, key=lambda s: abs(s["end"] - coarse_end))
        refined_end = candidate["end"]

    return max(0.0, refined_start), min(duration_sec, refined_end)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast InsightFace sermon-bound candidate detector.")
    parser.add_argument("video_path", help="Path to source video.")
    parser.add_argument("--out", required=True, help="Path to output JSON.")
    parser.add_argument("--model-name", default=os.getenv("FACE_MODEL_NAME", "buffalo_l"))
    parser.add_argument("--det-size", type=int, default=int(os.getenv("FACE_DET_SIZE", "480")))
    parser.add_argument("--step-sec", type=float, default=float(os.getenv("FACE_SCAN_STEP_SEC", "2.0")))
    parser.add_argument("--fine-step-sec", type=float, default=float(os.getenv("FACE_SCAN_FINE_STEP_SEC", "0.5")))
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=float(os.getenv("FACE_CLUSTER_SIM_THRESHOLD", "0.48")),
    )
    parser.add_argument(
        "--boundary-similarity-threshold",
        type=float,
        default=float(os.getenv("FACE_BOUNDARY_SIM_THRESHOLD", "0.45")),
    )
    parser.add_argument("--bridge-gap-sec", type=float, default=float(os.getenv("FACE_BRIDGE_GAP_SEC", "35")))
    parser.add_argument("--det-score-threshold", type=float, default=float(os.getenv("FACE_DET_SCORE_THRESHOLD", "0.45")))
    parser.add_argument("--min-face-area-ratio", type=float, default=float(os.getenv("FACE_MIN_AREA_RATIO", "0.01")))
    parser.add_argument("--max-faces-per-frame", type=int, default=int(os.getenv("FACE_MAX_FACES_PER_FRAME", "2")))
    parser.add_argument(
        "--dense-window-sec",
        type=float,
        default=float(os.getenv("FACE_DENSE_WINDOW_SEC", "30")),
    )
    parser.add_argument(
        "--disable-dense-refine",
        action="store_true",
        default=str(os.getenv("FACE_DISABLE_DENSE_REFINE", "false")).lower() == "true",
    )
    parser.add_argument(
        "--significant-min-sec",
        type=float,
        default=float(os.getenv("FACE_SIGNIFICANT_SEGMENT_MIN_SEC", "120")),
    )
    parser.add_argument(
        "--significant-min-longest-ratio",
        type=float,
        default=float(os.getenv("FACE_SIGNIFICANT_SEGMENT_MIN_LONGEST_RATIO", "0.12")),
    )
    parser.add_argument(
        "--start-backtrack-max-sec",
        type=float,
        default=float(os.getenv("FACE_START_BACKTRACK_MAX_SEC", "120")),
    )
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")

    cv2, np = _safe_imports()

    try:
        from insightface.app import FaceAnalysis  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "InsightFace is not installed in this Python env. Install with: "
            "pip install insightface onnxruntime"
        ) from exc

    app = FaceAnalysis(name=args.model_name, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    duration_sec, clusters = scan_faces(
        video_path=args.video_path,
        app=app,
        sample_step_sec=args.step_sec,
        similarity_threshold=args.similarity_threshold,
        det_score_threshold=args.det_score_threshold,
        min_face_area_ratio=args.min_face_area_ratio,
        max_faces_per_frame=args.max_faces_per_frame,
    )

    scored = score_clusters(clusters, duration_sec, args.step_sec, args.bridge_gap_sec)
    if not scored:
        raise RuntimeError("No stable face identities detected. Adjust thresholds or check video framing.")

    winner = scored[0]
    winner_segments = winner["segments"]
    main_segment = winner_segments[0]
    longest = float(max(1.0, winner["longest_run_sec"]))
    min_sig = max(float(args.significant_min_sec), longest * float(args.significant_min_longest_ratio))
    significant_segments = [s for s in winner_segments if float(s["duration"]) >= min_sig]
    if not significant_segments:
        significant_segments = [main_segment]

    main_start = float(main_segment["start"])
    start_pool = [
        s for s in significant_segments if float(s["start"]) >= (main_start - float(args.start_backtrack_max_sec))
    ]
    if not start_pool:
        start_pool = significant_segments
    coarse_start = min(float(s["start"]) for s in start_pool)
    coarse_end = max(float(s["end"]) for s in significant_segments)

    if args.disable_dense_refine:
        refined_start, refined_end = coarse_start, coarse_end
    else:
        refined_start, refined_end = refine_bounds_from_dense_scan(
            video_path=args.video_path,
            app=app,
            selected_centroid=winner["centroid"],
            coarse_start=coarse_start,
            coarse_end=coarse_end,
            duration_sec=duration_sec,
            coarse_step_sec=args.step_sec,
            fine_step_sec=args.fine_step_sec,
            similarity_threshold=args.boundary_similarity_threshold,
            dense_window_sec=args.dense_window_sec,
        )

    start_candidates = sorted({float(s["start"]) for s in significant_segments})
    end_candidates = sorted({float(s["end"]) for s in significant_segments})

    out = {
        "model": f"insightface/{args.model_name}",
        "config": {
            "step_sec": args.step_sec,
            "fine_step_sec": args.fine_step_sec,
            "similarity_threshold": args.similarity_threshold,
            "boundary_similarity_threshold": args.boundary_similarity_threshold,
            "bridge_gap_sec": args.bridge_gap_sec,
            "det_score_threshold": args.det_score_threshold,
            "min_face_area_ratio": args.min_face_area_ratio,
            "max_faces_per_frame": args.max_faces_per_frame,
            "det_size": args.det_size,
            "significant_min_sec": args.significant_min_sec,
            "significant_min_longest_ratio": args.significant_min_longest_ratio,
            "start_backtrack_max_sec": args.start_backtrack_max_sec,
            "disable_dense_refine": args.disable_dense_refine,
            "dense_window_sec": args.dense_window_sec,
        },
        "video_duration_sec": duration_sec,
        "start": refined_start,
        "end": refined_end,
        "start_candidates": start_candidates,
        "end_candidates": end_candidates,
        "selected_identity": {
            "id": winner["id"],
            "score": winner["score"],
            "total_presence_sec": winner["total_presence_sec"],
            "longest_run_sec": winner["longest_run_sec"],
            "avg_center_score": winner["avg_center_score"],
            "avg_area_ratio": winner["avg_area_ratio"],
            "mid_ratio": winner["mid_ratio"],
            "centroid": winner["centroid"],
            "main_segment": main_segment,
            "top_segments": winner_segments[:5],
            "significant_segments": significant_segments,
            "coarse_start_from_significant": coarse_start,
            "coarse_end_from_significant": coarse_end,
        },
        "top_identities": [
            {
                "id": c["id"],
                "score": c["score"],
                "total_presence_sec": c["total_presence_sec"],
                "longest_run_sec": c["longest_run_sec"],
                "main_segment": c["segments"][0] if c["segments"] else None,
            }
            for c in scored[:5]
        ],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({"ok": True, "out": args.out, "start": refined_start, "end": refined_end}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(str(exc), file=sys.stderr)
        sys.exit(1)
