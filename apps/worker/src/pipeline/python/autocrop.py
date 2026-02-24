import sys
import json
import cv2
import os
import math


def _cosine(a, b):
    if a is None or b is None:
        return -1.0
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(x) * float(x) for x in b))
    if na == 0 or nb == 0:
        return -1.0
    return sum(float(x) * float(y) for x, y in zip(a, b)) / (na * nb)

def _largest_face_center_x(face_cascade, frame, target_detection_height=360, prev_center=None, target_center=0.5):
    height = frame.shape[0]
    if height <= 0:
        return None
    scale = target_detection_height / height
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    small_width = small_frame.shape[1]
    max_area = max([f[2] * f[3] for f in faces]) if len(faces) > 0 else 1.0
    best_cx = None
    best_score = -1e9
    for (xf, yf, wf, hf) in faces:
        cx = (xf + wf / 2) / small_width
        area = wf * hf
        area_score = (area / max_area) if max_area > 0 else 0.0
        prev_score = 1.0 - abs(cx - prev_center) if prev_center is not None else 0.0
        center_score = 1.0 - abs(cx - target_center)
        # weighted multi-objective selection:
        # - keep largest face preference
        # - strongly prefer temporal continuity when available
        # - slight preference for global target center
        if prev_center is not None:
            score = 0.35 * area_score + 0.55 * prev_score + 0.10 * center_score
        else:
            score = 0.65 * area_score + 0.35 * center_score
        if score > best_score:
            best_score = score
            best_cx = cx
    return best_cx

def detect_face_opencv(video_path, limit_seconds=0):
    print(f"Analyzing {video_path} for crop...", file=sys.stderr)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Analyze 1 frame per 2 seconds
    step = int(fps * 2) 
    
    limit_frames = int(limit_seconds * fps) if limit_seconds > 0 else total_frames
    
    centers_x = []
    
    frame_idx = 0
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success:
            break

        center_x = _largest_face_center_x(face_cascade, frame)
        if center_x is not None:
            centers_x.append(center_x)
        
        frame_idx += step
        if frame_idx >= total_frames or frame_idx >= limit_frames:
            break
            
    cap.release()
    
    if not centers_x:
        return 0.5
        
    centers_x.sort()
    # Use median
    return centers_x[len(centers_x) // 2]


def _interp_center(points, t):
    if not points:
        return 0.5
    if t <= points[0]["t"]:
        return float(points[0]["center_x"])
    if t >= points[-1]["t"]:
        return float(points[-1]["center_x"])
    lo = 0
    hi = len(points) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if points[mid]["t"] <= t:
            lo = mid
        else:
            hi = mid
    a = points[lo]
    b = points[hi]
    dt = max(1e-6, float(b["t"] - a["t"]))
    u = (t - float(a["t"])) / dt
    return float(a["center_x"] + (b["center_x"] - a["center_x"]) * u)


def _apply_motion_filter(
    points,
    end_rel,
    motion_dt,
    max_speed_per_sec,
    max_accel_per_sec2,
    follow_kp=7.0,
    follow_kd=4.0,
    lookahead_sec=1.2
):
    if not points:
        return []
    dt = max(0.04, float(motion_dt))
    max_speed = max(0.002, float(max_speed_per_sec))
    max_accel = max(0.002, float(max_accel_per_sec2))
    t = 0.0
    x = float(points[0]["center_x"])
    v = 0.0
    out = [{"t": 0.0, "center_x": x}]
    kp = max(0.1, float(follow_kp))
    kd = max(0.0, float(follow_kd))
    la = max(0.0, float(lookahead_sec))
    while t < end_rel:
        t_next = min(end_rel, t + dt)
        target_t = min(end_rel, t_next + la)
        target = _interp_center(points, target_t)
        err = target - x
        # Critically-damped style follower: smooth camera-head motion without snap.
        accel = kp * err - kd * v
        if accel > max_accel:
            accel = max_accel
        elif accel < -max_accel:
            accel = -max_accel
        v = v + accel * dt
        if v > max_speed:
            v = max_speed
        elif v < -max_speed:
            v = -max_speed
        x = x + v * dt
        x = max(0.0, min(1.0, x))
        out.append({"t": float(t_next), "center_x": float(x)})
        t = t_next
    return out


def _build_camera_keyframes(
    points,
    end_rel,
    keyframe_sec=1.5,
    min_move=0.012,
    max_hold_sec=4.0
):
    if not points:
        return []
    end_rel = max(0.0, float(end_rel))
    spacing = max(0.2, float(keyframe_sec))
    min_move = max(0.001, float(min_move))
    max_hold = max(spacing, float(max_hold_sec))

    start_x = _interp_center(points, 0.0)
    out = [{"t": 0.0, "center_x": float(start_x)}]
    last_t = 0.0
    last_x = float(start_x)

    t = spacing
    while t < end_rel:
        x = _interp_center(points, t)
        if abs(x - last_x) >= min_move or (t - last_t) >= max_hold:
            out.append({"t": float(t), "center_x": float(x)})
            last_t = float(t)
            last_x = float(x)
        t += spacing

    end_x = _interp_center(points, end_rel)
    if abs(end_x - last_x) >= (min_move * 0.5) or (end_rel - last_t) >= (max_hold * 0.5):
        out.append({"t": float(end_rel), "center_x": float(end_x)})
    elif out[-1]["t"] < end_rel:
        out.append({"t": float(end_rel), "center_x": float(out[-1]["center_x"])})

    # Ensure monotonic unique times
    dedup = []
    prev_t = -1.0
    for p in out:
        tt = float(p["t"])
        if tt <= prev_t:
            continue
        dedup.append({"t": tt, "center_x": float(p["center_x"])})
        prev_t = tt
    return dedup

def detect_face_track(
    video_path,
    start_sec=0.0,
    duration_sec=0.0,
    window_sec=1.0,
    smooth_alpha=0.35,
    target_center=0.5,
    max_delta_per_sec=0.12,
    face_pass_json=None,
    identity_min_sim=0.35,
    identity_det_size=256,
    deadband=0.015,
    hold_windows=2,
    motion_dt=0.10,
    max_accel_per_sec2=0.06,
    follow_kp=7.0,
    follow_kd=4.0,
    composition_face_weight=0.75,
    lookahead_sec=1.2,
    keyframe_sec=1.5,
    keyframe_min_move=0.012,
    keyframe_max_hold_sec=4.0
):
    print(
        f"Tracking face center on {video_path} from {start_sec:.2f}s for {duration_sec:.2f}s (window={window_sec:.2f}s)...",
        file=sys.stderr
    )
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise IOError("Invalid FPS detected")

    full_duration = total_frames / fps if total_frames > 0 else 0
    start_sec = max(0.0, float(start_sec))
    if duration_sec <= 0:
        duration_sec = max(0.0, full_duration - start_sec)
    end_sec = min(full_duration, start_sec + float(duration_sec)) if full_duration > 0 else start_sec + float(duration_sec)
    if end_sec <= start_sec:
        cap.release()
        return []

    step = max(1, int(fps * max(0.2, float(window_sec))))
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    if end_frame <= start_frame:
        cap.release()
        return []

    raw = []
    last_center = float(target_center)
    has_last = False
    pending_dir = 0
    pending_count = 0
    identity_emb = None
    identity_anchor_times = []
    insightface_app = None
    np = None
    if face_pass_json and os.path.exists(face_pass_json):
        try:
            with open(face_pass_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
            maybe_centroid = (
                payload.get("selected_identity", {}).get("centroid")
                if isinstance(payload, dict)
                else None
            )
            if isinstance(maybe_centroid, list) and len(maybe_centroid) >= 64:
                identity_emb = [float(v) for v in maybe_centroid]
                n = math.sqrt(sum(v * v for v in identity_emb))
                if n > 0:
                    identity_emb = [v / n for v in identity_emb]
            elif isinstance(payload, dict):
                seg = payload.get("selected_identity", {}).get("main_segment")
                if isinstance(seg, dict):
                    seg_start = float(seg.get("start", start_sec))
                    seg_end = float(seg.get("end", seg_start + 1.0))
                    mid = (seg_start + seg_end) / 2.0
                    identity_anchor_times = [
                        max(0.0, seg_start + 1.0),
                        max(0.0, seg_start + 5.0),
                        max(0.0, mid)
                    ]
        except Exception:
            identity_emb = None
    if identity_emb is not None:
        try:
            import numpy as np  # type: ignore
            from insightface.app import FaceAnalysis  # type: ignore
            insightface_app = FaceAnalysis(name=os.getenv("FACE_MODEL_NAME", "buffalo_l"), providers=["CPUExecutionProvider"])
            ds = max(128, int(identity_det_size))
            insightface_app.prepare(ctx_id=0, det_size=(ds, ds))
            print(
                f"Identity-anchored crop enabled (sim>={float(identity_min_sim):.2f}, det_size={ds}) from {os.path.basename(face_pass_json)}",
                file=sys.stderr
            )
        except Exception as e:
            print(f"Identity-anchored crop unavailable ({e}); using OpenCV fallback.", file=sys.stderr)
            identity_emb = None
            insightface_app = None
    elif identity_anchor_times:
        try:
            import numpy as np  # type: ignore
            from insightface.app import FaceAnalysis  # type: ignore
            insightface_app = FaceAnalysis(name=os.getenv("FACE_MODEL_NAME", "buffalo_l"), providers=["CPUExecutionProvider"])
            ds = max(128, int(identity_det_size))
            insightface_app.prepare(ctx_id=0, det_size=(ds, ds))
            full_duration = total_frames / fps if total_frames > 0 else 0.0
            best_emb = None
            best_area = 0.0
            for t_abs in identity_anchor_times:
                if full_duration > 0:
                    t_abs = max(0.0, min(full_duration - 0.01, t_abs))
                cap.set(cv2.CAP_PROP_POS_MSEC, float(t_abs) * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                faces = insightface_app.get(frame) or []
                if not faces:
                    continue
                h, w = frame.shape[:2]
                frame_area = float(max(1, w * h))
                for face in faces:
                    emb = getattr(face, "embedding", None)
                    bbox = getattr(face, "bbox", None)
                    if emb is None or bbox is None:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                    area_ratio = (max(1.0, x2 - x1) * max(1.0, y2 - y1)) / frame_area
                    if area_ratio > best_area:
                        v = [float(x) for x in emb]
                        n = math.sqrt(sum(x * x for x in v))
                        if n > 0:
                            best_area = area_ratio
                            best_emb = [x / n for x in v]
            if best_emb is not None:
                identity_emb = best_emb
                print(
                    f"Identity-anchored crop enabled via anchors from {os.path.basename(face_pass_json)} (sim>={float(identity_min_sim):.2f})",
                    file=sys.stderr
                )
            else:
                insightface_app = None
        except Exception as e:
            print(f"Identity anchor extraction unavailable ({e}); using OpenCV fallback.", file=sys.stderr)
            insightface_app = None
            identity_emb = None

    frame_idx = start_frame
    while frame_idx <= end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        cx = None
        if insightface_app is not None and identity_emb is not None and np is not None:
            try:
                faces = insightface_app.get(frame) or []
                best = None
                best_score = -1e9
                h, w = frame.shape[:2]
                frame_area = float(max(1, w * h))
                for face in faces:
                    emb = getattr(face, "embedding", None)
                    bbox = getattr(face, "bbox", None)
                    if emb is None or bbox is None:
                        continue
                    sim = _cosine(identity_emb, emb)
                    if sim < float(identity_min_sim):
                        continue
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                    bw = max(1.0, x2 - x1)
                    bh = max(1.0, y2 - y1)
                    area_ratio = (bw * bh) / frame_area
                    cxx = (x1 + x2) / 2.0 / max(1.0, w)
                    prev_score = 1.0 - abs(cxx - last_center) if has_last else 0.0
                    center_score = 1.0 - abs(cxx - float(target_center))
                    score = 0.70 * sim + 0.15 * prev_score + 0.10 * center_score + 0.05 * min(1.0, area_ratio * 20.0)
                    if score > best_score:
                        best_score = score
                        best = cxx
                if best is not None:
                    cx = float(best)
            except Exception:
                cx = None
        if cx is None:
            cx = _largest_face_center_x(
                face_cascade,
                frame,
                prev_center=(last_center if has_last else None),
                target_center=float(target_center)
            )
        if cx is None:
            cx = last_center if has_last else 0.5
        # Composition blend: avoid hard face-centering, preserve more body/stage context.
        fw = max(0.0, min(1.0, float(composition_face_weight)))
        cx = fw * float(cx) + (1.0 - fw) * float(target_center)
        if has_last:
            dt_sec = max(0.2, float(window_sec))
            max_step = max(0.01, float(max_delta_per_sec) * dt_sec)
            if abs(cx - last_center) > max_step:
                # clamp abrupt jumps that usually mean wrong person (e.g., band/background face)
                cx = last_center + max_step if cx > last_center else last_center - max_step
            # deadband: ignore tiny changes that cause visual jitter
            delta = cx - last_center
            if abs(delta) < float(deadband):
                cx = last_center
                pending_dir = 0
                pending_count = 0
            else:
                # hold: require sustained movement direction for a few windows
                direction = 1 if delta > 0 else -1
                if pending_dir != direction:
                    pending_dir = direction
                    pending_count = 1
                    cx = last_center
                else:
                    pending_count += 1
                    if pending_count < int(max(1, hold_windows)):
                        cx = last_center
                    else:
                        pending_count = int(max(1, hold_windows))
        last_center = cx
        has_last = True
        t_rel = max(0.0, (frame_idx / fps) - start_sec)
        raw.append({"t": t_rel, "center_x": float(cx)})
        frame_idx += step

    cap.release()
    if not raw:
        return []

    alpha = max(0.05, min(0.95, float(smooth_alpha)))
    smoothed = []
    prev = raw[0]["center_x"]
    for p in raw:
        v = p["center_x"] * alpha + prev * (1.0 - alpha)
        prev = v
        smoothed.append({"t": float(p["t"]), "center_x": float(v)})

    if smoothed[0]["t"] > 0:
        smoothed.insert(0, {"t": 0.0, "center_x": float(smoothed[0]["center_x"])})

    end_rel = max(0.0, end_sec - start_sec)
    if smoothed[-1]["t"] < end_rel:
        smoothed.append({"t": float(end_rel), "center_x": float(smoothed[-1]["center_x"])})
    filtered = _apply_motion_filter(
        smoothed,
        end_rel=end_rel,
        motion_dt=float(motion_dt),
        max_speed_per_sec=float(max_delta_per_sec),
        max_accel_per_sec2=float(max_accel_per_sec2),
        follow_kp=float(follow_kp),
        follow_kd=float(follow_kd),
        lookahead_sec=float(lookahead_sec)
    )
    final_track = filtered if filtered else smoothed
    camera_keyframes = _build_camera_keyframes(
        final_track,
        end_rel=end_rel,
        keyframe_sec=float(keyframe_sec),
        min_move=float(keyframe_min_move),
        max_hold_sec=float(keyframe_max_hold_sec)
    )
    return final_track, camera_keyframes

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_file")
    parser.add_argument("--limit_seconds", type=int, default=0)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--start_sec", type=float, default=0.0)
    parser.add_argument("--duration_sec", type=float, default=0.0)
    parser.add_argument("--window_sec", type=float, default=1.0)
    parser.add_argument("--smooth_alpha", type=float, default=0.35)
    parser.add_argument("--target_center", type=float, default=0.5)
    parser.add_argument("--max_delta_per_sec", type=float, default=0.12)
    parser.add_argument("--face_pass_json", type=str, default=None)
    parser.add_argument("--identity_min_sim", type=float, default=0.35)
    parser.add_argument("--identity_det_size", type=int, default=256)
    parser.add_argument("--deadband", type=float, default=0.015)
    parser.add_argument("--hold_windows", type=int, default=2)
    parser.add_argument("--motion_dt", type=float, default=0.10)
    parser.add_argument("--max_accel_per_sec2", type=float, default=0.06)
    parser.add_argument("--follow_kp", type=float, default=7.0)
    parser.add_argument("--follow_kd", type=float, default=4.0)
    parser.add_argument("--composition_face_weight", type=float, default=0.75)
    parser.add_argument("--lookahead_sec", type=float, default=1.2)
    parser.add_argument("--keyframe_sec", type=float, default=1.5)
    parser.add_argument("--keyframe_min_move", type=float, default=0.012)
    parser.add_argument("--keyframe_max_hold_sec", type=float, default=4.0)
    args = parser.parse_args()
    
    video_file = args.video_file
    limit_seconds = args.limit_seconds
    
    try:
        if args.track:
            track, camera_keyframes = detect_face_track(
                video_file,
                start_sec=args.start_sec,
                duration_sec=args.duration_sec,
                window_sec=args.window_sec,
                smooth_alpha=args.smooth_alpha,
                target_center=args.target_center,
                max_delta_per_sec=args.max_delta_per_sec,
                face_pass_json=args.face_pass_json,
                identity_min_sim=args.identity_min_sim,
                identity_det_size=args.identity_det_size,
                deadband=args.deadband,
                hold_windows=args.hold_windows,
                motion_dt=args.motion_dt,
                max_accel_per_sec2=args.max_accel_per_sec2,
                follow_kp=args.follow_kp,
                follow_kd=args.follow_kd,
                composition_face_weight=args.composition_face_weight,
                lookahead_sec=args.lookahead_sec,
                keyframe_sec=args.keyframe_sec,
                keyframe_min_move=args.keyframe_min_move,
                keyframe_max_hold_sec=args.keyframe_max_hold_sec
            )
            if not track:
                print(json.dumps({"center_x": 0.5, "track": []}))
            else:
                centers = sorted([p["center_x"] for p in track])
                median = centers[len(centers) // 2]
                print(json.dumps({"center_x": median, "track": track, "camera_keyframes": camera_keyframes}))
        else:
            center_x = detect_face_opencv(video_file, limit_seconds)
            print(json.dumps({"center_x": center_x}))
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)
