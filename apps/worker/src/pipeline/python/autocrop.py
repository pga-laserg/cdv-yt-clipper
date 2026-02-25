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


def _normalize_vec(v):
    if v is None:
        return None
    vv = [float(x) for x in v]
    n = math.sqrt(sum(x * x for x in vv))
    if n <= 0:
        return None
    return [x / n for x in vv]


def _calc_mouth_motion(prev_gray, curr_gray, bbox):
    """
    ClipsAI-inspired speech proxy:
    estimate mouth activity from frame-to-frame pixel motion in lower face region.
    """
    if prev_gray is None or curr_gray is None or bbox is None:
        return 0.0
    h, w = curr_gray.shape[:2]
    x1, y1, x2, y2 = [int(float(v)) for v in bbox]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    bw = x2 - x1
    bh = y2 - y1
    if bw < 12 or bh < 12:
        return 0.0
    # Lower-central facial region where mouth motion dominates.
    mx1 = x1 + int(0.15 * bw)
    mx2 = x1 + int(0.85 * bw)
    my1 = y1 + int(0.55 * bh)
    my2 = y1 + int(0.92 * bh)
    mx1 = max(0, min(w - 1, mx1))
    mx2 = max(0, min(w, mx2))
    my1 = max(0, min(h - 1, my1))
    my2 = max(0, min(h, my2))
    if mx2 - mx1 < 4 or my2 - my1 < 4:
        return 0.0
    prev_patch = prev_gray[my1:my2, mx1:mx2]
    curr_patch = curr_gray[my1:my2, mx1:mx2]
    if prev_patch.shape != curr_patch.shape or prev_patch.size == 0:
        return 0.0
    diff = cv2.absdiff(curr_patch, prev_patch)
    return float(diff.mean() / 255.0)

def _largest_face_center(face_cascade, frame, target_detection_height=360, prev_center=None, target_center=0.5):
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
    small_height = small_frame.shape[0]
    max_area = max([f[2] * f[3] for f in faces]) if len(faces) > 0 else 1.0
    best = None
    best_score = -1e9
    for (xf, yf, wf, hf) in faces:
        cx = (xf + wf / 2) / small_width
        cy = (yf + hf / 2) / max(1, small_height)
        fh = hf / max(1, small_height)
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
            best = (cx, cy, fh)
    return best


def _largest_face_center_x(face_cascade, frame, target_detection_height=360, prev_center=None, target_center=0.5):
    out = _largest_face_center(
        face_cascade,
        frame,
        target_detection_height=target_detection_height,
        prev_center=prev_center,
        target_center=target_center
    )
    if out is None:
        return None
    return out[0]

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


def _interp_field(points, key, t, default_value=0.0):
    if not points:
        return float(default_value)
    if t <= points[0]["t"]:
        return float(points[0].get(key, default_value))
    if t >= points[-1]["t"]:
        return float(points[-1].get(key, default_value))
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
    av = float(a.get(key, default_value))
    bv = float(b.get(key, default_value))
    dt = max(1e-6, float(b["t"] - a["t"]))
    u = (t - float(a["t"])) / dt
    return float(av + (bv - av) * u)


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


def _apply_hold_ramp(
    points,
    threshold=0.03,
    confirm_sec=0.3,
    ramp_sec=0.9,
    sample_dt=0.10
):
    if not points:
        return []
    pts = sorted(points, key=lambda p: p["t"])
    out = []
    current = float(pts[0]["center_x"])
    last_t = float(pts[0]["t"])
    pending_start = None
    pending_target = None
    thr = max(0.001, float(threshold))
    confirm = max(0.05, float(confirm_sec))
    ramp = max(0.05, float(ramp_sec))
    dt = max(0.02, float(sample_dt))

    def smoothstep(u: float) -> float:
        u = max(0.0, min(1.0, u))
        return u * u * (3 - 2 * u)

    i = 0
    while i < len(pts):
        t = float(pts[i]["t"])
        x = float(pts[i]["center_x"])
        delta = x - current
        if abs(delta) <= thr:
            pending_start = None
            pending_target = None
            out.append({"t": t, "center_x": current})
            last_t = t
            i += 1
            continue
        # start or continue confirmation window
        if pending_start is None:
            pending_start = t
            pending_target = x
            out.append({"t": t, "center_x": current})
            last_t = t
            i += 1
            continue
        if t - pending_start < confirm:
            out.append({"t": t, "center_x": current})
            last_t = t
            i += 1
            continue
        # commit to ramp toward pending_target
        target = pending_target if pending_target is not None else x
        ramp_start = t
        ramp_end = t + ramp
        tt = ramp_start
        while tt <= ramp_end + 1e-6:
            u = (tt - ramp_start) / ramp
            y = current + (target - current) * smoothstep(u)
            out.append({"t": tt, "center_x": y})
            tt += dt
        current = target
        last_t = ramp_end
        pending_start = None
        pending_target = None
        i += 1
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
    keyframe_max_hold_sec=4.0,
    active_id_weight=0.20,
    talk_weight=0.15
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
        return [], [], []

    step = max(1, int(fps * max(0.2, float(window_sec))))
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    if end_frame <= start_frame:
        cap.release()
        return [], [], []

    raw = []
    last_center = float(target_center)
    last_center_y = 0.5
    last_face_h = 0.18
    has_last = False
    pending_dir = 0
    pending_count = 0
    identity_emb = None
    identity_anchor_times = []
    insightface_app = None
    allow_unanchored = os.getenv("VERTICAL_DYNAMIC_CROP_ALLOW_UNANCHORED", "true").lower() == "true"
    center_gate = float(os.getenv("VERTICAL_DYNAMIC_CROP_CENTER_GATE", 0.18))
    identity_margin = float(os.getenv("VERTICAL_DYNAMIC_CROP_IDENTITY_MARGIN", 0.06))
    min_face_h_abs = float(os.getenv("VERTICAL_DYNAMIC_CROP_MIN_FACE_H", 0.07))
    min_rel_face_h = float(os.getenv("VERTICAL_DYNAMIC_CROP_MIN_REL_FACE_H", 0.55))
    np = None
    active_emb = None
    prev_gray = None
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
                identity_emb = _normalize_vec(maybe_centroid)
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
                        norm_emb = _normalize_vec(emb)
                        if norm_emb is not None:
                            best_area = area_ratio
                            best_emb = norm_emb
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

    # ClipsAI-inspired unanchored mode: still run face analysis (without identity seed)
    # so continuity + talk activity can decide focus among multiple visible faces.
    if allow_unanchored and insightface_app is None and (float(active_id_weight) > 0.0 or float(talk_weight) > 0.0):
        try:
            import numpy as np  # type: ignore
            from insightface.app import FaceAnalysis  # type: ignore
            insightface_app = FaceAnalysis(name=os.getenv("FACE_MODEL_NAME", "buffalo_l"), providers=["CPUExecutionProvider"])
            ds = max(128, int(identity_det_size))
            insightface_app.prepare(ctx_id=0, det_size=(ds, ds))
            print(
                f"Unanchored insightface mode enabled (active_id_weight={float(active_id_weight):.2f}, talk_weight={float(talk_weight):.2f}, det_size={ds})",
                file=sys.stderr
            )
        except Exception as e:
            print(f"Unanchored insightface unavailable ({e}); using OpenCV fallback.", file=sys.stderr)
            insightface_app = None

    # Scene cut detection: compare simple color histogram across sampled frames
    def _is_cut(prev_frame, curr_frame, thresh=0.35):
        if prev_frame is None or curr_frame is None:
            return False
        prev_hist = cv2.calcHist([prev_frame], [0], None, [32], [0, 256])
        curr_hist = cv2.calcHist([curr_frame], [0], None, [32], [0, 256])
        cv2.normalize(prev_hist, prev_hist)
        cv2.normalize(curr_hist, curr_hist)
        diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
        return diff >= thresh

    frame_idx = start_frame
    last_sample_frame = -999999
    prev_scene_frame = None
    cut_fast_mode_frames = 0
    scene_fixed_center = None
    scene_cut_times = []
    scene_cut_min_gap = float(os.getenv("VERTICAL_SCENE_CUT_MIN_GAP_SEC", 0.35))
    scene_hold_enabled = os.getenv("VERTICAL_SCENE_HOLD_ENABLED", "true").lower() == "true"
    scene_hold_window = float(os.getenv("VERTICAL_SCENE_HOLD_WINDOW_SEC", 2.0))
    scene_hold_samples = []
    scene_detect_window_frames = int(max(1, fps * scene_hold_window))
    while frame_idx <= end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cx = None
        cy = None
        fh = None
        selected_emb = None

        # detect cuts every ~0.5s of source footage
        if frame_idx - last_sample_frame >= int(max(1, fps * 0.5)):
            if _is_cut(prev_scene_frame, frame, thresh=0.38):
                t_cut = max(0.0, (frame_idx / fps) - start_sec)
                if (not scene_cut_times) or abs(float(t_cut) - float(scene_cut_times[-1])) >= max(0.05, scene_cut_min_gap):
                    scene_cut_times.append(float(t_cut))
                # on cut: reset to identity anchor center, enable fast catch-up window
                if identity_emb is not None:
                    cx = last_center if has_last else target_center
                if scene_hold_enabled:
                    scene_fixed_center = cx if cx is not None else (last_center if has_last else target_center)
                    scene_hold_samples = []
                cut_fast_mode_frames = int(max(1, fps * 1.5))
            prev_scene_frame = frame.copy()
            last_sample_frame = frame_idx
        if insightface_app is not None and np is not None:
            try:
                faces = insightface_app.get(frame) or []
                candidates = []
                best_score = -1e9
                h, w = frame.shape[:2]
                frame_area = float(max(1, w * h))
                for face in faces:
                    emb = getattr(face, "embedding", None)
                    bbox = getattr(face, "bbox", None)
                    if emb is None or bbox is None:
                        continue
                    norm_emb = _normalize_vec(emb)
                    if norm_emb is None:
                        continue
                    identity_sim = _cosine(identity_emb, norm_emb) if identity_emb is not None else 0.0
                    if identity_emb is not None and identity_sim < float(identity_min_sim):
                        continue
                    active_sim = _cosine(active_emb, norm_emb) if active_emb is not None else 0.0
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                    bw = max(1.0, x2 - x1)
                    bh = max(1.0, y2 - y1)
                    area_ratio = (bw * bh) / frame_area
                    cxx = (x1 + x2) / 2.0 / max(1.0, w)
                    cyy = (y1 + y2) / 2.0 / max(1.0, h)
                    fhh = bh / max(1.0, h)
                    if identity_emb is not None and abs(cxx - last_center) > center_gate:
                        continue
                    prev_score = 1.0 - abs(cxx - last_center) if has_last else 0.0
                    center_score = 1.0 - abs(cxx - float(target_center))
                    talk_raw = _calc_mouth_motion(prev_gray, curr_gray, bbox)
                    candidates.append({
                        "cxx": float(cxx),
                        "cyy": float(cyy),
                        "fhh": float(max(0.01, min(0.8, fhh))),
                        "identity_sim": float(max(-1.0, identity_sim)),
                        "active_sim": float(max(-1.0, active_sim)),
                        "prev_score": float(max(0.0, min(1.0, prev_score))),
                        "center_score": float(max(0.0, min(1.0, center_score))),
                        "area_score": float(max(0.0, min(1.0, area_ratio * 20.0))),
                        "talk_raw": float(max(0.0, talk_raw)),
                        "emb": norm_emb,
                    })

                if candidates:
                    if identity_emb is not None:
                        best_identity = max(c["identity_sim"] for c in candidates)
                        best_fhh = max(c["fhh"] for c in candidates)
                        dyn_min_h = float(min_face_h_abs)
                        if has_last:
                            dyn_min_h = max(dyn_min_h, float(last_face_h) * float(min_rel_face_h))
                        anchored = []
                        for c in candidates:
                            if c["identity_sim"] < (best_identity - float(identity_margin)):
                                continue
                            # Reject persistent tiny/background faces, but allow fallback when all faces are small.
                            if c["fhh"] < dyn_min_h and c["fhh"] < best_fhh * 0.85:
                                continue
                            anchored.append(c)
                        if anchored:
                            candidates = anchored

                    max_talk = max(c["talk_raw"] for c in candidates)
                    talk_scale = max(1e-6, max_talk)
                    for c in candidates:
                        talk_score = c["talk_raw"] / talk_scale
                        if identity_emb is not None:
                            # Identity-anchored mode: preserve speaker lock and use talk as a tie-breaker.
                            score = (
                                0.50 * c["identity_sim"] +
                                float(active_id_weight) * c["active_sim"] +
                                0.14 * c["prev_score"] +
                                0.06 * c["center_score"] +
                                0.10 * c["area_score"] +
                                float(talk_weight) * talk_score
                            )
                        else:
                            # Unanchored mode: continuity/talk drive the decision.
                            score = (
                                float(active_id_weight) * c["active_sim"] +
                                0.45 * c["prev_score"] +
                                0.15 * c["center_score"] +
                                0.20 * c["area_score"] +
                                float(talk_weight) * talk_score
                            )
                        c["score"] = float(score)
                        if score > best_score:
                            best_score = score
                            cx = c["cxx"]
                            cy = c["cyy"]
                            fh = c["fhh"]
                            selected_emb = c["emb"]
            except Exception:
                cx = None
                cy = None
                fh = None
                selected_emb = None
        if cx is None:
            face_info = _largest_face_center(
                face_cascade,
                frame,
                prev_center=(last_center if has_last else None),
                target_center=float(target_center)
            )
            if face_info is not None:
                cx, cy, fh = face_info
        if cx is None:
            cx = last_center if has_last else 0.5
        if cy is None:
            cy = last_center_y if has_last else 0.5
        if fh is None:
            fh = last_face_h

        if selected_emb is not None:
            if active_emb is None:
                active_emb = selected_emb
            else:
                # EMA on selected identity embedding to reduce accidental target switching.
                alpha_emb = 0.20
                blended = [
                    (1.0 - alpha_emb) * float(a) + alpha_emb * float(b)
                    for a, b in zip(active_emb, selected_emb)
                ]
                active_emb = _normalize_vec(blended) or active_emb

        # Composition blend: avoid hard face-centering, preserve more body/stage context.
        fw = max(0.0, min(1.0, float(composition_face_weight)))
        cx = fw * float(cx) + (1.0 - fw) * float(target_center)
        if scene_hold_enabled and scene_fixed_center is not None:
            cx = scene_fixed_center
        if has_last:
            dt_sec = max(0.2, float(window_sec))
            local_max_delta = float(max_delta_per_sec)
            local_max_accel = float(max_accel_per_sec2)
            if cut_fast_mode_frames > 0:
                local_max_delta = max(local_max_delta, float(os.getenv("VERTICAL_DYNAMIC_CROP_CUT_MAX_DELTA", 0.12)))
                local_max_accel = max(local_max_accel, float(os.getenv("VERTICAL_DYNAMIC_CROP_CUT_MAX_ACCEL", 0.08)))
            max_step = max(0.01, local_max_delta * dt_sec)
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
        last_center_y = cy
        last_face_h = fh
        has_last = True
        t_rel = max(0.0, (frame_idx / fps) - start_sec)
        if scene_hold_enabled and scene_fixed_center is not None:
            scene_hold_samples.append(float(cx))
            # update scene-fixed center using median over hold window duration
            max_samples = max(3, scene_detect_window_frames)
            if len(scene_hold_samples) > max_samples:
                scene_hold_samples = scene_hold_samples[-max_samples:]
            scene_fixed_center = sorted(scene_hold_samples)[len(scene_hold_samples) // 2]

        raw.append({"t": t_rel, "center_x": float(cx), "center_y": float(cy), "face_h": float(fh)})
        prev_gray = curr_gray
        frame_idx += step
        if cut_fast_mode_frames > 0:
            cut_fast_mode_frames -= step

    cap.release()
    if not raw:
        return [], [], []

    alpha = max(0.05, min(0.95, float(smooth_alpha)))
    smoothed = []
    prev_x = raw[0]["center_x"]
    prev_y = raw[0].get("center_y", 0.5)
    prev_h = raw[0].get("face_h", 0.18)
    for p in raw:
        vx = p["center_x"] * alpha + prev_x * (1.0 - alpha)
        vy = p.get("center_y", prev_y) * alpha + prev_y * (1.0 - alpha)
        vh = p.get("face_h", prev_h) * alpha + prev_h * (1.0 - alpha)
        prev_x = vx
        prev_y = vy
        prev_h = vh
        smoothed.append({"t": float(p["t"]), "center_x": float(vx), "center_y": float(vy), "face_h": float(vh)})

    if smoothed[0]["t"] > 0:
        smoothed.insert(0, {
            "t": 0.0,
            "center_x": float(smoothed[0]["center_x"]),
            "center_y": float(smoothed[0].get("center_y", 0.5)),
            "face_h": float(smoothed[0].get("face_h", 0.18))
        })

    end_rel = max(0.0, end_sec - start_sec)
    if smoothed[-1]["t"] < end_rel:
        smoothed.append({
            "t": float(end_rel),
            "center_x": float(smoothed[-1]["center_x"]),
            "center_y": float(smoothed[-1].get("center_y", 0.5)),
            "face_h": float(smoothed[-1].get("face_h", 0.18))
        })
    use_hold_ramp = os.getenv("VERTICAL_DYNAMIC_CROP_USE_HOLD_RAMP", "false").lower() == "true"
    if use_hold_ramp:
        threshold = float(os.getenv("VERTICAL_DYNAMIC_CROP_RAMP_THRESHOLD", 0.03))
        confirm_sec = float(os.getenv("VERTICAL_DYNAMIC_CROP_RAMP_CONFIRM_SEC", 0.3))
        ramp_sec = float(os.getenv("VERTICAL_DYNAMIC_CROP_RAMP_SEC", 0.9))
        ramp_dt = float(os.getenv("VERTICAL_DYNAMIC_CROP_RAMP_DT", 0.10))
        ramp_track = _apply_hold_ramp(
            [{"t": p["t"], "center_x": p["center_x"]} for p in smoothed],
            threshold=threshold,
            confirm_sec=confirm_sec,
            ramp_sec=ramp_sec,
            sample_dt=ramp_dt
        )
        filtered = ramp_track if ramp_track else smoothed
    else:
        filtered = _apply_motion_filter(
            [{"t": p["t"], "center_x": p["center_x"]} for p in smoothed],
            end_rel=end_rel,
            motion_dt=float(motion_dt),
            max_speed_per_sec=float(max_delta_per_sec),
            max_accel_per_sec2=float(max_accel_per_sec2),
            follow_kp=float(follow_kp),
            follow_kd=float(follow_kd),
            lookahead_sec=float(lookahead_sec)
        )
    final_x_track = filtered if filtered else [{"t": p["t"], "center_x": p["center_x"]} for p in smoothed]
    final_track = []
    for p in final_x_track:
        tt = float(p["t"])
        final_track.append({
            "t": tt,
            "center_x": float(p["center_x"]),
            "center_y": _interp_field(smoothed, "center_y", tt, 0.5),
            "face_h": _interp_field(smoothed, "face_h", tt, 0.18)
        })
    camera_keyframes = _build_camera_keyframes(
        [{"t": p["t"], "center_x": p["center_x"]} for p in final_track],
        end_rel=end_rel,
        keyframe_sec=float(keyframe_sec),
        min_move=float(keyframe_min_move),
        max_hold_sec=float(keyframe_max_hold_sec)
    )
    scene_cuts = [float(t) for t in scene_cut_times if 0.0 < float(t) < float(end_rel)]
    return final_track, camera_keyframes, scene_cuts

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
    parser.add_argument("--active_id_weight", type=float, default=0.20)
    parser.add_argument("--talk_weight", type=float, default=0.15)
    args = parser.parse_args()
    
    video_file = args.video_file
    limit_seconds = args.limit_seconds
    
    try:
        if args.track:
            track, camera_keyframes, scene_cuts = detect_face_track(
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
                keyframe_max_hold_sec=args.keyframe_max_hold_sec,
                active_id_weight=args.active_id_weight,
                talk_weight=args.talk_weight
            )
            if not track:
                print(json.dumps({"center_x": 0.5, "track": []}))
            else:
                centers = sorted([p["center_x"] for p in track])
                median = centers[len(centers) // 2]
                print(json.dumps({"center_x": median, "track": track, "camera_keyframes": camera_keyframes, "scene_cuts_sec": scene_cuts}))
        else:
            center_x = detect_face_opencv(video_file, limit_seconds)
            print(json.dumps({"center_x": center_x}))
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)
