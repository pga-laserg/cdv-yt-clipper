from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrackState:
    track_id: int
    region: str
    x1: int
    y1: int
    x2: int
    y2: int
    start_sec: float
    end_sec: float
    last_seen_sec: float
    hits: int = 1
    misses: int = 0
    score: float = 0.0


def iou_box(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = float((x2 - x1) * (y2 - y1))
    aa = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
    bb = float(max(1, (bx2 - bx1) * (by2 - by1)))
    return inter / max(1e-6, (aa + bb - inter))


class SimpleBoxTracker:
    def __init__(self, region: str, iou_threshold: float = 0.35, max_misses: int = 3) -> None:
        self.region = str(region)
        self.iou_threshold = float(iou_threshold)
        self.max_misses = int(max_misses)
        self.next_id = 1
        self.active: list[TrackState] = []
        self.finished: list[TrackState] = []

    def _new_track(self, bbox: tuple[int, int, int, int], score: float, t_sec: float) -> TrackState:
        x1, y1, x2, y2 = bbox
        tr = TrackState(
            track_id=self.next_id,
            region=self.region,
            x1=int(x1),
            y1=int(y1),
            x2=int(x2),
            y2=int(y2),
            start_sec=float(t_sec),
            end_sec=float(t_sec),
            last_seen_sec=float(t_sec),
            hits=1,
            misses=0,
            score=float(score),
        )
        self.next_id += 1
        return tr

    def update(
        self,
        detections: list[tuple[int, int, int, int, float]],
        t_sec: float,
    ) -> list[tuple[int, tuple[int, int, int, int], float]]:
        # Returns assignments: (track_id, bbox, score) for current detections.
        assignments: list[tuple[int, tuple[int, int, int, int], float]] = []
        if not detections and not self.active:
            return assignments

        det_used = [False] * len(detections)
        track_used = [False] * len(self.active)

        # Greedy matching by IoU.
        pairs: list[tuple[float, int, int]] = []
        for ti, tr in enumerate(self.active):
            tb = (tr.x1, tr.y1, tr.x2, tr.y2)
            for di, det in enumerate(detections):
                db = (int(det[0]), int(det[1]), int(det[2]), int(det[3]))
                s = iou_box(tb, db)
                if s >= self.iou_threshold:
                    pairs.append((s, ti, di))
        pairs.sort(key=lambda x: x[0], reverse=True)

        for _, ti, di in pairs:
            if track_used[ti] or det_used[di]:
                continue
            track_used[ti] = True
            det_used[di] = True
            tr = self.active[ti]
            x1, y1, x2, y2, dscore = detections[di]
            # Smooth box update to reduce jitter.
            tr.x1 = int(round(0.6 * tr.x1 + 0.4 * x1))
            tr.y1 = int(round(0.6 * tr.y1 + 0.4 * y1))
            tr.x2 = int(round(0.6 * tr.x2 + 0.4 * x2))
            tr.y2 = int(round(0.6 * tr.y2 + 0.4 * y2))
            tr.end_sec = float(t_sec)
            tr.last_seen_sec = float(t_sec)
            tr.hits += 1
            tr.misses = 0
            tr.score = max(float(tr.score), float(dscore))
            assignments.append((tr.track_id, (tr.x1, tr.y1, tr.x2, tr.y2), float(dscore)))

        # Unmatched active tracks get miss counts.
        still_active: list[TrackState] = []
        for ti, tr in enumerate(self.active):
            if not track_used[ti]:
                tr.misses += 1
            if tr.misses > self.max_misses:
                self.finished.append(tr)
            else:
                still_active.append(tr)
        self.active = still_active

        # Unmatched detections become new tracks.
        for di, det in enumerate(detections):
            if det_used[di]:
                continue
            x1, y1, x2, y2, dscore = det
            tr = self._new_track((int(x1), int(y1), int(x2), int(y2)), float(dscore), float(t_sec))
            self.active.append(tr)
            assignments.append((tr.track_id, (tr.x1, tr.y1, tr.x2, tr.y2), float(dscore)))

        return assignments

    def finalize(self) -> list[TrackState]:
        if self.active:
            self.finished.extend(self.active)
            self.active = []
        return list(self.finished)

