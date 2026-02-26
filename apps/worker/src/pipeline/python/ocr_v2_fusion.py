from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from difflib import SequenceMatcher


try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:  # pragma: no cover
    fuzz = None


@dataclass
class OcrObservation:
    track_key: str
    start: float
    end: float
    text: str
    confidence: float
    region: str


def clean_text(text: str) -> str:
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return t


def normalize_text(text: str) -> str:
    t = clean_text(text).lower()
    t = re.sub(r"[^a-z0-9áéíóúñü ]+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if fuzz is not None:
        return float(fuzz.ratio(a, b)) / 100.0
    return SequenceMatcher(None, a, b).ratio()


def fuse_observations(
    observations: list[OcrObservation],
    min_samples: int = 2,
    similarity_threshold: float = 0.84,
) -> dict[str, dict]:
    by_track: dict[str, list[OcrObservation]] = {}
    for obs in observations:
        if not clean_text(obs.text):
            continue
        by_track.setdefault(obs.track_key, []).append(obs)

    out: dict[str, dict] = {}
    for track_key, items in by_track.items():
        if len(items) < max(1, int(min_samples)):
            continue
        items = sorted(items, key=lambda x: x.start)
        norm_texts = [normalize_text(x.text) for x in items]
        base_counts = Counter(t for t in norm_texts if t)
        if not base_counts:
            continue
        # Cluster by similarity, then pick highest-support cluster.
        clusters: list[list[int]] = []
        for i, nt in enumerate(norm_texts):
            if not nt:
                continue
            assigned = False
            for cluster in clusters:
                ref = norm_texts[cluster[0]]
                if text_similarity(nt, ref) >= similarity_threshold:
                    cluster.append(i)
                    assigned = True
                    break
            if not assigned:
                clusters.append([i])
        clusters.sort(key=len, reverse=True)
        best = clusters[0]
        # Pick representative text: highest confidence among best cluster.
        best_items = [items[i] for i in best]
        best_items.sort(key=lambda x: (float(x.confidence), len(clean_text(x.text))), reverse=True)
        rep = clean_text(best_items[0].text)
        avg_conf = sum(float(x.confidence) for x in best_items) / max(1, len(best_items))
        out[track_key] = {
            "track_key": track_key,
            "text": rep,
            "norm_text": normalize_text(rep),
            "confidence": round(float(avg_conf), 3),
            "samples": len(best_items),
            "start": float(items[0].start),
            "end": float(items[-1].end),
            "region": str(best_items[0].region),
        }
    return out

