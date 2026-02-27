import argparse
import base64
import json
import math
import os
import re
import shutil
import sys
import time
from typing import Any, Literal
import urllib.error
import urllib.request

import cv2
import numpy as np


OcrType = Literal["speaker_name", "bible_verse", "sermon_title", "song_lyric", "on_screen_text"]
OcrRegion = Literal["lower_third", "slide", "full"]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def parse_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    v = str(raw).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def sec_or_none(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return max(0.0, float(value))


def format_duration(sec: float) -> str:
    s = int(max(0.0, sec))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    return f"{h:02d}:{m:02d}:{ss:02d}"


def clean_text(text: str) -> str:
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return t


def normalize_key(text: str) -> str:
    t = clean_text(text).lower()
    t = re.sub(r"[^a-z0-9áéíóúñü ]+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def preprocess_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )
    return thr


def classify_text(text: str, region: OcrRegion) -> OcrType:
    t = clean_text(text)
    if not t:
        return "on_screen_text"

    verse_rx = re.compile(
        r"\b(gen(?:e?si?s)?|éxodo|exodo|lev[ií]tico|n[úu]meros|deuteronomio|josu[eé]|"
        r"jueces|rut|samuel|reyes|cr[oó]nicas|esdras|nehem[ií]as|ester|job|salmo[s]?|"
        r"proverbios|isa[ií]as|jerem[ií]as|ezequiel|daniel|oseas|joel|am[oó]s|abd[ií]as|"
        r"jon[aá]s|miqueas|nah[uú]m|habacuc|sof[oó]n[ií]as|hageo|zacar[ií]as|malaqu[ií]as|"
        r"mateo|marcos|lucas|juan|hechos|romanos|corintios|g[aá]latas|efesios|filipenses|"
        r"colosenses|tesalonicenses|timoteo|tito|filem[oó]n|hebreos|santiago|pedro|judas|apocalipsis)"
        r"\s+\d{1,3}[:.]\d{1,3}(?:\s*[-–]\s*\d{1,3})?\b",
        re.IGNORECASE,
    )
    if verse_rx.search(t):
        return "bible_verse"

    lower_name_rx = re.compile(
        r"\b(pr\.?|pastor|anciano|hno\.?|hna\.?|elder)\b",
        re.IGNORECASE,
    )
    words = [w for w in re.split(r"\s+", t) if w]
    sermon_title_rx = re.compile(
        r"\b(tema|t[ií]tulo|serie|mensaje|predicaci[oó]n|serm[oó]n)\b",
        re.IGNORECASE,
    )
    # Keep title keyword routing for non-slide regions only. Slide title semantics are delegated
    # to downstream LLM/context steps to avoid brittle keyword-only tagging.
    if region != "slide" and sermon_title_rx.search(t):
        return "sermon_title"

    if region == "lower_third":
        looks_name = (
            len(words) >= 2
            and len(words) <= 7
            and all(len(w) >= 2 for w in words)
            and sum(1 for w in words if w[:1].isupper()) >= 1
        )
        if lower_name_rx.search(t) or looks_name:
            return "speaker_name"
    if region == "slide" and lower_name_rx.search(t):
        tokens = re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñÜü]+", t)
        has_name_shape = bool(
            re.search(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+){1,3}\b", t)
        )
        has_phrase = len(tokens) >= 6
        if has_name_shape or has_phrase:
            return "speaker_name"

    lyric_rx = re.compile(
        r"\b(cantemos|alabemos|adoraci[oó]n|alabanza|santo|gloria|jes[uú]s|se[ñn]or)\b",
        re.IGNORECASE,
    )
    if lyric_rx.search(t) and len(words) >= 4:
        return "song_lyric"

    return "on_screen_text"


def detect_scene_cut(prev_frame, curr_frame, threshold: float) -> bool:
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_hist = cv2.calcHist([prev_gray], [0], None, [32], [0, 256])
    curr_hist = cv2.calcHist([curr_gray], [0], None, [32], [0, 256])
    cv2.normalize(prev_hist, prev_hist)
    cv2.normalize(curr_hist, curr_hist)
    diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
    return float(diff) >= threshold


def estimate_text_likelihood(roi, region: OcrRegion, resize_max_width: int = 640) -> tuple[float, dict[str, float]]:
    if roi is None or roi.size == 0:
        return 0.0, {
            "edge_density": 0.0,
            "component_density": 0.0,
            "contrast": 0.0,
            "score": 0.0,
        }

    h, w = roi.shape[:2]
    if resize_max_width > 0 and w > resize_max_width:
        scale = resize_max_width / float(w)
        roi_small = cv2.resize(roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        roi_small = roi

    gray = cv2.cvtColor(roi_small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    edge_density = float(cv2.countNonZero(edges)) / float(max(1, edges.shape[0] * edges.shape[1]))
    edge_norm = clamp((edge_density - 0.008) / 0.115, 0.0, 1.0)

    thr = preprocess_roi(roi_small)
    inv = cv2.bitwise_not(thr)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    roi_area = float(max(1, inv.shape[0] * inv.shape[1]))
    min_area = max(6, int(roi_area * 0.00003))
    max_area = max(min_area + 1, int(roi_area * 0.012))
    component_count = 0
    for idx in range(1, int(num_labels)):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        cw = int(stats[idx, cv2.CC_STAT_WIDTH])
        ch = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if cw < 2 or ch < 2:
            continue
        aspect = float(cw) / float(max(1, ch))
        if 0.15 <= aspect <= 18.0:
            component_count += 1
    component_density = min(1.0, float(component_count) / 85.0)

    row_ratio = np.count_nonzero(inv > 0, axis=1).astype(np.float32) / float(max(1, inv.shape[1]))
    active_rows = row_ratio > 0.035
    active_row_ratio = float(np.count_nonzero(active_rows)) / float(max(1, active_rows.shape[0]))
    row_band_norm = clamp(active_row_ratio / 0.35, 0.0, 1.0)

    runs = 0
    run_lengths: list[int] = []
    run_len = 0
    for flag in active_rows.tolist():
        if flag:
            run_len += 1
        elif run_len > 0:
            runs += 1
            run_lengths.append(run_len)
            run_len = 0
    if run_len > 0:
        runs += 1
        run_lengths.append(run_len)
    mean_run = float(sum(run_lengths) / max(1, len(run_lengths)))
    run_count_norm = clamp((8.0 - min(8.0, float(runs))) / 8.0, 0.0, 1.0)
    run_len_norm = clamp(mean_run / 14.0, 0.0, 1.0)
    line_structure = clamp(0.5 * run_count_norm + 0.5 * run_len_norm, 0.0, 1.0)

    _, stddev = cv2.meanStdDev(gray)
    contrast = float(stddev[0][0]) if stddev is not None else 0.0
    contrast_norm = clamp(contrast / 72.0, 0.0, 1.0)

    if region == "lower_third":
        score = clamp(
            0.42 * edge_norm + 0.32 * component_density + 0.18 * row_band_norm + 0.08 * line_structure,
            0.0,
            1.0,
        )
    elif region == "slide":
        score = clamp(
            0.40 * edge_norm + 0.28 * component_density + 0.22 * row_band_norm + 0.10 * line_structure,
            0.0,
            1.0,
        )
    else:
        score = clamp(
            0.52 * edge_norm + 0.24 * component_density + 0.16 * row_band_norm + 0.08 * line_structure,
            0.0,
            1.0,
        )
    return score, {
        "edge_density": round(edge_density, 6),
        "component_density": round(component_density, 6),
        "contrast": round(contrast_norm, 6),
        "row_band": round(row_band_norm, 6),
        "line_structure": round(line_structure, 6),
        "score": round(score, 6),
    }


def merge_presence_observations(
    observations: list[dict[str, Any]],
    sample_sec: float,
    merge_gap_sec: float,
    min_duration_sec: float = 2.0,
) -> list[dict[str, Any]]:
    if not observations:
        return []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for obs in observations:
        region = str(obs.get("region", "full"))
        grouped.setdefault(region, []).append(obs)

    out: list[dict[str, Any]] = []
    for region, items in grouped.items():
        items = sorted(items, key=lambda x: float(x["t"]))
        current: dict[str, Any] | None = None
        for obs in items:
            t = float(obs["t"])
            score = float(obs["score"])
            scene_cut = bool(obs.get("scene_cut", False))
            if current is None:
                current = {
                    "region": region,
                    "start": t,
                    "end": t + sample_sec,
                    "score_sum": score,
                    "score_peak": score,
                    "samples": 1,
                    "scene_cut_hits": 1 if scene_cut else 0,
                }
                continue
            gap = t - float(current["end"])
            if gap <= merge_gap_sec:
                current["end"] = t + sample_sec
                current["score_sum"] += score
                current["score_peak"] = max(float(current["score_peak"]), score)
                current["samples"] += 1
                current["scene_cut_hits"] += 1 if scene_cut else 0
            else:
                dur = float(current["end"]) - float(current["start"])
                if dur >= min_duration_sec:
                    out.append(
                        {
                            "region": region,
                            "start": float(current["start"]),
                            "end": float(current["end"]),
                            "confidence": round(float(current["score_sum"]) / max(1, int(current["samples"])), 3),
                            "peak_confidence": round(float(current["score_peak"]), 3),
                            "samples": int(current["samples"]),
                            "near_scene_cut": bool(current["scene_cut_hits"] > 0),
                        }
                    )
                current = {
                    "region": region,
                    "start": t,
                    "end": t + sample_sec,
                    "score_sum": score,
                    "score_peak": score,
                    "samples": 1,
                    "scene_cut_hits": 1 if scene_cut else 0,
                }
        if current is not None:
            dur = float(current["end"]) - float(current["start"])
            if dur >= min_duration_sec:
                out.append(
                    {
                        "region": region,
                        "start": float(current["start"]),
                        "end": float(current["end"]),
                        "confidence": round(float(current["score_sum"]) / max(1, int(current["samples"])), 3),
                        "peak_confidence": round(float(current["score_peak"]), 3),
                        "samples": int(current["samples"]),
                        "near_scene_cut": bool(current["scene_cut_hits"] > 0),
                    }
                )
    return sorted(out, key=lambda x: (float(x["start"]), str(x["region"])))


def _parse_lang_hints(lang_hint: str) -> list[str]:
    hints: list[str] = []
    for raw in str(lang_hint or "").split(","):
        code = str(raw).strip().lower().replace("_", "-")
        if not code:
            continue
        hints.append(code)
    seen = set()
    out: list[str] = []
    for code in hints:
        if code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


def init_ocr_engine(prefer_backend: str, lang_hint: str):
    backend = "none"
    engine: Any = None
    errors: list[str] = []

    normalized = str(prefer_backend or "auto").strip().lower()
    gcv_aliases = {
        "gcv",
        "google_vision",
        "google-vision",
        "vision",
        "gcv_text_detection",
        "google_vision_text_detection",
    }
    openai_aliases = {
        "openai",
        "openai_text_detection",
        "openai_vision",
        "openai_vision_text",
    }
    auto_include_gcv = parse_bool(os.getenv("OCR_AUTO_INCLUDE_GCV"), default=False)
    gcv_fallback_enabled = parse_bool(os.getenv("OCR_GCV_FALLBACK"), default=True)
    openai_fallback_enabled = parse_bool(os.getenv("OCR_OPENAI_FALLBACK"), default=True)
    disable_gcv = parse_bool(os.getenv("OCR_DISABLE_GCV"), default=False)

    wants_gcv = (normalized in gcv_aliases) or (normalized == "auto" and auto_include_gcv)
    wants_openai = normalized in openai_aliases
    wants_easyocr = normalized in ("auto", "easyocr")
    wants_tesseract = normalized in ("auto", "tesseract")
    if normalized in gcv_aliases and gcv_fallback_enabled:
        if openai_fallback_enabled:
            wants_openai = True
        wants_easyocr = True
        wants_tesseract = True

    if wants_gcv:
        if disable_gcv:
            errors.append("google-cloud-vision unavailable: disabled via OCR_DISABLE_GCV=true")
        else:
            try:
                from google.cloud import vision  # type: ignore

                endpoint = str(os.getenv("OCR_GCV_ENDPOINT", "")).strip()
                if endpoint:
                    client = vision.ImageAnnotatorClient(client_options={"api_endpoint": endpoint})
                else:
                    client = vision.ImageAnnotatorClient()
                engine = {
                    "client": client,
                    "vision": vision,
                    "lang_hints": _parse_lang_hints(lang_hint),
                }
                backend = "gcv_text_detection"
                return backend, engine, errors
            except Exception as exc:  # pragma: no cover
                errors.append(f"google-cloud-vision unavailable: {exc}")

    if wants_openai:
        api_key = str(os.getenv("OPENAI_API_KEY", "")).strip()
        if not api_key:
            errors.append("openai vision unavailable: OPENAI_API_KEY missing")
        else:
            model = str(
                os.getenv("OCR_OPENAI_VISION_MODEL")
                or os.getenv("BOUNDARY_LLM_MODEL")
                or os.getenv("ANALYZE_OPENAI_MODEL")
                or "gpt-5-mini"
            ).strip()
            if not model:
                model = "gpt-5-mini"
            engine = {
                "api_key": api_key,
                "base_url": str(os.getenv("OCR_OPENAI_BASE_URL", "https://api.openai.com/v1")).strip().rstrip("/"),
                "model": model,
                "timeout_sec": float(os.getenv("OCR_OPENAI_TIMEOUT_SEC", "45")),
                "image_detail": str(os.getenv("OCR_OPENAI_IMAGE_DETAIL", "high")).strip().lower() or "high",
            }
            backend = "openai_text_detection"
            return backend, engine, errors

    if wants_easyocr:
        try:
            import easyocr  # type: ignore

            lang_codes = [x.strip() for x in lang_hint.split(",") if x.strip()]
            # EasyOCR language codes are usually "es", "en".
            if not lang_codes:
                lang_codes = ["es", "en"]
            engine = easyocr.Reader(lang_codes, gpu=False, verbose=False)
            backend = "easyocr"
            return backend, engine, errors
        except Exception as exc:  # pragma: no cover
            errors.append(f"easyocr unavailable: {exc}")

    if wants_tesseract:
        try:
            import pytesseract  # type: ignore

            tess_bin = os.getenv("TESSERACT_BIN", "").strip()
            candidate_bins = [
                tess_bin,
                getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract"),
                "tesseract",
                "/opt/homebrew/bin/tesseract",
                "/usr/local/bin/tesseract",
            ]
            resolved = ""
            for candidate in candidate_bins:
                c = str(candidate or "").strip()
                if not c:
                    continue
                if os.path.isabs(c) and os.path.exists(c):
                    resolved = c
                    break
                found = shutil.which(c)
                if found:
                    resolved = found
                    break
            if not resolved:
                raise RuntimeError("tesseract binary not found (set TESSERACT_BIN or add to PATH)")
            pytesseract.pytesseract.tesseract_cmd = resolved
            engine = pytesseract
            backend = "tesseract"
            return backend, engine, errors
        except Exception as exc:  # pragma: no cover
            errors.append(f"tesseract unavailable: {exc}")

    return backend, engine, errors


def _extract_gcv_confidence_and_lang(response: Any) -> tuple[float, list[str]]:
    confs: list[float] = []
    langs: list[str] = []
    full = getattr(response, "full_text_annotation", None)
    pages = getattr(full, "pages", None) if full is not None else None
    if pages:
        for page in pages:
            # language hints are available in properties across levels.
            prop = getattr(page, "property", None)
            det_langs = getattr(prop, "detected_languages", None) if prop is not None else None
            if det_langs:
                for dl in det_langs:
                    code = str(getattr(dl, "language_code", "") or "").strip()
                    if code:
                        langs.append(code)
            blocks = getattr(page, "blocks", None) or []
            for block in blocks:
                paragraphs = getattr(block, "paragraphs", None) or []
                for para in paragraphs:
                    words = getattr(para, "words", None) or []
                    for word in words:
                        w_conf = getattr(word, "confidence", None)
                        if isinstance(w_conf, (int, float)):
                            confs.append(float(w_conf))
                        symbols = getattr(word, "symbols", None) or []
                        for sym in symbols:
                            s_conf = getattr(sym, "confidence", None)
                            if isinstance(s_conf, (int, float)):
                                confs.append(float(s_conf))
                            sprop = getattr(sym, "property", None)
                            det_langs = getattr(sprop, "detected_languages", None) if sprop is not None else None
                            if det_langs:
                                for dl in det_langs:
                                    code = str(getattr(dl, "language_code", "") or "").strip()
                                    if code:
                                        langs.append(code)
    avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
    # De-dup lang codes preserving order.
    seen = set()
    uniq_langs: list[str] = []
    for code in langs:
        if code in seen:
            continue
        seen.add(code)
        uniq_langs.append(code)
    return avg_conf, uniq_langs


def _run_gcv_text_detection(engine: Any, img, lang_hint: str) -> tuple[str, float, dict[str, Any]]:
    if not isinstance(engine, dict):
        return "", 0.0, {}
    client = engine.get("client")
    vision = engine.get("vision")
    if client is None or vision is None:
        return "", 0.0, {}

    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        return "", 0.0, {}
    image = vision.Image(content=enc.tobytes())
    lang_hints = engine.get("lang_hints")
    if not isinstance(lang_hints, list):
        lang_hints = _parse_lang_hints(lang_hint)
    image_context = {"language_hints": lang_hints} if lang_hints else None

    response = client.text_detection(image=image, image_context=image_context)
    err = getattr(getattr(response, "error", None), "message", "")
    if err:
        raise RuntimeError(f"google vision text_detection failed: {err}")

    text = ""
    full = getattr(response, "full_text_annotation", None)
    full_text = str(getattr(full, "text", "") or "").strip() if full is not None else ""
    if full_text:
        text = full_text
    else:
        annotations = getattr(response, "text_annotations", None) or []
        if annotations:
            text = str(getattr(annotations[0], "description", "") or "").strip()
    text = clean_text(text)
    if not text:
        return "", 0.0, {
            "provider": "google_vision",
            "method": "TEXT_DETECTION",
            "detected_languages": [],
            "word_confidence_mean": 0.0,
        }

    mean_conf, langs = _extract_gcv_confidence_and_lang(response)
    # Some videos/models may not expose word-level confidence for TEXT_DETECTION.
    conf = mean_conf if mean_conf > 0 else 0.80
    conf = float(max(0.0, min(1.0, conf)))
    meta = {
        "provider": "google_vision",
        "method": "TEXT_DETECTION",
        "detected_languages": langs,
        "word_confidence_mean": float(round(mean_conf, 6)),
    }
    return text, conf, meta


def _parse_openai_ocr_content(raw_content: Any) -> str:
    if isinstance(raw_content, list):
        parts: list[str] = []
        for item in raw_content:
            if isinstance(item, dict):
                tx = str(item.get("text", "")).strip()
                if tx:
                    parts.append(tx)
        return clean_text(" ".join(parts))
    if not isinstance(raw_content, str):
        return ""
    content = raw_content.strip()
    if not content:
        return ""

    # Strip fenced markdown blocks when present.
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", content, flags=re.IGNORECASE)
    if fence:
        content = fence.group(1).strip()

    parsed_obj: dict[str, Any] | None = None
    try:
        maybe = json.loads(content)
        if isinstance(maybe, dict):
            parsed_obj = maybe
    except Exception:
        left = content.find("{")
        right = content.rfind("}")
        if left >= 0 and right > left:
            try:
                maybe = json.loads(content[left : right + 1])
                if isinstance(maybe, dict):
                    parsed_obj = maybe
            except Exception:
                parsed_obj = None

    if isinstance(parsed_obj, dict):
        for key in ("text", "transcript", "ocr_text", "content"):
            val = parsed_obj.get(key)
            if isinstance(val, str) and clean_text(val):
                return clean_text(val)
        return ""
    return clean_text(content)


def _run_openai_text_detection(engine: Any, img, lang_hint: str) -> tuple[str, float, dict[str, Any]]:
    if not isinstance(engine, dict):
        return "", 0.0, {}
    api_key = str(engine.get("api_key", "")).strip()
    if not api_key:
        return "", 0.0, {}
    model = str(engine.get("model", "gpt-5-mini")).strip() or "gpt-5-mini"
    base_url = str(engine.get("base_url", "https://api.openai.com/v1")).strip().rstrip("/")
    timeout_sec = float(engine.get("timeout_sec", 45))
    image_detail = str(engine.get("image_detail", "high")).strip().lower() or "high"
    if image_detail not in ("low", "high", "auto"):
        image_detail = "high"

    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        return "", 0.0, {}
    b64 = base64.b64encode(enc.tobytes()).decode("ascii")
    prompt = (
        "Extract all visible text from this image exactly as seen. "
        "Return strict JSON only: {\"text\":\"...\"}. "
        "Use '\\n' for line breaks. If no text is visible, return {\"text\":\"\"}."
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": image_detail,
                        },
                    },
                ],
            }
        ],
    }

    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=max(5.0, timeout_sec)) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        raise RuntimeError(f"openai vision request failed: HTTP {exc.code} {detail[:400]}") from exc

    data = json.loads(body) if body else {}
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        raise RuntimeError(f"openai vision request failed: {data['error'].get('message', 'unknown error')}")

    choices = data.get("choices") if isinstance(data, dict) else None
    message = choices[0].get("message", {}) if isinstance(choices, list) and choices else {}
    raw_content = message.get("content")
    text = _parse_openai_ocr_content(raw_content)
    if not text:
        return "", 0.0, {
            "provider": "openai",
            "method": "chat_completions_vision",
            "model": model,
        }

    alnum_chars = sum(1 for ch in text if ch.isalnum())
    alnum_ratio = float(alnum_chars) / float(max(1, len(text)))
    length_norm = clamp(len(text) / 420.0, 0.0, 1.0)
    confidence = clamp(0.55 + 0.30 * length_norm + 0.15 * alnum_ratio, 0.0, 0.95)
    usage = data.get("usage") if isinstance(data, dict) else None
    meta = {
        "provider": "openai",
        "method": "chat_completions_vision",
        "model": model,
        "word_confidence_mean": round(confidence, 6),
        "usage": usage if isinstance(usage, dict) else {},
    }
    return clean_text(text), float(confidence), meta


def run_ocr(
    backend: str,
    engine: Any,
    roi_img,
    lang_hint: str,
    *,
    psm: int | None = None,
    upscale: float = 1.0,
    return_meta: bool = False,
) -> tuple[str, float] | tuple[str, float, dict[str, Any]]:
    if roi_img is None or roi_img.size == 0:
        return ("", 0.0, {}) if return_meta else ("", 0.0)
    img = roi_img
    if isinstance(upscale, (int, float)) and float(upscale) > 1.01:
        try:
            h, w = roi_img.shape[:2]
            img = cv2.resize(
                roi_img,
                (max(2, int(round(w * float(upscale)))), max(2, int(round(h * float(upscale))))),
                interpolation=cv2.INTER_CUBIC,
            )
        except Exception:
            img = roi_img

    if backend == "gcv_text_detection":
        try:
            text, conf, meta = _run_gcv_text_detection(engine, img, lang_hint)
            return (text, conf, meta) if return_meta else (text, conf)
        except Exception:
            return ("", 0.0, {}) if return_meta else ("", 0.0)

    if backend == "openai_text_detection":
        try:
            text, conf, meta = _run_openai_text_detection(engine, img, lang_hint)
            return (text, conf, meta) if return_meta else (text, conf)
        except Exception:
            return ("", 0.0, {}) if return_meta else ("", 0.0)

    if backend == "easyocr":
        # EasyOCR output shape varies by mode:
        # - paragraph=False -> [bbox, text, confidence]
        # - paragraph=True  -> often [bbox, text]
        # Run paragraph=False first to keep confidences when available.
        all_texts: list[str] = []
        all_confs: list[float] = []
        for paragraph_mode in (False, True):
            out = engine.readtext(img, detail=1, paragraph=paragraph_mode) or []
            if not out:
                continue
            for item in out:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                tx = clean_text(item[1])
                if not tx:
                    continue
                conf: float
                if len(item) >= 3 and isinstance(item[2], (int, float, np.floating)):
                    conf = float(item[2])
                else:
                    # Paragraph mode may omit confidence; keep moderate default.
                    conf = 0.55
                all_texts.append(tx)
                all_confs.append(float(max(0.0, min(1.0, conf))))
            if all_texts:
                break
        if not all_texts:
            return ("", 0.0, {}) if return_meta else ("", 0.0)
        text = clean_text(" ".join(all_texts))
        conf = float(sum(all_confs) / max(1, len(all_confs)))
        return (text, conf, {"provider": "easyocr"}) if return_meta else (text, conf)

    if backend == "tesseract":
        pytesseract = engine
        selected_psm = int(psm) if isinstance(psm, int) else 6
        if selected_psm < 3 or selected_psm > 13:
            selected_psm = 6
        config = f"--oem 1 --psm {selected_psm}"
        lang = os.getenv("TESSERACT_LANG", "spa+eng")
        try:
            data = pytesseract.image_to_data(
                img,
                lang=lang,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
            texts: list[str] = []
            confs: list[float] = []
            n = len(data.get("text", []))
            for i in range(n):
                tx = clean_text(data["text"][i])
                if not tx:
                    continue
                try:
                    c = float(data.get("conf", ["-1"])[i])
                except Exception:
                    c = -1.0
                if c < 0:
                    continue
                texts.append(tx)
                confs.append(c / 100.0)
            text = clean_text(" ".join(texts))
            conf = float(sum(confs) / max(1, len(confs)))
            return (text, conf, {"provider": "tesseract"}) if return_meta else (text, conf)
        except Exception:
            return ("", 0.0, {}) if return_meta else ("", 0.0)

    return ("", 0.0, {}) if return_meta else ("", 0.0)


def merge_observations(
    observations: list[dict[str, Any]],
    sample_sec: float,
    merge_gap_sec: float,
    scene_cuts: list[float],
    near_scene_cut_sec: float,
) -> list[dict[str, Any]]:
    if not observations:
        return []
    observations = sorted(observations, key=lambda x: (float(x["t"]), x["type"], x["region"]))
    out: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for obs in observations:
        key = (obs["norm"], obs["type"], obs["region"])
        if current is None:
            current = {
                "start": float(obs["t"]),
                "end": float(obs["t"]) + sample_sec,
                "text": obs["text"],
                "norm": obs["norm"],
                "type": obs["type"],
                "region": obs["region"],
                "confidence_sum": float(obs["confidence"]),
                "samples": 1,
            }
            continue

        curr_key = (current["norm"], current["type"], current["region"])
        gap = float(obs["t"]) - float(current["end"])
        if key == curr_key and gap <= merge_gap_sec:
            current["end"] = float(obs["t"]) + sample_sec
            current["confidence_sum"] += float(obs["confidence"])
            current["samples"] += 1
        else:
            avg_conf = float(current["confidence_sum"]) / max(1, int(current["samples"]))
            center = 0.5 * (float(current["start"]) + float(current["end"]))
            near_cut = any(abs(center - c) <= near_scene_cut_sec for c in scene_cuts)
            out.append(
                {
                    "start": float(current["start"]),
                    "end": float(current["end"]),
                    "text": str(current["text"]),
                    "type": str(current["type"]),
                    "region": str(current["region"]),
                    "confidence": round(avg_conf, 3),
                    "near_scene_cut": bool(near_cut),
                }
            )
            current = {
                "start": float(obs["t"]),
                "end": float(obs["t"]) + sample_sec,
                "text": obs["text"],
                "norm": obs["norm"],
                "type": obs["type"],
                "region": obs["region"],
                "confidence_sum": float(obs["confidence"]),
                "samples": 1,
            }

    if current is not None:
        avg_conf = float(current["confidence_sum"]) / max(1, int(current["samples"]))
        center = 0.5 * (float(current["start"]) + float(current["end"]))
        near_cut = any(abs(center - c) <= near_scene_cut_sec for c in scene_cuts)
        out.append(
            {
                "start": float(current["start"]),
                "end": float(current["end"]),
                "text": str(current["text"]),
                "type": str(current["type"]),
                "region": str(current["region"]),
                "confidence": round(avg_conf, 3),
                "near_scene_cut": bool(near_cut),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract OCR signals (lower-thirds/slides/lyrics) plus scene cuts for chapter/boundary cues."
    )
    parser.add_argument("video_path", help="Path to source video.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--sample-sec", type=float, default=float(os.getenv("OCR_SAMPLE_SEC", "2.5")))
    parser.add_argument("--max-sec", type=float, default=float(os.getenv("OCR_MAX_DURATION_SEC", "0")))
    parser.add_argument("--scene-cut-threshold", type=float, default=float(os.getenv("OCR_SCENE_CUT_THRESHOLD", "0.32")))
    parser.add_argument("--merge-gap-sec", type=float, default=float(os.getenv("OCR_MERGE_GAP_SEC", "4.5")))
    parser.add_argument("--near-scene-cut-sec", type=float, default=float(os.getenv("OCR_NEAR_SCENE_CUT_SEC", "3.0")))
    parser.add_argument("--backend", type=str, default=os.getenv("OCR_BACKEND", "auto"))
    parser.add_argument("--lang", type=str, default=os.getenv("OCR_LANG", "es,en"))
    parser.add_argument("--targeted", type=str, default=os.getenv("OCR_TARGETED", "true"))
    parser.add_argument("--target-threshold-lower-third", type=float, default=float(os.getenv("OCR_TARGET_THRESHOLD_LOWER_THIRD", "0.16")))
    parser.add_argument("--target-threshold-slide", type=float, default=float(os.getenv("OCR_TARGET_THRESHOLD_SLIDE", "0.20")))
    parser.add_argument("--target-threshold-full", type=float, default=float(os.getenv("OCR_TARGET_THRESHOLD_FULL", "0.24")))
    parser.add_argument("--target-persist-relax", type=float, default=float(os.getenv("OCR_TARGET_PERSIST_RELAX", "0.82")))
    parser.add_argument("--target-force-every-samples", type=int, default=int(os.getenv("OCR_TARGET_FORCE_EVERY_SAMPLES", "6")))
    parser.add_argument("--target-max-rois-per-sample", type=int, default=int(os.getenv("OCR_TARGET_MAX_ROIS_PER_SAMPLE", "2")))
    parser.add_argument("--target-scene-cut-boost", type=float, default=float(os.getenv("OCR_TARGET_SCENE_CUT_BOOST", "0.05")))
    parser.add_argument("--target-resize-max-width", type=int, default=int(os.getenv("OCR_TARGET_RESIZE_MAX_WIDTH", "640")))
    parser.add_argument("--target-emit-presence", type=str, default=os.getenv("OCR_TARGET_EMIT_PRESENCE", "true"))
    parser.add_argument("--target-presence-threshold", type=float, default=float(os.getenv("OCR_TARGET_PRESENCE_THRESHOLD", "0.14")))
    parser.add_argument("--target-presence-min-duration-sec", type=float, default=float(os.getenv("OCR_TARGET_PRESENCE_MIN_DURATION_SEC", "3.0")))
    parser.add_argument("--target-presence-merge-gap-sec", type=float, default=float(os.getenv("OCR_TARGET_PRESENCE_MERGE_GAP_SEC", "8.0")))
    parser.add_argument("--ocr-min-text-confidence", type=float, default=float(os.getenv("OCR_MIN_TEXT_CONFIDENCE", "0.10")))
    parser.add_argument("--ocr-preproc-fallback", type=str, default=os.getenv("OCR_PREPROC_FALLBACK", "true"))
    parser.add_argument("--regions", type=str, default=os.getenv("OCR_REGIONS", "lower_third,slide,full"))
    parser.add_argument("--lower-third-y1", type=float, default=float(os.getenv("OCR_LOWER_THIRD_Y1", "0.68")))
    parser.add_argument("--lower-third-y2", type=float, default=float(os.getenv("OCR_LOWER_THIRD_Y2", "0.92")))
    parser.add_argument("--lower-third-x1", type=float, default=float(os.getenv("OCR_LOWER_THIRD_X1", "0.06")))
    parser.add_argument("--lower-third-x2", type=float, default=float(os.getenv("OCR_LOWER_THIRD_X2", "0.94")))
    parser.add_argument("--progress-every-samples", type=int, default=int(os.getenv("OCR_PROGRESS_EVERY_SAMPLES", "0")))
    parser.add_argument("--progress-json", type=str, default=os.getenv("OCR_PROGRESS_JSON", "true"))
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")
    if args.sample_sec <= 0:
        raise ValueError("sample-sec must be > 0")

    targeted_enabled = parse_bool(args.targeted, default=True)
    region_thresholds: dict[str, float] = {
        "lower_third": float(args.target_threshold_lower_third),
        "slide": float(args.target_threshold_slide),
        "full": float(args.target_threshold_full),
    }
    target_persist_relax = clamp(float(args.target_persist_relax), 0.0, 1.0)
    target_force_every_samples = max(0, int(args.target_force_every_samples))
    target_max_rois = max(1, int(args.target_max_rois_per_sample))
    target_scene_cut_boost = max(0.0, float(args.target_scene_cut_boost))
    target_resize_max_width = max(160, int(args.target_resize_max_width))
    target_emit_presence = parse_bool(args.target_emit_presence, default=True)
    target_presence_threshold = clamp(float(args.target_presence_threshold), 0.0, 1.0)
    target_presence_min_duration_sec = max(0.5, float(args.target_presence_min_duration_sec))
    target_presence_merge_gap_sec = max(float(args.sample_sec), float(args.target_presence_merge_gap_sec))
    ocr_min_text_confidence = clamp(float(args.ocr_min_text_confidence), 0.0, 1.0)
    ocr_preproc_fallback = parse_bool(args.ocr_preproc_fallback, default=True)
    enabled_regions = {x.strip().lower() for x in str(args.regions or "").split(",") if x.strip()}
    if not enabled_regions:
        enabled_regions = {"lower_third", "slide", "full"}
    valid_regions = {"lower_third", "slide", "full"}
    enabled_regions = enabled_regions.intersection(valid_regions)
    if not enabled_regions:
        enabled_regions = {"lower_third"}
    lower_third_y1 = clamp(float(args.lower_third_y1), 0.0, 1.0)
    lower_third_y2 = clamp(float(args.lower_third_y2), 0.0, 1.0)
    lower_third_x1 = clamp(float(args.lower_third_x1), 0.0, 1.0)
    lower_third_x2 = clamp(float(args.lower_third_x2), 0.0, 1.0)
    if lower_third_y2 <= lower_third_y1:
        lower_third_y1, lower_third_y2 = 0.68, 0.92
    if lower_third_x2 <= lower_third_x1:
        lower_third_x1, lower_third_x2 = 0.06, 0.94

    backend, engine, engine_errors = init_ocr_engine(args.backend.strip().lower(), args.lang)
    if backend == "none":
        print(f"[ocr-events] No OCR backend available. errors={engine_errors}", file=sys.stderr)
        out_obj = {
            "source": "ocr-none",
            "duration_sec": 0,
            "sample_sec": args.sample_sec,
            "scene_cuts_sec": [],
            "segments": [],
            "detected_text_windows": [],
            "errors": engine_errors,
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        print(json.dumps({"ok": True, "segments": 0, "source": "ocr-none"}))
        return

    cap = cv2.VideoCapture(args.video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = float(total_frames / fps) if fps > 0 and total_frames > 0 else 0.0
    max_sec = duration_sec if args.max_sec <= 0 else min(duration_sec, float(args.max_sec))
    if max_sec <= 0:
        raise RuntimeError("Could not determine video duration for OCR pass.")

    print(
        f"[ocr-events] source={args.video_path} backend={backend} duration={duration_sec:.1f}s "
        f"max_sec={max_sec:.1f}s sample={args.sample_sec}s",
        file=sys.stderr,
    )

    observations: list[dict[str, Any]] = []
    presence_observations: list[dict[str, Any]] = []
    scene_cuts: list[float] = []
    prev_frame = None
    prev_selected_regions: set[str] = set()
    # Sampling is endpoint-inclusive (t=0 and t=max_sec when aligned), so expected
    # samples are floor(max_sec / step) + 1.
    total_samples = max(1, int(math.floor(max_sec / args.sample_sec + 1e-9)) + 1)
    progress_every_samples = int(args.progress_every_samples)
    progress_json = parse_bool(args.progress_json, default=True)
    if progress_every_samples <= 0:
        # Default to 5% progress ticks when not explicitly configured.
        progress_every_samples = max(1, int(math.ceil(total_samples * 0.05)))
    sample_idx = 0
    started_at = time.monotonic()
    targeting_stats = {
        "samples": 0,
        "roi_candidates": 0,
        "selected_rois": 0,
        "skipped_rois": 0,
        "ocr_calls": 0,
        "ocr_calls_raw": 0,
        "ocr_calls_preproc": 0,
        "forced_probes": 0,
        "persist_hits": 0,
        "scene_cut_samples": 0,
        "presence_hits": 0,
    }
    region_score_sums: dict[str, float] = {"lower_third": 0.0, "slide": 0.0, "full": 0.0}
    region_score_counts: dict[str, int] = {"lower_third": 0, "slide": 0, "full": 0}

    # Sequential frame traversal is significantly faster and more stable than
    # repeated random seeks on long GOP streams (e.g., AV1 YouTube sources).
    sample_every_frames = max(1, int(round(args.sample_sec * fps))) if fps > 0 else 1
    max_frame_idx = int(max_sec * fps) if fps > 0 else total_frames
    next_sample_frame = 0
    frame_idx = 0

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

        h, w = frame.shape[:2]
        scene_cut_now = False
        if prev_frame is not None and detect_scene_cut(prev_frame, frame, float(args.scene_cut_threshold)):
            if len(scene_cuts) == 0 or abs(float(t) - scene_cuts[-1]) >= 0.25:
                scene_cuts.append(float(t))
            scene_cut_now = True
        prev_frame = frame.copy()
        if scene_cut_now:
            targeting_stats["scene_cut_samples"] += 1

        lower_y1 = int(lower_third_y1 * h)
        lower_y2 = int(lower_third_y2 * h)
        lower_x1 = int(lower_third_x1 * w)
        lower_x2 = int(lower_third_x2 * w)
        rois: list[tuple[OcrRegion, Any]] = []
        if "lower_third" in enabled_regions:
            rois.append(("lower_third", frame[lower_y1:lower_y2, lower_x1:lower_x2]))
        if "slide" in enabled_regions:
            rois.append(("slide", frame[int(0.05 * h) : int(0.50 * h), 0:w]))
        if "full" in enabled_regions:
            rois.append(("full", frame))

        candidates: list[dict[str, Any]] = []
        for region, roi in rois:
            if roi is None or roi.size == 0:
                continue
            score, likelihood_parts = estimate_text_likelihood(roi, region, resize_max_width=target_resize_max_width)
            threshold = float(region_thresholds.get(region, 0.2))
            adjusted_threshold = max(0.0, threshold - (target_scene_cut_boost if scene_cut_now else 0.0))
            selected = score >= adjusted_threshold
            if not selected and region in prev_selected_regions and score >= adjusted_threshold * target_persist_relax:
                selected = True
                targeting_stats["persist_hits"] += 1
            if target_emit_presence and score >= target_presence_threshold:
                presence_observations.append(
                    {
                        "t": float(t),
                        "region": str(region),
                        "score": float(score),
                        "scene_cut": bool(scene_cut_now),
                    }
                )
                targeting_stats["presence_hits"] += 1
            region_score_sums[region] = float(region_score_sums.get(region, 0.0)) + float(score)
            region_score_counts[region] = int(region_score_counts.get(region, 0)) + 1
            candidates.append(
                {
                    "region": region,
                    "roi": roi,
                    "score": float(score),
                    "threshold": float(adjusted_threshold),
                    "selected": bool(selected),
                    "likelihood": likelihood_parts,
                }
            )

        if targeted_enabled:
            selected = [c for c in candidates if bool(c["selected"])]
            if len(selected) == 0 and target_force_every_samples > 0 and (sample_idx % target_force_every_samples == 0):
                best = max(candidates, key=lambda c: float(c["score"])) if candidates else None
                if best is not None:
                    selected = [best]
                    targeting_stats["forced_probes"] += 1
            if len(selected) > target_max_rois:
                selected = sorted(selected, key=lambda c: float(c["score"]), reverse=True)[:target_max_rois]
            selected_regions = {str(c["region"]) for c in selected}
            prev_selected_regions = selected_regions
        else:
            selected = list(candidates)
            prev_selected_regions = {str(c["region"]) for c in selected}

        targeting_stats["samples"] += 1
        targeting_stats["roi_candidates"] += len(candidates)
        targeting_stats["selected_rois"] += len(selected)
        targeting_stats["skipped_rois"] += max(0, len(candidates) - len(selected))
        targeting_stats["ocr_calls"] += len(selected)

        for candidate in selected:
            region = candidate["region"]
            roi = candidate["roi"]
            proc = preprocess_roi(roi)
            attempts: list[tuple[str, Any]]
            if backend in ("easyocr", "gcv_text_detection"):
                attempts = [("raw", roi), ("preproc", proc)] if ocr_preproc_fallback else [("raw", roi)]
            else:
                attempts = [("preproc", proc), ("raw", roi)] if ocr_preproc_fallback else [("preproc", proc)]

            best_text = ""
            best_conf = 0.0
            best_quality = -1.0
            best_input = ""

            for input_kind, input_img in attempts:
                if input_kind == "raw":
                    targeting_stats["ocr_calls_raw"] += 1
                else:
                    targeting_stats["ocr_calls_preproc"] += 1

                cand_text, cand_conf = run_ocr(backend, engine, input_img, args.lang)
                cand_text = clean_text(cand_text)
                if len(cand_text) < 4:
                    continue
                if not re.search(r"[A-Za-zÁÉÍÓÚáéíóúÑñ0-9]", cand_text):
                    continue

                cand_conf = float(max(0.0, min(1.0, cand_conf)))
                quality = cand_conf + min(1.0, len(cand_text) / 120.0) * 0.40
                # Allow long strings even when confidence is soft, but prioritize confidence.
                if cand_conf >= ocr_min_text_confidence or len(cand_text) >= 12:
                    if quality > best_quality:
                        best_quality = quality
                        best_text = cand_text
                        best_conf = cand_conf
                        best_input = input_kind

            if len(best_text) < 4:
                continue

            ttype = classify_text(best_text, region)
            observations.append(
                {
                    "t": float(t),
                    "text": best_text,
                    "norm": normalize_key(best_text),
                    "type": ttype,
                    "region": region,
                    "confidence": float(max(0.0, min(1.0, best_conf))),
                    "ocr_input": best_input,
                }
            )

        sample_idx += 1
        if sample_idx == 1 or sample_idx == total_samples or sample_idx % progress_every_samples == 0:
            pct = (sample_idx / max(1, total_samples)) * 100.0
            elapsed_sec = time.monotonic() - started_at
            speed = sample_idx / elapsed_sec if elapsed_sec > 0 else 0.0
            frames_per_sec = frame_idx / elapsed_sec if elapsed_sec > 0 else 0.0
            video_speed_x = t / elapsed_sec if elapsed_sec > 0 else 0.0
            remaining_samples = max(0, total_samples - sample_idx)
            eta_sec = (remaining_samples / speed) if speed > 0 else 0.0
            print(
                f"[ocr-events] progress={pct:.1f}% sampled={sample_idx}/{total_samples} "
                f"video_t={format_duration(t)}/{format_duration(max_sec)} "
                f"obs={len(observations)} scene_cuts={len(scene_cuts)} "
                f"ocr_calls={targeting_stats['ocr_calls']}/{max(1, targeting_stats['roi_candidates'])} "
                f"elapsed={format_duration(elapsed_sec)} eta={format_duration(eta_sec)} "
                f"samples_per_sec={speed:.3f} frames_per_sec={frames_per_sec:.1f} video_speed_x={video_speed_x:.2f}x",
                file=sys.stderr,
            )
            if progress_json:
                progress_payload = {
                    "type": "ocr_progress",
                    "percent": round(pct, 3),
                    "sampled": int(sample_idx),
                    "total_samples": int(total_samples),
                    "video_time_sec": sec_or_none(t),
                    "video_duration_sec": sec_or_none(max_sec),
                    "observations": int(len(observations)),
                    "scene_cuts": int(len(scene_cuts)),
                    "ocr_calls": int(targeting_stats["ocr_calls"]),
                    "roi_candidates": int(targeting_stats["roi_candidates"]),
                    "elapsed_sec": sec_or_none(elapsed_sec),
                    "eta_sec": sec_or_none(eta_sec),
                    "samples_per_sec": round(float(speed), 6),
                    "frames_per_sec": round(float(frames_per_sec), 6),
                    "video_speed_x": round(float(video_speed_x), 6),
                    "targeted": bool(targeted_enabled),
                }
                print(f"[ocr-events-progress] {json.dumps(progress_payload, ensure_ascii=False)}", file=sys.stderr)
        frame_idx += 1

    cap.release()

    merged = merge_observations(
        observations,
        sample_sec=float(args.sample_sec),
        merge_gap_sec=float(args.merge_gap_sec),
        scene_cuts=scene_cuts,
        near_scene_cut_sec=float(args.near_scene_cut_sec),
    )
    # Drop very short low-confidence generic text.
    merged = [
        s
        for s in merged
        if (s["type"] != "on_screen_text" or (float(s["end"]) - float(s["start"]) >= max(2.0, args.sample_sec * 1.5)))
    ]
    detected_text_windows = merge_presence_observations(
        observations=presence_observations if target_emit_presence else [],
        sample_sec=float(args.sample_sec),
        merge_gap_sec=float(target_presence_merge_gap_sec),
        min_duration_sec=float(target_presence_min_duration_sec),
    )

    out_obj = {
        "source": f"ocr-{backend}-v1",
        "duration_sec": duration_sec,
        "sample_sec": float(args.sample_sec),
        "scene_cuts_sec": scene_cuts,
        "targeting": {
            "enabled": bool(targeted_enabled),
            "thresholds": region_thresholds,
            "persist_relax": target_persist_relax,
            "force_every_samples": target_force_every_samples,
            "max_rois_per_sample": target_max_rois,
            "scene_cut_boost": target_scene_cut_boost,
            "resize_max_width": target_resize_max_width,
            "presence": {
                "emit": bool(target_emit_presence),
                "threshold": float(target_presence_threshold),
                "min_duration_sec": float(target_presence_min_duration_sec),
                "merge_gap_sec": float(target_presence_merge_gap_sec),
            },
            "ocr": {
                "min_text_confidence": float(ocr_min_text_confidence),
                "preproc_fallback": bool(ocr_preproc_fallback),
            },
            "roi": {
                "regions": sorted(enabled_regions),
                "lower_third_y1": float(lower_third_y1),
                "lower_third_y2": float(lower_third_y2),
                "lower_third_x1": float(lower_third_x1),
                "lower_third_x2": float(lower_third_x2),
            },
            "stats": targeting_stats,
            "avg_score_by_region": {
                region: round(
                    float(region_score_sums.get(region, 0.0)) / max(1, int(region_score_counts.get(region, 0))),
                    4,
                )
                for region in ("lower_third", "slide", "full")
            },
        },
        "segments": merged,
        "detected_text_windows": detected_text_windows,
        "errors": engine_errors,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    by_type: dict[str, int] = {}
    for s in merged:
        k = str(s.get("type", "unknown"))
        by_type[k] = by_type.get(k, 0) + 1
    print(
        f"[ocr-events] wrote segments={len(merged)} scene_cuts={len(scene_cuts)} types={by_type} -> {args.out}",
        file=sys.stderr,
    )
    print(json.dumps({"ok": True, "source": out_obj["source"], "segments": len(merged), "scene_cuts": len(scene_cuts)}))


if __name__ == "__main__":
    main()
