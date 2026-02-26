from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class TextBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    detector: str


def clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int] | None:
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w - 1, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h - 1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def box_area(box: TextBox) -> float:
    return float(max(0, box.x2 - box.x1) * max(0, box.y2 - box.y1))


def iou(a: TextBox, b: TextBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = float((x2 - x1) * (y2 - y1))
    union = box_area(a) + box_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def nms(boxes: list[TextBox], iou_threshold: float = 0.25) -> list[TextBox]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: float(b.score), reverse=True)
    out: list[TextBox] = []
    for box in boxes:
        if all(iou(box, kept) < iou_threshold for kept in out):
            out.append(box)
    return out


def load_east_model(model_path: str | None) -> Any | None:
    if not model_path:
        return None
    try:
        return cv2.dnn.readNet(model_path)
    except Exception:
        return None


def _decode_east(scores: np.ndarray, geometry: np.ndarray, min_conf: float) -> tuple[list[list[float]], list[float]]:
    rects: list[list[float]] = []
    confs: list[float] = []
    num_rows = int(scores.shape[2])
    num_cols = int(scores.shape[3])
    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x0_data = geometry[0, 0, y]
        x1_data = geometry[0, 1, y]
        x2_data = geometry[0, 2, y]
        x3_data = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]
        for x in range(num_cols):
            score = float(scores_data[x])
            if score < min_conf:
                continue
            offset_x = float(x * 4.0)
            offset_y = float(y * 4.0)
            angle = float(angles_data[x])
            cos_a = float(np.cos(angle))
            sin_a = float(np.sin(angle))
            h = float(x0_data[x] + x2_data[x])
            w = float(x1_data[x] + x3_data[x])
            end_x = offset_x + (cos_a * x1_data[x]) + (sin_a * x2_data[x])
            end_y = offset_y - (sin_a * x1_data[x]) + (cos_a * x2_data[x])
            start_x = end_x - w
            start_y = end_y - h
            rects.append([start_x, start_y, end_x, end_y])
            confs.append(score)
    return rects, confs


def detect_text_boxes_east(
    bgr: np.ndarray,
    east_net: Any,
    min_conf: float = 0.55,
    max_boxes: int = 8,
    min_w: int = 24,
    min_h: int = 12,
    min_aspect: float = 1.2,
    max_aspect: float = 16.0,
) -> list[TextBox]:
    h, w = bgr.shape[:2]
    if h < 32 or w < 32:
        return []
    in_w = int(max(32, (w // 32) * 32))
    in_h = int(max(32, (h // 32) * 32))
    r_w = w / float(in_w)
    r_h = h / float(in_h)
    resized = cv2.resize(bgr, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

    blob = cv2.dnn.blobFromImage(
        resized,
        scalefactor=1.0,
        size=(in_w, in_h),
        mean=(123.68, 116.78, 103.94),
        swapRB=True,
        crop=False,
    )
    east_net.setInput(blob)
    scores, geometry = east_net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
    rects, confs = _decode_east(scores, geometry, min_conf=min_conf)
    if not rects:
        return []

    boxes: list[TextBox] = []
    for r, c in zip(rects, confs):
        x1 = int(round(float(r[0]) * r_w))
        y1 = int(round(float(r[1]) * r_h))
        x2 = int(round(float(r[2]) * r_w))
        y2 = int(round(float(r[3]) * r_h))
        clipped = clip_box(x1, y1, x2, y2, w, h)
        if clipped is None:
            continue
        cx1, cy1, cx2, cy2 = clipped
        bw = cx2 - cx1
        bh = cy2 - cy1
        if bw < max(8, int(min_w)) or bh < max(6, int(min_h)):
            continue
        aspect = float(bw) / float(max(1, bh))
        if aspect < float(min_aspect) or aspect > float(max_aspect):
            continue
        boxes.append(TextBox(cx1, cy1, cx2, cy2, float(c), detector="east"))
    return nms(boxes, iou_threshold=0.22)[: max(1, int(max_boxes))]


def detect_text_boxes_mser(
    bgr: np.ndarray,
    min_conf: float = 0.35,
    max_boxes: int = 8,
    min_w: int = 24,
    min_h: int = 12,
    min_aspect: float = 1.5,
    max_aspect: float = 14.0,
    min_area_ratio: float = 0.00012,
    max_area_ratio: float = 0.04,
) -> list[TextBox]:
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    mser.setDelta(5)
    mser.setMinArea(max(40, int((h * w) * float(min_area_ratio))))
    mser.setMaxArea(max(120, int((h * w) * float(max_area_ratio))))
    _, bboxes = mser.detectRegions(gray)
    if bboxes is None:
        return []

    boxes: list[TextBox] = []
    for x, y, bw, bh in bboxes:
        if bw < max(8, int(min_w)) or bh < max(6, int(min_h)):
            continue
        aspect = float(bw) / float(max(1, bh))
        if aspect < float(min_aspect) or aspect > float(max_aspect):
            continue
        clipped = clip_box(x, y, x + bw, y + bh, w, h)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        edges = cv2.Canny(patch, 80, 180)
        edge_density = float(cv2.countNonZero(edges)) / float(max(1, patch.shape[0] * patch.shape[1]))
        conf = max(0.0, min(1.0, (edge_density - 0.02) / 0.22))
        if conf < min_conf:
            continue
        boxes.append(TextBox(x1, y1, x2, y2, conf, detector="mser"))
    return nms(boxes, iou_threshold=0.25)[: max(1, int(max_boxes))]


def detect_text_boxes(
    bgr: np.ndarray,
    mode: str = "hybrid",
    min_conf: float = 0.30,
    east_min_conf: float = 0.55,
    mser_min_conf: float = 0.35,
    max_boxes: int = 8,
    min_w: int = 24,
    min_h: int = 12,
    min_aspect: float = 1.2,
    max_aspect: float = 16.0,
    east_net: Any | None = None,
) -> list[TextBox]:
    if bgr is None or bgr.size == 0:
        return []
    normalized_mode = str(mode or "hybrid").strip().lower()
    use_mser = normalized_mode in ("hybrid", "mser")
    use_east = normalized_mode in ("hybrid", "east") and east_net is not None

    boxes: list[TextBox] = []
    if use_mser:
        boxes.extend(
            detect_text_boxes_mser(
                bgr,
                min_conf=max(float(min_conf), float(mser_min_conf)),
                max_boxes=max_boxes,
                min_w=min_w,
                min_h=min_h,
                min_aspect=max(1.0, min_aspect),
                max_aspect=max(2.0, max_aspect),
            )
        )
    if use_east:
        boxes.extend(
            detect_text_boxes_east(
                bgr,
                east_net=east_net,
                min_conf=max(float(min_conf), float(east_min_conf)),
                max_boxes=max_boxes,
                min_w=min_w,
                min_h=min_h,
                min_aspect=min_aspect,
                max_aspect=max_aspect,
            )
        )
    return nms(boxes, iou_threshold=0.22)[: max(1, int(max_boxes))]
