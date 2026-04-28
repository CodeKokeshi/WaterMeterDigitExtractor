"""OpenCV helpers for finding a digit strip and refining it to 4 points."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def bbox_to_quad(bbox_xyxy: list[float] | tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )


def _clamp_bbox(
    bbox_xyxy: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = int(np.clip(x1, 0, max(width - 1, 0)))
    y1 = int(np.clip(y1, 0, max(height - 1, 0)))
    x2 = int(np.clip(x2, 0, max(width, 0)))
    y2 = int(np.clip(y2, 0, max(height, 0)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _expand_bbox(
    bbox_xyxy: tuple[int, int, int, int],
    width: int,
    height: int,
    margin_ratio: float,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox_xyxy
    bw = x2 - x1
    bh = y2 - y1
    mx = int(round(bw * margin_ratio))
    my = int(round(bh * margin_ratio))
    return _clamp_bbox((x1 - mx, y1 - my, x2 + mx, y2 + my), width, height)


def _build_candidate_masks(gray: np.ndarray) -> list[np.ndarray]:
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    block = max(15, ((min(h, w) // 8) * 2) + 1)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        6,
    )

    kernel_w = max(17, min(w // 5, 61))
    kernel_h = max(3, min(h // 14, 11))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    opened = cv2.morphologyEx(
        closed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    edges = cv2.Canny(blur, 40, 140)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    edge_mask = cv2.dilate(edges, edge_kernel, iterations=1)
    edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    merged = cv2.bitwise_or(opened, edge_mask)
    return [opened, edge_mask, merged]


def _score_contour(
    contour: np.ndarray,
    crop_shape: tuple[int, int],
) -> tuple[float, np.ndarray] | None:
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), _angle = rect
    if w <= 1.0 or h <= 1.0:
        return None

    long_side = max(w, h)
    short_side = min(w, h)
    aspect = long_side / max(short_side, 1.0)
    rect_area = w * h
    contour_area = cv2.contourArea(contour)
    crop_h, crop_w = crop_shape
    area_ratio = rect_area / max(float(crop_w * crop_h), 1.0)
    fill_ratio = contour_area / max(rect_area, 1.0)

    if aspect < 2.0 or aspect > 12.0:
        return None
    if short_side < max(8.0, crop_h * 0.035):
        return None
    if not (0.01 <= area_ratio <= 0.85):
        return None
    if fill_ratio < 0.20:
        return None

    crop_center = np.array([crop_w / 2.0, crop_h / 2.0], dtype=np.float32)
    center = np.array([cx, cy], dtype=np.float32)
    center_dist = float(np.linalg.norm(center - crop_center))
    max_dist = max(float(np.linalg.norm(crop_center)), 1.0)
    center_score = 1.0 - min(center_dist / max_dist, 1.0)

    aspect_score = 1.0 - min(abs(aspect - 5.0) / 5.0, 1.0)
    fill_score = min(max((fill_ratio - 0.20) / 0.60, 0.0), 1.0)
    area_score = 1.0 - min(abs(area_ratio - 0.12) / 0.20, 1.0)

    score = (
        (0.45 * aspect_score)
        + (0.20 * fill_score)
        + (0.20 * center_score)
        + (0.15 * area_score)
    )
    return score, cv2.boxPoints(rect).astype(np.float32)


def find_digit_strip_quad(
    image: np.ndarray,
    search_bbox_xyxy: tuple[int, int, int, int] | None = None,
    margin_ratio: float = 0.12,
) -> dict[str, Any] | None:
    if image is None or image.size == 0:
        return None

    full_h, full_w = image.shape[:2]
    if search_bbox_xyxy is None:
        crop_bbox = (0, 0, full_w, full_h)
    else:
        crop_bbox = _expand_bbox(search_bbox_xyxy, full_w, full_h, margin_ratio)
        if crop_bbox is None:
            return None

    x1, y1, x2, y2 = crop_bbox
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    best_score = -1.0
    best_box: np.ndarray | None = None
    for mask in _build_candidate_masks(gray):
        contours, _hier = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 50.0:
                continue
            scored = _score_contour(contour, gray.shape[:2])
            if scored is None:
                continue
            score, box = scored
            if score > best_score:
                best_score = score
                best_box = box

    if best_box is None:
        return None

    best_box[:, 0] += float(x1)
    best_box[:, 1] += float(y1)
    best_box[:, 0] = np.clip(best_box[:, 0], 0, max(full_w - 1, 0))
    best_box[:, 1] = np.clip(best_box[:, 1], 0, max(full_h - 1, 0))

    return {
        "points": order_points(best_box),
        "score": float(best_score),
        "search_bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
    }
