"""Detect individual digit-like quadrilateral regions in a full image."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


TARGET_SIZE = 28


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def _line_coefficients(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return np.array(
        [
            float(p1[1] - p2[1]),
            float(p2[0] - p1[0]),
            float((p1[0] * p2[1]) - (p2[0] * p1[1])),
        ],
        dtype=np.float32,
    )


def _parallel_score(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a[:2]) * np.linalg.norm(b[:2])), 1e-6)
    return abs(float(np.dot(a[:2], b[:2])) / denom)


def _bbox_area_xyxy(bbox_xyxy: list[float] | tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection_area_xyxy(
    box_a: list[float] | tuple[float, float, float, float],
    box_b: list[float] | tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    return max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)


def _expand_bbox_xyxy(
    bbox_xyxy: list[float] | tuple[float, float, float, float],
    image_shape: tuple[int, ...],
    margin_ratio: float = 0.08,
) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    image_h, image_w = image_shape[:2]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    mx = bw * margin_ratio
    my = bh * margin_ratio
    return [
        float(np.clip(x1 - mx, 0.0, max(float(image_w - 1), 0.0))),
        float(np.clip(y1 - my, 0.0, max(float(image_h - 1), 0.0))),
        float(np.clip(x2 + mx, 1.0, max(float(image_w), 1.0))),
        float(np.clip(y2 + my, 1.0, max(float(image_h), 1.0))),
    ]


def _candidate_is_inside_limiters(
    bbox_xywh: list[int],
    limiter_boxes: list[list[float]] | None,
) -> tuple[bool, int]:
    if not limiter_boxes:
        return True, -1

    x, y, w, h = [int(v) for v in bbox_xywh]
    candidate_xyxy = [float(x), float(y), float(x + w), float(y + h)]
    candidate_area = max(_bbox_area_xyxy(candidate_xyxy), 1.0)
    cx = float(x + (w / 2.0))
    cy = float(y + (h / 2.0))

    best_index = -1
    best_overlap = 0.0
    for idx, limiter in enumerate(limiter_boxes):
        overlap = _intersection_area_xyxy(candidate_xyxy, limiter) / candidate_area
        if overlap > best_overlap:
            best_overlap = overlap
            best_index = idx
        lx1, ly1, lx2, ly2 = [float(v) for v in limiter]
        if lx1 <= cx <= lx2 and ly1 <= cy <= ly2 and overlap >= 0.40:
            return True, idx

    return best_overlap >= 0.70, best_index


def classify_quad_shape(points: np.ndarray) -> str:
    ordered = order_points(points.astype(np.float32))
    tl, tr, br, bl = ordered
    top = _line_coefficients(tl, tr)
    bottom = _line_coefficients(bl, br)
    left = _line_coefficients(tl, bl)
    right = _line_coefficients(tr, br)

    top_bottom_parallel = _parallel_score(top, bottom) > 0.96
    left_right_parallel = _parallel_score(left, right) > 0.96

    if top_bottom_parallel and left_right_parallel:
        top_vec = tr - tl
        left_vec = bl - tl
        denom = max(float(np.linalg.norm(top_vec) * np.linalg.norm(left_vec)), 1e-6)
        cosine = abs(float(np.dot(top_vec, left_vec)) / denom)
        if cosine < 0.15:
            return "rectangle"
        return "parallelogram"

    if top_bottom_parallel or left_right_parallel:
        return "trapezium"

    return "quadrilateral"


def _build_digit_masks(gray: np.ndarray) -> list[np.ndarray]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape[:2]
    block = max(15, ((min(h, w) // 20) * 2) + 1)

    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        4,
    )

    otsu_threshold, otsu = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    _ = otsu_threshold

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(2, w // 240), max(3, h // 80)),
    )
    adaptive_clean = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    otsu_clean = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    edge_mask = cv2.Canny(blur, 40, 120)
    edge_mask = cv2.dilate(
        edge_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    return [adaptive_clean, otsu_clean, edge_mask]


def _contour_to_quad(contour: np.ndarray) -> np.ndarray | None:
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0.0:
        return None

    hull = cv2.convexHull(contour)
    approx = cv2.approxPolyDP(hull, 0.04 * perimeter, True)
    if len(approx) == 4:
        return order_points(approx.reshape(4, 2).astype(np.float32))

    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), _angle = rect
    if w <= 1.0 or h <= 1.0:
        return None
    return order_points(cv2.boxPoints(rect).astype(np.float32))


def extract_digit_patch(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    ordered = order_points(points.astype(np.float32))
    dst = np.array(
        [
            [0.0, 0.0],
            [TARGET_SIZE - 1.0, 0.0],
            [TARGET_SIZE - 1.0, TARGET_SIZE - 1.0],
            [0.0, TARGET_SIZE - 1.0],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, matrix, (TARGET_SIZE, TARGET_SIZE))

    if len(warped.shape) == 3:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    else:
        gray = warped.copy()

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    patch = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    return patch


def detect_digit_candidates(
    image: np.ndarray,
    limiter_boxes: list[list[float]] | None = None,
    max_candidates: int = 40,
) -> list[dict[str, Any]]:
    if image is None or image.size == 0:
        return []

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    image_h, image_w = gray.shape[:2]
    image_area = float(max(image_h * image_w, 1))
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()

    expanded_limiters = [
        _expand_bbox_xyxy(limiter, image.shape, margin_ratio=0.08)
        for limiter in (limiter_boxes or [])
    ]

    for mask in _build_digit_masks(gray):
        contours, _hier = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour_area = float(cv2.contourArea(contour))
            if contour_area < image_area * 0.00015 or contour_area > image_area * 0.08:
                continue

            quad = _contour_to_quad(contour)
            if quad is None:
                continue

            rect = cv2.minAreaRect(contour)
            (_cx, _cy), (w, h), _angle = rect
            if w <= 2.0 or h <= 2.0:
                continue

            long_side = max(float(w), float(h))
            short_side = min(float(w), float(h))
            aspect_ratio = long_side / max(short_side, 1.0)
            if aspect_ratio > 4.5:
                continue

            x, y, bw, bh = cv2.boundingRect(np.round(quad).astype(np.int32))
            if bw <= 3 or bh <= 3:
                continue
            inside_limiters, limiter_index = _candidate_is_inside_limiters(
                [int(x), int(y), int(bw), int(bh)],
                expanded_limiters,
            )
            if not inside_limiters:
                continue
            bbox_key = (int(x), int(y), int(bw), int(bh))
            if bbox_key in seen:
                continue
            seen.add(bbox_key)

            fill_ratio = contour_area / max(float(w * h), 1.0)
            if fill_ratio < 0.08 or fill_ratio > 0.95:
                continue

            center_x = float(np.mean(quad[:, 0]))
            center_y = float(np.mean(quad[:, 1]))
            area_ratio = contour_area / image_area
            base_score = (
                (0.35 * min(fill_ratio / 0.45, 1.0))
                + (0.25 * (1.0 - min(abs(aspect_ratio - 1.8) / 1.8, 1.0)))
                + (0.20 * min(area_ratio / 0.01, 1.0))
                + (0.20 * (1.0 - min(abs((center_y / max(float(image_h), 1.0)) - 0.5) / 0.5, 1.0)))
            )

            candidates.append(
                {
                    "points": quad,
                    "bbox_xywh": [int(x), int(y), int(bw), int(bh)],
                    "shape_type": classify_quad_shape(quad),
                    "aspect_ratio": float(aspect_ratio),
                    "fill_ratio": float(fill_ratio),
                    "center": [center_x, center_y],
                    "height": float(short_side if bh < bw else long_side),
                    "width": float(long_side),
                    "detector_score": float(base_score),
                    "limiter_index": int(limiter_index),
                }
            )

    if not candidates:
        return []

    candidates.sort(
        key=lambda item: (
            float(item.get("center", [0.0, 0.0])[0]),
            float(item.get("center", [0.0, 0.0])[1]),
        )
    )

    for idx, candidate in enumerate(candidates):
        center_x, center_y = [float(v) for v in candidate.get("center", [0.0, 0.0])]
        height = max(float(candidate.get("height", 1.0)), 1.0)
        width = max(float(candidate.get("width", 1.0)), 1.0)
        neighbor_count = 0
        for jdx, other in enumerate(candidates):
            if idx == jdx:
                continue
            other_center_x, other_center_y = [float(v) for v in other.get("center", [0.0, 0.0])]
            other_height = max(float(other.get("height", 1.0)), 1.0)
            if int(candidate.get("limiter_index", -1)) != int(other.get("limiter_index", -1)):
                continue
            if abs(other_center_y - center_y) > height * 0.75:
                continue
            if abs(other_center_x - center_x) > width * 3.0:
                continue
            height_ratio = max(height, other_height) / max(min(height, other_height), 1.0)
            if height_ratio > 1.45:
                continue
            neighbor_count += 1

        candidate["neighbor_count"] = neighbor_count
        candidate["final_score"] = float(candidate["detector_score"]) + min(neighbor_count * 0.18, 0.72)
        candidate["image"] = extract_digit_patch(image, np.asarray(candidate["points"], dtype=np.float32))

    ranked = sorted(
        candidates,
        key=lambda item: (
            float(item.get("final_score", 0.0)),
            float(item.get("detector_score", 0.0)),
        ),
        reverse=True,
    )

    return ranked[:max(int(max_candidates), 1)]
