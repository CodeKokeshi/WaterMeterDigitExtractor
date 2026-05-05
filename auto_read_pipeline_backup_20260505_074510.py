"""Helpers for auto-finding and auto-reading digit strips."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


FINAL_W = 140
FINAL_H = 28


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


@dataclass(frozen=True)
class SlidingWindow:
    x: int
    y: int
    size: int


def generate_bbox_candidates(
    bbox_xyxy: tuple[float, float, float, float],
    image_shape: tuple[int, ...],
) -> list[dict[str, Any]]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    h, w = image_shape[:2]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    variants = [
        ("base", 1.00, 1.00, 0.00, 0.00),
        ("expand", 1.12, 1.18, 0.00, 0.00),
        ("expand_more", 1.22, 1.30, 0.00, 0.00),
        ("tight", 0.92, 0.90, 0.00, 0.00),
        ("wide", 1.20, 1.00, 0.00, 0.00),
        ("tall", 1.00, 1.18, 0.00, 0.00),
        ("left", 1.08, 1.10, -0.06, 0.00),
        ("right", 1.08, 1.10, 0.06, 0.00),
        ("up", 1.08, 1.10, 0.00, -0.06),
        ("down", 1.08, 1.10, 0.00, 0.06),
    ]

    results: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for name, sx, sy, ox, oy in variants:
        cand_cx = cx + (width * ox)
        cand_cy = cy + (height * oy)
        cand_w = width * sx
        cand_h = height * sy
        cand_x1 = int(round(cand_cx - (cand_w / 2.0)))
        cand_y1 = int(round(cand_cy - (cand_h / 2.0)))
        cand_x2 = int(round(cand_cx + (cand_w / 2.0)))
        cand_y2 = int(round(cand_cy + (cand_h / 2.0)))
        cand_x1 = int(np.clip(cand_x1, 0, max(w - 1, 0)))
        cand_y1 = int(np.clip(cand_y1, 0, max(h - 1, 0)))
        cand_x2 = int(np.clip(cand_x2, 1, max(w, 1)))
        cand_y2 = int(np.clip(cand_y2, 1, max(h, 1)))
        if cand_x2 <= cand_x1 or cand_y2 <= cand_y1:
            continue
        digest = (cand_x1, cand_y1, cand_x2, cand_y2)
        if digest in seen:
            continue
        seen.add(digest)
        results.append({
            "name": name,
            "bbox_xyxy": [cand_x1, cand_y1, cand_x2, cand_y2],
        })
    return results


def build_sliding_windows(
    image_width: int,
    image_height: int,
    min_window_size: int = 320,
    max_windows: int = 24,
) -> list[SlidingWindow]:
    short_side = min(image_width, image_height)
    if short_side <= 0:
        return []

    min_size = min(short_side, max(64, min_window_size))
    scale_factors = [1.0, 0.85, 0.70, 0.55, 0.40]
    sizes: list[int] = []
    for factor in scale_factors:
        size = int(round(short_side * factor))
        size = max(min_size, min(size, short_side))
        if size not in sizes:
            sizes.append(size)

    windows: list[SlidingWindow] = []
    for size in sizes:
        if size >= image_width and size >= image_height:
            windows.append(SlidingWindow(0, 0, short_side))
            continue

        step = max(int(round(size * 0.35)), 1)
        max_x = max(image_width - size, 0)
        max_y = max(image_height - size, 0)
        xs = list(range(0, max_x + 1, step))
        ys = list(range(0, max_y + 1, step))
        if not xs or xs[-1] != max_x:
            xs.append(max_x)
        if not ys or ys[-1] != max_y:
            ys.append(max_y)

        for y in ys:
            for x in xs:
                windows.append(SlidingWindow(x, y, size))

    deduped: list[SlidingWindow] = []
    seen: set[tuple[int, int, int]] = set()
    for window in windows:
        key = (window.x, window.y, window.size)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(window)

    if len(deduped) <= max_windows:
        return deduped

    image_center = np.array([image_width / 2.0, image_height / 2.0], dtype=np.float32)

    def window_rank(window: SlidingWindow) -> tuple[float, int]:
        center = np.array(
            [window.x + (window.size / 2.0), window.y + (window.size / 2.0)],
            dtype=np.float32,
        )
        dist = float(np.linalg.norm(center - image_center))
        return dist, -window.size

    return sorted(deduped, key=window_rank)[:max_windows]


def generate_strip_candidates(strip: np.ndarray) -> list[dict[str, Any]]:
    if strip is None or strip.size == 0:
        return []

    if len(strip.shape) == 3:
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    else:
        gray = strip.copy()

    base = cv2.resize(gray, (FINAL_W, FINAL_H), interpolation=cv2.INTER_AREA)
    canvas = cv2.resize(base, (FINAL_W * 4, FINAL_H * 4), interpolation=cv2.INTER_CUBIC)
    center = (canvas.shape[1] / 2.0, canvas.shape[0] / 2.0)

    variants = [
        ("base", 0.0, 0.00, 1.00, 0),
        ("rot_m2", -2.0, 0.00, 1.00, 0),
        ("rot_p2", 2.0, 0.00, 1.00, 0),
        ("shear_m", 0.0, -0.08, 1.00, 0),
        ("shear_p", 0.0, 0.08, 1.00, 0),
        ("tight", 0.0, 0.00, 0.92, 0),
        ("wide", 0.0, 0.00, 1.08, 0),
        ("up", 0.0, 0.00, 1.00, -4),
        ("down", 0.0, 0.00, 1.00, 4),
    ]

    candidates: list[dict[str, Any]] = []
    seen_hashes: set[bytes] = set()
    for name, angle, shear, scale_y, shift_y in variants:
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
        shear_matrix = np.array(
            [[1.0, shear, 0.0], [0.0, scale_y, float(shift_y)], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        final_matrix = (shear_matrix @ matrix)[:2, :]
        warped = cv2.warpAffine(
            canvas,
            final_matrix,
            (canvas.shape[1], canvas.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        candidate = cv2.resize(warped, (FINAL_W, FINAL_H), interpolation=cv2.INTER_AREA)
        digest = candidate.tobytes()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        candidates.append({"name": name, "image": candidate})

    return candidates


def generate_quad_candidates(
    points: np.ndarray,
    image_shape: tuple[int, ...],
) -> list[dict[str, Any]]:
    if points is None or getattr(points, "shape", None) != (4, 2):
        return []

    ordered = order_points(points.astype(np.float32))
    tl, tr, br, bl = ordered

    top_vec = tr - tl
    bottom_vec = br - bl
    left_vec = bl - tl
    right_vec = br - tr

    top_len = max(float(np.linalg.norm(top_vec)), 1.0)
    bottom_len = max(float(np.linalg.norm(bottom_vec)), 1.0)
    left_len = max(float(np.linalg.norm(left_vec)), 1.0)
    right_len = max(float(np.linalg.norm(right_vec)), 1.0)

    top_u = top_vec / top_len
    bottom_u = bottom_vec / bottom_len
    left_u = left_vec / left_len
    right_u = right_vec / right_len

    avg_w = (top_len + bottom_len) / 2.0
    avg_h = (left_len + right_len) / 2.0
    h, w = image_shape[:2]

    variants = [
        ("base", 0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
        ("tight_h", -0.08, -0.08, 0.00, 0.00, 0.00, 0.00),
        ("loose_h", 0.10, 0.10, 0.00, 0.00, 0.00, 0.00),
        ("top_up", 0.12, 0.00, 0.00, 0.00, 0.00, 0.00),
        ("bottom_down", 0.00, 0.12, 0.00, 0.00, 0.00, 0.00),
        ("left_wide", 0.00, 0.00, 0.08, 0.00, 0.00, 0.00),
        ("right_wide", 0.00, 0.00, 0.00, 0.08, 0.00, 0.00),
        ("wide_all", 0.00, 0.00, 0.06, 0.06, 0.00, 0.00),
        ("shear_l", 0.00, 0.00, 0.00, 0.00, -0.06, 0.06),
        ("shear_r", 0.00, 0.00, 0.00, 0.00, 0.06, -0.06),
        ("top_left", 0.08, 0.00, 0.05, 0.00, -0.03, 0.03),
        ("top_right", 0.08, 0.00, 0.00, 0.05, 0.03, -0.03),
    ]

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for (
        name,
        top_expand,
        bottom_expand,
        left_expand,
        right_expand,
        top_shift,
        bottom_shift,
    ) in variants:
        candidate = np.array(
            [
                tl - (top_u * avg_w * left_expand) - (left_u * avg_h * top_expand) + (top_u * avg_h * top_shift),
                tr + (top_u * avg_w * right_expand) - (right_u * avg_h * top_expand) + (top_u * avg_h * top_shift),
                br + (bottom_u * avg_w * right_expand) + (right_u * avg_h * bottom_expand) + (bottom_u * avg_h * bottom_shift),
                bl - (bottom_u * avg_w * left_expand) + (left_u * avg_h * bottom_expand) + (bottom_u * avg_h * bottom_shift),
            ],
            dtype=np.float32,
        )

        candidate[:, 0] = np.clip(candidate[:, 0], 0, max(w - 1, 0))
        candidate[:, 1] = np.clip(candidate[:, 1], 0, max(h - 1, 0))
        candidate = order_points(candidate)
        digest = tuple(int(round(v)) for v in candidate.ravel())
        if digest in seen:
            continue
        seen.add(digest)
        candidates.append({"name": name, "points": candidate})

    return candidates


def vote_prediction_candidates(
    candidates: list[dict[str, Any]],
    top_k: int = 5,
) -> dict[str, Any]:
    if not candidates:
        return {
            "voted_label": "",
            "top_candidates": [],
            "vote_details": [],
        }

    ranked = sorted(
        candidates,
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )[:max(int(top_k), 1)]

    if not ranked:
        return {
            "voted_label": "",
            "top_candidates": [],
            "vote_details": [],
        }

    max_len = max(len(str(item.get("predicted_label", ""))) for item in ranked)
    voted_chars: list[str] = []
    vote_details: list[dict[str, Any]] = []

    for idx in range(max_len):
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for candidate in ranked:
            label = str(candidate.get("predicted_label", ""))
            if idx >= len(label):
                continue
            char = label[idx]
            score = float(candidate.get("score", 0.0))
            totals[char] = totals.get(char, 0.0) + score
            counts[char] = counts.get(char, 0) + 1

        if not totals:
            voted_chars.append("?")
            continue

        winner = max(
            totals.keys(),
            key=lambda key: (counts.get(key, 0), totals.get(key, 0.0), key),
        )
        voted_chars.append(winner)
        vote_details.append(
            {
                "position": idx,
                "winner": winner,
                "totals": totals,
                "counts": counts,
            }
        )

    return {
        "voted_label": "".join(voted_chars),
        "top_candidates": ranked,
        "vote_details": vote_details,
    }


def score_prediction_confidences(confidences: list[float]) -> float:
    if not confidences:
        return 0.0
    scores = np.asarray(confidences, dtype=np.float32)
    mean_score = float(scores.mean())
    min_score = float(scores.min())
    return (0.65 * mean_score) + (0.35 * min_score)
