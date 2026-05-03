"""
Streamlined guidebox-only DigitExtractor UI with canonical workspace framing.
"""

from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QColor,
    QImage,
    QPainter,
    QPen,
    QResizeEvent,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialogButtonBox,
    QFileDialog,
    QDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from guidebox_workspace import (
    WorkspaceFrame,
    WorkspaceRenderResult,
    build_default_workspace_frame,
    clone_workspace_frame,
    render_workspace_view,
    update_workspace_frame_for_size,
)
from main_deprecated import (
    IMAGE_EXTENSIONS,
    NUM_SEGMENTS,
    ROI_RAW_DIR_NAME,
    PreviewWidget,
    UNREADABLE_FOLDER_NAME,
    UNREADABLE_LABEL_CHAR,
    is_digit_or_unreadable_label,
    prepare_guidebox_strip,
    read_image_any,
)

# Folder names that should never be treated as a "class" folder when the
# balancer scans a parent directory. They are auxiliary outputs of this
# tool, hidden / system folders, or virtual environments.
_BALANCER_IGNORED_SUBFOLDERS = {
    ROI_RAW_DIR_NAME,
    "__pycache__",
    ".git",
    ".github",
    ".venv",
    "venv-314",
    "trained_models",
    "trained_yolo_models",
    "fixed_models",
    "assets",
}

# Caption-style class folder names supported by the trainer.
# Examples that match:
#   "0 - Full"
#   "0 - Going 1"
#   "0 - Rolling from 9"
#   "1 - Full"
#   "1 - Going 2"
#   "1 - Rolling from 0"
# The leading single digit (group 1) is the class label; the rest of the
# name (group 2) is the caption ("Full", "Going 1", "Rolling from 9", ...).
# The same regex is used by lenet_backend._iter_digit_folders so the UI's
# validation matches what the training backend will actually consume.
_CAPTION_FOLDER_RE = re.compile(r"^(\d)\s*-\s*(.+)$")
from digit_ml_commands import (
    MlCommandWorker,
    build_lenet_train_command,
    build_lenet_predict_command,
    get_ml_backend_script_path,
    get_python_version,
    is_supported_tensorflow_backend,
    write_temp_strip_image,
)
from digit_ml_dialogs import LeNetTrainingDialog

# Model folder defaults
LENET_MODEL_DIR_NAME = "trained_models"


# =============================================================================
# Adaptive Dataset Balancer (no skew / no rotation / no perspective)
# =============================================================================
#
# Goals (per user requirements):
#   * Detect ALL class folders, including special ones like
#       "1 - Full", "1 - Going 2", "1 - Rolling from 0"
#     in addition to the regular 0..9 / Unreadable.
#   * Balance every class to a target count chosen by the user (default:
#     median of the current distribution — gentle, neither inflates tiny
#     classes too aggressively nor punishes large ones too hard).
#   * Augmentation is allowed to: thicken / thin the strokes adaptively
#     (driven by the white-pixel ratio of each image), add adaptive
#     graininess (sigma scales with how "empty" the canvas is), nudge by
#     ±1 px, and apply mild brightness/contrast/blur jitter.
#   * Augmentation is NOT allowed to: skew, rotate, flip, or apply
#     perspective transforms.
#   * Downsampling uses farthest-point sampling on tiny image features so
#     that the most VISUALLY DISTINCT samples are kept and only the most
#     redundant near-duplicates are dropped — this protects rare/unique
#     images even when a class has to be heavily downsampled.
#   * A tolerance band lets near-balanced classes pass through unchanged,
#     so we don't punish already-good data for the sake of an exact target.
# =============================================================================


def _adaptive_no_skew_augment(img: np.ndarray) -> np.ndarray:
    """Diversify a digit image WITHOUT any skew / rotation / perspective.

    Adaptive operations (selection driven by current image statistics):
      * Stroke thicken / thin via 3x3 cross morphology, biased by the
        white-pixel ratio of the input image.
      * Adaptive Gaussian + sparse salt-and-pepper grain (sigma scales
        inversely with ink ratio so denser strokes stay readable).
      * Tiny ±1 px translation (centroid jitter, NOT skew).
      * Subtle brightness / contrast jitter.
      * Occasional mild blur or mild sharpen for edge-texture variety.

    The result is the same shape and dtype as the input.
    """
    aug = img.copy()
    if aug.ndim == 3:
        aug = cv2.cvtColor(aug, cv2.COLOR_BGR2GRAY)
    h, w = aug.shape[:2]

    # 1) Coverage analysis: what proportion of the canvas is "ink".
    ink_mask = aug > 127
    ink_ratio = float(ink_mask.sum()) / (h * w + 1e-6)

    rng = np.random.default_rng()
    # Healthy MNIST-ish band — outside this we bias morphology.
    target_min, target_max = 0.10, 0.30
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # 2) Adaptive thicken / thin (NEVER skew).
    if ink_ratio < target_min:
        # Stroke too thin → bias toward thickening.
        iters = 1 if ink_ratio > target_min * 0.5 else 2
        if rng.random() < 0.85:
            aug = cv2.dilate(aug, kernel, iterations=iters)
    elif ink_ratio > target_max:
        # Stroke too thick → bias toward thinning.
        iters = 1 if ink_ratio < target_max * 1.5 else 2
        if rng.random() < 0.85:
            aug = cv2.erode(aug, kernel, iterations=iters)
    else:
        # Healthy band: ~40% chance of a single-iter morph, else leave alone.
        roll = rng.random()
        if roll < 0.20:
            aug = cv2.dilate(aug, kernel, iterations=1)
        elif roll < 0.40:
            aug = cv2.erode(aug, kernel, iterations=1)

    # 3) Adaptive grain — sigma proportional to (1 - ink_ratio) so that
    # high-coverage shapes get gentler noise (avoids muddying), and low-
    # coverage shapes get a little more variety.
    base_sigma = float(np.clip(6.0 * (1.0 - ink_ratio), 2.0, 6.0))
    noise = rng.normal(0.0, base_sigma, (h, w)).astype(np.float32)
    aug_f = aug.astype(np.float32) + noise
    # Sparse salt-and-pepper (~0.5%–1.5% of pixels). Half salt, half pepper.
    sp_density = float(rng.uniform(0.005, 0.015))
    n_sp = int(sp_density * h * w)
    if n_sp > 0:
        ys = rng.integers(0, h, size=n_sp)
        xs = rng.integers(0, w, size=n_sp)
        half = n_sp // 2
        aug_f[ys[:half], xs[:half]] = 255.0
        aug_f[ys[half:], xs[half:]] = 0.0
    aug = np.clip(aug_f, 0, 255).astype(np.uint8)

    # 4) Tiny translation (±1 px) — NOT skew.
    tx = int(rng.integers(-1, 2))
    ty = int(rng.integers(-1, 2))
    if tx or ty:
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        aug = cv2.warpAffine(
            aug, M, (w, h), borderMode=cv2.BORDER_REPLICATE
        )

    # 5) Subtle brightness / contrast jitter.
    alpha = float(rng.uniform(0.92, 1.08))   # contrast
    beta = float(rng.uniform(-6.0, 6.0))     # brightness
    aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

    # 6) Occasional gentle blur OR mild sharpen (rare, edge-texture variety).
    roll = rng.random()
    if roll < 0.15:
        aug = cv2.GaussianBlur(aug, (3, 3), sigmaX=0.5)
    elif roll < 0.30:
        ksharp = np.array(
            [[0, -1, 0],
             [-1, 5, -1],
             [0, -1, 0]],
            dtype=np.float32,
        )
        aug = cv2.filter2D(aug, -1, ksharp)

    return aug


def _ensure_digit_canvas_28x28(img: np.ndarray) -> np.ndarray:
    """Normalize arbitrary input to a uint8 28x28 grayscale canvas."""
    out = img.copy()
    if out.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    if out.shape[:2] != (28, 28):
        out = cv2.resize(out, (28, 28), interpolation=cv2.INTER_AREA)
    return out


def _analyze_digit_body(img: np.ndarray) -> dict:
    """Collect safety metrics that gate thinning / soft hollowing."""
    out = _ensure_digit_canvas_28x28(img)
    h, w = out.shape[:2]
    ink_mask = out > 60
    ink_count = int(ink_mask.sum())
    if ink_count == 0:
        return {
            "ink_ratio": 0.0,
            "bbox_w": 0,
            "bbox_h": 0,
            "fill_ratio": 0.0,
            "edge_margin": 0,
            "ink_mean": 0.0,
            "bg_mean": float(out.mean()),
            "component_count": 0,
            "interior_ratio": 0.0,
            "tier": "fragile",
        }

    ys, xs = np.where(ink_mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox_w = x1 - x0 + 1
    bbox_h = y1 - y0 + 1
    bbox_area = max(1, bbox_w * bbox_h)
    ink_ratio = float(ink_count) / float(h * w)
    fill_ratio = float(ink_count) / float(bbox_area)
    edge_margin = int(min(x0, y0, w - 1 - x1, h - 1 - y1))
    ink_mean = float(out[ink_mask].mean()) if ink_count > 0 else 0.0
    bg_mean = float(out[~ink_mask].mean()) if ink_count < h * w else 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    eroded_mask = cv2.erode(
        ink_mask.astype(np.uint8) * 255, kernel, iterations=1
    ) > 0
    interior_ratio = (
        float(eroded_mask.sum()) / float(ink_count) if ink_count > 0 else 0.0
    )

    num_labels, _labels = cv2.connectedComponents(
        ink_mask.astype(np.uint8), connectivity=8
    )
    component_count = max(0, int(num_labels) - 1)

    tier = "fragile"
    if (
        ink_ratio >= 0.21
        and fill_ratio >= 0.42
        and interior_ratio >= 0.22
        and bbox_w >= 8
        and bbox_h >= 11
        and ink_mean >= 135.0
        and component_count <= 2
        and edge_margin >= 1
    ):
        tier = "strong"
    elif (
        ink_ratio >= 0.11
        and fill_ratio >= 0.28
        and bbox_w >= 6
        and bbox_h >= 9
        and ink_mean >= 105.0
        and component_count <= 3
    ):
        tier = "medium"

    if (
        ink_ratio < 0.08
        or fill_ratio < 0.18
        or bbox_w < 5
        or bbox_h < 7
        or ink_mean < 90.0
        or component_count >= 4
    ):
        tier = "fragile"

    return {
        "ink_ratio": ink_ratio,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "fill_ratio": fill_ratio,
        "edge_margin": edge_margin,
        "ink_mean": ink_mean,
        "bg_mean": bg_mean,
        "component_count": component_count,
        "interior_ratio": interior_ratio,
        "tier": tier,
    }


def _adaptive_digit_body_style(
    img: np.ndarray,
    analysis: dict | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Thin / soften dense digits, but back off on fragile 28x28 samples."""
    out = _ensure_digit_canvas_28x28(img)
    if rng is None:
        rng = np.random.default_rng()
    if analysis is None:
        analysis = _analyze_digit_body(out)

    tier = analysis.get("tier", "fragile")
    if tier == "fragile":
        return out, analysis

    ink_mask = out > 60
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    eroded = cv2.erode(out, kernel, iterations=1)
    candidate = out.copy()

    if tier == "strong":
        candidate = cv2.addWeighted(out, 0.72, eroded, 0.28, 0.0)
        dt = cv2.distanceTransform(
            ink_mask.astype(np.uint8) * 255, cv2.DIST_L2, 3
        )
        interior_mask = dt >= 1.35
        if int(interior_mask.sum()) >= 8:
            reduction = float(rng.uniform(18.0, 34.0))
            candidate_f = candidate.astype(np.float32)
            candidate_f[interior_mask] -= reduction
            if rng.random() < 0.65:
                sparse = (
                    interior_mask
                    & (rng.random(candidate.shape[:2]) < rng.uniform(0.06, 0.14))
                )
                candidate_f[sparse] -= float(rng.uniform(6.0, 14.0))
            candidate = np.clip(candidate_f, 0, 255).astype(np.uint8)
    else:
        candidate = cv2.addWeighted(out, 0.84, eroded, 0.16, 0.0)
        dt = cv2.distanceTransform(
            ink_mask.astype(np.uint8) * 255, cv2.DIST_L2, 3
        )
        interior_mask = dt >= 1.8
        if int(interior_mask.sum()) >= 6:
            reduction = float(rng.uniform(8.0, 18.0))
            candidate_f = candidate.astype(np.float32)
            candidate_f[interior_mask] -= reduction
            candidate = np.clip(candidate_f, 0, 255).astype(np.uint8)

    after = _analyze_digit_body(candidate)
    min_ink_ratio = max(0.055, float(analysis["ink_ratio"]) * 0.68)
    max_components = max(int(analysis["component_count"]) + 1, 3)
    if after["ink_ratio"] < min_ink_ratio:
        return out, analysis
    if after["component_count"] > max_components:
        return out, analysis
    if after["ink_mean"] < max(80.0, float(analysis["ink_mean"]) - 42.0):
        return out, analysis

    return candidate, after


# =============================================================================
# Border Wobble Augmenter (28x28 cell-border emulation)
# =============================================================================
#
# Goal: take a clean MNIST-style 28x28 digit (online dataset, no border)
# and add the look of a real-world meter cell:
#
#   * a thin rectangle around the digit, ~1 px wide,
#   * placed just OUTSIDE the digit's ink bounding box
#       (tight 1-2 px padding on left/right/bottom; ~2-4 px on top
#        because the digit slides through the cell and the top/bottom
#        gap is naturally wider in real captures),
#   * mid-gray (~140-220) — never bright white, so it never dominates
#     the digit itself,
#   * partially erased: ~10-30% of the border pixels are randomly
#     skipped along each side,
#   * mildly wobbly: each pixel along a side can shift +/-1 px
#     perpendicular to the line, so the border is never perfectly
#     straight,
#   * digit-safe: the border is composed via a max() blend so an
#     existing bright ink pixel is NEVER darkened by border placement,
#   * occasional corner ticks and a faint dusting of gray grain on the
#     bare canvas mimic printing/imaging texture from the reference
#     screenshots.
#
# Operates strictly at 28x28 — anything else is INTER_AREA-resized to 28
# before processing so dataset sizes stay consistent with our LeNet
# input shape.
# =============================================================================


def _apply_wobbly_borders_legacy_bbox_noise(
    img: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Wrap a 28x28 digit with a faint, broken, wobbly rectangle border.

    See the section comment above for design rationale. Returns a uint8
    grayscale image of the same shape as the input.
    """
    if img is None or img.size == 0:
        return img
    out = img.copy()
    if out.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    h, w = out.shape[:2]
    if rng is None:
        rng = np.random.default_rng()

    # 1) Locate ink. Threshold is permissive so we still catch faint strokes.
    ink_mask = out > 60
    ys, xs = np.where(ink_mask)
    if ys.size < 3:
        # Nothing to box around — fall back to a small centered rect so
        # the image still picks up a visible cell-style border.
        y0, y1 = 6, h - 7
        x0, x1 = 6, w - 7
    else:
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())

    # 2) Adaptive padding. Top is a touch wider than sides/bottom.
    pad_left = int(rng.integers(1, 3))
    pad_right = int(rng.integers(1, 3))
    pad_bot = int(rng.integers(1, 3))
    pad_top = int(rng.integers(2, 5))

    bx0 = max(0, x0 - pad_left)
    by0 = max(0, y0 - pad_top)
    bx1 = min(w - 1, x1 + pad_right)
    by1 = min(h - 1, y1 + pad_bot)

    if bx1 - bx0 < 4 or by1 - by0 < 4:
        return out

    # 3) Per-image base intensity and per-side gap probability. The
    # narrow (140-220) band keeps borders subordinate to the digit.
    base_intensity = int(rng.integers(140, 220))
    gap_p = float(rng.uniform(0.10, 0.30))

    def _stamp(px: int, py: int) -> None:
        if 0 <= px < w and 0 <= py < h:
            jitter = int(rng.integers(-25, 26))
            val = int(np.clip(base_intensity + jitter, 90, 240))
            # max-blend: never darken an existing brighter (ink) pixel.
            cur = int(out[py, px])
            if val > cur:
                out[py, px] = val

    # 4) Top edge (with +/-1 px perpendicular wobble).
    for x in range(bx0, bx1 + 1):
        if rng.random() < gap_p:
            continue
        wob = int(rng.integers(-1, 2))
        _stamp(x, by0 + wob)

    # 5) Bottom edge.
    for x in range(bx0, bx1 + 1):
        if rng.random() < gap_p:
            continue
        wob = int(rng.integers(-1, 2))
        _stamp(x, by1 + wob)

    # 6) Left edge.
    for y in range(by0, by1 + 1):
        if rng.random() < gap_p:
            continue
        wob = int(rng.integers(-1, 2))
        _stamp(bx0 + wob, y)

    # 7) Right edge.
    for y in range(by0, by1 + 1):
        if rng.random() < gap_p:
            continue
        wob = int(rng.integers(-1, 2))
        _stamp(bx1 + wob, y)

    # 8) Optional corner ticks — small printing imperfections at the
    # rectangle joins.
    for cx, cy in ((bx0, by0), (bx1, by0), (bx0, by1), (bx1, by1)):
        if rng.random() < 0.4:
            dx = int(rng.integers(-1, 2))
            dy = int(rng.integers(-1, 2))
            _stamp(cx + dx, cy + dy)

    # 9) Faint dusting of gray grain on the bare canvas outside the
    # digit, mimicking printed-ink texture seen in the reference.
    n_grain = max(1, int(0.012 * h * w))
    gys = rng.integers(0, h, size=n_grain)
    gxs = rng.integers(0, w, size=n_grain)
    for gy, gx in zip(gys, gxs):
        if int(out[gy, gx]) < 60:
            val = int(rng.integers(50, 110))
            if val > int(out[gy, gx]):
                out[gy, gx] = val

    return out


def _apply_wobbly_borders_legacy_adaptive_noise(
    img: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Wrap a digit with an adaptive faint, broken, wobbly rectangle border."""
    if img is None or img.size == 0:
        return img
    if rng is None:
        rng = np.random.default_rng()

    styled, analysis = _adaptive_digit_body_style(img, rng=rng)
    out = styled.copy()
    h, w = out.shape[:2]

    ink_mask = out > 60
    ys, xs = np.where(ink_mask)
    if ys.size < 3:
        y0, y1 = 6, h - 7
        x0, x1 = 6, w - 7
    else:
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())

    edge_margin = int(analysis.get("edge_margin", 0))
    tier = analysis.get("tier", "fragile")
    if edge_margin <= 0:
        pad_left = pad_right = pad_bot = 1
        pad_top = 2
        wobble_span = 0
    elif edge_margin == 1:
        pad_left = pad_right = pad_bot = 1
        pad_top = 2
        wobble_span = 0 if tier == "fragile" else 1
    else:
        pad_left = int(rng.integers(1, 3))
        pad_right = int(rng.integers(1, 3))
        pad_bot = int(rng.integers(1, 3))
        pad_top = int(rng.integers(2, 5))
        wobble_span = 1

    bx0 = max(0, x0 - pad_left)
    by0 = max(0, y0 - pad_top)
    bx1 = min(w - 1, x1 + pad_right)
    by1 = min(h - 1, y1 + pad_bot)
    if bx1 - bx0 < 4 or by1 - by0 < 4:
        return out

    if tier == "strong":
        base_intensity = int(rng.integers(150, 210))
        gap_p = float(rng.uniform(0.14, 0.26))
    elif tier == "medium":
        base_intensity = int(rng.integers(145, 205))
        gap_p = float(rng.uniform(0.12, 0.22))
    else:
        base_intensity = int(rng.integers(140, 195))
        gap_p = float(rng.uniform(0.08, 0.18))

    def _stamp(px: int, py: int) -> None:
        if 0 <= px < w and 0 <= py < h:
            jitter = int(rng.integers(-18, 19))
            val = int(np.clip(base_intensity + jitter, 90, 235))
            if val > int(out[py, px]):
                out[py, px] = val

    def _wobble() -> int:
        if wobble_span <= 0:
            return 0
        return int(rng.integers(-wobble_span, wobble_span + 1))

    for x in range(bx0, bx1 + 1):
        if rng.random() >= gap_p:
            _stamp(x, by0 + _wobble())
    for x in range(bx0, bx1 + 1):
        if rng.random() >= gap_p:
            _stamp(x, by1 + _wobble())
    for y in range(by0, by1 + 1):
        if rng.random() >= gap_p:
            _stamp(bx0 + _wobble(), y)
    for y in range(by0, by1 + 1):
        if rng.random() >= gap_p:
            _stamp(bx1 + _wobble(), y)

    for cx, cy in ((bx0, by0), (bx1, by0), (bx0, by1), (bx1, by1)):
        if rng.random() < (0.30 if tier == "fragile" else 0.45):
            _stamp(cx + _wobble(), cy + _wobble())

    n_grain = max(1, int(0.010 * h * w))
    if tier == "strong":
        n_grain = max(n_grain, 10)
    gys = rng.integers(0, h, size=n_grain)
    gxs = rng.integers(0, w, size=n_grain)
    for gy, gx in zip(gys, gxs):
        if int(out[gy, gx]) < 60:
            val = int(rng.integers(45, 95))
            if val > int(out[gy, gx]):
                out[gy, gx] = val

    return out


_AUTHENTIC_FRAME_PRESETS = (
    {
        "name": "clean_rect",
        "dx": 0,
        "dy": 0,
        "clear_l": 2,
        "clear_r": 2,
        "clear_t": 3,
        "clear_b": 2,
        "top_mode": "full",
        "left_shift": 0,
        "right_shift": 0,
        "bottom_shift": 0,
        "max_gaps": 1,
        "corner_loss_p": 0.15,
    },
    {
        "name": "open_top",
        "dx": 0,
        "dy": 0,
        "clear_l": 2,
        "clear_r": 2,
        "clear_t": 3,
        "clear_b": 2,
        "top_mode": "soft_open",
        "left_shift": 0,
        "right_shift": 0,
        "bottom_shift": 0,
        "max_gaps": 1,
        "corner_loss_p": 0.20,
    },
    {
        "name": "broken_side",
        "dx": 0,
        "dy": 0,
        "clear_l": 2,
        "clear_r": 2,
        "clear_t": 4,
        "clear_b": 2,
        "top_mode": "full",
        "left_shift": 0,
        "right_shift": 0,
        "bottom_shift": 0,
        "max_gaps": 2,
        "corner_loss_p": 0.30,
    },
    {
        "name": "offset_rails",
        "dx": 0,
        "dy": 0,
        "clear_l": 2,
        "clear_r": 3,
        "clear_t": 3,
        "clear_b": 2,
        "top_mode": "full",
        "left_shift": -1,
        "right_shift": 1,
        "bottom_shift": 0,
        "max_gaps": 1,
        "corner_loss_p": 0.20,
    },
)


def _get_digit_bbox(img: np.ndarray, threshold: int = 60) -> tuple[int, int, int, int]:
    """Return the tight ink bbox, or a centered fallback when the canvas is empty."""
    mask = img > threshold
    ys, xs = np.where(mask)
    if ys.size < 3:
        return 8, 5, 19, 23
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _stamp_frame_point(
    out: np.ndarray,
    px: int,
    py: int,
    base_intensity: int,
    rng: np.random.Generator,
) -> None:
    """Write one border pixel with small intensity drift, never darkening ink."""
    h, w = out.shape[:2]
    if 0 <= px < w and 0 <= py < h:
        val = int(np.clip(base_intensity + int(rng.integers(-8, 9)), 120, 210))
        if val > int(out[py, px]):
            out[py, px] = val


def _apply_contiguous_gaps(
    points: list[tuple[int, int]],
    max_gaps: int,
    rng: np.random.Generator,
    side_len: int,
) -> list[tuple[int, int]]:
    """Remove short contiguous runs to mimic scan dropouts, not pixel noise."""
    if len(points) <= 4 or max_gaps <= 0:
        return points
    keep = np.ones(len(points), dtype=bool)
    n_gaps = int(rng.integers(0, max_gaps + 1))
    for _ in range(n_gaps):
        gap_len = int(rng.integers(1, 4 if side_len >= 10 else 3))
        lo = 1
        hi = max(lo + 1, len(points) - gap_len - 1)
        if hi <= lo:
            continue
        start = int(rng.integers(lo, hi))
        keep[start:start + gap_len] = False
    return [pt for idx, pt in enumerate(points) if keep[idx]]


def _build_horizontal_points(
    x0: int,
    x1: int,
    y: int,
    mode: str,
    side_shift: int,
    max_gaps: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    points = [(x, y + side_shift) for x in range(x0, x1 + 1)]
    if mode == "soft_open" and len(points) >= 10:
        center = len(points) // 2
        gap_span = int(rng.integers(2, 5))
        open_lo = max(1, center - gap_span)
        open_hi = min(len(points) - 1, center + gap_span)
        points = [
            pt for idx, pt in enumerate(points)
            if not (open_lo <= idx < open_hi and rng.random() < 0.85)
        ]
    return _apply_contiguous_gaps(points, max_gaps, rng, x1 - x0 + 1)


def _build_vertical_points(
    x: int,
    y0: int,
    y1: int,
    side_shift: int,
    max_gaps: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    points = [(x + side_shift, y) for y in range(y0, y1 + 1)]
    return _apply_contiguous_gaps(points, max_gaps, rng, y1 - y0 + 1)


def _compute_authentic_frame_rect(
    img: np.ndarray,
    preset: dict,
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    """Choose a constrained frame rect with safer spacing than raw bbox expansion."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = _get_digit_bbox(img)
    bbox_w = x1 - x0 + 1
    bbox_h = y1 - y0 + 1
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)

    clear_l = preset["clear_l"] + int(rng.integers(0, 2))
    clear_r = preset["clear_r"] + int(rng.integers(0, 2))
    clear_t = preset["clear_t"] + int(rng.integers(0, 2))
    clear_b = preset["clear_b"] + int(rng.integers(0, 2))

    frame_w = int(np.clip(bbox_w + clear_l + clear_r, 13, 21))
    frame_h = int(np.clip(bbox_h + clear_t + clear_b, 18, 27))
    left = int(round(cx - (frame_w - 1) / 2.0 + preset["dx"]))
    top = int(round(cy - (frame_h - 1) / 2.0 + preset["dy"]))
    right = left + frame_w - 1
    bottom = top + frame_h - 1

    desired_left = x0 - clear_l
    desired_right = x1 + clear_r
    desired_top = y0 - clear_t
    desired_bottom = y1 + clear_b
    if left > desired_left:
        shift = left - desired_left
        left -= shift
        right -= shift
    if right < desired_right:
        shift = desired_right - right
        left += shift
        right += shift
    if top > desired_top:
        shift = top - desired_top
        top -= shift
        bottom -= shift
    if bottom < desired_bottom:
        shift = desired_bottom - bottom
        top += shift
        bottom += shift

    if left < 1:
        right += 1 - left
        left = 1
    if right > w - 2:
        shift = right - (w - 2)
        left -= shift
        right -= shift
    if top < 1:
        bottom += 1 - top
        top = 1
    if bottom > h - 1:
        shift = bottom - (h - 1)
        top -= shift
        bottom -= shift

    left = int(np.clip(left, 0, w - 5))
    right = int(np.clip(right, left + 4, w - 1))
    top = int(np.clip(top, 0, h - 6))
    bottom = int(np.clip(bottom, top + 5, h - 1))
    return left, top, right, bottom


def _draw_authentic_frame(
    out: np.ndarray,
    rect: tuple[int, int, int, int],
    preset: dict,
    base_intensity: int,
    rng: np.random.Generator,
) -> None:
    """Draw a frame with segment-level defects and light rail offsets."""
    left, top, right, bottom = rect
    max_gaps = int(preset["max_gaps"])
    top_points = _build_horizontal_points(
        left,
        right,
        top,
        preset["top_mode"],
        0,
        1 if preset["top_mode"] == "full" else 0,
        rng,
    )
    bottom_points = _build_horizontal_points(
        left,
        right,
        bottom,
        "full",
        int(preset["bottom_shift"]),
        max_gaps,
        rng,
    )
    left_points = _build_vertical_points(
        left,
        top,
        bottom,
        int(preset["left_shift"]),
        max_gaps,
        rng,
    )
    right_points = _build_vertical_points(
        right,
        top,
        bottom,
        int(preset["right_shift"]),
        max_gaps,
        rng,
    )

    for px, py in top_points + bottom_points + left_points + right_points:
        _stamp_frame_point(out, px, py, base_intensity, rng)

    for cx, cy in ((left, top), (right, top), (left, bottom), (right, bottom)):
        if rng.random() >= float(preset["corner_loss_p"]):
            _stamp_frame_point(out, cx, cy, base_intensity, rng)


def _add_frame_grain(
    out: np.ndarray,
    frame_rect: tuple[int, int, int, int],
    rng: np.random.Generator,
) -> None:
    """Add faint background speckle outside the digit and near the frame."""
    left, top, right, bottom = frame_rect
    h, w = out.shape[:2]
    n_grain = int(rng.integers(4, 10))
    for _ in range(n_grain):
        gx = int(rng.integers(max(0, left - 1), min(w, right + 2)))
        gy = int(rng.integers(max(0, top - 1), min(h, bottom + 1)))
        if int(out[gy, gx]) < 45:
            val = int(rng.integers(28, 70))
            if val > int(out[gy, gx]):
                out[gy, gx] = val


def _apply_wobbly_borders(
    img: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a more authentic meter-frame border around a 28x28 digit."""
    if img is None or img.size == 0:
        return img
    if rng is None:
        rng = np.random.default_rng()

    analysis = _analyze_digit_body(img)
    out = _ensure_digit_canvas_28x28(img)
    if analysis.get("tier") == "strong" and rng.random() < 0.55:
        out, _ = _adaptive_digit_body_style(img, analysis=analysis, rng=rng)
    elif analysis.get("tier") == "medium" and rng.random() < 0.20:
        out, _ = _adaptive_digit_body_style(img, analysis=analysis, rng=rng)

    preset = _AUTHENTIC_FRAME_PRESETS[
        int(rng.integers(0, len(_AUTHENTIC_FRAME_PRESETS)))
    ]
    frame_rect = _compute_authentic_frame_rect(out, preset, rng)

    if analysis.get("tier") == "strong":
        base_intensity = int(rng.integers(150, 182))
    elif analysis.get("tier") == "medium":
        base_intensity = int(rng.integers(146, 176))
    else:
        base_intensity = int(rng.integers(142, 170))

    _draw_authentic_frame(out, frame_rect, preset, base_intensity, rng)
    _add_frame_grain(out, frame_rect, rng)
    return out


# =============================================================================
# Context-aware reader (Full / Going X / Rolling from X / Middle of A and B)
# =============================================================================
#
# The trained LeNet has 10 output classes (digits 0-9). During dataset
# preparation the caption variants all map to the same label (the
# leading digit), so the model learns to output the *dominant* digit.
# Captions the reader can surface from the per-segment softmax:
#
#   * "<top1> - Full"
#       -> top-1 dominates clearly (top-2 share is small). Bias = top.
#   * "<top1> - Going <X>"
#       -> top-2 is the NEXT digit on the wheel:  X = (top1 + 1) mod 10.
#       -> The wheel is rolling forward; X is climbing into view from
#         the bottom of the cell. Bias = top.
#   * "<top1> - Rolling from <X>"
#       -> top-2 is the PREVIOUS digit on the wheel: X = (top1 - 1) mod 10.
#       -> top1 has just rolled in from the bottom; X is the previous
#         reading still fading out at the top. Bias = bottom.
#   * "Middle of <A> and <B>"
#       -> top-2 is a wheel-neighbor of top-1 *and* the two together hold
#         most of the probability mass with a near-tie ratio. The wheel
#         is sitting halfway between two adjacent digits (the user's
#         "bias to middle" / confusing case). The pair is always written
#         in meter-progression order: A is the lower wheel position and
#         B is the next one (e.g. "Middle of 9 and 0" — wraps cyclically,
#         "Middle of 1 and 2", etc.).
#   * "Uncertain"
#       -> top-1 is itself low confidence, OR the runner-up is not a
#         wheel neighbor. The model genuinely does not know.
#
# Digit wheels are CYCLIC: 0's neighbors are 9 (previous) and 1 (next),
# 9's neighbors are 8 (previous) and 0 (next).
# =============================================================================

# Confidence thresholds. Calibrated for the LeNet softmax, easy to tune
# from one place if the trained model behaves differently.
READER_FULL_THRESHOLD = 0.85       # top-1 alone wins outright
READER_UNCERTAIN_THRESHOLD = 0.50  # top-1 below this -> Uncertain
READER_TRANSITION_RATIO = 0.20     # top2/top1 above this -> Going/Rolling/Middle
READER_MIDDLE_RATIO = 0.70         # top2/top1 above this AND neighbors -> Middle
READER_MIDDLE_MASS = 0.60          # p1 + p2 must clear this for Middle


def infer_digit_context(
    probs: list[float] | np.ndarray,
) -> dict:
    """Classify a single 10-way softmax into digit + caption context.

    Returns a dict with:
        digit       int    top-1 predicted digit (0..9)
        caption     str    "Full" | "Going X" | "Rolling from X"
                            | "Middle of A and B" | "Uncertain"
        top1_prob   float  softmax of top-1 digit
        top2_digit  int    runner-up digit
        top2_prob   float  softmax of runner-up
        ratio       float  top2_prob / top1_prob
        pair        tuple[int, int] | None
                            (A, B) for Middle (in meter-progression order),
                            otherwise None.
    """
    probs_arr = np.asarray(probs, dtype=np.float32).reshape(-1)
    if probs_arr.size != 10:
        return {
            "digit": -1,
            "caption": "Uncertain",
            "top1_prob": 0.0,
            "top2_digit": -1,
            "top2_prob": 0.0,
            "ratio": 0.0,
            "pair": None,
        }

    order = np.argsort(probs_arr)[::-1]   # descending
    top1 = int(order[0])
    top2 = int(order[1])
    p1 = float(probs_arr[top1])
    p2 = float(probs_arr[top2])
    ratio = (p2 / p1) if p1 > 1e-9 else 0.0

    next_digit = (top1 + 1) % 10
    prev_digit = (top1 - 1) % 10
    is_neighbor = top2 == next_digit or top2 == prev_digit

    # Resolve the meter-progression pair (A, B) when top2 is a neighbor.
    # Convention: A is the earlier wheel position, B is the next one
    # (cyclic). For top1=9, top2=0  -> (9, 0). For top1=0, top2=9 -> (9, 0).
    pair: tuple[int, int] | None = None
    if is_neighbor:
        if top2 == next_digit:
            pair = (top1, top2)
        else:
            pair = (top2, top1)

    caption: str

    # 1) Middle has priority over the uncertainty floor: a near-tie on
    #    two adjacent digits with most of the mass concentrated there is
    #    a confident "halfway" call, not a mush.
    if (
        is_neighbor
        and ratio >= READER_MIDDLE_RATIO
        and (p1 + p2) >= READER_MIDDLE_MASS
        and pair is not None
    ):
        caption = f"Middle of {pair[0]} and {pair[1]}"
    # 2) Below the uncertainty floor the model itself is not committed.
    elif p1 < READER_UNCERTAIN_THRESHOLD:
        caption = "Uncertain"
    # 3) Strong top-1 -> Full.
    elif p1 >= READER_FULL_THRESHOLD or ratio < READER_TRANSITION_RATIO:
        caption = "Full"
    # 4) Transition: a moderate runner-up that IS a wheel neighbor.
    elif is_neighbor and pair is not None:
        if top2 == next_digit:
            caption = f"Going {top2}"
        else:
            caption = f"Rolling from {top2}"
    else:
        # Runner-up is not a wheel neighbor -> genuine ambiguity.
        caption = "Uncertain"

    return {
        "digit": top1,
        "caption": caption,
        "top1_prob": p1,
        "top2_digit": top2,
        "top2_prob": p2,
        "ratio": ratio,
        "pair": pair,
    }


def _diverse_downsample_paths(
    image_paths: list[Path],
    target_count: int,
) -> list[Path]:
    """Farthest-point sampling that preserves visually distinct images.

    Builds a tiny 32x32 normalized feature for each image and greedily
    picks the next sample that is farthest from any already-kept sample.
    The dropped images are guaranteed to be the most redundant near-
    duplicates relative to the kept set — UNIQUE images are protected.
    Falls back to random fill if all remaining points are duplicates of
    existing selections.
    """
    feats: list[tuple[Path, np.ndarray]] = []
    for p in image_paths:
        img = read_image_any(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        thumb = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        thumb = cv2.GaussianBlur(thumb, (3, 3), 0)
        thumb = thumb.astype(np.float32) / 255.0
        feats.append((p, thumb.flatten()))

    if len(feats) == 0:
        return []
    if len(feats) <= target_count:
        return [p for p, _ in feats]

    paths = [p for p, _ in feats]
    feat_mat = np.stack([f for _, f in feats], axis=0).astype(np.float32)
    n = feat_mat.shape[0]

    rng = np.random.default_rng()
    start = int(rng.integers(0, n))
    selected = [start]
    diffs = feat_mat - feat_mat[start]
    min_dist = np.einsum("ij,ij->i", diffs, diffs)
    min_dist[start] = -1.0  # never re-pick the same index

    while len(selected) < target_count:
        nxt = int(np.argmax(min_dist))
        if min_dist[nxt] <= 0.0:
            # Remaining images are duplicates of the kept set; fill the
            # last few slots with random unselected indices so we still
            # hit the target count (very rare branch).
            taken = set(selected)
            remaining = [i for i in range(n) if i not in taken]
            rng.shuffle(remaining)
            need = target_count - len(selected)
            selected.extend(remaining[:need])
            break
        selected.append(nxt)
        diffs = feat_mat - feat_mat[nxt]
        d = np.einsum("ij,ij->i", diffs, diffs)
        min_dist = np.minimum(min_dist, d)
        min_dist[nxt] = -1.0

    return [paths[i] for i in selected]


class BalanceDialog(QDialog):
    """Adaptive balancer dialog that detects ALL class folders.

    Beyond plain 0..9 / Unreadable, this also handles special variants
    such as '1 - Full', '1 - Going 2', and '1 - Rolling from 0'.
    """

    def __init__(self, category_counts: dict[str, int], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Adaptive Dataset Balancer (no-skew)")
        self.setModal(True)
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("<b>Detected Class Distribution:</b>"))

        stats_text = ""
        total_images = sum(category_counts.values())
        for cat, count in sorted(category_counts.items()):
            stats_text += f"'{cat}': {count} images\n"
        stats_label = QLabel(stats_text or "(no classes detected)")
        stats_label.setStyleSheet(
            "font-family: monospace; color: #9fc5e8;"
        )
        layout.addWidget(stats_label)

        layout.addWidget(QLabel(
            f"<i>Total images: {total_images} across "
            f"{len(category_counts)} classes</i>\n"
        ))

        counts = list(category_counts.values())
        if not counts:
            self.reject()
            return

        median_val = int(np.median(counts))
        mean_val = int(np.mean(counts))
        max_val = max(counts)
        min_val = max(min(counts), 1)

        # Recommendation: median is gentler than mean. Avoid pushing tiny
        # classes by more than ~20x (overfitting territory).
        recommended = median_val if median_val >= 100 else max(
            100, min(median_val * 2 if median_val else 100, mean_val)
        )
        max_safe_inflate = max(min_val * 20, recommended)
        recommended = min(recommended, max_safe_inflate)

        info = QLabel(
            f"<b>Adaptive Analysis:</b><br>"
            f"&bull; Largest class: {max_val}<br>"
            f"&bull; Smallest class: {min_val}<br>"
            f"&bull; Mean: {mean_val} &nbsp;&bull;&nbsp; Median: {median_val}<br><br>"
            f"<i>Recommended target: {recommended}</i><br>"
            "<i>(Median is the gentle bet — it neither inflates tiny "
            "classes by more than ~20x nor culls large classes too "
            "hard.)</i>"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("<b>Target count per class:</b>"))
        self._target_spin = QSpinBox()
        self._target_spin.setRange(10, 100000)
        self._target_spin.setValue(int(recommended))
        self._target_spin.setFixedWidth(120)
        target_row.addWidget(self._target_spin)
        target_row.addStretch()
        layout.addLayout(target_row)

        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("Keep-as-is tolerance (%):"))
        self._tol_spin = QSpinBox()
        self._tol_spin.setRange(0, 200)
        self._tol_spin.setValue(20)
        self._tol_spin.setFixedWidth(80)
        self._tol_spin.setToolTip(
            "If a class is within this percentage of the target, it is\n"
            "kept as-is (no down/up sampling). Protects valuable, near-\n"
            "balanced data from unnecessary modification."
        )
        tol_row.addWidget(self._tol_spin)
        tol_row.addStretch()
        layout.addLayout(tol_row)

        notice = QLabel(
            "<span style='color:#f0d27a;'>Augmentation: thicken / thin "
            "(adaptive), grain (adaptive), ±1 px nudge, tiny "
            "brightness/contrast jitter.<br>"
            "No skew. No rotation. No perspective. No flips.</span>"
        )
        notice.setWordWrap(True)
        layout.addWidget(notice)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_target(self) -> int:
        return int(self._target_spin.value())

    def get_tolerance(self) -> float:
        return float(self._tol_spin.value()) / 100.0


class WobblyBorderDialog(QDialog):
    """Configuration dialog for the Border Wobble augmenter.

    Lets the user pick how many bordered variants to generate per source
    image, whether to also copy the originals into the output, and
    whether to preserve the input folder tree (per-class subfolders) or
    flatten everything into a single output directory.
    """

    def __init__(self, source_count: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Wobbly Borders")
        self.setModal(True)
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            f"<b>Source images detected:</b> {source_count}"
        ))

        info = QLabel(
            "<i>Generates several believable meter-frame candidates per "
            "28x28 digit using constrained border presets, safer spacing "
            "from the number, and subtle scan-like defects instead of "
            "noisy hand-drawn wobble.</i>"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        n_row = QHBoxLayout()
        n_row.addWidget(QLabel("<b>Variants per source image:</b>"))
        self._variants_spin = QSpinBox()
        self._variants_spin.setRange(1, 20)
        self._variants_spin.setValue(4)
        self._variants_spin.setFixedWidth(80)
        self._variants_spin.setToolTip(
            "Each input image is saved with this many constrained border "
            "candidates. Higher values give you more believable options "
            "to review."
        )
        n_row.addWidget(self._variants_spin)
        n_row.addStretch()
        layout.addLayout(n_row)

        self._keep_originals = QCheckBox(
            "Also copy the original (unbordered) image to the output"
        )
        self._keep_originals.setChecked(False)
        layout.addWidget(self._keep_originals)

        self._preserve_classes = QCheckBox(
            "Preserve class subfolder structure"
        )
        self._preserve_classes.setChecked(True)
        self._preserve_classes.setToolTip(
            "If on, the output mirrors the input folder tree (each "
            "class stays in its own subfolder). If off, every bordered "
            "image is dumped into a single flat output folder."
        )
        layout.addWidget(self._preserve_classes)

        notice = QLabel(
            "<span style='color:#f0d27a;'>Borders use a few authentic "
            "frame presets with cleaner rails, short broken segments, "
            "and safer clearance from the digit. Dense digits may still "
            "be softened slightly, while thin digits are mostly left "
            "alone. Original ink is never darkened.</span>"
        )
        notice.setWordWrap(True)
        layout.addWidget(notice)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_variants(self) -> int:
        return int(self._variants_spin.value())

    def keep_originals(self) -> bool:
        return self._keep_originals.isChecked()

    def preserve_classes(self) -> bool:
        return self._preserve_classes.isChecked()


class GuideboxWorkspaceView(QWidget):
    frame_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 420)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self._source_image: np.ndarray | None = None
        self._frame: WorkspaceFrame | None = None
        self._render_result: WorkspaceRenderResult | None = None
        self._dragging = False
        self._last_mouse_pos = QPoint()

    def _current_workspace_size(self) -> tuple[int, int]:
        rect = self.contentsRect()
        return max(rect.width(), 1), max(rect.height(), 1)

    def set_source_image(self, image: np.ndarray | None):
        self._source_image = None if image is None else image.copy()
        if self._frame is None:
            self._frame = build_default_workspace_frame(self._current_workspace_size())
        else:
            self._frame = update_workspace_frame_for_size(self._frame, self._current_workspace_size())
        self._rerender(emit_signal=False)

    def reset_view(self, rotation_deg: float = 0.0):
        self._frame = build_default_workspace_frame(self._current_workspace_size(), rotation_deg=rotation_deg)
        self._rerender(emit_signal=False)

    def set_workspace_frame(self, frame: WorkspaceFrame, emit_signal: bool = True):
        self._frame = clone_workspace_frame(frame)
        self._frame = update_workspace_frame_for_size(self._frame, self._current_workspace_size())
        self._rerender(emit_signal=emit_signal)

    def get_workspace_frame(self) -> WorkspaceFrame | None:
        if self._frame is None:
            return None
        return clone_workspace_frame(self._frame)

    def fit_to_view(self):
        if self._frame is None:
            return
        self._frame.scale = 1.0
        self._frame.translate_x = 0.0
        self._frame.translate_y = 0.0
        self._rerender()

    def set_rotation(self, angle_deg: float):
        if self._frame is None:
            self._frame = build_default_workspace_frame(self._current_workspace_size(), rotation_deg=angle_deg)
        else:
            self._frame.rotation_deg = float(angle_deg)
        self._rerender()

    def get_rotation(self) -> float:
        return 0.0 if self._frame is None else float(self._frame.rotation_deg)

    def get_guidebox_crop(self) -> np.ndarray | None:
        if self._render_result is None or self._render_result.guidebox_crop is None:
            return None
        return self._render_result.guidebox_crop.copy()

    def _rerender(self, emit_signal: bool = True):
        if self._frame is None:
            self._frame = build_default_workspace_frame(self._current_workspace_size())
        else:
            self._frame = update_workspace_frame_for_size(self._frame, self._current_workspace_size())
        self._render_result = render_workspace_view(self._source_image, self._frame)
        self.update()
        if emit_signal:
            self.frame_changed.emit()

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        if self._frame is not None:
            self._rerender()

    def wheelEvent(self, event: QWheelEvent):
        if self._frame is None:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self._frame.scale = max(0.05, min(self._frame.scale * factor, 50.0))
        self._rerender()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._frame is not None:
            self._dragging = True
            self._last_mouse_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and self._frame is not None:
            delta = event.pos() - self._last_mouse_pos
            self._frame.translate_x += float(delta.x())
            self._frame.translate_y += float(delta.y())
            self._last_mouse_pos = event.pos()
            self._rerender()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(24, 24, 24))

        if self._render_result is not None:
            workspace_image = self._render_result.workspace_image
            if workspace_image is not None and workspace_image.size:
                rgb = cv2.cvtColor(workspace_image, cv2.COLOR_BGR2RGB)
                image = QImage(
                    rgb.data,
                    rgb.shape[1],
                    rgb.shape[0],
                    rgb.shape[1] * 3,
                    QImage.Format.Format_RGB888,
                ).copy()
                painter.drawImage(self.rect(), image)

        guidebox_rect = self._guidebox_rect_widget()
        shade = QColor(0, 0, 0, 110)
        full_rect = QRectF(self.rect())
        if guidebox_rect.isValid():
            top_rect = QRectF(full_rect.left(), full_rect.top(), full_rect.width(), max(0.0, guidebox_rect.top() - full_rect.top()))
            bottom_rect = QRectF(full_rect.left(), guidebox_rect.bottom(), full_rect.width(), max(0.0, full_rect.bottom() - guidebox_rect.bottom()))
            left_rect = QRectF(full_rect.left(), guidebox_rect.top(), max(0.0, guidebox_rect.left() - full_rect.left()), guidebox_rect.height())
            right_rect = QRectF(guidebox_rect.right(), guidebox_rect.top(), max(0.0, full_rect.right() - guidebox_rect.right()), guidebox_rect.height())
            painter.fillRect(top_rect, shade)
            painter.fillRect(bottom_rect, shade)
            painter.fillRect(left_rect, shade)
            painter.fillRect(right_rect, shade)
        else:
            painter.fillRect(full_rect, shade)

        border_pen = QPen(QColor(0, 220, 255, 230), 2)
        painter.setPen(border_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(guidebox_rect, 6.0, 6.0)

        column_pen = QPen(QColor(255, 255, 255, 90), 1, Qt.PenStyle.DashLine)
        painter.setPen(column_pen)
        for idx in range(1, NUM_SEGMENTS):
            x = guidebox_rect.left() + (guidebox_rect.width() * idx / NUM_SEGMENTS)
            painter.drawLine(int(round(x)), int(round(guidebox_rect.top())), int(round(x)), int(round(guidebox_rect.bottom())))

        painter.setPen(QColor(220, 245, 255, 230))
        text_rect = QRectF(
            guidebox_rect.left(),
            max(guidebox_rect.top() - 28.0, 4.0),
            guidebox_rect.width(),
            22.0,
        )
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, "5:1 guidebox")

    def _guidebox_rect_widget(self) -> QRectF:
        if self._frame is None:
            return QRectF()
        x, y, w, h = self._frame.guidebox_rect_workspace
        return QRectF(x, y, w, h)


class StreamlinedMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DigitExtractor — Streamlined Guidebox")
        self.resize(1500, 900)

        self._output_dir = ""
        self._batch_mode = False
        self._batch_template: WorkspaceFrame | None = None
        self._batch_processed = 0
        self._batch_saved_segments = 0
        self._batch_roi_raw = 0
        self._batch_errors = 0
        self._batch_skipped = 0
        self._auto_preview_timer = QTimer(self)
        self._auto_preview_timer.setSingleShot(True)
        self._auto_preview_timer.timeout.connect(self._preview_current_guidebox)

        # --- LeNet ML state ---
        self._ml_worker: MlCommandWorker | None = None
        self._ml_progress: QProgressDialog | None = None
        self._ml_progress_title: str = "Working"
        self._ml_log_lines: list[str] = []
        self._ml_backend_python: str = sys.executable
        self._testing_model_path: str = ""
        self._last_trained_model_dir: str = str(Path.cwd() / LENET_MODEL_DIR_NAME)
        self._last_tflite_model_dir: str = self._last_trained_model_dir

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        sidebar = QGroupBox("Images")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)

        self._btn_open_folder = QPushButton("Open Folder")
        self._btn_set_output = QPushButton("Set Output Folder")
        self._output_label = QLabel("Output: not set")
        self._output_label.setWordWrap(True)
        self._file_list = QListWidget()

        sidebar_layout.addWidget(self._btn_open_folder)
        sidebar_layout.addWidget(self._btn_set_output)
        sidebar_layout.addWidget(self._output_label)
        sidebar_layout.addWidget(self._file_list, stretch=1)
        sidebar.setMaximumWidth(320)
        outer.addWidget(sidebar, stretch=0)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(8)

        top_bar = QGroupBox("Alignment")
        top_layout = QGridLayout(top_bar)

        self._btn_fit = QPushButton("Fit View")
        self._btn_preview = QPushButton("Preview Strip")
        self._btn_save_current = QPushButton("Save Current")
        self._batch_checkbox = QCheckBox("Batch Mode")
        self._lenet_checkbox = QCheckBox("Show LeNet")

        self._rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self._rotation_slider.setRange(0, 359)
        self._rotation_slider.setValue(0)
        self._rotation_slider.setEnabled(False)
        self._rotation_value = QLabel("0°")
        self._rotation_value.setFixedWidth(40)

        top_layout.addWidget(self._btn_fit, 0, 0)
        top_layout.addWidget(self._btn_preview, 0, 1)
        top_layout.addWidget(self._btn_save_current, 0, 2)
        top_layout.addWidget(self._batch_checkbox, 0, 3)
        top_layout.addWidget(self._lenet_checkbox, 0, 4)
        top_layout.addWidget(QLabel("Rotate"), 1, 0)
        top_layout.addWidget(self._rotation_slider, 1, 1, 1, 2)
        top_layout.addWidget(self._rotation_value, 1, 3)
        center_layout.addWidget(top_bar, stretch=0)

        body = QHBoxLayout()
        body.setSpacing(8)

        self._viewer = GuideboxWorkspaceView()
        body.addWidget(self._viewer, stretch=4)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        self._preview = PreviewWidget()
        preview_layout.addWidget(self._preview)
        right_layout.addWidget(preview_group, stretch=3)

        single_group = QGroupBox("Current Image")
        single_layout = QFormLayout(single_group)
        self._current_name_label = QLabel("No image selected")
        self._current_name_label.setWordWrap(True)
        self._single_label_entry = QLineEdit()
        self._single_label_entry.setPlaceholderText("5 characters using 0-9 and X")
        single_layout.addRow("Image", self._current_name_label)
        single_layout.addRow("Label", self._single_label_entry)
        right_layout.addWidget(single_group, stretch=0)

        # --- LeNet group: Train, select model, and Read the guidebox ---
        lenet_group = QGroupBox("LeNet — Read & Train")
        lenet_layout = QVBoxLayout(lenet_group)
        lenet_layout.setSpacing(6)

        # Model path display
        self._model_path_label = QLabel("Model: not selected")
        self._model_path_label.setWordWrap(True)
        self._model_path_label.setStyleSheet("color: #aaa; font-size: 11px;")
        lenet_layout.addWidget(self._model_path_label)

        # Model browse + train row
        model_row = QHBoxLayout()
        self._btn_browse_model = QPushButton("Select Model")
        self._btn_train_lenet = QPushButton("Train LeNet")
        model_row.addWidget(self._btn_browse_model)
        model_row.addWidget(self._btn_train_lenet)
        lenet_layout.addLayout(model_row)

        # Read button — reads the current 5:1 guidebox and predicts 5 digits
        self._btn_read = QPushButton("Read Guidebox")
        self._btn_read.setEnabled(False)
        self._btn_read.setToolTip(
            "Select a model first, then click to read the 5:1 guidebox.\n"
            "The guidebox crop is sliced into 5 segments and predicted by LeNet."
        )
        self._btn_read.setStyleSheet(
            "QPushButton { background: #1a6b3c; color: white; font-weight: bold; padding: 6px; border-radius: 4px; }"
            "QPushButton:disabled { background: #333; color: #666; }"
            "QPushButton:hover:enabled { background: #228b52; }"
        )
        lenet_layout.addWidget(self._btn_read)

        # Result label — shows predicted digits and confidence per slot
        self._read_result_label = QLabel("— no prediction yet —")
        self._read_result_label.setWordWrap(True)
        self._read_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._read_result_label.setStyleSheet(
            "background: #111; border: 1px solid #444; padding: 6px; "
            "font-family: monospace; font-size: 13px; color: #c8f0c8;"
        )
        self._read_result_label.setMinimumHeight(48)
        lenet_layout.addWidget(self._read_result_label)

        # LeNet group is hidden by default — toggle with "Show LeNet" checkbox
        lenet_group.setVisible(False)
        self._lenet_group = lenet_group
        right_layout.addWidget(lenet_group, stretch=1)

        batch_group = QGroupBox("Batch Processing")
        batch_layout = QVBoxLayout(batch_group)
        self._batch_state_label = QLabel("Batch mode is off.")
        self._batch_state_label.setWordWrap(True)
        self._batch_template_label = QLabel("Template: not captured")
        self._batch_template_label.setWordWrap(True)
        self._btn_capture_template = QPushButton("Use Current Framing For Remaining")
        self._btn_batch_save_next = QPushButton("Save + Next")
        self._btn_batch_skip = QPushButton("Skip This Image")
        self._btn_batch_finish = QPushButton("Finish Batch")
        self._batch_label_entry = QLineEdit()
        self._batch_label_entry.setPlaceholderText("5 characters using 0-9 and X")
        self._batch_reuse_previous = QCheckBox("Reuse previous label when empty")
        self._batch_reuse_previous.setChecked(True)
        self._previous_label_label = QLabel("Previous label: none")
        self._previous_label_label.setWordWrap(True)

        batch_layout.addWidget(self._batch_state_label)
        batch_layout.addWidget(self._batch_template_label)
        batch_layout.addWidget(self._btn_capture_template)
        batch_layout.addWidget(QLabel("Batch label"))
        batch_layout.addWidget(self._batch_label_entry)
        batch_layout.addWidget(self._batch_reuse_previous)
        batch_layout.addWidget(self._previous_label_label)
        batch_layout.addWidget(self._btn_batch_save_next)
        batch_layout.addWidget(self._btn_batch_skip)
        batch_layout.addWidget(self._btn_batch_finish)
        batch_layout.addStretch(1)
        right_layout.addWidget(batch_group, stretch=2)

        body.addWidget(right_panel, stretch=2)
        center_layout.addLayout(body, stretch=1)
        outer.addWidget(center, stretch=1)

        self.setStatusBar(QStatusBar(self))
        self._set_batch_controls_enabled(False)
        self._btn_preview.setEnabled(False)
        self._btn_save_current.setEnabled(False)
        self._btn_fit.setEnabled(False)

        # ------------------------------------------------------------------
        # Tools menu — host for the adaptive dataset balancer.
        # ------------------------------------------------------------------
        menubar = self.menuBar()
        tools_menu = menubar.addMenu("&Tools")

        self._act_balance = QAction(
            "Balance Dataset (Adaptive, No-Skew)…", self
        )
        self._act_balance.setStatusTip(
            "Balance every class folder (0-9, Unreadable, '1 - Full', "
            "'1 - Going 2', '1 - Rolling from 0', and any other "
            "image subfolder) using thickening/thinning, adaptive "
            "graininess, and a diversity-preserving downsampler."
        )
        self._act_balance.triggered.connect(self._on_balance_dataset)
        tools_menu.addAction(self._act_balance)

        self._act_wobbly = QAction(
            "Add Wobbly Borders (28x28 cell-style)\u2026", self
        )
        self._act_wobbly.setStatusTip(
            "Wrap each 28x28 digit with a faint, broken, wobbly "
            "rectangle border so the online dataset matches the "
            "look of real meter cells."
        )
        self._act_wobbly.triggered.connect(self._on_add_wobbly_borders)
        tools_menu.addAction(self._act_wobbly)

    def _connect_signals(self):
        self._btn_open_folder.clicked.connect(self._on_open_folder)
        self._btn_set_output.clicked.connect(self._on_set_output)
        self._btn_fit.clicked.connect(self._viewer.fit_to_view)
        self._btn_preview.clicked.connect(self._preview_current_guidebox)
        self._btn_save_current.clicked.connect(self._on_save_current)
        self._rotation_slider.valueChanged.connect(self._on_rotation_changed)
        self._file_list.currentRowChanged.connect(self._on_file_selected)
        self._batch_checkbox.toggled.connect(self._on_batch_toggled)
        self._btn_capture_template.clicked.connect(self._capture_batch_template)
        self._btn_batch_save_next.clicked.connect(self._on_batch_save_next)
        self._btn_batch_skip.clicked.connect(self._on_batch_skip)
        self._btn_batch_finish.clicked.connect(self._finish_batch)
        self._single_label_entry.returnPressed.connect(self._on_save_current)
        self._batch_label_entry.returnPressed.connect(self._on_batch_save_next)
        self._viewer.frame_changed.connect(self._on_viewer_frame_changed)

        # LeNet panel toggle
        self._lenet_checkbox.toggled.connect(self._lenet_group.setVisible)

        # LeNet signals
        self._btn_browse_model.clicked.connect(self._on_browse_model)
        self._btn_train_lenet.clicked.connect(self._on_train_lenet)
        self._btn_read.clicked.connect(self._on_read_guidebox)

    # -------------------------------------------------------------------------
    # LeNet — Model selection
    # -------------------------------------------------------------------------

    def _on_browse_model(self):
        """Let the user pick a .tflite or .keras model for testing."""
        start_dir = self._last_tflite_model_dir or self._last_trained_model_dir or str(Path.cwd())
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select LeNet-5 Model",
            start_dir,
            "Models (*.tflite *.keras);;All Files (*)",
        )
        if not model_path:
            return

        self._testing_model_path = model_path
        self._model_path_label.setText(f"Model: {Path(model_path).name}")
        self._model_path_label.setToolTip(model_path)
        self._btn_read.setEnabled(True)
        self.statusBar().showMessage(f"Model selected: {Path(model_path).name}")

    # -------------------------------------------------------------------------
    # LeNet — Read guidebox (predict 5 digits from current 5:1 guidebox crop)
    # -------------------------------------------------------------------------

    def _on_read_guidebox(self):
        """Read the current 5:1 guidebox, slice into 5 segments, run LeNet predict."""
        if not self._testing_model_path or not Path(self._testing_model_path).is_file():
            QMessageBox.warning(
                self,
                "No Model Selected",
                "Select a .tflite or .keras model with the 'Select Model' button first.",
            )
            return

        if self._ml_worker is not None and self._ml_worker.isRunning():
            QMessageBox.information(
                self,
                "Busy",
                "A training or prediction task is already running. Please wait.",
            )
            return

        # Get the guidebox crop from the viewer
        guide_crop = self._viewer.get_guidebox_crop()
        if guide_crop is None or guide_crop.size == 0:
            QMessageBox.warning(
                self,
                "Empty Guidebox",
                "The guidebox crop is empty.\n"
                "Load an image and align the digit strip inside the 5:1 guidebox first.",
            )
            return

        # Convert the 5:1 guidebox crop into the 140x28 strip that LeNet expects
        try:
            strip = prepare_guidebox_strip(guide_crop)
        except Exception as exc:
            QMessageBox.warning(self, "Strip Error", str(exc))
            return

        # Update the preview with the strip so the user can see what LeNet will read
        self._preview.set_strip(strip)

        # Write the strip to a temp file for the backend
        try:
            temp_image_path = write_temp_strip_image(strip)
        except Exception as exc:
            QMessageBox.critical(self, "Temp Write Error", str(exc))
            return

        self._read_result_label.setText("Reading… please wait.")
        self._btn_read.setEnabled(False)

        # Build the predict command (strip is already inverted by prepare_guidebox_strip)
        command = build_lenet_predict_command(
            backend_python=self._ml_backend_python,
            model_path=self._testing_model_path,
            image_path=temp_image_path,
            expected_label="",
            invert_input=False,
        )

        self._start_ml_worker(
            command,
            "Reading Guidebox",
            "Running LeNet prediction on the 5:1 guidebox...",
            self._on_lenet_read_finished,
        )

    def _on_lenet_read_finished(self, result: dict):
        """Show the predicted 5-digit label with rolling-context per slot.

        The backend now returns 'probs' (a 5x10 softmax matrix). For each
        segment we run :func:`infer_digit_context` to classify it as
        Full / Going X / Rolling from X / Uncertain, and we render both
        the dominant digit and the caption. If 'probs' is missing (older
        backend output), we fall back to the legacy single-confidence
        display so old runs still render something useful.
        """
        self._btn_read.setEnabled(bool(self._testing_model_path))
        predicted_label = str(result.get("predicted_label", "?????"))
        confidences = list(result.get("confidences", []))
        probs_matrix = result.get("probs")

        # Preferred path: full per-segment softmax available.
        if (
            isinstance(probs_matrix, list)
            and len(probs_matrix) == NUM_SEGMENTS
            and all(isinstance(row, list) and len(row) == 10 for row in probs_matrix)
        ):
            contexts = [infer_digit_context(row) for row in probs_matrix]

            # Reconcile predicted_label with our top-1 read so the
            # display is internally consistent. (They should match for a
            # current backend, but be defensive against rounding.)
            digits_from_probs = "".join(
                str(ctx["digit"]) if ctx["digit"] >= 0 else "?"
                for ctx in contexts
            )
            if digits_from_probs and "?" not in digits_from_probs:
                predicted_label = digits_from_probs

            cell_lines: list[str] = []
            transition_count = 0
            middle_count = 0
            uncertain_count = 0
            for i, ctx in enumerate(contexts):
                digit = ctx["digit"]
                caption = ctx["caption"]
                p1 = ctx["top1_prob"] * 100.0
                p2 = ctx["top2_prob"] * 100.0
                if caption == "Full":
                    cell_lines.append(
                        f"  cell {i + 1}: {digit} — Full ({p1:.1f}%)"
                    )
                elif caption.startswith("Middle of"):
                    # "Middle of A and B" — tag with the dominant digit so
                    # the user still sees what the model picked as top-1.
                    middle_count += 1
                    pair = ctx.get("pair")
                    if isinstance(pair, tuple) and len(pair) == 2:
                        a, b = pair
                        # Show the share that lands on each of the two
                        # paired digits explicitly.
                        a_share = p1 if digit == a else p2
                        b_share = p1 if digit == b else p2
                        cell_lines.append(
                            f"  cell {i + 1}: {digit} — {caption} "
                            f"({a}: {a_share:.1f}% / {b}: {b_share:.1f}%)"
                        )
                    else:
                        cell_lines.append(
                            f"  cell {i + 1}: {digit} — {caption} "
                            f"(top1 {p1:.1f}% / top2 {p2:.1f}%)"
                        )
                elif caption == "Uncertain":
                    uncertain_count += 1
                    cell_lines.append(
                        f"  cell {i + 1}: {digit if digit >= 0 else '?'} — "
                        f"Uncertain (top1 {p1:.1f}%, top2 {p2:.1f}%)"
                    )
                else:
                    # "Going X" or "Rolling from X"
                    transition_count += 1
                    cell_lines.append(
                        f"  cell {i + 1}: {digit} — {caption} "
                        f"(top1 {p1:.1f}% / top2 {p2:.1f}%)"
                    )

            summary_bits: list[str] = []
            if transition_count:
                summary_bits.append(f"{transition_count} in transition")
            if middle_count:
                summary_bits.append(f"{middle_count} middle")
            if uncertain_count:
                summary_bits.append(f"{uncertain_count} uncertain")
            summary_suffix = (
                f"  ({', '.join(summary_bits)})" if summary_bits else ""
            )

            full_text = (
                f"{predicted_label}{summary_suffix}\n"
                + "\n".join(cell_lines)
            )
            self._read_result_label.setText(full_text)
            self.statusBar().showMessage(
                f"LeNet read: {predicted_label}"
                + (f" — {', '.join(summary_bits)}" if summary_bits else "")
            )
            return

        # Legacy fallback: only top-1 confidences available.
        if confidences and len(confidences) == NUM_SEGMENTS:
            digit_parts = []
            for i, (digit, conf) in enumerate(zip(predicted_label, confidences)):
                digit_parts.append(f"{digit} ({float(conf) * 100:.1f}%)")
            display = " | ".join(digit_parts)
            full_text = f"{predicted_label}\n{display}"
        else:
            full_text = predicted_label

        self._read_result_label.setText(full_text)
        self.statusBar().showMessage(f"LeNet read: {predicted_label}")

    # -------------------------------------------------------------------------
    # LeNet — Training
    # -------------------------------------------------------------------------

    def _on_train_lenet(self):
        """Open the training dialog and kick off the LeNet training backend."""
        dataset_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Digit Dataset Folder (legacy 0-9 or caption-style)",
        )
        if not dataset_dir:
            return

        # Validate that every digit (0-9) is represented by at least one
        # folder. Two folder-naming conventions are accepted (and may even
        # coexist in the same dataset directory):
        #
        #   Legacy:        "0", "1", ..., "9"
        #   Caption-style: "0 - Full", "0 - Going 1", "0 - Rolling from 9",
        #                  "1 - Full", "1 - Going 2", "1 - Rolling from 0",
        #                  ... and so on for every digit.
        #
        # The training backend (lenet_backend._iter_digit_folders) already
        # merges every folder whose name starts with the same digit into
        # that digit's class, so the label is always the dominant digit.
        # We only need to confirm here that no digit is left without any
        # folder at all.
        dataset_path = Path(dataset_dir)
        digits_present: dict[int, list[str]] = {d: [] for d in range(10)}
        try:
            entries = sorted(p for p in dataset_path.iterdir() if p.is_dir())
        except OSError as exc:
            QMessageBox.warning(
                self, "Cannot read dataset folder", str(exc)
            )
            return

        for entry in entries:
            name = entry.name
            # Caption-style: "0 - Full", "0 - Going 1", ...
            m = _CAPTION_FOLDER_RE.match(name)
            if m:
                digits_present[int(m.group(1))].append(name)
                continue
            # Legacy: a single-digit folder name.
            if name.isdigit() and len(name) == 1:
                digits_present[int(name)].append(name)

        missing = sorted(d for d, folders in digits_present.items() if not folders)
        if missing:
            QMessageBox.warning(
                self,
                "Incomplete Dataset",
                "Training requires every digit (0-9) to be represented "
                "by at least one folder. Either the legacy '0'..'9' "
                "format or the caption-style "
                "'0 - Full', '0 - Going 1', '0 - Rolling from 9' "
                "(and the same for every digit) is accepted; the two "
                "may also coexist in the same dataset folder.\n\n"
                f"Missing digit(s): {', '.join(str(d) for d in missing)}",
            )
            return

        output_dir = self._last_trained_model_dir or str(Path.cwd() / LENET_MODEL_DIR_NAME)
        tflite_dir = self._last_tflite_model_dir or output_dir

        dialog = LeNetTrainingDialog(
            dataset_dir=dataset_dir,
            backend_python=self._ml_backend_python,
            keras_output_dir=output_dir,
            tflite_output_dir=tflite_dir,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        config = dialog.get_config()
        self._ml_backend_python = str(config["backend_python"])

        version = get_python_version(self._ml_backend_python)
        if not is_supported_tensorflow_backend(version):
            ver_str = f"{version[0]}.{version[1]}" if version else "(could not detect)"
            QMessageBox.warning(
                self,
                "Unsupported Python Version",
                f"TensorFlow training needs Python 3.10–3.13.\n\nDetected: {ver_str}",
            )
            return

        self._last_trained_model_dir = str(config["keras_output_dir"])
        self._last_tflite_model_dir = str(config["tflite_output_dir"])

        command = build_lenet_train_command(
            backend_python=str(config["backend_python"]),
            dataset_dir=str(config["dataset_dir"]),
            keras_output_dir=str(config["keras_output_dir"]),
            tflite_output_dir=str(config["tflite_output_dir"]),
            epochs=int(config["epochs"]),
            batch_size=int(config["batch_size"]),
            dropout_rate=float(config.get("dropout_rate", -1.0)),
            learning_rate=float(config.get("learning_rate", 0.0)),
            early_stopping_patience=int(config.get("early_stopping_patience", -1)),
            validation_split=float(config["validation_split"]),
            seed=int(config["seed"]),
        )

        self._start_ml_worker(
            command,
            "Training LeNet-5",
            "Training LeNet-5 digit model...",
            self._on_lenet_training_finished,
        )

    def _on_lenet_training_finished(self, result: dict):
        """Show training results and auto-select the freshly trained model."""
        keras_path = str(result.get("keras_model_path", ""))
        tflite_path = str(result.get("tflite_model_path", ""))

        # Auto-select the new model so "Read" works immediately after training
        new_model = tflite_path or keras_path
        if new_model and Path(new_model).is_file():
            self._testing_model_path = new_model
            self._model_path_label.setText(f"Model: {Path(new_model).name}")
            self._model_path_label.setToolTip(new_model)
            self._btn_read.setEnabled(True)

        train_acc = float(result.get("train_accuracy", 0.0)) * 100.0
        val_acc = float(result.get("val_accuracy", 0.0)) * 100.0
        test_acc = float(result.get("test_accuracy", 0.0)) * 100.0
        dataset_size = int(result.get("dataset_size", 0))

        self.statusBar().showMessage(f"LeNet training complete — test accuracy: {test_acc:.2f}%")
        QMessageBox.information(
            self,
            "LeNet Training Complete",
            f"Dataset size: {dataset_size}\n"
            f"Train accuracy: {train_acc:.2f}%\n"
            f"Validation accuracy: {val_acc:.2f}%\n"
            f"Test accuracy: {test_acc:.2f}%\n\n"
            f"Keras model:  {keras_path or '(not saved)'}\n"
            f"TFLite model: {tflite_path or '(not saved)'}\n\n"
            + (f"Model auto-selected for reading: {Path(new_model).name}" if new_model else ""),
        )

    # -------------------------------------------------------------------------
    # ML worker helpers (shared by train and read)
    # -------------------------------------------------------------------------

    def _start_ml_worker(
        self,
        command: list[str],
        title: str,
        status_message: str,
        success_handler,
    ):
        if self._ml_worker is not None and self._ml_worker.isRunning():
            QMessageBox.information(
                self,
                "ML Task Running",
                "Please wait for the current task to finish.",
            )
            return

        # Determinate 0-100 progress dialog. The worker emits a real
        # percentage from parsed Keras output; before that arrives we
        # show 0% with a friendly stage message instead of streaming raw
        # TensorFlow log lines into the dialog title.
        progress = QProgressDialog(
            f"{title} — preparing…",
            "",                # no cancel button
            0,
            100,
            self,
        )
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        progress.show()

        self._ml_progress = progress
        self._ml_progress_title = title
        self._ml_log_lines = []
        self._ml_worker = MlCommandWorker(
            command, str(Path(__file__).resolve().parent)
        )
        self._ml_worker.result_ready.connect(success_handler)
        self._ml_worker.error.connect(self._on_ml_worker_error)
        self._ml_worker.log.connect(self._on_ml_worker_log)
        self._ml_worker.progress.connect(self._on_ml_worker_progress)
        self._ml_worker.finished.connect(self._cleanup_ml_worker)
        self.statusBar().showMessage(status_message)
        self._ml_worker.start()

    def _cleanup_ml_worker(self, *_args):
        if self._ml_progress is not None:
            # Snap to 100% on success so the bar visibly completes before
            # the dialog goes away.
            try:
                self._ml_progress.setValue(100)
            except Exception:
                pass
            self._ml_progress.close()
            self._ml_progress = None
        self._ml_worker = None

    def _on_ml_worker_progress(self, current: int, total: int, message: str):
        """Real progress from the MlCommandWorker.

        ``total`` is always 100 (percent) for now. ``message`` is a clean
        stage label like 'Epoch 3 of 10' or 'Saving model…'.
        """
        if self._ml_progress is None:
            return
        if total > 0:
            try:
                self._ml_progress.setRange(0, total)
                self._ml_progress.setValue(max(0, min(current, total)))
            except Exception:
                pass
        if message:
            label = (
                f"{getattr(self, '_ml_progress_title', 'Working')} — {message}"
            )
            self._ml_progress.setLabelText(label)
            self.statusBar().showMessage(message)

    def _on_ml_worker_log(self, message: str):
        """Capture every backend line for the error tail, but DO NOT push
        raw TensorFlow output into the progress dialog title — the dialog
        is driven by the cleaner ``progress`` signal instead.
        """
        text = str(message).strip()
        if not text:
            return
        self._ml_log_lines.append(text)
        # Cap the in-memory log to avoid runaway growth on a long train run.
        if len(self._ml_log_lines) > 2000:
            self._ml_log_lines = self._ml_log_lines[-1000:]

    def _on_ml_worker_error(self, message: str):
        # Re-enable the read button if it was disabled during a read operation
        if self._testing_model_path:
            self._btn_read.setEnabled(True)
        self._read_result_label.setText("— prediction failed —")
        self.statusBar().showMessage("ML task failed.")
        details = "\n".join(self._ml_log_lines[-25:])
        full_message = str(message)
        if details:
            full_message = f"{full_message}\n\nRecent log:\n{details}"
        QMessageBox.critical(self, "ML Task Failed", full_message)

    # -------------------------------------------------------------------------
    # Existing methods (unchanged)
    # -------------------------------------------------------------------------

    def _on_open_folder(self):
        folder = self._pick_directory("Select Image Folder")
        if not folder:
            return

        self._file_list.clear()
        files = sorted(
            (
                path for path in Path(folder).iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: (path.stat().st_mtime, path.name.lower()),
        )
        if not files:
            QMessageBox.information(self, "No Images", "No supported image files were found.")
            return

        for file_path in files:
            item = QListWidgetItem(file_path.name)
            item.setData(Qt.ItemDataRole.UserRole, str(file_path))
            self._file_list.addItem(item)

        self.statusBar().showMessage(f"Loaded {len(files)} image(s).")
        self._file_list.setCurrentRow(0)

    def _on_set_output(self):
        folder = self._pick_directory("Select Output Folder")
        if not folder:
            return
        self._output_dir = folder
        self._output_label.setText(f"Output: {folder}")
        self.statusBar().showMessage(f"Output folder set to {folder}")

    def _pick_directory(self, title: str) -> str:
        dialog = QFileDialog(self, title)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setDirectory(str(Path.cwd()))
        try:
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return ""
        except KeyboardInterrupt:
            self.statusBar().showMessage("Folder selection was interrupted. The app is still running.")
            return ""

        selected = dialog.selectedFiles()
        return selected[0] if selected else ""

    def _on_file_selected(self, row: int):
        if row < 0:
            self._current_name_label.setText("No image selected")
            self._preview.clear()
            self._btn_preview.setEnabled(False)
            self._btn_save_current.setEnabled(False)
            self._btn_fit.setEnabled(False)
            self._rotation_slider.setEnabled(False)
            self._viewer.set_source_image(None)
            return

        item = self._file_list.item(row)
        image_path = item.data(Qt.ItemDataRole.UserRole)
        image = read_image_any(image_path, cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.warning(self, "Load Error", f"Cannot read:\n{image_path}")
            return

        self._viewer.set_source_image(image)
        self._current_name_label.setText(item.text())
        self._btn_preview.setEnabled(True)
        self._btn_save_current.setEnabled(True)
        self._btn_fit.setEnabled(True)
        self._rotation_slider.setEnabled(True)

        if self._batch_mode and self._batch_template is not None:
            self._viewer.set_workspace_frame(self._batch_template, emit_signal=False)
            self._sync_rotation_slider_from_viewer()
            self._schedule_auto_preview()
        else:
            self._viewer.reset_view(0.0)
            self._rotation_slider.blockSignals(True)
            self._rotation_slider.setValue(0)
            self._rotation_slider.blockSignals(False)
            self._rotation_value.setText("0°")

        self._preview.clear()
        self._update_batch_state_label()
        self.statusBar().showMessage(
            f"Viewing {item.text()}. Align inside the fixed 5:1 guidebox, then preview or save."
        )

    def _on_rotation_changed(self, angle: int):
        self._rotation_value.setText(f"{angle}°")
        self._viewer.set_rotation(angle)

    def _on_viewer_frame_changed(self):
        self._sync_rotation_slider_from_viewer()
        if self._batch_mode:
            self._schedule_auto_preview()
        else:
            self._preview.clear()

    def _sync_rotation_slider_from_viewer(self):
        angle = int(round(self._viewer.get_rotation())) % 360
        self._rotation_slider.blockSignals(True)
        self._rotation_slider.setValue(angle)
        self._rotation_slider.blockSignals(False)
        self._rotation_value.setText(f"{angle}°")

    def _schedule_auto_preview(self):
        self._auto_preview_timer.start(120)

    def _preview_current_guidebox(self) -> np.ndarray | None:
        guide_crop = self._viewer.get_guidebox_crop()
        if guide_crop is None:
            self.statusBar().showMessage("Guidebox crop is empty. Move and zoom until the strip sits inside the box.")
            return None
        try:
            strip = prepare_guidebox_strip(guide_crop)
        except Exception as exc:
            QMessageBox.warning(self, "Preview Error", str(exc))
            return None
        self._preview.set_strip(strip)
        self.statusBar().showMessage("Guidebox preview updated.")
        return strip

    def _on_save_current(self):
        if self._batch_mode:
            self._on_batch_save_next()
            return

        label = self._single_label_entry.text().strip().upper()
        if not is_digit_or_unreadable_label(label):
            QMessageBox.warning(self, "Invalid Label", "Enter exactly 5 characters using digits and X.")
            return
        self._save_current_image_with_label(label)

    def _on_batch_toggled(self, enabled: bool):
        self._batch_mode = enabled
        self._set_batch_controls_enabled(enabled)
        if enabled:
            self._reset_batch_counters()
            self._capture_batch_template()
            self._schedule_auto_preview()
            self.statusBar().showMessage("Batch mode enabled. Adjust on the left, label on the right, then Save + Next.")
        else:
            self._batch_template = None
            self._update_batch_template_label()
            self._batch_state_label.setText("Batch mode is off.")
            self.statusBar().showMessage("Batch mode disabled.")

    def _capture_batch_template(self):
        frame = self._viewer.get_workspace_frame()
        if frame is None:
            QMessageBox.warning(self, "Missing Framing", "Align the current image inside the fixed 5:1 guidebox first.")
            return
        self._batch_template = clone_workspace_frame(frame)
        self._update_batch_template_label()
        self.statusBar().showMessage("Batch framing template updated from the current workspace frame.")

    def _on_batch_save_next(self):
        if not self._batch_mode:
            return

        label = self._batch_label_entry.text().strip().upper()
        if not label and self._batch_reuse_previous.isChecked():
            previous = self._single_label_entry.text().strip().upper()
            if is_digit_or_unreadable_label(previous):
                label = previous
        if not is_digit_or_unreadable_label(label):
            QMessageBox.warning(self, "Invalid Label", "Enter exactly 5 characters using digits and X.")
            return

        if not self._save_current_image_with_label(label):
            return

        self._single_label_entry.setText(label)
        self._batch_label_entry.clear()
        self._previous_label_label.setText(f"Previous label: {label}")
        self._batch_processed += 1

        current_row = self._file_list.currentRow()
        if current_row >= 0:
            self._file_list.takeItem(current_row)

        if self._file_list.count() == 0:
            self._finish_batch()
            return

        next_row = min(current_row, self._file_list.count() - 1)
        self._file_list.setCurrentRow(next_row)
        self.statusBar().showMessage("Saved current image and moved to the next one.")
        self._update_batch_state_label()

    def _on_batch_skip(self):
        if not self._batch_mode or self._file_list.count() == 0:
            return
        current_row = self._file_list.currentRow()
        if current_row < 0:
            return
        self._batch_skipped += 1
        next_row = (current_row + 1) % self._file_list.count()
        self._file_list.setCurrentRow(next_row)
        self.statusBar().showMessage("Skipped current image.")
        self._update_batch_state_label()

    def _finish_batch(self):
        if self._batch_mode:
            self._batch_checkbox.blockSignals(True)
            self._batch_checkbox.setChecked(False)
            self._batch_checkbox.blockSignals(False)
            self._batch_mode = False
            self._set_batch_controls_enabled(False)

        self._update_batch_template_label()
        self._batch_state_label.setText("Batch mode is off.")
        QMessageBox.information(
            self,
            "Batch Summary",
            f"Processed images: {self._batch_processed}\n"
            f"Skipped images: {self._batch_skipped}\n"
            f"Saved segments: {self._batch_saved_segments}\n"
            f"ROI raw saved: {self._batch_roi_raw}\n"
            f"Errors: {self._batch_errors}\n"
            f"Output: {self._output_dir or '(not set)'}",
        )
        self.statusBar().showMessage("Batch finished.")

    def _save_current_image_with_label(self, label: str) -> bool:
        if not self._output_dir:
            self._on_set_output()
            if not self._output_dir:
                return False

        strip = self._preview_current_guidebox()
        if strip is None:
            return False

        segments = self._preview.get_segments()
        if len(segments) != NUM_SEGMENTS:
            QMessageBox.warning(self, "No Segments", "Preview the current guidebox crop first.")
            return False

        guide_crop = self._viewer.get_guidebox_crop()
        if guide_crop is None:
            QMessageBox.warning(self, "Missing Crop", "Guidebox crop is empty.")
            return False

        saved_segments, folders_used, write_errors = self._save_segments_with_label(segments, label)
        roi_raw_saved, roi_errors = self._save_roi_exports(guide_crop, label)

        self._batch_saved_segments += saved_segments
        self._batch_roi_raw += roi_raw_saved
        self._batch_errors += write_errors + roi_errors

        self.statusBar().showMessage(
            f"Saved label {label}. Segments: {saved_segments}, ROI raw: {roi_raw_saved}."
        )

        if not self._batch_mode:
            QMessageBox.information(
                self,
                "Saved",
                f"Saved {saved_segments} segment(s)\n"
                f"ROI raw saved: {roi_raw_saved}\n"
                f"Folders: {', '.join(sorted(folders_used)) if folders_used else '(none)'}\n"
                f"Errors: {write_errors + roi_errors}",
            )
        return True

    def _save_roi_exports(self, guide_crop: np.ndarray, label: str) -> tuple[int, int]:
        if guide_crop is None or guide_crop.size == 0:
            return 0, 1

        raw_dir = Path(self._output_dir) / ROI_RAW_DIR_NAME
        raw_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"{label}_{uuid.uuid4().hex[:10]}"
        raw_ok = cv2.imwrite(str(raw_dir / f"{base_name}_raw.png"), guide_crop.copy())
        return int(raw_ok), int(not raw_ok)

    def _save_segments_with_label(self, segments: list[np.ndarray], label: str) -> tuple[int, set[str], int]:
        saved = 0
        folders_used: set[str] = set()
        write_errors = 0

        for index, segment in enumerate(segments):
            char = label[index]
            folder_name = self._label_char_to_category_folder(char)
            folders_used.add(folder_name)
            char_dir = Path(self._output_dir) / folder_name
            char_dir.mkdir(parents=True, exist_ok=True)
            save_path = char_dir / f"segment_{uuid.uuid4().hex[:8]}.png"
            if cv2.imwrite(str(save_path), segment):
                saved += 1
            else:
                write_errors += 1

        return saved, folders_used, write_errors

    @staticmethod
    def _label_char_to_category_folder(label_char: str) -> str:
        if label_char.upper() == UNREADABLE_LABEL_CHAR:
            return UNREADABLE_FOLDER_NAME
        return label_char

    def _set_batch_controls_enabled(self, enabled: bool):
        self._btn_capture_template.setEnabled(enabled)
        self._btn_batch_save_next.setEnabled(enabled)
        self._btn_batch_skip.setEnabled(enabled)
        self._btn_batch_finish.setEnabled(enabled)
        self._batch_label_entry.setEnabled(enabled)
        self._batch_reuse_previous.setEnabled(enabled)

    def _reset_batch_counters(self):
        self._batch_processed = 0
        self._batch_saved_segments = 0
        self._batch_roi_raw = 0
        self._batch_errors = 0
        self._batch_skipped = 0
        self._update_batch_state_label()

    def _update_batch_state_label(self):
        if not self._batch_mode:
            return
        remaining = self._file_list.count()
        current_name = self._current_name_label.text()
        self._batch_state_label.setText(
            f"Adjust on the left, label on the right.\n"
            f"Current: {current_name}\n"
            f"Processed: {self._batch_processed} | Remaining: {remaining} | Skipped: {self._batch_skipped}"
        )

    def _update_batch_template_label(self):
        if self._batch_template is None:
            self._batch_template_label.setText("Template: not captured")
            return
        self._batch_template_label.setText(
            "Template captured from current workspace frame.\n"
            f"Rotation: {int(round(self._batch_template.rotation_deg)) % 360} deg | "
            f"Scale: {self._batch_template.scale:.3f} | "
            f"Pan: ({self._batch_template.translate_x:.1f}, {self._batch_template.translate_y:.1f})"
        )

    # ------------------------------------------------------------------
    # Adaptive dataset balancer (Tools menu)
    # ------------------------------------------------------------------
    def _validate_extended_class_parent(
        self,
        parent_folder: str,
    ) -> tuple[bool, str, list[str]]:
        """Discover every subfolder of ``parent_folder`` that holds images.

        Accepts ANY subfolder name (single digits 0..9, "Unreadable",
        "1 - Full", "1 - Going 2", "1 - Rolling from 0", and any other
        custom class folder). Hidden folders, virtual envs, model
        outputs and the ROI raw output folder are ignored.
        """
        try:
            children = [p for p in Path(parent_folder).iterdir() if p.is_dir()]
        except OSError as exc:
            return False, f"Cannot access selected folder:\n{exc}", []

        if not children:
            return (
                False,
                "Selected folder has no subfolders. It must contain at "
                "least one class subfolder (e.g. 0-9, Unreadable, "
                "'1 - Full', '1 - Going 2', '1 - Rolling from 0').",
                [],
            )

        valid: list[str] = []
        for d in children:
            if d.name.startswith(".") or d.name.startswith("__"):
                continue
            if d.name in _BALANCER_IGNORED_SUBFOLDERS:
                continue
            try:
                has_image = any(
                    p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                    for p in d.iterdir()
                )
            except OSError:
                has_image = False
            if has_image:
                valid.append(d.name)

        if not valid:
            return (
                False,
                "No class subfolder contained any images. Expected "
                "subfolders like 0-9, Unreadable, or '1 - Full', "
                "'1 - Going 2', '1 - Rolling from 0', etc.",
                [],
            )

        valid.sort()
        return True, "", valid

    def _on_balance_dataset(self):
        input_parent = QFileDialog.getExistingDirectory(
            self,
            "Select Imbalanced Source Folder (any class subfolders)",
        )
        if not input_parent:
            return

        valid, message, category_folders = (
            self._validate_extended_class_parent(input_parent)
        )
        if not valid:
            QMessageBox.warning(self, "Invalid Input Folder", message)
            return

        # Pre-scan: count images per class so the dialog can recommend
        # an adaptive target.
        category_counts: dict[str, int] = {}
        for cat in category_folders:
            cat_path = Path(input_parent) / cat
            try:
                category_counts[cat] = sum(
                    1
                    for p in cat_path.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                )
            except OSError:
                category_counts[cat] = 0

        dialog = BalanceDialog(category_counts, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        target_count = dialog.get_target()
        tolerance = dialog.get_tolerance()

        output_parent = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Balanced Data"
        )
        if not output_parent:
            return

        if Path(output_parent).resolve() == Path(input_parent).resolve():
            QMessageBox.warning(
                self,
                "Choose a different output",
                "Output folder must be different from the input folder.",
            )
            return

        # Progress dialog so the user gets feedback per class.
        progress = QProgressDialog(
            "Balancing dataset…",
            "Cancel",
            0,
            len(category_folders),
            self,
        )
        progress.setWindowTitle("Adaptive Balancer")
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        log: list[str] = []
        cancelled = False
        for idx, cat in enumerate(category_folders, start=1):
            if progress.wasCanceled():
                cancelled = True
                break
            progress.setLabelText(f"Balancing class '{cat}'…")
            QApplication.processEvents()
            try:
                line = self._balance_one_category(
                    Path(input_parent) / cat,
                    Path(output_parent) / cat,
                    target_count,
                    tolerance,
                )
            except Exception as exc:
                line = f"[{cat}] ERROR: {exc}"
            log.append(line)
            progress.setValue(idx)
            QApplication.processEvents()

        progress.close()

        title = "Balancing Cancelled" if cancelled else "Balancing Complete"
        summary = "\n".join(log) if log else "(no classes processed)"
        QMessageBox.information(
            self,
            title,
            f"Target per class: {target_count}\n"
            f"Tolerance: ±{int(tolerance * 100)}%\n\n"
            f"Per-class summary:\n{summary}",
        )
        self.statusBar().showMessage(
            "Dataset balancing cancelled."
            if cancelled
            else f"Dataset balanced to ~{target_count} per class."
        )

    def _balance_one_category(
        self,
        src_dir: Path,
        dst_dir: Path,
        target_count: int,
        tolerance: float,
    ) -> str:
        """Balance a single class folder.

        Returns a human-readable summary line for the per-class log.
        """
        dst_dir.mkdir(parents=True, exist_ok=True)

        image_paths = [
            p
            for p in src_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        current_count = len(image_paths)
        cat = src_dir.name
        if current_count == 0:
            return f"[{cat}] skipped — 0 images."

        lo = int(target_count * (1.0 - tolerance))
        hi = int(target_count * (1.0 + tolerance))

        if lo <= current_count <= hi:
            self._copy_balanced_originals(image_paths, dst_dir)
            return (
                f"[{cat}] {current_count} → {current_count} "
                f"(within ±{int(tolerance * 100)}% tolerance, kept as-is)."
            )

        if current_count > target_count:
            kept = _diverse_downsample_paths(image_paths, target_count)
            self._copy_balanced_originals(kept, dst_dir)
            dropped = current_count - len(kept)
            return (
                f"[{cat}] {current_count} → {len(kept)} "
                f"(dropped {dropped} most-redundant near-duplicates; "
                f"uniques preserved)."
            )

        # Oversample with adaptive, no-skew augmentation.
        self._copy_balanced_originals(image_paths, dst_dir)
        shortfall = target_count - current_count
        self._generate_no_skew_augmentations(image_paths, dst_dir, shortfall)
        return (
            f"[{cat}] {current_count} → {target_count} "
            f"(+{shortfall} adaptive aug: thicken/thin/grain only)."
        )

    @staticmethod
    def _copy_balanced_originals(paths: list[Path], dst_dir: Path) -> None:
        for p in paths:
            img = read_image_any(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            out_path = dst_dir / p.name
            # Avoid clobbering a same-named file from another sample.
            if out_path.exists():
                out_path = dst_dir / f"{p.stem}_{uuid.uuid4().hex[:6]}{p.suffix}"
            cv2.imwrite(str(out_path), img)

    @staticmethod
    def _generate_no_skew_augmentations(
        base_paths: list[Path],
        dst_dir: Path,
        shortfall: int,
    ) -> None:
        if shortfall <= 0 or not base_paths:
            return
        rng = np.random.default_rng()
        ordered = list(base_paths)
        rng.shuffle(ordered)
        for i in range(shortfall):
            src = ordered[i % len(ordered)]
            img = read_image_any(str(src), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            aug = _adaptive_no_skew_augment(img)
            new_name = f"bal_aug_{uuid.uuid4().hex[:8]}.png"
            cv2.imwrite(str(dst_dir / new_name), aug)

    def _on_add_wobbly_borders(self):
        input_root = QFileDialog.getExistingDirectory(
            self,
            "Select Source Folder for Wobbly Border Conversion",
        )
        if not input_root:
            return

        image_entries = self._collect_wobbly_border_entries(Path(input_root))
        if not image_entries:
            QMessageBox.warning(
                self,
                "No Images Found",
                "The selected folder did not contain any supported images.",
            )
            return

        dialog = WobblyBorderDialog(len(image_entries), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        output_root = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder for Bordered Dataset",
        )
        if not output_root:
            return

        if Path(output_root).resolve() == Path(input_root).resolve():
            QMessageBox.warning(
                self,
                "Choose a different output",
                "Output folder must be different from the input folder.",
            )
            return

        variants = dialog.get_variants()
        keep_originals = dialog.keep_originals()
        preserve_classes = dialog.preserve_classes()
        total_steps = len(image_entries) * variants
        if keep_originals:
            total_steps += len(image_entries)

        progress = QProgressDialog(
            "Generating bordered variants...",
            "Cancel",
            0,
            total_steps,
            self,
        )
        progress.setWindowTitle("Add Wobbly Borders")
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        rng = np.random.default_rng()
        bordered_written = 0
        originals_written = 0
        skipped = 0
        errors = 0
        steps = 0

        for src_path, rel_parent in image_entries:
            if progress.wasCanceled():
                break

            dst_dir = Path(output_root)
            if preserve_classes and rel_parent != Path("."):
                dst_dir = dst_dir / rel_parent
            dst_dir.mkdir(parents=True, exist_ok=True)

            img = read_image_any(str(src_path), cv2.IMREAD_GRAYSCALE)
            if img is None or getattr(img, "size", 0) == 0:
                skipped += 1
                continue

            prepared = _ensure_digit_canvas_28x28(img)

            if keep_originals:
                if self._save_wobbly_output(
                    dst_dir,
                    src_path.stem,
                    prepared,
                    suffix="orig",
                ):
                    originals_written += 1
                else:
                    errors += 1
                steps += 1
                progress.setValue(steps)
                progress.setLabelText(f"Copying original {src_path.name}...")
                QApplication.processEvents()
                if progress.wasCanceled():
                    break

            for variant_idx in range(variants):
                bordered = _apply_wobbly_borders(prepared, rng=rng)
                saved = self._save_wobbly_output(
                    dst_dir,
                    src_path.stem,
                    bordered,
                    suffix=f"wobbly_v{variant_idx + 1:02d}",
                )
                if saved:
                    bordered_written += 1
                else:
                    errors += 1
                steps += 1
                progress.setValue(steps)
                progress.setLabelText(
                    f"Bordering {src_path.name} ({variant_idx + 1}/{variants})..."
                )
                QApplication.processEvents()
                if progress.wasCanceled():
                    break

        progress.close()

        cancelled = steps < total_steps and progress.wasCanceled()
        QMessageBox.information(
            self,
            "Border Conversion Cancelled" if cancelled else "Border Conversion Complete",
            f"Source images: {len(image_entries)}\n"
            f"Bordered variants written: {bordered_written}\n"
            f"Originals copied: {originals_written}\n"
            f"Skipped bad images: {skipped}\n"
            f"Write errors: {errors}\n"
            f"Output: {output_root}",
        )
        self.statusBar().showMessage(
            "Wobbly border conversion cancelled."
            if cancelled
            else f"Wobbly border conversion wrote {bordered_written} bordered image(s)."
        )

    @staticmethod
    def _collect_wobbly_border_entries(root: Path) -> list[tuple[Path, Path]]:
        entries: list[tuple[Path, Path]] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                rel_parent = path.parent.relative_to(root)
            except ValueError:
                rel_parent = Path(".")
            if any(
                part.startswith(".")
                or part.startswith("__")
                or part in _BALANCER_IGNORED_SUBFOLDERS
                for part in rel_parent.parts
            ):
                continue
            entries.append((path, rel_parent))
        entries.sort(key=lambda item: str(item[0]).lower())
        return entries

    @staticmethod
    def _save_wobbly_output(
        dst_dir: Path,
        base_stem: str,
        img: np.ndarray,
        suffix: str,
    ) -> bool:
        safe_img = _ensure_digit_canvas_28x28(img)
        out_path = dst_dir / f"{base_stem}_{suffix}.png"
        if out_path.exists():
            out_path = dst_dir / f"{base_stem}_{suffix}_{uuid.uuid4().hex[:6]}.png"
        return bool(cv2.imwrite(str(out_path), safe_img))


def main():
    app = QApplication(sys.argv)
    window = StreamlinedMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
