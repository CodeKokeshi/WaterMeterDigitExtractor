"""
Dual-mode digit dataset exporter.

This standalone app keeps the existing main.py untouched while combining:
  * manual 4-point perspective extraction
  * fixed 5:1 guidebox extraction

Outputs per saved sample:
  * LeNet-ready 32x32 digit crops + processed preview strip
  * YOLO 640x640 context image + 6 labels by default
  * raw native digit crops + upscaled raw digit crops
  * separate must-avoid negative crops
  * metadata sidecar JSON
"""

from __future__ import annotations

import importlib
import json
import sys
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRectF, QSettings, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QResizeEvent,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
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


HANDLE_RADIUS = 7
HANDLE_COLOR = QColor(0, 200, 255, 220)
HANDLE_HOVER_COLOR = QColor(255, 100, 0, 220)
LINE_COLOR = QColor(0, 255, 100, 180)
LINE_WIDTH = 2

NUM_SEGMENTS = 5
NUM_MUST_AVOID_SAMPLES = 10
LENET_DIGIT_SIZE = 32
FINAL_W = 140
FINAL_H = 28
WARP_HI_W = 500
WARP_HI_H = 100
YOLO_SIZE = 640
RAW_UPSCALE_FACTOR = 4
UNREADABLE_LABEL_CHAR = "X"
UNREADABLE_FOLDER_NAME = "Unreadable"
STRIP_CLASS_ID = 0
DIGIT_CLASS_OFFSET = 1
UNREADABLE_DIGIT_CLASS_ID = 11
DEFAULT_CONTEXT_MARGIN_RATIO = 0.20

ROI_640_DIR_NAME = "ROI_640"
ROI_640_LABELS_DIR_NAME = "ROI_640_labels"
YOLO_VISUALS_DIR_NAME = "YoloVisuals"
LENET_DIGITS_STRIPS_DIR_NAME = "lenet_digits_strip"
LENET_DIGITS_SEGMENTS_DIR_NAME = "lenet_digits_segments"
RAW_DIGITS_SEGMENTS_DIR_NAME = "raw_digits_segments"
RAW_DIGITS_SEGMENTS_UPSCALED_DIR_NAME = "raw_digits_segments_upscaled"
LENET_MUST_AVOID_SEGMENTS_DIR_NAME = "lenet_must_avoid_segments"
RAW_MUST_AVOID_SEGMENTS_DIR_NAME = "raw_must_avoid_segments"
RAW_MUST_AVOID_SEGMENTS_UPSCALED_DIR_NAME = "raw_must_avoid_segments_upscaled"
METADATA_DIR_NAME = "sample_metadata"
YOLO_CLASS_MAP_FILE_NAME = "yolo_class_map.json"
YOLO_CLASSES_FILE_NAME = "yolo_classes.txt"
CLASSIFIER_DATASET_INFO_FILE_NAME = "classifier_dataset_info.json"

SETTINGS_ORG = "DigitExtractor"
SETTINGS_APP = "DatasetExporter"
SETTINGS_LAST_IMAGE_DIR = "paths/last_image_dir"
SETTINGS_LAST_OUTPUT_DIR = "paths/last_output_dir"
SETTINGS_BATCH_MODE = "ui/batch_mode"

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".heif"
}

HEIF_DECODER_AVAILABLE = False
_PIL_IMAGE_MODULE = None


class ExtractionMode(str, Enum):
    MANUAL_PERSPECTIVE = "manual_perspective"
    FIXED_GUIDEBOX = "fixed_guidebox"


@dataclass
class ExtractionResult:
    mode: ExtractionMode
    source_path: str
    source_image_shape: tuple[int, int, int]
    label: str
    search_bbox_xywh: tuple[float, float, float, float]
    strip_quad: np.ndarray
    digit_quads: list[np.ndarray]
    strip_bbox_xywh: tuple[float, float, float, float]
    digit_bboxes_xywh: list[tuple[float, float, float, float]]
    processed_digit_crops: list[np.ndarray]
    processed_preview_strip: np.ndarray
    raw_digit_crops: list[np.ndarray]
    raw_digit_crops_upscaled: list[np.ndarray]
    must_avoid_boxes_xywh: list[tuple[float, float, float, float]]
    must_avoid_processed_crops: list[np.ndarray]
    must_avoid_raw_crops: list[np.ndarray]
    must_avoid_raw_crops_upscaled: list[np.ndarray]
    yolo_image: np.ndarray
    yolo_lines: list[str]
    yolo_boxes_xywh: list[tuple[float, float, float, float]]
    yolo_context_bbox_xywh: tuple[int, int, int, int]
    mode_geometry: dict[str, object]


@dataclass
class ManualPointsTemplate:
    rotation_angle: int
    normalized_points: np.ndarray
    reference_size: tuple[int, int]


def _ensure_heif_decoder() -> bool:
    global HEIF_DECODER_AVAILABLE, _PIL_IMAGE_MODULE
    if HEIF_DECODER_AVAILABLE:
        return True

    try:
        pillow_heif = importlib.import_module("pillow_heif")
        pil_image = importlib.import_module("PIL.Image")
        pillow_heif.register_heif_opener()
        _PIL_IMAGE_MODULE = pil_image
        HEIF_DECODER_AVAILABLE = True
        return True
    except Exception:
        return False


def read_image_any(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    image = cv2.imread(path, flags)
    if image is not None:
        return image

    ext = Path(path).suffix.lower()
    if ext not in {".heic", ".heif"} or not _ensure_heif_decoder():
        return None

    try:
        with _PIL_IMAGE_MODULE.open(path) as pil_img:
            if flags == cv2.IMREAD_GRAYSCALE:
                return np.array(pil_img.convert("L"))

            if flags == cv2.IMREAD_UNCHANGED:
                if "A" in pil_img.getbands():
                    rgba = np.array(pil_img.convert("RGBA"))
                    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
                rgb = np.array(pil_img.convert("RGB"))
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            rgb = np.array(pil_img.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def is_digit_or_unreadable_label(label: str) -> bool:
    normalized = label.strip().upper()
    return len(normalized) == NUM_SEGMENTS and all(
        ch.isdigit() or ch == UNREADABLE_LABEL_CHAR for ch in normalized
    )


def label_char_to_folder(label_char: str) -> str:
    if label_char.upper() == UNREADABLE_LABEL_CHAR:
        return UNREADABLE_FOLDER_NAME
    return label_char


def label_char_to_yolo_class_id(label_char: str) -> int:
    if label_char.upper() == UNREADABLE_LABEL_CHAR:
        return UNREADABLE_DIGIT_CLASS_ID
    return DIGIT_CLASS_OFFSET + int(label_char)


def build_yolo_class_map() -> dict[str, object]:
    class_map: dict[str, object] = {
        "0": {
            "name": "digit_strip",
            "kind": "detector_target",
            "meaning": "full 5-digit strip bounding box",
        }
    }
    for digit in range(10):
        class_id = str(DIGIT_CLASS_OFFSET + digit)
        class_map[class_id] = {
            "name": f"digit_{digit}",
            "kind": "detector_target",
            "meaning": f"single digit box for value {digit}",
        }
    class_map[str(UNREADABLE_DIGIT_CLASS_ID)] = {
        "name": "digit_unreadable",
        "kind": "detector_target",
        "meaning": f"single digit box labeled {UNREADABLE_LABEL_CHAR}",
    }
    return class_map


def build_yolo_classes_txt_lines() -> list[str]:
    return [
        "digit_strip",
        *[f"digit_{digit}" for digit in range(10)],
        "digit_unreadable",
    ]


def build_classifier_dataset_info() -> dict[str, object]:
    return {
        "task": "digit_classification_with_rejects",
        "positive_dataset": {
            "processed_digits_dir": LENET_DIGITS_SEGMENTS_DIR_NAME,
            "processed_digits_strip_dir": LENET_DIGITS_STRIPS_DIR_NAME,
            "raw_digits_dir": RAW_DIGITS_SEGMENTS_DIR_NAME,
            "raw_digits_upscaled_dir": RAW_DIGITS_SEGMENTS_UPSCALED_DIR_NAME,
            "classes": [str(digit) for digit in range(10)] + [UNREADABLE_FOLDER_NAME],
            "notes": "Positive classifier crops are saved by digit folder name. Unreadable slots are stored under Unreadable.",
        },
        "must_avoid_dataset": {
            "role": "reject_class",
            "processed_dir": LENET_MUST_AVOID_SEGMENTS_DIR_NAME,
            "raw_dir": RAW_MUST_AVOID_SEGMENTS_DIR_NAME,
            "raw_upscaled_dir": RAW_MUST_AVOID_SEGMENTS_UPSCALED_DIR_NAME,
            "samples_per_image": NUM_MUST_AVOID_SAMPLES,
            "notes": "Must Avoid is intentionally kept separate from YOLO classes and is exported as standalone negative crops, not strips.",
        },
        "recommended_training_split": {
            "detector": "Train YOLO on ROI_640 + ROI_640_labels only.",
            "classifier": "Train digit classifier on lenet_digits_segments positives plus lenet_must_avoid_segments as reject samples.",
        },
    }


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def split_strip_segments(strip: np.ndarray) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for i in range(NUM_SEGMENTS):
        x0 = i * LENET_DIGIT_SIZE
        segments.append(strip[:, x0:x0 + LENET_DIGIT_SIZE].copy())
    return segments


def normalize_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    x_den = max(width - 1, 1)
    y_den = max(height - 1, 1)
    normalized = points.astype(np.float32).copy()
    normalized[:, 0] = np.clip(normalized[:, 0] / x_den, 0.0, 1.0)
    normalized[:, 1] = np.clip(normalized[:, 1] / y_den, 0.0, 1.0)
    return normalized


def denormalize_points(normalized_points: np.ndarray, width: int, height: int) -> np.ndarray:
    x_max = max(width - 1, 1)
    y_max = max(height - 1, 1)
    points = normalized_points.astype(np.float32).copy()
    points[:, 0] = np.clip(points[:, 0] * x_max, 0, x_max)
    points[:, 1] = np.clip(points[:, 1] * y_max, 0, y_max)
    return points


def prepare_guidebox_strip(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Empty guidebox crop.")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    binary = cv2.medianBlur(binary, 3)
    binary = cv2.bitwise_not(binary)
    interpolation = cv2.INTER_AREA if binary.shape[1] >= FINAL_W else cv2.INTER_LINEAR
    return cv2.resize(binary, (FINAL_W, FINAL_H), interpolation=interpolation)


def prepare_manual_processed_strip(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    src = order_points(points)
    dst = np.array(
        [
            [0, 0],
            [WARP_HI_W - 1, 0],
            [WARP_HI_W - 1, WARP_HI_H - 1],
            [0, WARP_HI_H - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (WARP_HI_W, WARP_HI_H))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    binary = cv2.medianBlur(binary, 3)
    binary = cv2.bitwise_not(binary)
    interpolation = cv2.INTER_AREA if binary.shape[1] >= FINAL_W else cv2.INTER_LINEAR
    return cv2.resize(binary, (FINAL_W, FINAL_H), interpolation=interpolation)


def crop_image_xywh(image: np.ndarray, bbox_xywh: tuple[float, float, float, float]) -> np.ndarray:
    h, w = image.shape[:2]
    x, y, bw, bh = bbox_xywh
    x1 = max(0, min(int(round(x)), w - 1))
    y1 = max(0, min(int(round(y)), h - 1))
    x2 = max(x1 + 1, min(int(round(x + bw)), w))
    y2 = max(y1 + 1, min(int(round(y + bh)), h))
    crop = image[y1:y2, x1:x2].copy()
    return crop if crop.size else image[max(0, y1):max(y1 + 1, y1 + 1), max(0, x1):max(x1 + 1, x1 + 1)].copy()


def rects_intersect(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (
        ax + aw <= bx
        or bx + bw <= ax
        or ay + ah <= by
        or by + bh <= ay
    )


def sample_must_avoid_boxes(
    image_shape: tuple[int, int, int],
    strip_bbox_xywh: tuple[float, float, float, float],
    digit_bboxes_xywh: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    image_h, image_w = image_shape[:2]
    sx, sy, sw, sh = strip_bbox_xywh
    margin = max(4.0, min(sw, sh) * 0.08)
    results: list[tuple[float, float, float, float]] = []

    preferred_orders = [
        ("top", "bottom", "left", "right"),
        ("bottom", "top", "right", "left"),
        ("left", "right", "top", "bottom"),
        ("right", "left", "bottom", "top"),
        ("top", "right", "bottom", "left"),
        ("bottom", "left", "top", "right"),
        ("left", "top", "right", "bottom"),
        ("right", "bottom", "left", "top"),
        ("top", "left", "bottom", "right"),
        ("bottom", "right", "top", "left"),
    ]

    def candidate_for_direction(
        box: tuple[float, float, float, float],
        side: float,
        direction: str,
    ) -> tuple[float, float, float, float]:
        x, y, w, h = box
        center_x = x + (w / 2.0)
        center_y = y + (h / 2.0)
        if direction == "top":
            return center_x - (side / 2.0), sy - side - margin, side, side
        if direction == "bottom":
            return center_x - (side / 2.0), sy + sh + margin, side, side
        if direction == "left":
            return sx - side - margin, center_y - (side / 2.0), side, side
        return sx + sw + margin, center_y - (side / 2.0), side, side

    def is_valid(candidate: tuple[float, float, float, float]) -> bool:
        x, y, w, h = candidate
        if x < 0 or y < 0 or x + w > image_w or y + h > image_h:
            return False
        if rects_intersect(candidate, strip_bbox_xywh):
            return False
        if any(rects_intersect(candidate, digit_box) for digit_box in digit_bboxes_xywh):
            return False
        if any(rects_intersect(candidate, existing) for existing in results):
            return False
        return True

    for index in range(NUM_MUST_AVOID_SAMPLES):
        digit_box = digit_bboxes_xywh[index % len(digit_bboxes_xywh)]
        _, _, bw, bh = digit_box
        side = float(max(bw, bh))
        matched_box: tuple[float, float, float, float] | None = None
        for direction in preferred_orders[index % len(preferred_orders)]:
            candidate = candidate_for_direction(digit_box, side, direction)
            if is_valid(candidate):
                matched_box = candidate
                break

        if matched_box is None:
            step = max(int(round(side / 2.0)), 8)
            limit_x = max(int(image_w - side), 0)
            limit_y = max(int(image_h - side), 0)
            for y0 in range(0, limit_y + 1, step):
                for x0 in range(0, limit_x + 1, step):
                    candidate = (float(x0), float(y0), side, side)
                    if is_valid(candidate):
                        matched_box = candidate
                        break
                if matched_box is not None:
                    break

        if matched_box is None:
            raise ValueError("Unable to sample 5 must-avoid negatives outside the declared strip.")
        results.append(matched_box)

    return results


def prepare_digit_crop_for_lenet(image: np.ndarray, size: int = LENET_DIGIT_SIZE) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Empty digit crop.")
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    binary = cv2.medianBlur(binary, 3)
    binary = cv2.bitwise_not(binary)
    return cv2.resize(binary, (size, size), interpolation=cv2.INTER_AREA)


def build_processed_digit_contact_sheet(segments: list[np.ndarray]) -> np.ndarray:
    if not segments:
        return np.zeros((LENET_DIGIT_SIZE, LENET_DIGIT_SIZE * NUM_SEGMENTS), dtype=np.uint8)
    return np.hstack([segment.copy() for segment in segments])


def gray_segment_to_pixmap(
    gray_segment: np.ndarray,
    width: int,
    height: int,
    interpolation: int = cv2.INTER_NEAREST,
) -> QPixmap:
    disp_seg = cv2.resize(gray_segment, (width, height), interpolation=interpolation)
    qimg = QImage(
        disp_seg.data,
        width,
        height,
        width,
        QImage.Format.Format_Grayscale8,
    )
    return QPixmap.fromImage(qimg.copy())


def color_image_to_pixmap(image: np.ndarray, width: int, height: int) -> QPixmap:
    if image.ndim == 2:
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
        qimg = QImage(
            resized.data,
            width,
            height,
            width,
            QImage.Format.Format_Grayscale8,
        )
        return QPixmap.fromImage(qimg.copy())

    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    qimg = QImage(
        rgb.data,
        width,
        height,
        width * 3,
        QImage.Format.Format_RGB888,
    )
    return QPixmap.fromImage(qimg.copy())


def get_quad_bbox_xywh(points: np.ndarray, width: int, height: int) -> tuple[float, float, float, float]:
    ordered = order_points(points)
    pts = np.round(ordered).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, max(width - 1, 0))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(height - 1, 0))
    x, y, bw, bh = cv2.boundingRect(pts)
    return float(x), float(y), float(bw), float(bh)


def build_yolo_bbox_line(
    class_id: int,
    x: float,
    y: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> str:
    cx = (x + (width / 2.0)) / max(float(image_width), 1.0)
    cy = (y + (height / 2.0)) / max(float(image_height), 1.0)
    bw = width / max(float(image_width), 1.0)
    bh = height / max(float(image_height), 1.0)
    cx = float(np.clip(cx, 0.0, 1.0))
    cy = float(np.clip(cy, 0.0, 1.0))
    bw = float(np.clip(bw, 0.0, 1.0))
    bh = float(np.clip(bh, 0.0, 1.0))
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def make_square_context_crop(
    image: np.ndarray,
    boxes_xywh: list[tuple[float, float, float, float]],
    margin_ratio: float = DEFAULT_CONTEXT_MARGIN_RATIO,
) -> tuple[np.ndarray, list[tuple[float, float, float, float]], tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    xs = [box[0] for box in boxes_xywh]
    ys = [box[1] for box in boxes_xywh]
    x2s = [box[0] + box[2] for box in boxes_xywh]
    y2s = [box[1] + box[3] for box in boxes_xywh]

    min_x = min(xs)
    min_y = min(ys)
    max_x = max(x2s)
    max_y = max(y2s)

    tight_w = max_x - min_x
    tight_h = max_y - min_y
    side = int(round(max(tight_w, tight_h) * (1.0 + (2.0 * margin_ratio))))
    side = max(side, int(round(tight_w)), int(round(tight_h)), 1)

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    x0 = int(round(cx - (side / 2.0)))
    y0 = int(round(cy - (side / 2.0)))
    x1 = x0 + side
    y1 = y0 + side

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        padded = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT_101,
        )
    else:
        padded = image

    x0 += pad_left
    y0 += pad_top
    x1 += pad_left
    y1 += pad_top
    crop = padded[y0:y1, x0:x1].copy()

    translated_boxes: list[tuple[float, float, float, float]] = []
    for x, y, bw, bh in boxes_xywh:
        translated_boxes.append(
            (
                float((x + pad_left) - x0),
                float((y + pad_top) - y0),
                float(bw),
                float(bh),
            )
        )

    return crop, translated_boxes, (int(x0 - pad_left), int(y0 - pad_top), int(side), int(side))


def resize_boxes_xywh(
    boxes_xywh: list[tuple[float, float, float, float]],
    src_size: tuple[int, int],
    dst_size: int = YOLO_SIZE,
) -> list[tuple[float, float, float, float]]:
    src_h, src_w = src_size
    scale_x = dst_size / max(float(src_w), 1.0)
    scale_y = dst_size / max(float(src_h), 1.0)
    return [
        (x * scale_x, y * scale_y, bw * scale_x, bh * scale_y)
        for x, y, bw, bh in boxes_xywh
    ]


def draw_boxes_preview(
    image: np.ndarray,
    boxes_xywh: list[tuple[float, float, float, float]],
    class_ids: list[int],
) -> np.ndarray:
    overlay = image.copy()
    palette = [
        (0, 255, 255),
        (80, 220, 80),
        (255, 160, 0),
        (255, 100, 100),
        (140, 140, 255),
        (255, 60, 220),
    ]
    for index, (box, class_id) in enumerate(zip(boxes_xywh, class_ids)):
        x, y, bw, bh = [int(round(v)) for v in box]
        color = palette[index % len(palette)]
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, 2, cv2.LINE_AA)
        label = "strip" if class_id == STRIP_CLASS_ID else f"d:{class_id}"
        cv2.putText(
            overlay,
            label,
            (x + 4, max(18, y + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay


def crop_raw_segments_from_strip(strip_image: np.ndarray) -> list[np.ndarray]:
    h, w = strip_image.shape[:2]
    segments: list[np.ndarray] = []
    for index in range(NUM_SEGMENTS):
        x0 = int(round(index * w / NUM_SEGMENTS))
        x1 = int(round((index + 1) * w / NUM_SEGMENTS))
        segment = strip_image[:, x0:x1].copy()
        segments.append(segment)
    return segments


def upscale_segments(segments: list[np.ndarray], factor: int = RAW_UPSCALE_FACTOR) -> list[np.ndarray]:
    upscaled: list[np.ndarray] = []
    for segment in segments:
        h, w = segment.shape[:2]
        target_w = max(w * factor, 1)
        target_h = max(h * factor, 1)
        interpolation = cv2.INTER_LANCZOS4 if min(h, w) <= 48 else cv2.INTER_CUBIC
        enlarged = cv2.resize(segment, (target_w, target_h), interpolation=interpolation)
        # Light unsharp mask so enlarged digits stay crisp instead of muddy.
        blur = cv2.GaussianBlur(enlarged, (0, 0), sigmaX=0.9, sigmaY=0.9)
        sharpened = cv2.addWeighted(enlarged, 1.18, blur, -0.18, 0)
        upscaled.append(np.clip(sharpened, 0, 255).astype(np.uint8))
    return upscaled


def perspective_warp_strip(image: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = order_points(points.astype(np.float32))
    dst = np.array(
        [
            [0, 0],
            [WARP_HI_W - 1, 0],
            [WARP_HI_W - 1, WARP_HI_H - 1],
            [0, WARP_HI_H - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    inverse = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, matrix, (WARP_HI_W, WARP_HI_H))
    return warped, matrix, inverse


def map_partition_quads_back(inverse_matrix: np.ndarray) -> list[np.ndarray]:
    quads: list[np.ndarray] = []
    cell_w = WARP_HI_W / float(NUM_SEGMENTS)
    for index in range(NUM_SEGMENTS):
        x0 = index * cell_w
        x1 = (index + 1) * cell_w
        quad = np.array(
            [[x0, 0], [x1, 0], [x1, WARP_HI_H - 1], [x0, WARP_HI_H - 1]],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(quad, inverse_matrix).reshape(4, 2)
        quads.append(order_points(mapped))
    return quads


def build_equal_partition_quads_from_bbox(
    bbox_xywh: tuple[float, float, float, float],
) -> list[np.ndarray]:
    x, y, w, h = bbox_xywh
    quads: list[np.ndarray] = []
    cell_w = w / float(NUM_SEGMENTS)
    for index in range(NUM_SEGMENTS):
        x0 = x + (index * cell_w)
        x1 = x + ((index + 1) * cell_w)
        quad = np.array(
            [[x0, y], [x1, y], [x1, y + h], [x0, y + h]],
            dtype=np.float32,
        )
        quads.append(order_points(quad))
    return quads


def make_extraction_result(
    *,
    mode: ExtractionMode,
    source_path: str,
    source_image: np.ndarray,
    label: str,
    search_bbox_xywh: tuple[float, float, float, float],
    strip_quad: np.ndarray,
    digit_quads: list[np.ndarray],
    mode_geometry: dict[str, object],
) -> ExtractionResult:
    source_h, source_w = source_image.shape[:2]
    search_bbox = tuple(float(v) for v in search_bbox_xywh)
    strip_bbox = get_quad_bbox_xywh(strip_quad, source_w, source_h)
    digit_bboxes = [get_quad_bbox_xywh(quad, source_w, source_h) for quad in digit_quads]
    raw_digit_crops = [crop_image_xywh(source_image, box) for box in digit_bboxes]
    processed_digit_crops = [
        prepare_digit_crop_for_lenet(crop, size=LENET_DIGIT_SIZE)
        for crop in raw_digit_crops
    ]
    raw_digit_crops_upscaled = upscale_segments(raw_digit_crops)
    processed_preview_strip = build_processed_digit_contact_sheet(processed_digit_crops)
    must_avoid_boxes = sample_must_avoid_boxes(source_image.shape, strip_bbox, digit_bboxes)
    must_avoid_raw_crops = [crop_image_xywh(source_image, box) for box in must_avoid_boxes]
    must_avoid_processed_crops = [
        prepare_digit_crop_for_lenet(crop, size=LENET_DIGIT_SIZE)
        for crop in must_avoid_raw_crops
    ]
    must_avoid_raw_crops_upscaled = upscale_segments(must_avoid_raw_crops)

    context_crop, translated_boxes, context_bbox = make_square_context_crop(
        source_image,
        [search_bbox] + digit_bboxes,
    )
    yolo_image = cv2.resize(context_crop, (YOLO_SIZE, YOLO_SIZE), interpolation=cv2.INTER_AREA)
    yolo_boxes = resize_boxes_xywh(translated_boxes, context_crop.shape[:2], YOLO_SIZE)

    class_ids = [STRIP_CLASS_ID] + [label_char_to_yolo_class_id(ch) for ch in label]
    yolo_lines = [
        build_yolo_bbox_line(
            class_id=class_ids[index],
            x=box[0],
            y=box[1],
            width=box[2],
            height=box[3],
            image_width=YOLO_SIZE,
            image_height=YOLO_SIZE,
        )
        for index, box in enumerate(yolo_boxes)
    ]

    return ExtractionResult(
        mode=mode,
        source_path=source_path,
        source_image_shape=tuple(int(v) for v in source_image.shape),
        label=label,
        search_bbox_xywh=search_bbox,
        strip_quad=strip_quad.astype(np.float32),
        digit_quads=[quad.astype(np.float32) for quad in digit_quads],
        strip_bbox_xywh=strip_bbox,
        digit_bboxes_xywh=digit_bboxes,
        processed_digit_crops=processed_digit_crops,
        processed_preview_strip=processed_preview_strip,
        raw_digit_crops=raw_digit_crops,
        raw_digit_crops_upscaled=raw_digit_crops_upscaled,
        must_avoid_boxes_xywh=must_avoid_boxes,
        must_avoid_processed_crops=must_avoid_processed_crops,
        must_avoid_raw_crops=must_avoid_raw_crops,
        must_avoid_raw_crops_upscaled=must_avoid_raw_crops_upscaled,
        yolo_image=yolo_image,
        yolo_lines=yolo_lines,
        yolo_boxes_xywh=yolo_boxes,
        yolo_context_bbox_xywh=context_bbox,
        mode_geometry=mode_geometry,
    )


class DraggableHandle(QGraphicsEllipseItem):
    def __init__(self, x: float, y: float, index: int, parent_view: "ManualPerspectiveView"):
        r = HANDLE_RADIUS
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.setPos(x, y)
        self.index = index
        self.parent_view = parent_view
        self.setBrush(QBrush(HANDLE_COLOR))
        self.setPen(QPen(Qt.GlobalColor.white, 1))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setZValue(20)

        labels = ["TL", "TR", "BR", "BL"]
        self._label = QGraphicsTextItem(labels[index], self)
        self._label.setDefaultTextColor(Qt.GlobalColor.yellow)
        font = QFont("Consolas", 8, QFont.Weight.Bold)
        self._label.setFont(font)
        self._label.setPos(r + 2, -r - 2)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.parent_view.update_lines()
            self.parent_view.geometry_changed.emit()
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event):
        self.setBrush(QBrush(HANDLE_HOVER_COLOR))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(QBrush(HANDLE_COLOR))
        super().hoverLeaveEvent(event)


class ManualPerspectiveView(QGraphicsView):
    geometry_changed = pyqtSignal()
    points_ready = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._original_cv_image: np.ndarray | None = None
        self._cv_image: np.ndarray | None = None
        self._rotation_angle = 0
        self._zoom_factor = 1.0
        self._handles: list[DraggableHandle] = []
        self._lines: list[QGraphicsLineItem] = []
        self._placing = False
        self._dragging_selection = False
        self._drag_last_scene_pos = QPointF()

    def load_image(self, image: np.ndarray | None):
        self._original_cv_image = None if image is None else image.copy()
        self._cv_image = None if image is None else image.copy()
        self._rotation_angle = 0
        self._clear_selection()
        self._placing = False
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        if self._cv_image is None:
            self._scene.clear()
            self._pixmap_item = None
            return
        self._render_cv_image(self._cv_image)

    def _render_cv_image(self, img: np.ndarray):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self._scene.clear()
        self._handles.clear()
        self._lines.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect().toRectF()))
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_factor = 1.0

    @staticmethod
    def _rotate_image(image: np.ndarray, angle_deg: int) -> np.ndarray:
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        cos_v = abs(matrix[0, 0])
        sin_v = abs(matrix[0, 1])
        new_w = int((h * sin_v) + (w * cos_v))
        new_h = int((h * cos_v) + (w * sin_v))
        matrix[0, 2] += (new_w / 2.0) - center[0]
        matrix[1, 2] += (new_h / 2.0) - center[1]
        return cv2.warpAffine(
            image,
            matrix,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def set_rotation(self, angle_deg: int):
        if self._original_cv_image is None:
            return
        normalized = int(angle_deg) % 360
        if normalized == self._rotation_angle:
            return
        self._rotation_angle = normalized
        if normalized == 0:
            self._cv_image = self._original_cv_image.copy()
        else:
            self._cv_image = self._rotate_image(self._original_cv_image, normalized)
        self._render_cv_image(self._cv_image)
        self.geometry_changed.emit()

    def get_rotation(self) -> int:
        return self._rotation_angle

    def fit_to_view(self):
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom_factor = 1.0

    def get_cv_image(self) -> np.ndarray | None:
        return None if self._cv_image is None else self._cv_image.copy()

    def start_selection(self):
        if self._pixmap_item is None:
            return
        self._clear_selection()
        self._placing = True
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.geometry_changed.emit()

    def clear_points(self):
        self._clear_selection()
        self._placing = False
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.geometry_changed.emit()

    def get_points(self) -> np.ndarray | None:
        if len(self._handles) != 4:
            return None
        raw = np.array([[h.pos().x(), h.pos().y()] for h in self._handles], dtype=np.float32)
        return order_points(raw)

    def set_points(self, points: np.ndarray) -> bool:
        if self._pixmap_item is None or points is None or points.shape != (4, 2):
            return False
        ordered = order_points(points.astype(np.float32))
        scene_rect = self._scene.sceneRect()
        self._clear_selection()
        self._placing = False
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        for idx, point in enumerate(ordered):
            x = float(np.clip(point[0], scene_rect.left(), scene_rect.right()))
            y = float(np.clip(point[1], scene_rect.top(), scene_rect.bottom()))
            handle = DraggableHandle(x, y, idx, self)
            self._scene.addItem(handle)
            self._handles.append(handle)
        self._create_lines()
        self.points_ready.emit()
        self.geometry_changed.emit()
        return True

    def _clear_selection(self):
        for handle in self._handles:
            self._scene.removeItem(handle)
        for line in self._lines:
            self._scene.removeItem(line)
        self._handles.clear()
        self._lines.clear()
        self._dragging_selection = False

    def _is_over_handle(self, scene_pos: QPointF) -> bool:
        item = self._scene.itemAt(scene_pos, self.transform())
        if isinstance(item, DraggableHandle):
            return True
        return item is not None and isinstance(item.parentItem(), DraggableHandle)

    def _selection_contains_point(self, scene_pos: QPointF) -> bool:
        if len(self._handles) != 4:
            return False
        polygon = QPolygonF([handle.pos() for handle in self._handles])
        path = QPainterPath()
        path.addPolygon(polygon)
        return path.contains(scene_pos)

    def _move_selection_by(self, delta: QPointF):
        if len(self._handles) != 4:
            return
        scene_rect = self._scene.sceneRect()
        xs = [handle.pos().x() for handle in self._handles]
        ys = [handle.pos().y() for handle in self._handles]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        dx = float(delta.x())
        dy = float(delta.y())
        dx = max(dx, scene_rect.left() - min_x)
        dx = min(dx, scene_rect.right() - max_x)
        dy = max(dy, scene_rect.top() - min_y)
        dy = min(dy, scene_rect.bottom() - max_y)
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return
        for handle in self._handles:
            pos = handle.pos()
            handle.setPos(pos.x() + dx, pos.y() + dy)
        self.update_lines()

    def _add_handle(self, scene_pos: QPointF):
        idx = len(self._handles)
        handle = DraggableHandle(scene_pos.x(), scene_pos.y(), idx, self)
        self._scene.addItem(handle)
        self._handles.append(handle)
        if len(self._handles) == 4:
            self._placing = False
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._create_lines()
            self._reorder_handles()
            self.points_ready.emit()
        self.geometry_changed.emit()

    def _reorder_handles(self):
        pts = np.array([[h.pos().x(), h.pos().y()] for h in self._handles], dtype=np.float32)
        ordered = order_points(pts)
        labels = ["TL", "TR", "BR", "BL"]
        for handle in self._handles:
            pos = np.array([handle.pos().x(), handle.pos().y()])
            for index in range(4):
                if np.allclose(pos, ordered[index], atol=0.5):
                    handle.index = index
                    handle._label.setPlainText(labels[index])
                    break
        self._handles.sort(key=lambda handle: handle.index)

    def _create_lines(self):
        pen = QPen(LINE_COLOR, LINE_WIDTH)
        pen.setCosmetic(True)
        for _ in range(4):
            line = QGraphicsLineItem()
            line.setPen(pen)
            line.setZValue(10)
            self._scene.addItem(line)
            self._lines.append(line)
        self.update_lines()

    def update_lines(self):
        if len(self._handles) != 4 or len(self._lines) != 4:
            return
        pts = [handle.pos() for handle in self._handles]
        for index in range(4):
            nxt = (index + 1) % 4
            self._lines[index].setLine(pts[index].x(), pts[index].y(), pts[nxt].x(), pts[nxt].y())

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        self._zoom_factor = max(0.1, min(self._zoom_factor * factor, 20.0))

    def mousePressEvent(self, event):
        if (
            not self._placing
            and event.button() == Qt.MouseButton.LeftButton
            and len(self._handles) == 4
        ):
            scene_pos = self.mapToScene(event.pos())
            if self._selection_contains_point(scene_pos) and not self._is_over_handle(scene_pos):
                self._dragging_selection = True
                self._drag_last_scene_pos = scene_pos
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
        if self._placing and event.button() == Qt.MouseButton.LeftButton:
            self._add_handle(self.mapToScene(event.pos()))
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging_selection:
            scene_pos = self.mapToScene(event.pos())
            delta = scene_pos - self._drag_last_scene_pos
            self._move_selection_by(delta)
            self._drag_last_scene_pos = scene_pos
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging_selection and event.button() == Qt.MouseButton.LeftButton:
            self._dragging_selection = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.geometry_changed.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.clear_points()
        elif event.key() == Qt.Key.Key_F:
            self.fit_to_view()
        super().keyPressEvent(event)


class ExportGuideboxView(QWidget):
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

    def get_workspace_image(self) -> np.ndarray | None:
        if self._render_result is None or self._render_result.workspace_image is None:
            return None
        return self._render_result.workspace_image.copy()

    def get_guidebox_crop(self) -> np.ndarray | None:
        if self._render_result is None or self._render_result.guidebox_crop is None:
            return None
        return self._render_result.guidebox_crop.copy()

    def get_guidebox_rect_xywh(self) -> tuple[int, int, int, int] | None:
        if self._render_result is None:
            return None
        x1, y1, x2, y2 = self._render_result.guidebox_rect_workspace
        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

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

    def _guidebox_rect_widget(self) -> QRectF:
        if self._frame is None:
            return QRectF()
        x, y, w, h = self._frame.guidebox_rect_workspace
        return QRectF(x, y, w, h)

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


class ProcessedPreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._strip_label = QLabel("No LeNet-ready preview yet")
        self._strip_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._strip_label.setStyleSheet("background: #222; border: 1px solid #555; padding: 4px;")
        self._strip_label.setMinimumHeight(96)
        self._strip_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._strip_label)

        self._strip_img: np.ndarray | None = None

    def set_data(self, strip: np.ndarray):
        self._strip_img = strip
        self._refresh()

    def clear(self):
        self._strip_img = None
        self._strip_label.setText("No LeNet-ready preview yet")
        self._strip_label.setPixmap(QPixmap())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._strip_img is not None:
            self._refresh()

    def _refresh(self):
        if self._strip_img is None:
            return
        strip_rect = self._strip_label.contentsRect()
        strip_width = max(strip_rect.width(), FINAL_W * 2)
        strip_height = max(strip_rect.height(), FINAL_H * 2)
        self._strip_label.setText("")
        self._strip_label.setPixmap(
            gray_segment_to_pixmap(self._strip_img, strip_width, strip_height, interpolation=cv2.INTER_NEAREST)
        )


class RawSegmentsPreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(QLabel("Native raw segments"))
        native_row = QHBoxLayout()
        self._native_labels: list[QLabel] = []
        for _ in range(NUM_SEGMENTS):
            label = QLabel()
            label.setMinimumSize(70, 70)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("background: #151515; border: 1px solid #444;")
            native_row.addWidget(label)
            self._native_labels.append(label)
        layout.addLayout(native_row)

        layout.addWidget(QLabel("Upscaled raw segments"))
        upscaled_row = QHBoxLayout()
        self._upscaled_labels: list[QLabel] = []
        for _ in range(NUM_SEGMENTS):
            label = QLabel()
            label.setMinimumSize(70, 70)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("background: #151515; border: 1px solid #444;")
            upscaled_row.addWidget(label)
            self._upscaled_labels.append(label)
        layout.addLayout(upscaled_row)

        self._native_segments: list[np.ndarray] = []
        self._upscaled_segments: list[np.ndarray] = []

    def set_data(self, native_segments: list[np.ndarray], upscaled_segments: list[np.ndarray]):
        self._native_segments = [segment.copy() for segment in native_segments]
        self._upscaled_segments = [segment.copy() for segment in upscaled_segments]
        self._refresh()

    def clear(self):
        self._native_segments = []
        self._upscaled_segments = []
        for label in self._native_labels + self._upscaled_labels:
            label.setPixmap(QPixmap())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._native_segments:
            self._refresh()

    def _refresh(self):
        for label, segment in zip(self._native_labels, self._native_segments):
            rect = label.contentsRect()
            label.setPixmap(color_image_to_pixmap(segment, max(rect.width(), 70), max(rect.height(), 70)))
        for label, segment in zip(self._upscaled_labels, self._upscaled_segments):
            rect = label.contentsRect()
            label.setPixmap(color_image_to_pixmap(segment, max(rect.width(), 70), max(rect.height(), 70)))


class YoloPreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self._image_label = QLabel("No YOLO preview yet")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("background: #111; border: 1px solid #444;")
        self._image_label.setMinimumSize(260, 260)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self._image_label)
        self._image: np.ndarray | None = None

    def set_image(self, image: np.ndarray):
        self._image = image.copy()
        self._refresh()

    def clear(self):
        self._image = None
        self._image_label.setText("No YOLO preview yet")
        self._image_label.setPixmap(QPixmap())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        side = max(260, self.width() - 8)
        self._image_label.setFixedHeight(side)
        if self._image is not None:
            self._refresh()

    def _refresh(self):
        if self._image is None:
            return
        rect = self._image_label.contentsRect()
        side = max(min(rect.width(), rect.height()), 260)
        self._image_label.setText("")
        self._image_label.setPixmap(color_image_to_pixmap(self._image, side, side))


class DatasetExporterMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DigitExtractor Dataset Exporter")
        self.resize(1680, 980)
        self._settings = QSettings(SETTINGS_ORG, SETTINGS_APP)

        self._output_dir = ""
        self._current_image_path = ""
        self._current_image: np.ndarray | None = None
        self._current_result: ExtractionResult | None = None
        self._right_scroll_state = (0, 0)
        self._batch_mode = self._read_batch_mode_setting()
        self._guidebox_batch_template: WorkspaceFrame | None = None
        self._manual_batch_template: ManualPointsTemplate | None = None
        self._batch_processed = 0
        self._batch_errors = 0
        self._batch_skipped = 0
        self._batch_saved_exports = 0
        self._auto_preview_timer = QTimer(self)
        self._auto_preview_timer.setSingleShot(True)
        self._auto_preview_timer.timeout.connect(self._refresh_preview)
        self._syncing_label_fields = False
        self._last_image_dir = self._read_existing_dir_setting(SETTINGS_LAST_IMAGE_DIR)
        self._last_output_dir = self._read_existing_dir_setting(SETTINGS_LAST_OUTPUT_DIR)

        self._build_ui()
        self._connect_signals()
        self._restore_persisted_ui_state()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        sidebar = QGroupBox("Images")
        sidebar_layout = QVBoxLayout(sidebar)
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

        body_splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(body_splitter, stretch=1)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(8)
        body_splitter.addWidget(center)

        top_bar = QGroupBox("Alignment")
        top_layout = QGridLayout(top_bar)
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Manual Perspective", ExtractionMode.MANUAL_PERSPECTIVE.value)
        self._mode_combo.addItem("Fixed Guidebox", ExtractionMode.FIXED_GUIDEBOX.value)
        self._btn_fit = QPushButton("Fit View")
        self._btn_preview = QPushButton("Preview Outputs")
        self._btn_save_current = QPushButton("Save Current")
        self._btn_manual_points = QPushButton("Plot 4 Points")
        self._btn_manual_clear = QPushButton("Clear Points")
        self._batch_checkbox = QCheckBox("Batch Mode")
        self._rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self._rotation_slider.setRange(0, 359)
        self._rotation_slider.setValue(0)
        self._rotation_value = QLabel("0 deg")
        self._rotation_value.setFixedWidth(56)

        top_layout.addWidget(QLabel("Mode"), 0, 0)
        top_layout.addWidget(self._mode_combo, 0, 1)
        top_layout.addWidget(self._btn_fit, 0, 2)
        top_layout.addWidget(self._btn_preview, 0, 3)
        top_layout.addWidget(self._btn_save_current, 0, 4)
        top_layout.addWidget(self._batch_checkbox, 0, 5)
        top_layout.addWidget(self._btn_manual_points, 1, 0)
        top_layout.addWidget(self._btn_manual_clear, 1, 1)
        top_layout.addWidget(QLabel("Rotate"), 1, 2)
        top_layout.addWidget(self._rotation_slider, 1, 3, 1, 2)
        top_layout.addWidget(self._rotation_value, 1, 5)
        center_layout.addWidget(top_bar, stretch=0)

        self._viewer_stack = QStackedWidget()
        self._manual_view = ManualPerspectiveView()
        self._guidebox_view = ExportGuideboxView()
        self._viewer_stack.addWidget(self._manual_view)
        self._viewer_stack.addWidget(self._guidebox_view)
        center_layout.addWidget(self._viewer_stack, stretch=1)

        self._right_scroll_area = QScrollArea()
        self._right_scroll_area.setWidgetResizable(True)
        self._right_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._right_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._right_scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        body_splitter.addWidget(self._right_scroll_area)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        self._right_scroll_area.setWidget(right_panel)
        body_splitter.setSizes([1080, 600])

        self._label_group = QGroupBox("Labeling")
        label_layout = QFormLayout(self._label_group)
        self._current_name_label = QLabel("No image selected")
        self._current_name_label.setWordWrap(True)
        self._single_label_entry = QLineEdit()
        self._single_label_entry.setPlaceholderText("5 chars using 0-9 and X")
        digit_row = QHBoxLayout()
        self._digit_entries: list[QLineEdit] = []
        for _ in range(NUM_SEGMENTS):
            entry = QLineEdit()
            entry.setMaxLength(1)
            entry.setFixedWidth(42)
            entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
            entry.setPlaceholderText("0")
            self._digit_entries.append(entry)
            digit_row.addWidget(entry)
        label_layout.addRow("Image", self._current_name_label)
        label_layout.addRow("5-char label", self._single_label_entry)
        label_layout.addRow("Per-digit", digit_row)
        right_layout.addWidget(self._label_group, stretch=0)

        processed_group = QGroupBox("LeNet-ready Preview")
        processed_layout = QVBoxLayout(processed_group)
        self._processed_preview = ProcessedPreviewWidget()
        processed_layout.addWidget(self._processed_preview)
        right_layout.addWidget(processed_group, stretch=2)

        yolo_group = QGroupBox("YOLO 640 Preview")
        yolo_layout = QVBoxLayout(yolo_group)
        self._yolo_preview = YoloPreviewWidget()
        yolo_layout.addWidget(self._yolo_preview)
        right_layout.addWidget(yolo_group, stretch=2)

        raw_group = QGroupBox("Unprocessed Segments")
        raw_layout = QVBoxLayout(raw_group)
        self._raw_preview = RawSegmentsPreviewWidget()
        raw_layout.addWidget(self._raw_preview)
        right_layout.addWidget(raw_group, stretch=2)

        self._batch_group = QGroupBox("Batch Processing")
        batch_layout = QVBoxLayout(self._batch_group)
        self._batch_state_label = QLabel("Batch mode is off.")
        self._batch_state_label.setWordWrap(True)
        self._batch_template_label = QLabel("Template: not captured")
        self._batch_template_label.setWordWrap(True)
        self._btn_capture_template = QPushButton("Use Current Template For Remaining")
        self._btn_batch_save_next = QPushButton("Save + Next")
        self._btn_batch_skip = QPushButton("Skip This Image")
        self._btn_batch_finish = QPushButton("Finish Batch")
        self._batch_label_entry = QLineEdit()
        self._batch_label_entry.setPlaceholderText("Leave empty to reuse previous label")
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
        right_layout.addWidget(self._batch_group, stretch=0)
        right_layout.addStretch(1)

        self.setStatusBar(QStatusBar(self))
        self._set_batch_controls_enabled(False)
        self._set_manual_buttons_enabled(True)
        self._btn_preview.setEnabled(False)
        self._btn_save_current.setEnabled(False)
        self._btn_fit.setEnabled(False)
        self._rotation_slider.setEnabled(False)
        self._batch_checkbox.setChecked(self._batch_mode)
        self._apply_batch_section_visibility()

        tools_menu = self.menuBar().addMenu("&Tools")
        act_refresh = QAction("Refresh Preview", self)
        act_refresh.triggered.connect(self._refresh_preview)
        tools_menu.addAction(act_refresh)

    def _connect_signals(self):
        self._btn_open_folder.clicked.connect(self._on_open_folder)
        self._btn_set_output.clicked.connect(self._on_set_output)
        self._file_list.currentRowChanged.connect(self._on_file_selected)
        self._btn_fit.clicked.connect(self._fit_active_view)
        self._btn_preview.clicked.connect(self._refresh_preview)
        self._btn_save_current.clicked.connect(self._on_save_current)
        self._btn_manual_points.clicked.connect(self._manual_view.start_selection)
        self._btn_manual_clear.clicked.connect(self._manual_view.clear_points)
        self._rotation_slider.valueChanged.connect(self._on_rotation_changed)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._batch_checkbox.toggled.connect(self._on_batch_toggled)
        self._btn_capture_template.clicked.connect(self._capture_batch_template)
        self._btn_batch_save_next.clicked.connect(self._on_batch_save_next)
        self._btn_batch_skip.clicked.connect(self._on_batch_skip)
        self._btn_batch_finish.clicked.connect(self._finish_batch)
        self._batch_label_entry.returnPressed.connect(self._on_batch_save_next)
        self._single_label_entry.returnPressed.connect(self._on_save_current)

        self._manual_view.geometry_changed.connect(self._on_manual_geometry_changed)
        self._manual_view.points_ready.connect(self._schedule_auto_preview)
        self._guidebox_view.frame_changed.connect(self._on_guidebox_frame_changed)

        self._single_label_entry.textChanged.connect(self._on_full_label_changed)
        for entry in self._digit_entries:
            entry.textChanged.connect(self._on_digit_fields_changed)

        self._right_scroll_area.verticalScrollBar().valueChanged.connect(self._remember_right_scroll_state)
        self._right_scroll_area.horizontalScrollBar().valueChanged.connect(self._remember_right_scroll_state)

    def _current_mode(self) -> ExtractionMode:
        value = self._mode_combo.currentData()
        return ExtractionMode(value)

    def _read_existing_dir_setting(self, key: str) -> str:
        value = str(self._settings.value(key, "", type=str) or "")
        return value if value and Path(value).exists() else ""

    def _read_batch_mode_setting(self) -> bool:
        value = self._settings.value(SETTINGS_BATCH_MODE, True)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in {"0", "false", "no", ""}

    def _restore_persisted_ui_state(self):
        if self._last_output_dir:
            self._output_dir = self._last_output_dir
            self._output_label.setText(f"Output: {self._last_output_dir}")
        if self._batch_mode:
            self._batch_state_label.setText("Batch mode is on. Load an image and capture a template to continue.")
        self._update_batch_state_label()
        self._update_batch_template_label()

    def _apply_batch_section_visibility(self):
        show_batch = self._batch_mode
        self._label_group.setVisible(not show_batch)
        self._batch_group.setVisible(show_batch)
        self._preserve_right_scroll_position()

    def _remember_right_scroll_state(self):
        self._right_scroll_state = (
            self._right_scroll_area.horizontalScrollBar().value(),
            self._right_scroll_area.verticalScrollBar().value(),
        )

    def _restore_right_scroll_state(self):
        h_value, v_value = self._right_scroll_state
        self._right_scroll_area.horizontalScrollBar().setValue(h_value)
        self._right_scroll_area.verticalScrollBar().setValue(v_value)

    def _preserve_right_scroll_position(self):
        self._remember_right_scroll_state()
        QTimer.singleShot(0, self._restore_right_scroll_state)

    def _set_manual_buttons_enabled(self, enabled: bool):
        self._btn_manual_points.setEnabled(enabled)
        self._btn_manual_clear.setEnabled(enabled)

    def _set_batch_controls_enabled(self, enabled: bool):
        self._btn_capture_template.setEnabled(enabled)
        self._btn_batch_save_next.setEnabled(enabled)
        self._btn_batch_skip.setEnabled(enabled)
        self._btn_batch_finish.setEnabled(enabled)
        self._batch_label_entry.setEnabled(enabled)
        self._batch_reuse_previous.setEnabled(enabled)

    def _ensure_output_dir(self) -> bool:
        if self._output_dir:
            return True
        self._on_set_output()
        if self._output_dir:
            return True
        self.statusBar().showMessage("Saving canceled because no output folder is set.")
        return False

    def _capture_manual_batch_template(self) -> ManualPointsTemplate | None:
        image = self._manual_view.get_cv_image()
        points = self._manual_view.get_points()
        if image is None or points is None:
            return None
        h, w = image.shape[:2]
        return ManualPointsTemplate(
            rotation_angle=int(self._manual_view.get_rotation()) % 360,
            normalized_points=normalize_points(points, w, h),
            reference_size=(int(w), int(h)),
        )

    def _apply_manual_batch_template_to_current_image(self) -> bool:
        if self._manual_batch_template is None:
            return False
        template = self._manual_batch_template
        self._manual_view.set_rotation(int(template.rotation_angle) % 360)
        image = self._manual_view.get_cv_image()
        if image is None:
            return False
        h, w = image.shape[:2]
        points = denormalize_points(template.normalized_points, w, h)
        return self._manual_view.set_points(points)

    def _has_valid_preview_geometry(self) -> bool:
        if self._current_image is None or not self._current_image_path:
            return False
        if self._current_mode() == ExtractionMode.MANUAL_PERSPECTIVE:
            return self._manual_view.get_points() is not None
        return self._guidebox_view.get_guidebox_crop() is not None

    def _update_preview_for_geometry_change(self):
        self._current_result = None
        if self._has_valid_preview_geometry():
            self._schedule_auto_preview()
        else:
            self._clear_preview()

    def _pick_directory(self, title: str, start_dir: str = "") -> str:
        dialog = QFileDialog(self, title)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        base_dir = start_dir if start_dir and Path(start_dir).exists() else str(Path.cwd())
        dialog.setDirectory(base_dir)
        if dialog.exec() != QFileDialog.DialogCode.Accepted:
            return ""
        selected = dialog.selectedFiles()
        return selected[0] if selected else ""

    def _on_open_folder(self):
        folder = self._pick_directory("Select Image Folder", self._last_image_dir)
        if not folder:
            return
        self._last_image_dir = folder
        self._settings.setValue(SETTINGS_LAST_IMAGE_DIR, folder)
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
        folder = self._pick_directory("Select Output Folder", self._last_output_dir or self._last_image_dir)
        if not folder:
            return
        self._output_dir = folder
        self._last_output_dir = folder
        self._settings.setValue(SETTINGS_LAST_OUTPUT_DIR, folder)
        self._output_label.setText(f"Output: {folder}")
        self.statusBar().showMessage(f"Output folder set to {folder}")

    def _on_file_selected(self, row: int):
        self._clear_preview()
        self._current_result = None
        if row < 0:
            self._current_image_path = ""
            self._current_image = None
            self._current_name_label.setText("No image selected")
            self._btn_preview.setEnabled(False)
            self._btn_save_current.setEnabled(False)
            self._btn_fit.setEnabled(False)
            self._rotation_slider.setEnabled(False)
            self._manual_view.load_image(None)
            self._guidebox_view.set_source_image(None)
            return

        item = self._file_list.item(row)
        image_path = str(item.data(Qt.ItemDataRole.UserRole))
        image = read_image_any(image_path, cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.warning(self, "Load Error", f"Cannot read:\n{image_path}")
            return

        self._current_image_path = image_path
        self._current_image = image
        self._current_name_label.setText(item.text())
        self._btn_preview.setEnabled(True)
        self._btn_save_current.setEnabled(True)
        self._btn_fit.setEnabled(True)
        self._rotation_slider.setEnabled(True)
        self._rotation_slider.blockSignals(True)
        self._rotation_slider.setValue(0)
        self._rotation_slider.blockSignals(False)
        self._rotation_value.setText("0 deg")

        self._manual_view.load_image(image)
        self._guidebox_view.set_source_image(image)
        self._guidebox_view.reset_view(0.0)
        if self._batch_mode:
            if self._current_mode() == ExtractionMode.FIXED_GUIDEBOX and self._guidebox_batch_template is not None:
                self._guidebox_view.set_workspace_frame(self._guidebox_batch_template, emit_signal=False)
            elif self._current_mode() == ExtractionMode.MANUAL_PERSPECTIVE and self._manual_batch_template is not None:
                self._apply_manual_batch_template_to_current_image()
            self._sync_rotation_slider_from_active_view()

        self._update_batch_state_label()
        self._update_mode_widgets()
        self._update_preview_for_geometry_change()
        self.statusBar().showMessage(f"Viewing {item.text()}.")

    def _fit_active_view(self):
        if self._current_mode() == ExtractionMode.MANUAL_PERSPECTIVE:
            self._manual_view.fit_to_view()
        else:
            self._guidebox_view.fit_to_view()

    def _on_rotation_changed(self, angle: int):
        self._rotation_value.setText(f"{angle} deg")
        if self._current_mode() == ExtractionMode.MANUAL_PERSPECTIVE:
            self._manual_view.set_rotation(angle)
        else:
            self._guidebox_view.set_rotation(angle)

    def _on_mode_changed(self):
        if self._batch_mode and self._current_image is not None:
            if self._current_mode() == ExtractionMode.FIXED_GUIDEBOX and self._guidebox_batch_template is not None:
                self._guidebox_view.set_workspace_frame(self._guidebox_batch_template, emit_signal=False)
            elif self._current_mode() == ExtractionMode.MANUAL_PERSPECTIVE and self._manual_batch_template is not None:
                self._apply_manual_batch_template_to_current_image()
        self._update_mode_widgets()
        self._update_batch_template_label()
        self._update_batch_state_label()
        self._sync_rotation_slider_from_active_view()
        self._status_for_current_mode()
        self._update_preview_for_geometry_change()
        self._preserve_right_scroll_position()

    def _update_mode_widgets(self):
        is_manual = self._current_mode() == ExtractionMode.MANUAL_PERSPECTIVE
        self._viewer_stack.setCurrentWidget(self._manual_view if is_manual else self._guidebox_view)
        self._set_manual_buttons_enabled(is_manual)
        self._batch_checkbox.setEnabled(True)
        self._apply_batch_section_visibility()
        if self._batch_mode:
            self._set_batch_controls_enabled(True)

    def _status_for_current_mode(self):
        if self._current_mode() == ExtractionMode.MANUAL_PERSPECTIVE:
            self.statusBar().showMessage("Manual perspective mode: plot 4 corners, then preview outputs.")
        else:
            self.statusBar().showMessage("Fixed guidebox mode: align inside the 5:1 frame, then preview outputs.")

    def _sync_rotation_slider_from_active_view(self):
        angle = self._manual_view.get_rotation() if self._current_mode() == ExtractionMode.MANUAL_PERSPECTIVE else int(round(self._guidebox_view.get_rotation())) % 360
        self._rotation_slider.blockSignals(True)
        self._rotation_slider.setValue(angle)
        self._rotation_slider.blockSignals(False)
        self._rotation_value.setText(f"{angle} deg")

    def _on_manual_geometry_changed(self):
        self._update_preview_for_geometry_change()

    def _on_guidebox_frame_changed(self):
        self._sync_rotation_slider_from_active_view()
        self._update_preview_for_geometry_change()

    def _schedule_auto_preview(self):
        self._auto_preview_timer.start(120)

    def _on_full_label_changed(self, text: str):
        if self._syncing_label_fields:
            return
        self._syncing_label_fields = True
        normalized = text.strip().upper()[:NUM_SEGMENTS]
        self._single_label_entry.blockSignals(True)
        self._single_label_entry.setText(normalized)
        self._single_label_entry.blockSignals(False)
        for idx, entry in enumerate(self._digit_entries):
            entry.blockSignals(True)
            entry.setText(normalized[idx] if idx < len(normalized) else "")
            entry.blockSignals(False)
        self._syncing_label_fields = False

    def _on_digit_fields_changed(self):
        if self._syncing_label_fields:
            return
        self._syncing_label_fields = True
        chars: list[str] = []
        for entry in self._digit_entries:
            text = entry.text().strip().upper()
            if text and not (text.isdigit() or text == UNREADABLE_LABEL_CHAR):
                text = ""
                entry.blockSignals(True)
                entry.setText("")
                entry.blockSignals(False)
            chars.append(text[:1] if text else "")
        combined = "".join(chars)
        self._single_label_entry.blockSignals(True)
        self._single_label_entry.setText(combined)
        self._single_label_entry.blockSignals(False)
        self._syncing_label_fields = False

    def _current_label(self) -> str:
        return self._single_label_entry.text().strip().upper()

    def _validate_label_or_warn(self, label: str) -> bool:
        if not is_digit_or_unreadable_label(label):
            QMessageBox.warning(self, "Invalid Label", "Enter exactly 5 characters using digits and X.")
            return False
        return True

    def _clear_preview(self):
        self._preserve_right_scroll_position()
        self._processed_preview.clear()
        self._yolo_preview.clear()
        self._raw_preview.clear()

    def _refresh_preview(self):
        if self._current_image is None or not self._current_image_path:
            return
        self._preserve_right_scroll_position()
        label = self._current_label()
        if not is_digit_or_unreadable_label(label):
            self.statusBar().showMessage("Previewing geometry with temporary placeholder label 00000.")
            label = "00000"
        try:
            result = self._extract_current_result(label)
        except ValueError as exc:
            self._current_result = None
            self._clear_preview()
            self.statusBar().showMessage(str(exc))
            return
        self._current_result = result
        self._processed_preview.set_data(result.processed_preview_strip)
        yolo_overlay = draw_boxes_preview(
            result.yolo_image,
            result.yolo_boxes_xywh,
            [STRIP_CLASS_ID] + [label_char_to_yolo_class_id(ch) for ch in result.label],
        )
        self._yolo_preview.set_image(yolo_overlay)
        self._raw_preview.set_data(result.raw_digit_crops, result.raw_digit_crops_upscaled)
        self.statusBar().showMessage("All output previews updated from the current digit boxes.")

    def _extract_current_result(self, label: str) -> ExtractionResult:
        if self._current_image is None or not self._current_image_path:
            raise ValueError("No image selected.")
        if self._current_mode() == ExtractionMode.MANUAL_PERSPECTIVE:
            return self._extract_manual_result(label)
        return self._extract_guidebox_result(label)

    def _extract_manual_result(self, label: str) -> ExtractionResult:
        image = self._manual_view.get_cv_image()
        if image is None:
            raise ValueError("Manual viewer has no image loaded.")
        points = self._manual_view.get_points()
        if points is None:
            raise ValueError("Manual mode needs 4 plotted points before preview/save.")
        ordered_points = order_points(points)
        _warped_strip, _matrix, inverse = perspective_warp_strip(image, ordered_points)
        digit_quads = map_partition_quads_back(inverse)
        search_bbox = get_quad_bbox_xywh(ordered_points, image.shape[1], image.shape[0])
        return make_extraction_result(
            mode=ExtractionMode.MANUAL_PERSPECTIVE,
            source_path=self._current_image_path,
            source_image=image,
            label=label,
            search_bbox_xywh=search_bbox,
            strip_quad=ordered_points,
            digit_quads=digit_quads,
            mode_geometry={
                "rotation_deg": int(self._manual_view.get_rotation()),
                "points": ordered_points.tolist(),
                "warp_size": [WARP_HI_W, WARP_HI_H],
            },
        )

    def _extract_guidebox_result(self, label: str) -> ExtractionResult:
        workspace_image = self._guidebox_view.get_workspace_image()
        guide_crop = self._guidebox_view.get_guidebox_crop()
        guide_rect = self._guidebox_view.get_guidebox_rect_xywh()
        if workspace_image is None or guide_crop is None or guide_rect is None:
            raise ValueError("Guidebox mode needs a loaded image and a valid guidebox crop.")

        gx, gy, gw, gh = guide_rect
        search_bbox = (float(gx), float(gy), float(gw), float(gh))
        strip_quad = np.array(
            [[gx, gy], [gx + gw, gy], [gx + gw, gy + gh], [gx, gy + gh]],
            dtype=np.float32,
        )
        digit_quads = build_equal_partition_quads_from_bbox(search_bbox)
        frame = self._guidebox_view.get_workspace_frame()
        return make_extraction_result(
            mode=ExtractionMode.FIXED_GUIDEBOX,
            source_path=self._current_image_path,
            source_image=workspace_image,
            label=label,
            search_bbox_xywh=search_bbox,
            strip_quad=strip_quad,
            digit_quads=digit_quads,
            mode_geometry={
                "rotation_deg": int(round(self._guidebox_view.get_rotation())) % 360,
                "guidebox_rect_xywh": [gx, gy, gw, gh],
                "yolo_search_bbox_xywh": [gx, gy, gw, gh],
                "workspace_frame": {
                    "rotation_deg": None if frame is None else float(frame.rotation_deg),
                    "scale": None if frame is None else float(frame.scale),
                    "translate_x": None if frame is None else float(frame.translate_x),
                    "translate_y": None if frame is None else float(frame.translate_y),
                    "workspace_size": None if frame is None else list(frame.workspace_size),
                },
            },
        )

    def _on_save_current(self):
        label = self._current_label()
        if not self._validate_label_or_warn(label):
            return
        if not self._ensure_output_dir():
            return
        try:
            result = self._extract_current_result(label)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot Save", str(exc))
            return
        self._current_result = result
        save_summary = self._save_result(result)
        if not self._batch_mode:
            QMessageBox.information(self, "Saved", save_summary)
        self.statusBar().showMessage(f"Saved sample {label}.")

    def _save_result(self, result: ExtractionResult) -> str:
        if not self._output_dir:
            raise ValueError("Output folder is not set.")
        out_dir = Path(self._output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._write_dataset_info_files(out_dir)
        uid = uuid.uuid4().hex[:10]
        base_name = f"{result.label}_{uid}"

        lenet_digits_strips_dir = out_dir / LENET_DIGITS_STRIPS_DIR_NAME
        lenet_digits_segments_dir = out_dir / LENET_DIGITS_SEGMENTS_DIR_NAME
        yolo_images_dir = out_dir / ROI_640_DIR_NAME
        yolo_labels_dir = out_dir / ROI_640_LABELS_DIR_NAME
        yolo_visuals_dir = out_dir / YOLO_VISUALS_DIR_NAME
        raw_digits_segments_dir = out_dir / RAW_DIGITS_SEGMENTS_DIR_NAME
        raw_digits_upscaled_dir = out_dir / RAW_DIGITS_SEGMENTS_UPSCALED_DIR_NAME
        lenet_must_avoid_segments_dir = out_dir / LENET_MUST_AVOID_SEGMENTS_DIR_NAME
        raw_must_avoid_segments_dir = out_dir / RAW_MUST_AVOID_SEGMENTS_DIR_NAME
        raw_must_avoid_upscaled_dir = out_dir / RAW_MUST_AVOID_SEGMENTS_UPSCALED_DIR_NAME
        metadata_dir = out_dir / METADATA_DIR_NAME

        for path in [
            lenet_digits_strips_dir,
            lenet_digits_segments_dir,
            yolo_images_dir,
            yolo_labels_dir,
            yolo_visuals_dir,
            raw_digits_segments_dir,
            raw_digits_upscaled_dir,
            lenet_must_avoid_segments_dir,
            raw_must_avoid_segments_dir,
            raw_must_avoid_upscaled_dir,
            metadata_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        processed_preview_path = lenet_digits_strips_dir / f"{base_name}_digit_preview.png"
        cv2.imwrite(str(processed_preview_path), result.processed_preview_strip)

        processed_segment_paths: list[str] = []
        raw_segment_paths: list[str] = []
        raw_upscaled_paths: list[str] = []
        must_avoid_processed_paths: list[str] = []
        must_avoid_raw_paths: list[str] = []
        must_avoid_raw_upscaled_paths: list[str] = []
        for index, (processed, raw_native, raw_upscaled, label_char) in enumerate(
            zip(result.processed_digit_crops, result.raw_digit_crops, result.raw_digit_crops_upscaled, result.label)
        ):
            folder_name = label_char_to_folder(label_char)
            processed_dir = lenet_digits_segments_dir / folder_name
            raw_dir = raw_digits_segments_dir / folder_name
            upscaled_dir = raw_digits_upscaled_dir / folder_name
            processed_dir.mkdir(parents=True, exist_ok=True)
            raw_dir.mkdir(parents=True, exist_ok=True)
            upscaled_dir.mkdir(parents=True, exist_ok=True)

            processed_path = processed_dir / f"{base_name}_slot{index + 1}_{label_char}.png"
            raw_path = raw_dir / f"{base_name}_slot{index + 1}_{label_char}.png"
            upscaled_path = upscaled_dir / f"{base_name}_slot{index + 1}_{label_char}.png"
            cv2.imwrite(str(processed_path), processed)
            cv2.imwrite(str(raw_path), raw_native)
            cv2.imwrite(str(upscaled_path), raw_upscaled)
            processed_segment_paths.append(str(processed_path))
            raw_segment_paths.append(str(raw_path))
            raw_upscaled_paths.append(str(upscaled_path))

        lenet_must_avoid_segments_dir.mkdir(parents=True, exist_ok=True)
        raw_must_avoid_segments_dir.mkdir(parents=True, exist_ok=True)
        raw_must_avoid_upscaled_dir.mkdir(parents=True, exist_ok=True)
        for index, (processed, raw_native, raw_upscaled) in enumerate(
            zip(
                result.must_avoid_processed_crops,
                result.must_avoid_raw_crops,
                result.must_avoid_raw_crops_upscaled,
            )
        ):
            processed_path = lenet_must_avoid_segments_dir / f"{base_name}_mustavoid{index + 1}.png"
            raw_path = raw_must_avoid_segments_dir / f"{base_name}_mustavoid{index + 1}.png"
            upscaled_path = raw_must_avoid_upscaled_dir / f"{base_name}_mustavoid{index + 1}.png"
            cv2.imwrite(str(processed_path), processed)
            cv2.imwrite(str(raw_path), raw_native)
            cv2.imwrite(str(upscaled_path), raw_upscaled)
            must_avoid_processed_paths.append(str(processed_path))
            must_avoid_raw_paths.append(str(raw_path))
            must_avoid_raw_upscaled_paths.append(str(upscaled_path))

        yolo_image_path = yolo_images_dir / f"{base_name}_640.png"
        yolo_label_path = yolo_labels_dir / f"{base_name}_640.txt"
        yolo_visual_path = yolo_visuals_dir / f"{base_name}_640_visual.png"
        cv2.imwrite(str(yolo_image_path), result.yolo_image)
        yolo_label_path.write_text("\n".join(result.yolo_lines) + "\n", encoding="utf-8")
        yolo_visual = draw_boxes_preview(
            result.yolo_image,
            result.yolo_boxes_xywh,
            [STRIP_CLASS_ID] + [label_char_to_yolo_class_id(ch) for ch in result.label],
        )
        cv2.imwrite(str(yolo_visual_path), yolo_visual)

        metadata = {
            "sample_id": base_name,
            "mode": result.mode.value,
            "source_path": result.source_path,
            "source_image_shape": list(result.source_image_shape),
            "digit_crop_source_image": "manual_view_source" if result.mode == ExtractionMode.MANUAL_PERSPECTIVE else "guidebox_workspace_image",
            "label": result.label,
            "unreadable_flags": [char == UNREADABLE_LABEL_CHAR for char in result.label],
            "yolo_search_bbox_xywh": list(result.search_bbox_xywh),
            "strip_quad": result.strip_quad.tolist(),
            "digit_quads": [quad.tolist() for quad in result.digit_quads],
            "strip_bbox_xywh": list(result.strip_bbox_xywh),
            "digit_boxes_xywh": [list(box) for box in result.digit_bboxes_xywh],
            "must_avoid_boxes_xywh": [list(box) for box in result.must_avoid_boxes_xywh],
            "yolo_context_bbox_xywh": list(result.yolo_context_bbox_xywh),
            "yolo_boxes_xywh": [list(box) for box in result.yolo_boxes_xywh],
            "yolo_label_lines": result.yolo_lines,
            "mode_geometry": result.mode_geometry,
            "output_files": {
                "processed_digit_preview": str(processed_preview_path),
                "processed_digits_32x32": processed_segment_paths,
                "yolo_image": str(yolo_image_path),
                "yolo_label": str(yolo_label_path),
                "yolo_visual": str(yolo_visual_path),
                "raw_digit_crops": raw_segment_paths,
                "raw_digit_crops_upscaled": raw_upscaled_paths,
                "must_avoid_processed_32x32": must_avoid_processed_paths,
                "must_avoid_raw_crops": must_avoid_raw_paths,
                "must_avoid_raw_crops_upscaled": must_avoid_raw_upscaled_paths,
            },
        }
        metadata_path = metadata_dir / f"{base_name}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        self._batch_saved_exports += 1
        return (
            f"Saved sample: {base_name}\n"
            f"LeNet 32x32 digits: {len(processed_segment_paths)}\n"
            f"Digit preview: {processed_preview_path.name}\n"
            f"YOLO image: {yolo_image_path.name}\n"
            f"YOLO labels: {yolo_label_path.name}\n"
            f"YOLO visual: {yolo_visual_path.name}\n"
            f"Raw digit crops: {len(raw_segment_paths)}\n"
            f"Must Avoid negatives: {len(must_avoid_processed_paths)}\n"
            f"Metadata: {metadata_path.name}"
        )

    def _write_dataset_info_files(self, out_dir: Path):
        yolo_class_map_path = out_dir / YOLO_CLASS_MAP_FILE_NAME
        yolo_classes_path = out_dir / YOLO_CLASSES_FILE_NAME
        classifier_info_path = out_dir / CLASSIFIER_DATASET_INFO_FILE_NAME

        yolo_class_map = {
            "format": "yolo_axis_aligned_bbox",
            "class_id_map": build_yolo_class_map(),
            "notes": [
                "The first value on each YOLO label line is the class id.",
                "The remaining four values are normalized x_center, y_center, width, height.",
                "Must Avoid samples are intentionally not mixed into YOLO classes.",
            ],
        }
        yolo_class_map_path.write_text(json.dumps(yolo_class_map, indent=2), encoding="utf-8")
        yolo_classes_path.write_text("\n".join(build_yolo_classes_txt_lines()) + "\n", encoding="utf-8")
        classifier_info_path.write_text(
            json.dumps(build_classifier_dataset_info(), indent=2),
            encoding="utf-8",
        )

    def _on_batch_toggled(self, enabled: bool):
        self._batch_mode = enabled
        self._settings.setValue(SETTINGS_BATCH_MODE, enabled)
        self._set_batch_controls_enabled(enabled)
        self._apply_batch_section_visibility()
        if enabled:
            if self._current_image is None:
                self._batch_state_label.setText("Batch mode is on. Load an image and capture a template to continue.")
                self._update_batch_template_label()
                self.statusBar().showMessage("Batch mode enabled. Load an image and capture a template.")
                return
            self._reset_batch_counters()
            if not self._capture_batch_template():
                self._batch_checkbox.blockSignals(True)
                self._batch_checkbox.setChecked(False)
                self._batch_checkbox.blockSignals(False)
                self._batch_mode = False
                self._settings.setValue(SETTINGS_BATCH_MODE, False)
                self._set_batch_controls_enabled(False)
                self._apply_batch_section_visibility()
                self._batch_state_label.setText("Batch mode is off.")
                return
            self._update_preview_for_geometry_change()
            self.statusBar().showMessage(
                "Batch mode enabled. Save + Next will reuse the current mode template."
            )
        else:
            self._guidebox_batch_template = None
            self._manual_batch_template = None
            self._update_batch_template_label()
            self._batch_state_label.setText("Batch mode is off.")
            self.statusBar().showMessage("Batch mode disabled.")

    def _capture_batch_template(self) -> bool:
        if self._current_mode() == ExtractionMode.FIXED_GUIDEBOX:
            frame = self._guidebox_view.get_workspace_frame()
            if frame is None:
                QMessageBox.warning(self, "Missing Framing", "Align the current image inside the fixed 5:1 guidebox first.")
                return False
            self._guidebox_batch_template = clone_workspace_frame(frame)
            self._manual_batch_template = None
            self._update_batch_template_label()
            self.statusBar().showMessage("Batch guidebox template updated from the current frame.")
            return True

        template = self._capture_manual_batch_template()
        if template is None:
            QMessageBox.warning(
                self,
                "Missing Points",
                "Plot 4 valid points in manual mode before capturing a batch template.",
            )
            return False
        self._manual_batch_template = template
        self._guidebox_batch_template = None
        self._update_batch_template_label()
        self.statusBar().showMessage("Batch manual-point template updated from the current plotted quad.")
        return True

    def _on_batch_save_next(self):
        if not self._batch_mode:
            return
        if not self._ensure_output_dir():
            return
        if not self._capture_batch_template():
            return
        label = self._batch_label_entry.text().strip().upper()
        if not label and self._batch_reuse_previous.isChecked():
            previous = self._current_label()
            if is_digit_or_unreadable_label(previous):
                label = previous
        if not self._validate_label_or_warn(label):
            return
        self._single_label_entry.setText(label)
        try:
            result = self._extract_current_result(label)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot Save", str(exc))
            return
        self._save_result(result)
        self._previous_label_label.setText(f"Previous label: {label}")
        self._batch_label_entry.clear()
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
            self._settings.setValue(SETTINGS_BATCH_MODE, False)
            self._set_batch_controls_enabled(False)
            self._apply_batch_section_visibility()
        self._guidebox_batch_template = None
        self._manual_batch_template = None
        self._update_batch_template_label()
        self._batch_state_label.setText("Batch mode is off.")
        QMessageBox.information(
            self,
            "Batch Summary",
            f"Processed images: {self._batch_processed}\n"
            f"Skipped images: {self._batch_skipped}\n"
            f"Saved sample exports: {self._batch_saved_exports}\n"
            f"Errors: {self._batch_errors}\n"
            f"Output: {self._output_dir or '(not set)'}",
        )
        self.statusBar().showMessage("Batch finished.")

    def _reset_batch_counters(self):
        self._batch_processed = 0
        self._batch_errors = 0
        self._batch_skipped = 0
        self._batch_saved_exports = 0
        self._update_batch_state_label()

    def _update_batch_state_label(self):
        if not self._batch_mode:
            return
        remaining = self._file_list.count()
        current_name = self._current_name_label.text()
        if remaining == 0:
            self._batch_state_label.setText(
                f"{'Manual' if self._current_mode() == ExtractionMode.MANUAL_PERSPECTIVE else 'Guidebox'} batch mode.\n"
                "Load an image and capture a template to begin."
            )
            return
        self._batch_state_label.setText(
            f"{'Manual' if self._current_mode() == ExtractionMode.MANUAL_PERSPECTIVE else 'Guidebox'} batch mode.\n"
            f"Current: {current_name}\n"
            f"Processed: {self._batch_processed} | Remaining: {remaining} | Skipped: {self._batch_skipped}"
        )

    def _update_batch_template_label(self):
        if self._current_mode() == ExtractionMode.MANUAL_PERSPECTIVE and self._manual_batch_template is not None:
            template = self._manual_batch_template
            points_text = ", ".join(
                f"({point[0]:.3f}, {point[1]:.3f})" for point in template.normalized_points
            )
            self._batch_template_label.setText(
                "Template captured from current manual quad.\n"
                f"Rotation: {int(template.rotation_angle) % 360} deg | "
                f"Reference size: {template.reference_size[0]}x{template.reference_size[1]}\n"
                f"Normalized points: {points_text}"
            )
            return
        if self._current_mode() == ExtractionMode.FIXED_GUIDEBOX and self._guidebox_batch_template is not None:
            template = self._guidebox_batch_template
            self._batch_template_label.setText(
                "Template captured from current guidebox frame.\n"
                f"Rotation: {int(round(template.rotation_deg)) % 360} deg | "
                f"Scale: {template.scale:.3f} | "
                f"Pan: ({template.translate_x:.1f}, {template.translate_y:.1f})"
            )
            return
        if self._manual_batch_template is not None:
            template = self._manual_batch_template
            self._batch_template_label.setText(
                "Manual template available.\n"
                f"Rotation: {int(template.rotation_angle) % 360} deg | "
                f"Reference size: {template.reference_size[0]}x{template.reference_size[1]}"
            )
            return
        if self._guidebox_batch_template is not None:
            template = self._guidebox_batch_template
            self._batch_template_label.setText(
                "Guidebox template available.\n"
                f"Rotation: {int(round(template.rotation_deg)) % 360} deg | "
                f"Scale: {template.scale:.3f}"
            )
            return
        if self._guidebox_batch_template is None and self._manual_batch_template is None:
            self._batch_template_label.setText("Template: not captured")
            return


def main():
    app = QApplication(sys.argv)
    window = DatasetExporterMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
