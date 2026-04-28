"""
High-Precision Image Dataset Extractor
PyQt6 + OpenCV application for extracting and segmenting digit images.
"""

import sys
import os
import uuid
import importlib
import math
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from itertools import product

import cv2
import numpy as np
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from digit_ml_commands import (
    MlCommandWorker as ExternalMlCommandWorker,
    build_lenet_predict_batch_command,
    build_lenet_predict_digits_command,
    build_lenet_predict_command,
    build_lenet_train_command,
    build_yolo_predict_command,
    build_yolo_predict_windows_command,
    build_yolo_train_command,
    get_ml_backend_script_path,
    get_python_version,
    is_supported_tensorflow_backend,
    write_temp_image,
    write_temp_images,
    write_temp_strip_image,
)
from auto_read_pipeline import (
    generate_bbox_candidates,
    generate_quad_candidates,
    generate_strip_candidates,
    vote_prediction_candidates,
)
from digit_candidate_detection import detect_digit_candidates
from digit_strip_detection import bbox_to_quad, find_digit_strip_quad
from digit_ml_dialogs import (
    AutoReadResultsDialog,
    DigitDiagnosisDialog,
    LeNetTrainingDialog as ExternalLeNetTrainingDialog,
    YoloTrainingDialog as ExternalYoloTrainingDialog,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HANDLE_RADIUS = 7          # pixels on screen
HANDLE_COLOR = QColor(0, 200, 255, 220)
HANDLE_HOVER_COLOR = QColor(255, 100, 0, 220)
LINE_COLOR = QColor(0, 255, 100, 180)
LINE_WIDTH = 2
WARP_HI_W, WARP_HI_H = 500, 100       # high-res buffer
FINAL_W, FINAL_H = 140, 28             # final strip size
SEGMENT_SIZE = 28                       # each digit cell
NUM_SEGMENTS = 5
UNREADABLE_LABEL_CHAR = "X"
UNREADABLE_FOLDER_NAME = "Unreadable"
ROI_RAW_DIR_NAME = "ROI_raw"
ROI_640_DIR_NAME = "ROI_640"
ROI_640_LABELS_DIR_NAME = "ROI_640_labels"
ROI_SIZE = 640
ROI_YOLO_CLASS_ID = 0
ROI_CONTEXT_MARGIN_RATIO = 0.20
LENET_MODEL_DIR_NAME = "trained_models"
LENET_KERAS_FILENAME = "lenet5_digits.keras"
LENET_TFLITE_FILENAME = "lenet5_digits.tflite"
LENET_LABELS_FILENAME = "labels.json"
LENET_METRICS_FILENAME = "metrics.json"
IMAGE_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.heic', '.heif'
}
HEIF_DECODER_AVAILABLE = False
_PIL_IMAGE_MODULE = None


def _ensure_heif_decoder() -> bool:
    """Lazily initialize HEIC/HEIF decoder dependencies."""
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
    """Read common image formats with OpenCV, plus HEIC/HEIF via Pillow fallback."""
    image = cv2.imread(path, flags)
    if image is not None:
        return image

    ext = Path(path).suffix.lower()
    if ext not in {'.heic', '.heif'} or not _ensure_heif_decoder():
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


# ---------------------------------------------------------------------------
# Utility: order 4 points as TL, TR, BR, BL
# ---------------------------------------------------------------------------
def order_points(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points into Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]   # TL has smallest x+y
    rect[2] = pts[np.argmax(s)]   # BR has largest  x+y
    rect[1] = pts[np.argmin(d)]   # TR has smallest x-y
    rect[3] = pts[np.argmax(d)]   # BL has largest  x-y
    return rect


def extract_processed_strips(
    image: np.ndarray,
    points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return both binarized strip (for saving) and grayscale strip (for readable preview)."""
    src = order_points(points)
    dst = np.array([
        [0, 0],
        [WARP_HI_W - 1, 0],
        [WARP_HI_W - 1, WARP_HI_H - 1],
        [0, WARP_HI_H - 1]
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (WARP_HI_W, WARP_HI_H))

    if len(warped.shape) == 3:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    else:
        gray = warped

    preview_strip = cv2.resize(
        gray,
        (FINAL_W, FINAL_H),
        interpolation=cv2.INTER_AREA
    )

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )
    binary = cv2.medianBlur(binary, 3)
    binary_strip = cv2.resize(
        binary,
        (FINAL_W, FINAL_H),
        interpolation=cv2.INTER_AREA
    )

    return binary_strip, preview_strip


def split_strip_segments(strip: np.ndarray) -> list[np.ndarray]:
    """Split a 140x28 strip into 5 independent 28x28 cells."""
    segments: list[np.ndarray] = []
    for i in range(NUM_SEGMENTS):
        x0 = i * SEGMENT_SIZE
        segments.append(strip[:, x0:x0 + SEGMENT_SIZE].copy())
    return segments


def get_points_bounding_rect(
    points: np.ndarray,
    width: int,
    height: int
) -> tuple[int, int, int, int] | None:
    """Get a clamped axis-aligned bounding box from the 4 selected points."""
    if points is None or len(points) != 4:
        return None

    ordered = order_points(points)
    pts = np.round(ordered).astype(np.int32)
    if pts.shape != (4, 2):
        return None

    pts[:, 0] = np.clip(pts[:, 0], 0, max(width - 1, 0))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(height - 1, 0))

    x, y, bw, bh = cv2.boundingRect(pts)
    if bw <= 0 or bh <= 0:
        return None
    return int(x), int(y), int(bw), int(bh)


def extract_roi_crops_with_context(
    image: np.ndarray,
    points: np.ndarray,
    margin_ratio: float = ROI_CONTEXT_MARGIN_RATIO,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]] | None:
    """Return tight ROI crop and square context crop from the real scene."""
    if image is None:
        return None

    h, w = image.shape[:2]
    bbox = get_points_bounding_rect(points, w, h)
    if bbox is None:
        return None

    x, y, bw, bh = bbox
    raw_crop = image[y:y + bh, x:x + bw].copy()
    if raw_crop.size == 0:
        return None

    side = int(round(max(bw, bh) * (1.0 + (2.0 * margin_ratio))))
    side = max(side, bw, bh, 1)

    cx = x + (bw / 2.0)
    cy = y + (bh / 2.0)
    x0 = int(round(cx - (side / 2.0)))
    y0 = int(round(cy - (side / 2.0)))
    x1 = x0 + side
    y1 = y0 + side

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        image_for_crop = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT_101,
        )
    else:
        image_for_crop = image

    x0 += pad_left
    y0 += pad_top
    x1 += pad_left
    y1 += pad_top

    square_context_crop = image_for_crop[y0:y1, x0:x1].copy()
    if square_context_crop.size == 0:
        return None

    bbox_in_square = (
        (x + pad_left) - x0,
        (y + pad_top) - y0,
        bw,
        bh,
    )
    return raw_crop, square_context_crop, bbox_in_square


def letterbox_to_square_with_meta(
    image: np.ndarray,
    size: int = ROI_SIZE,
) -> tuple[np.ndarray, float, int, int]:
    """Resize a square scene crop to target size without synthetic fill."""
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        fallback = np.zeros((size, size, 3), dtype=np.uint8)
        return fallback, 1.0, 0, 0

    scale = size / max(float(w), float(h))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (size, size), interpolation=interpolation)
    return resized, scale, 0, 0


def build_yolo_bbox_line(
    x: float,
    y: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
    class_id: int = ROI_YOLO_CLASS_ID,
) -> str:
    """Create one YOLO label line from pixel XYWH box coordinates."""
    cx = (x + (width / 2.0)) / max(float(image_width), 1.0)
    cy = (y + (height / 2.0)) / max(float(image_height), 1.0)
    bw = width / max(float(image_width), 1.0)
    bh = height / max(float(image_height), 1.0)

    cx = float(np.clip(cx, 0.0, 1.0))
    cy = float(np.clip(cy, 0.0, 1.0))
    bw = float(np.clip(bw, 0.0, 1.0))
    bh = float(np.clip(bh, 0.0, 1.0))
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def normalize_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalize points to [0, 1] coordinates so they can be reused on other image sizes."""
    x_den = max(width - 1, 1)
    y_den = max(height - 1, 1)
    normalized = points.astype(np.float32).copy()
    normalized[:, 0] = np.clip(normalized[:, 0] / x_den, 0.0, 1.0)
    normalized[:, 1] = np.clip(normalized[:, 1] / y_den, 0.0, 1.0)
    return normalized


def denormalize_points(normalized_points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Map normalized [0, 1] points onto the target image size."""
    x_max = max(width - 1, 1)
    y_max = max(height - 1, 1)
    points = normalized_points.astype(np.float32).copy()
    points[:, 0] = np.clip(points[:, 0] * x_max, 0, x_max)
    points[:, 1] = np.clip(points[:, 1] * y_max, 0, y_max)
    return points


def is_digit_or_unreadable_label(label: str) -> bool:
    normalized = label.strip().upper()
    return len(normalized) == NUM_SEGMENTS and all(
        ch.isdigit() or ch == UNREADABLE_LABEL_CHAR for ch in normalized
    )


def gray_segment_to_pixmap(
    gray_segment: np.ndarray,
    width: int,
    height: int,
    interpolation: int = cv2.INTER_NEAREST
) -> QPixmap:
    disp_seg = cv2.resize(gray_segment, (width, height), interpolation=interpolation)
    qimg = QImage(disp_seg.data, width, height, width, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(qimg.copy())


def prepare_strip_image(image: np.ndarray) -> np.ndarray:
    """Normalize a digit-strip image to the expected 140x28 grayscale layout."""
    if image is None:
        raise ValueError("Empty strip image.")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if gray.size == 0:
        raise ValueError("Empty strip image.")

    interpolation = cv2.INTER_AREA if gray.shape[1] >= FINAL_W else cv2.INTER_LINEAR
    return cv2.resize(gray, (FINAL_W, FINAL_H), interpolation=interpolation)


def prepare_guidebox_strip(image: np.ndarray) -> np.ndarray:
    """Convert a guidebox crop into the inverted 140x28 strip format used by the reader."""
    if image is None or image.size == 0:
        raise ValueError("Empty guidebox crop.")

    if len(image.shape) == 3:
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


@dataclass
class GuideboxFramingTemplate:
    rotation_angle: int
    zoom_factor: float
    center_x_norm: float
    center_y_norm: float
    width_norm: float
    height_norm: float


def clamp_normalized_rect(
    x_norm: float,
    y_norm: float,
    width_norm: float,
    height_norm: float,
) -> tuple[float, float, float, float]:
    width_norm = float(max(width_norm, 1e-6))
    height_norm = float(max(height_norm, 1e-6))
    x_norm = float(min(max(x_norm, 0.0), 1.0 - width_norm))
    y_norm = float(min(max(y_norm, 0.0), 1.0 - height_norm))
    width_norm = float(min(width_norm, 1.0))
    height_norm = float(min(height_norm, 1.0))
    return x_norm, y_norm, width_norm, height_norm


def crop_image_with_normalized_rect(
    image: np.ndarray,
    x_norm: float,
    y_norm: float,
    width_norm: float,
    height_norm: float,
) -> np.ndarray | None:
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return None

    x_norm, y_norm, width_norm, height_norm = clamp_normalized_rect(
        x_norm,
        y_norm,
        width_norm,
        height_norm,
    )

    x1 = max(0, min(int(round(x_norm * w)), w - 1))
    y1 = max(0, min(int(round(y_norm * h)), h - 1))
    x2 = max(x1 + 1, min(int(round((x_norm + width_norm) * w)), w))
    y2 = max(y1 + 1, min(int(round((y_norm + height_norm) * h)), h))
    crop = image[y1:y2, x1:x2].copy()
    return crop if crop.size else None


def prepare_guidebox_dataset_image(image: np.ndarray) -> np.ndarray:
    """Letterbox the guidebox crop into a square reference image without YOLO metadata."""
    roi_640, _scale, _pad_left, _pad_top = letterbox_to_square_with_meta(image, ROI_SIZE)
    return roi_640


# ---------------------------------------------------------------------------
# CV Processing Worker (runs on a QThread to keep UI responsive)
# ---------------------------------------------------------------------------
class ProcessingSignals(QObject):
    finished = pyqtSignal(object)   # emits the 140x28 strip (numpy)
    error = pyqtSignal(str)


class WarpWorker(QThread):
    """Guidebox crop preprocessing in a background thread."""

    def __init__(self, guide_crop: np.ndarray):
        super().__init__()
        self.signals = ProcessingSignals()
        self.guide_crop = guide_crop

    def run(self):
        try:
            strip = prepare_guidebox_strip(self.guide_crop)
            self.signals.finished.emit(strip)
        except Exception as exc:
            self.signals.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Draggable Handle (QGraphicsEllipseItem)
# ---------------------------------------------------------------------------
class DraggableHandle(QGraphicsEllipseItem):
    """A circular handle the user can drag to fine-tune corner position."""

    def __init__(self, x: float, y: float, index: int, parent_view):
        r = HANDLE_RADIUS
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.setPos(x, y)
        self.index = index
        self.parent_view = parent_view
        self.setBrush(QBrush(HANDLE_COLOR))
        self.setPen(QPen(Qt.GlobalColor.white, 1))
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setZValue(20)

        # Label text
        labels = ["TL", "TR", "BR", "BL"]
        self._label = QGraphicsTextItem(labels[index], self)
        self._label.setDefaultTextColor(Qt.GlobalColor.yellow)
        font = QFont("Consolas", 8, QFont.Weight.Bold)
        self._label.setFont(font)
        self._label.setPos(r + 2, -r - 2)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.GraphicsItemChange.ItemPositionHasChanged:
            self.parent_view.update_lines()
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event):
        self.setBrush(QBrush(HANDLE_HOVER_COLOR))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(QBrush(HANDLE_COLOR))
        super().hoverLeaveEvent(event)


# ---------------------------------------------------------------------------
# Image Viewer (QGraphicsView)
# ---------------------------------------------------------------------------
class ImageViewer(QGraphicsView):
    """Zoomable / pannable image viewer with guidebox-first framing."""

    points_ready = pyqtSignal()   # emitted when 4 points placed

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._original_cv_image: np.ndarray | None = None
        self._cv_image: np.ndarray | None = None
        self._rotation_angle = 0
        self._zoom_factor = 1.0

        # Selection state
        self._handles: list[DraggableHandle] = []
        self._lines: list[QGraphicsLineItem] = []
        self._placing = False          # True while user is clicking corners
        self._dragging_selection = False
        self._drag_last_scene_pos = QPointF()
        self._guidebox_enabled = True

    # -- public API ----------------------------------------------------------

    def load_image(self, path: str):
        """Load an image from disk and display it."""
        img = read_image_any(path, cv2.IMREAD_COLOR)
        if img is None:
            if Path(path).suffix.lower() in {'.heic', '.heif'} and not HEIF_DECODER_AVAILABLE:
                QMessageBox.warning(
                    self,
                    "Load Error",
                    "Cannot read HEIC/HEIF image.\n"
                    "Install HEIC support dependencies (Pillow + pillow-heif)\n"
                    f"and try again:\n{path}",
                )
                return
            QMessageBox.warning(self, "Load Error", f"Cannot read:\n{path}")
            return
        self._original_cv_image = img
        self._cv_image = img.copy()
        self._rotation_angle = 0
        self._render_cv_image(self._cv_image)
        self._placing = False

    def set_rotation(self, angle_deg: int) -> bool:
        """Rotate displayed image to an absolute angle in degrees (0-359)."""
        if self._original_cv_image is None:
            return False

        normalized = int(angle_deg) % 360
        if normalized == self._rotation_angle:
            return False

        self._rotation_angle = normalized
        if normalized == 0:
            self._cv_image = self._original_cv_image.copy()
        else:
            self._cv_image = self._rotate_image(self._original_cv_image, normalized)

        self._render_cv_image(self._cv_image)
        self._clear_selection()
        self._placing = False
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        return True

    def _render_cv_image(self, img: np.ndarray):
        """Render the given OpenCV image onto the graphics scene."""
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
        """Rotate image around center while expanding canvas to keep full content."""
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

    def start_selection(self):
        """Enter point-placement mode (clear old selection)."""
        self._clear_selection()
        self._placing = True
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def get_points(self) -> np.ndarray | None:
        """Return ordered 4×2 float32 array or None."""
        if len(self._handles) != 4:
            return None
        raw = np.array(
            [[h.pos().x(), h.pos().y()] for h in self._handles],
            dtype=np.float32,
        )
        return order_points(raw)

    def get_cv_image(self) -> np.ndarray | None:
        return self._cv_image

    def set_guidebox_enabled(self, enabled: bool):
        self._guidebox_enabled = bool(enabled)
        self.viewport().update()

    def get_guidebox_viewport_rect(self) -> QRectF | None:
        if not self._guidebox_enabled:
            return None

        vp = self.viewport().rect()
        if vp.width() <= 20 or vp.height() <= 20:
            return None

        margin_w = max(int(vp.width() * 0.08), 24)
        margin_h = max(int(vp.height() * 0.18), 24)
        max_width = max(vp.width() - (2 * margin_w), 40)
        max_height = max(vp.height() - (2 * margin_h), 20)
        width = min(float(max_width), float(max_height) * 5.0)
        height = width / 5.0
        x = (vp.width() - width) / 2.0
        y = (vp.height() - height) / 2.0
        return QRectF(x, y, width, height)

    def get_guidebox_scene_rect(self) -> QRectF | None:
        if self._pixmap_item is None or self._cv_image is None:
            return None

        guide_rect = self.get_guidebox_viewport_rect()
        if guide_rect is None:
            return None

        top_left = self.mapToScene(QPoint(int(round(guide_rect.left())), int(round(guide_rect.top()))))
        bottom_right = self.mapToScene(QPoint(int(round(guide_rect.right())), int(round(guide_rect.bottom()))))
        scene_rect = QRectF(top_left, bottom_right).normalized()
        image_rect = self._scene.sceneRect()
        return scene_rect.intersected(image_rect)

    def get_guidebox_crop(self) -> np.ndarray | None:
        if self._cv_image is None:
            return None

        scene_rect = self.get_guidebox_scene_rect()
        if scene_rect is None or scene_rect.width() < 5 or scene_rect.height() < 5:
            return None

        x1 = max(int(round(scene_rect.left())), 0)
        y1 = max(int(round(scene_rect.top())), 0)
        x2 = min(int(round(scene_rect.right())), self._cv_image.shape[1])
        y2 = min(int(round(scene_rect.bottom())), self._cv_image.shape[0])
        if x2 <= x1 or y2 <= y1:
            return None
        crop = self._cv_image[y1:y2, x1:x2].copy()
        return crop if crop.size else None

    def capture_framing_template(self) -> GuideboxFramingTemplate | None:
        if self._cv_image is None:
            return None

        scene_rect = self.get_guidebox_scene_rect()
        if scene_rect is None or scene_rect.width() < 5 or scene_rect.height() < 5:
            return None

        h, w = self._cv_image.shape[:2]
        if h <= 0 or w <= 0:
            return None

        width_norm = float(scene_rect.width() / w)
        height_norm = float(scene_rect.height() / h)
        center_x_norm = float(scene_rect.center().x() / w)
        center_y_norm = float(scene_rect.center().y() / h)
        return GuideboxFramingTemplate(
            rotation_angle=int(self._rotation_angle) % 360,
            zoom_factor=float(self._zoom_factor),
            center_x_norm=center_x_norm,
            center_y_norm=center_y_norm,
            width_norm=width_norm,
            height_norm=height_norm,
        )

    def apply_framing_template(self, template: GuideboxFramingTemplate) -> bool:
        if self._cv_image is None:
            return False

        h, w = self._cv_image.shape[:2]
        current_rect = self.get_guidebox_scene_rect()
        if current_rect is None or current_rect.width() < 5 or current_rect.height() < 5:
            return False

        desired_width = max(float(template.width_norm) * w, 1.0)
        desired_height = max(float(template.height_norm) * h, 1.0)
        desired_left = (float(template.center_x_norm) * w) - (desired_width / 2.0)
        desired_top = (float(template.center_y_norm) * h) - (desired_height / 2.0)
        desired_center_x = desired_left + (desired_width / 2.0)
        desired_center_y = desired_top + (desired_height / 2.0)

        width_ratio = current_rect.width() / desired_width
        height_ratio = current_rect.height() / desired_height
        target_zoom = float(self._zoom_factor * ((width_ratio + height_ratio) / 2.0))
        self.set_zoom_factor(target_zoom)
        self.centerOn(desired_center_x, desired_center_y)

        corrected_rect = self.get_guidebox_scene_rect()
        if corrected_rect is not None and corrected_rect.width() >= 5 and corrected_rect.height() >= 5:
            dx = desired_center_x - corrected_rect.center().x()
            dy = desired_center_y - corrected_rect.center().y()
            if abs(dx) > 0.5 or abs(dy) > 0.5:
                self.centerOn(corrected_rect.center().x() + dx, corrected_rect.center().y() + dy)
        return True

    def get_zoom_factor(self) -> float:
        return self._zoom_factor

    def set_zoom_factor(self, zoom_factor: float):
        if self._pixmap_item is None:
            return
        target = float(max(0.1, min(zoom_factor, 20.0)))
        if abs(target - self._zoom_factor) < 1e-6:
            return
        if self._zoom_factor <= 0:
            self._zoom_factor = 1.0
        ratio = target / self._zoom_factor
        previous_transform_anchor = self.transformationAnchor()
        previous_resize_anchor = self.resizeAnchor()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.scale(ratio, ratio)
        self.setTransformationAnchor(previous_transform_anchor)
        self.setResizeAnchor(previous_resize_anchor)
        self._zoom_factor = target

    def set_points(self, points: np.ndarray) -> bool:
        """Set 4 handles programmatically so users can readjust instead of redeclare."""
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
        return True

    def fit_to_view(self):
        if self._pixmap_item:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom_factor = 1.0

    # -- internal helpers ----------------------------------------------------

    def _clear_selection(self):
        for h in self._handles:
            self._scene.removeItem(h)
        for l in self._lines:
            self._scene.removeItem(l)
        self._handles.clear()
        self._lines.clear()
        self._dragging_selection = False

    def _is_over_handle(self, scene_pos: QPointF) -> bool:
        item = self._scene.itemAt(scene_pos, self.transform())
        if isinstance(item, DraggableHandle):
            return True
        if item is not None and isinstance(item.parentItem(), DraggableHandle):
            return True
        return False

    def _selection_contains_point(self, scene_pos: QPointF) -> bool:
        if len(self._handles) != 4:
            return False
        polygon = QPolygonF([h.pos() for h in self._handles])
        path = QPainterPath()
        path.addPolygon(polygon)
        return path.contains(scene_pos)

    def _move_selection_by(self, delta: QPointF):
        if len(self._handles) != 4:
            return

        scene_rect = self._scene.sceneRect()
        xs = [h.pos().x() for h in self._handles]
        ys = [h.pos().y() for h in self._handles]
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
        h = DraggableHandle(scene_pos.x(), scene_pos.y(), idx, self)
        self._scene.addItem(h)
        self._handles.append(h)

        if len(self._handles) == 4:
            self._placing = False
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._create_lines()
            self._reorder_handles()
            self.points_ready.emit()

    def _reorder_handles(self):
        """Re-label handles after auto-sorting so the labels match."""
        pts = np.array(
            [[h.pos().x(), h.pos().y()] for h in self._handles],
            dtype=np.float32,
        )
        ordered = order_points(pts)
        labels = ["TL", "TR", "BR", "BL"]
        for i, h in enumerate(self._handles):
            pos = np.array([h.pos().x(), h.pos().y()])
            for j in range(4):
                if np.allclose(pos, ordered[j], atol=0.5):
                    h.index = j
                    h._label.setPlainText(labels[j])
                    break
        # Sort handles list by index so get_points returns them in order
        self._handles.sort(key=lambda h: h.index)

    def _create_lines(self):
        pen = QPen(LINE_COLOR, LINE_WIDTH)
        pen.setCosmetic(True)
        for i in range(4):
            line = QGraphicsLineItem()
            line.setPen(pen)
            line.setZValue(10)
            self._scene.addItem(line)
            self._lines.append(line)
        self.update_lines()

    def update_lines(self):
        if len(self._handles) != 4 or len(self._lines) != 4:
            return
        pts = [h.pos() for h in self._handles]
        for i in range(4):
            j = (i + 1) % 4
            self._lines[i].setLine(
                pts[i].x(), pts[i].y(),
                pts[j].x(), pts[j].y(),
            )

    # -- events --------------------------------------------------------------

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
            scene_pos = self.mapToScene(event.pos())
            self._add_handle(scene_pos)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging_selection:
            scene_pos = self.mapToScene(event.pos())
            delta = scene_pos - self._drag_last_scene_pos
            self._move_selection_by(delta)
            self._drag_last_scene_pos = self.mapToScene(event.pos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging_selection and event.button() == Qt.MouseButton.LeftButton:
            self._dragging_selection = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._clear_selection()
            self._placing = False
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif event.key() == Qt.Key.Key_F:
            self.fit_to_view()
        super().keyPressEvent(event)

    def drawForeground(self, painter: QPainter, rect: QRectF):
        super().drawForeground(painter, rect)
        if not self._guidebox_enabled or self._pixmap_item is None:
            return

        guide_rect = self.get_guidebox_viewport_rect()
        if guide_rect is None:
            return

        painter.save()
        painter.resetTransform()

        viewport_rect = QRectF(self.viewport().rect())
        overlay_path = QPainterPath()
        overlay_path.addRect(viewport_rect)
        cutout_path = QPainterPath()
        cutout_path.addRoundedRect(guide_rect, 6.0, 6.0)
        painter.fillPath(overlay_path.subtracted(cutout_path), QColor(0, 0, 0, 110))

        border_pen = QPen(QColor(0, 220, 255, 230), 2)
        painter.setPen(border_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(guide_rect, 6.0, 6.0)

        column_pen = QPen(QColor(255, 255, 255, 90), 1, Qt.PenStyle.DashLine)
        painter.setPen(column_pen)
        for idx in range(1, NUM_SEGMENTS):
            x = guide_rect.left() + (guide_rect.width() * idx / NUM_SEGMENTS)
            painter.drawLine(QPointF(x, guide_rect.top()), QPointF(x, guide_rect.bottom()))

        label = "5:1 guidebox"
        font = painter.font()
        font.setPointSize(max(font.pointSize(), 10))
        painter.setFont(font)
        text_rect = QRectF(
            guide_rect.left(),
            max(guide_rect.top() - 28.0, 4.0),
            guide_rect.width(),
            22.0,
        )
        painter.setPen(QColor(220, 245, 255, 230))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, label)
        painter.restore()


# ---------------------------------------------------------------------------
# Preview Widget — shows the processed 140×28 strip and segments
# ---------------------------------------------------------------------------
class PreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._strip_label = QLabel("No preview yet")
        self._strip_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._strip_label.setStyleSheet(
            "background: #222; border: 1px solid #555; padding: 4px;"
        )
        self._strip_label.setMinimumHeight(56)
        self._strip_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._strip_label)

        seg_box = QGroupBox("Segments (28×28)")
        seg_layout = QHBoxLayout(seg_box)
        seg_layout.setSpacing(8)
        self._seg_labels: list[QLabel] = []
        for i in range(NUM_SEGMENTS):
            lbl = QLabel()
            lbl.setMinimumSize(56, 56)
            lbl.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding
            )
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background: #1a1a1a; border: 1px solid #444;")
            seg_layout.addWidget(lbl)
            self._seg_labels.append(lbl)
        layout.addWidget(seg_box)

        self._strip_img: np.ndarray | None = None
        self._segments: list[np.ndarray] = []

    def set_strip(self, strip: np.ndarray):
        self._strip_img = strip
        self._segments = split_strip_segments(strip)
        self._refresh_preview_pixmaps()

    def get_segments(self) -> list[np.ndarray]:
        return self._segments

    def clear(self):
        self._strip_img = None
        self._segments = []
        self._strip_label.setText("No preview yet")
        self._strip_label.setPixmap(QPixmap())
        for lbl in self._seg_labels:
            lbl.setPixmap(QPixmap())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._strip_img is not None:
            self._refresh_preview_pixmaps()

    def _refresh_preview_pixmaps(self):
        if self._strip_img is None:
            return

        strip_rect = self._strip_label.contentsRect()
        strip_width = max(strip_rect.width(), FINAL_W * 2)
        strip_height = max(strip_rect.height(), FINAL_H * 2)
        self._strip_label.setText("")
        self._strip_label.setPixmap(
            gray_segment_to_pixmap(
                self._strip_img,
                strip_width,
                strip_height,
                interpolation=cv2.INTER_NEAREST,
            )
        )

        for i, seg in enumerate(self._segments):
            seg_rect = self._seg_labels[i].contentsRect()
            seg_width = max(seg_rect.width(), 56)
            seg_height = max(seg_rect.height(), 56)
            self._seg_labels[i].setPixmap(
                gray_segment_to_pixmap(seg, seg_width, seg_height)
            )

class BalanceDialog(QDialog):
    """Smart dialog that analyzes the dataset before asking for a target."""
    def __init__(self, category_counts: dict[str, int], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Adaptive Dataset Balancer")
        self.setModal(True)
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # 1. Show current stats
        layout.addWidget(QLabel("<b>Current Dataset Distribution:</b>"))
        
        stats_text = ""
        total_images = sum(category_counts.values())
        for cat, count in sorted(category_counts.items()):
            stats_text += f"Digit '{cat}': {count} images\n"
        
        stats_label = QLabel(stats_text)
        stats_label.setStyleSheet("font-family: monospace; color: #9fc5e8;")
        layout.addWidget(stats_label)
        layout.addWidget(QLabel(f"<i>Total Images: {total_images}</i>\n"))

        # 2. Calculate Adaptive Recommendations
        counts = list(category_counts.values())
        if not counts:
            self.reject()
            return
            
        median_val = int(np.median(counts))
        mean_val = int(np.mean(counts))
        max_val = max(counts)
        min_val = min(counts)

        # Safe recommendation: The Median is usually the safest bet for imbalanced data
        # It downsizes the massive outliers and safely augments the tiny ones.
        recommended_target = median_val
        if recommended_target < 100: 
            recommended_target = 100 # Ensure we have at least *some* volume

        info = QLabel(
            f"<b>Adaptive Analysis:</b>\n"
            f"• Largest class: {max_val}\n"
            f"• Smallest class: {min_val}\n"
            f"• Median size: {median_val}\n\n"
            f"<i>Recommendation: {recommended_target}</i>\n"
            f"(Warning: Inflating tiny classes by more than 20x can cause overfitting.)"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # 3. User Input
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("<b>Set Target Count per Class:</b>"))
        self.spinbox = QSpinBox()
        self.spinbox.setRange(10, 50000)
        self.spinbox.setValue(recommended_target)
        self.spinbox.setFixedWidth(100)
        input_layout.addWidget(self.spinbox)
        input_layout.addStretch()
        layout.addLayout(input_layout)

        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_target(self) -> int:
        return self.spinbox.value()

class BatchLabelDialog(QDialog):
    """Modal for fast per-image label entry during batch processing."""

    def __init__(
        self,
        preview_segments: list[np.ndarray],
        image_name: str,
        image_index: int,
        total_images: int,
        previous_label: str,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Batch Label Input")
        self.setModal(True)
        self.setMinimumWidth(760)

        self._previous_label = previous_label.strip().upper()
        self._resolved_label = ""
        self._readjust_requested = False

        layout = QVBoxLayout(self)

        header = QLabel(f"Image {image_index}/{total_images}: {image_name}")
        header.setWordWrap(True)
        layout.addWidget(header)

        instructions = QLabel(
            "Type X if the number is unreadable.\n"
            "For example 011X3 if the 4th digit is dirty."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #f0d27a;")
        layout.addWidget(instructions)

        if self._previous_label:
            hint = QLabel(
                "Press Enter on an empty input to reuse previous label: "
                f"{self._previous_label}"
            )
            hint.setWordWrap(True)
            hint.setStyleSheet("color: #9fc5e8;")
            layout.addWidget(hint)

        preview_box = QGroupBox("Segment Preview (non-binarized)")
        preview_layout = QHBoxLayout(preview_box)
        for seg in preview_segments:
            seg_label = QLabel()
            seg_label.setFixedSize(78, 78)
            seg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            seg_label.setStyleSheet("background: #111; border: 1px solid #444;")
            seg_label.setPixmap(
                gray_segment_to_pixmap(seg, 72, 72, interpolation=cv2.INTER_CUBIC)
            )
            preview_layout.addWidget(seg_label)
        layout.addWidget(preview_box)

        self._label_input = QLineEdit()
        self._label_input.setMaxLength(NUM_SEGMENTS)
        self._label_input.setPlaceholderText("Enter 5 chars using 0-9 and X")
        self._label_input.returnPressed.connect(self._on_accept)
        layout.addWidget(self._label_input)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._readjust_button = self._buttons.addButton(
            "Readjust Here",
            QDialogButtonBox.ButtonRole.ActionRole
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        self._readjust_button.clicked.connect(self._on_readjust)
        layout.addWidget(self._buttons)

        self._label_input.setFocus()

    def _on_accept(self):
        typed = self._label_input.text().strip().upper()

        if not typed:
            if self._previous_label:
                self._resolved_label = self._previous_label
                self.accept()
                return
            QMessageBox.warning(
                self,
                "Missing Label",
                "Please enter a 5-character label for the first image."
            )
            return

        if not is_digit_or_unreadable_label(typed):
            QMessageBox.warning(
                self,
                "Invalid Label",
                "Label must be exactly 5 characters using only digits (0-9) and X."
            )
            return

        self._resolved_label = typed
        self.accept()

    def _on_readjust(self):
        self._readjust_requested = True
        self.reject()

    def get_label(self) -> str:
        return self._resolved_label

    def readjust_requested(self) -> bool:
        return self._readjust_requested


class MlCommandWorker(QThread):
    """Run an external ML helper command without freezing the UI."""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, command: list[str], cwd: str):
        super().__init__()
        self._command = command
        self._cwd = cwd

    def run(self):
        try:
            completed = subprocess.run(
                self._command,
                cwd=self._cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except Exception as exc:
            self.error.emit(str(exc))
            return

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()

        if completed.returncode != 0:
            message = stderr or stdout or "Unknown ML backend error."
            self.error.emit(message)
            return

        if not stdout:
            self.finished.emit({})
            return

        try:
            self.finished.emit(json.loads(stdout))
        except json.JSONDecodeError:
            self.error.emit(stdout)


class LeNetTrainingDialog(QDialog):
    """Collect parameters for LeNet-style digit training."""

    def __init__(
        self,
        dataset_dir: str,
        backend_python: str,
        output_dir: str,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Train LeNet-5 Digit Model")
        self.setModal(True)
        self.setMinimumWidth(700)

        self._dataset_dir = dataset_dir
        self._backend_python = backend_python
        self._output_dir = output_dir

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Train a LeNet-style digit classifier from your 0-9 folders and export "
            "both a Keras model and a TFLite model for Android use."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        note = QLabel(
            "TensorFlow must run on a compatible Python backend. "
            "This app can stay on Python 3.14, but the training backend usually needs "
            "Python 3.10-3.13 with TensorFlow installed."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #f0d27a;")
        layout.addWidget(note)

        form = QFormLayout()

        dataset_row = QHBoxLayout()
        self._dataset_edit = QLineEdit(self._dataset_dir)
        dataset_btn = QPushButton("Browseâ€¦")
        dataset_btn.clicked.connect(self._browse_dataset)
        dataset_row.addWidget(self._dataset_edit)
        dataset_row.addWidget(dataset_btn)
        form.addRow("Dataset Folder:", self._wrap_layout(dataset_row))

        output_row = QHBoxLayout()
        self._output_edit = QLineEdit(self._output_dir)
        output_btn = QPushButton("Browseâ€¦")
        output_btn.clicked.connect(self._browse_output)
        output_row.addWidget(self._output_edit)
        output_row.addWidget(output_btn)
        form.addRow("Model Output:", self._wrap_layout(output_row))

        backend_row = QHBoxLayout()
        self._backend_edit = QLineEdit(self._backend_python)
        backend_btn = QPushButton("Browseâ€¦")
        backend_btn.clicked.connect(self._browse_backend)
        backend_row.addWidget(self._backend_edit)
        backend_row.addWidget(backend_btn)
        form.addRow("Backend Python:", self._wrap_layout(backend_row))

        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 500)
        self._epochs_spin.setValue(20)
        form.addRow("Epochs:", self._epochs_spin)

        self._batch_size_spin = QSpinBox()
        self._batch_size_spin.setRange(4, 512)
        self._batch_size_spin.setValue(32)
        form.addRow("Batch Size:", self._batch_size_spin)

        self._validation_spin = QDoubleSpinBox()
        self._validation_spin.setRange(0.05, 0.5)
        self._validation_spin.setSingleStep(0.05)
        self._validation_spin.setDecimals(2)
        self._validation_spin.setValue(0.20)
        form.addRow("Validation Split:", self._validation_spin)

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 999999)
        self._seed_spin.setValue(42)
        form.addRow("Random Seed:", self._seed_spin)

        layout.addLayout(form)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

    def _wrap_layout(self, row_layout: QHBoxLayout) -> QWidget:
        widget = QWidget()
        widget.setLayout(row_layout)
        return widget

    def _browse_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Select 0-9 Dataset Folder")
        if folder:
            self._dataset_edit.setText(folder)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Model Output Folder")
        if folder:
            self._output_edit.setText(folder)

    def _browse_backend(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Compatible Python Executable",
            self._backend_edit.text().strip() or "",
            "Python (*.exe);;All Files (*)",
        )
        if path:
            self._backend_edit.setText(path)

    def _on_accept(self):
        dataset_dir = self._dataset_edit.text().strip()
        output_dir = self._output_edit.text().strip()
        backend_python = self._backend_edit.text().strip()

        if not dataset_dir or not Path(dataset_dir).is_dir():
            QMessageBox.warning(self, "Missing Dataset", "Select a valid dataset folder.")
            return

        if not output_dir:
            QMessageBox.warning(self, "Missing Output", "Select an output folder.")
            return

        if not backend_python or not Path(backend_python).exists():
            QMessageBox.warning(
                self,
                "Missing Backend Python",
                "Select a compatible Python executable for TensorFlow training.",
            )
            return

        version = get_python_version(backend_python)
        if version is None:
            QMessageBox.warning(
                self,
                "Unreadable Python",
                "The selected Python executable could not be queried.",
            )
            return
        if version < (3, 10) or version > (3, 13):
            QMessageBox.warning(
                self,
                "Unsupported Python Version",
                "TensorFlow training backend should use Python 3.10 to 3.13.\n\n"
                f"Selected version: {version[0]}.{version[1]}",
            )
            return

        self.accept()

    def get_config(self) -> dict[str, object]:
        return {
            "dataset_dir": self._dataset_edit.text().strip(),
            "output_dir": self._output_edit.text().strip(),
            "backend_python": self._backend_edit.text().strip(),
            "epochs": int(self._epochs_spin.value()),
            "batch_size": int(self._batch_size_spin.value()),
            "validation_split": float(self._validation_spin.value()),
            "seed": int(self._seed_spin.value()),
        }


class LeNetTestingDialog(QDialog):
    """Run a trained LeNet/TFLite model against a 5-digit strip image."""

    run_requested = pyqtSignal(dict)

    def __init__(self, backend_python: str, model_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Test LeNet-5 Digit Model")
        self.setModal(True)
        self.setMinimumWidth(860)

        self._current_strip: np.ndarray | None = None
        self._current_segments: list[np.ndarray] = []

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Load a cropped number-strip image, optionally type the expected 5-digit label, "
            "then run the trained model through the same 5-slot segmentation used by this app."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()

        backend_row = QHBoxLayout()
        self._backend_edit = QLineEdit(backend_python)
        backend_btn = QPushButton("Browseâ€¦")
        backend_btn.clicked.connect(self._browse_backend)
        backend_row.addWidget(self._backend_edit)
        backend_row.addWidget(backend_btn)
        form.addRow("Backend Python:", self._wrap_layout(backend_row))

        model_row = QHBoxLayout()
        self._model_edit = QLineEdit(model_path)
        model_btn = QPushButton("Browseâ€¦")
        model_btn.clicked.connect(self._browse_model)
        model_row.addWidget(self._model_edit)
        model_row.addWidget(model_btn)
        form.addRow("Model File:", self._wrap_layout(model_row))

        image_row = QHBoxLayout()
        self._image_edit = QLineEdit("")
        image_btn = QPushButton("Browseâ€¦")
        image_btn.clicked.connect(self._browse_image)
        image_row.addWidget(self._image_edit)
        image_row.addWidget(image_btn)
        form.addRow("Strip Image:", self._wrap_layout(image_row))

        self._expected_edit = QLineEdit()
        self._expected_edit.setMaxLength(NUM_SEGMENTS)
        self._expected_edit.setPlaceholderText("Optional expected label, e.g. 38104")
        form.addRow("Expected Label:", self._expected_edit)

        layout.addLayout(form)

        preview_box = QGroupBox("Input Preview")
        preview_layout = QVBoxLayout(preview_box)
        self._strip_label = QLabel("No strip loaded")
        self._strip_label.setMinimumHeight(90)
        self._strip_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._strip_label.setStyleSheet("background: #111; border: 1px solid #444;")
        preview_layout.addWidget(self._strip_label)

        segments_row = QHBoxLayout()
        self._segment_boxes: list[QGroupBox] = []
        self._segment_labels: list[QLabel] = []
        for _ in range(NUM_SEGMENTS):
            box = QGroupBox("?")
            box_layout = QVBoxLayout(box)
            seg_label = QLabel()
            seg_label.setMinimumSize(78, 78)
            seg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            seg_label.setStyleSheet("background: #111; border: 1px solid #444;")
            box_layout.addWidget(seg_label)
            self._segment_boxes.append(box)
            self._segment_labels.append(seg_label)
            segments_row.addWidget(box)
        preview_layout.addLayout(segments_row)
        layout.addWidget(preview_box)

        self._result_label = QLabel("Run a test to see predictions.")
        self._result_label.setWordWrap(True)
        layout.addWidget(self._result_label)

        buttons_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Test")
        self._run_btn.clicked.connect(self._on_run_clicked)
        buttons_row.addWidget(self._run_btn)
        buttons_row.addStretch()
        cancel_btn = QPushButton("Close")
        cancel_btn.clicked.connect(self.reject)
        buttons_row.addWidget(cancel_btn)
        layout.addLayout(buttons_row)

    def _wrap_layout(self, row_layout: QHBoxLayout) -> QWidget:
        widget = QWidget()
        widget.setLayout(row_layout)
        return widget

    def _browse_backend(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Compatible Python Executable",
            self._backend_edit.text().strip() or "",
            "Python (*.exe);;All Files (*)",
        )
        if path:
            self._backend_edit.setText(path)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Trained Model",
            self._model_edit.text().strip() or "",
            "Models (*.tflite *.keras);;All Files (*)",
        )
        if path:
            self._model_edit.setText(path)

    def _browse_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Cropped Number Strip",
            self._image_edit.text().strip() or "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp *.heic *.heif);;All Files (*)",
        )
        if not path:
            return

        self._image_edit.setText(path)
        image = read_image_any(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            QMessageBox.warning(self, "Load Error", f"Cannot read:\n{path}")
            return

        try:
            self._current_strip = prepare_strip_image(image)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Strip", str(exc))
            return

        self._current_segments = split_strip_segments(self._current_strip)
        self._strip_label.setText("")
        self._strip_label.setPixmap(
            gray_segment_to_pixmap(
                self._current_strip,
                max(self._strip_label.width() - 12, FINAL_W * 3),
                max(self._strip_label.height() - 12, FINAL_H * 3),
                interpolation=cv2.INTER_NEAREST,
            )
        )

        for i, seg in enumerate(self._current_segments):
            self._segment_boxes[i].setTitle(f"Digit {i + 1}")
            self._segment_labels[i].setPixmap(gray_segment_to_pixmap(seg, 84, 84))

        self._result_label.setText("Strip loaded. Click Run Test to predict the 5 digits.")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._current_strip is not None:
            self._strip_label.setPixmap(
                gray_segment_to_pixmap(
                    self._current_strip,
                    max(self._strip_label.width() - 12, FINAL_W * 3),
                    max(self._strip_label.height() - 12, FINAL_H * 3),
                    interpolation=cv2.INTER_NEAREST,
                )
            )

    def _on_run_clicked(self):
        backend_python = self._backend_edit.text().strip()
        model_path = self._model_edit.text().strip()
        image_path = self._image_edit.text().strip()
        expected = self._expected_edit.text().strip()

        if not backend_python or not Path(backend_python).exists():
            QMessageBox.warning(
                self,
                "Missing Backend Python",
                "Select a compatible Python executable for model inference.",
            )
            return

        version = get_python_version(backend_python)
        if version is None:
            QMessageBox.warning(
                self,
                "Unreadable Python",
                "The selected Python executable could not be queried.",
            )
            return
        if version < (3, 10) or version > (3, 13):
            QMessageBox.warning(
                self,
                "Unsupported Python Version",
                "TensorFlow inference backend should use Python 3.10 to 3.13.\n\n"
                f"Selected version: {version[0]}.{version[1]}",
            )
            return

        if not model_path or not Path(model_path).is_file():
            QMessageBox.warning(self, "Missing Model", "Select a trained .tflite or .keras model.")
            return

        if not image_path or not Path(image_path).is_file():
            QMessageBox.warning(self, "Missing Image", "Select a cropped strip image to test.")
            return

        if expected and (len(expected) != NUM_SEGMENTS or not expected.isdigit()):
            QMessageBox.warning(
                self,
                "Invalid Expected Label",
                f"Expected label must be exactly {NUM_SEGMENTS} digits if provided.",
            )
            return

        self.run_requested.emit({
            "backend_python": self._backend_edit.text().strip(),
            "model_path": self._model_edit.text().strip(),
            "image_path": self._image_edit.text().strip(),
            "expected_label": self._expected_edit.text().strip(),
        })

    def set_busy(self, busy: bool):
        self._run_btn.setEnabled(not busy)
        self._result_label.setText("Running predictionâ€¦" if busy else self._result_label.text())

    def apply_result(self, result: dict[str, object]):
        predicted_label = str(result.get("predicted_label", ""))
        expected_label = str(result.get("expected_label", ""))
        confidences = result.get("confidences", [])
        self._run_btn.setEnabled(True)

        for i, score in enumerate(confidences[:NUM_SEGMENTS]):
            confidence_pct = float(score) * 100.0
            title = f"{predicted_label[i]} ({confidence_pct:.1f}%)" if i < len(predicted_label) else f"? ({confidence_pct:.1f}%)"
            self._segment_boxes[i].setTitle(title)

        if expected_label:
            match_text = "MATCH" if expected_label == predicted_label else "MISMATCH"
            self._result_label.setText(
                f"Prediction: {predicted_label} | Expected: {expected_label} | Result: {match_text}"
            )
        else:
            self._result_label.setText(f"Prediction: {predicted_label}")


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DigitExtractor — Image Dataset Extractor")
        self.resize(1280, 800)
        self._settings = QSettings("DigitExtractor", "DigitExtractor")
        self._output_dir = ""
        self._worker: WarpWorker | None = None
        self._ml_worker: ExternalMlCommandWorker | None = None
        self._ml_progress: QProgressDialog | None = None
        self._ml_log_lines: list[str] = []
        self._ml_backend_python = sys.executable
        self._last_trained_model_dir = str(Path.cwd() / LENET_MODEL_DIR_NAME)
        self._last_tflite_model_dir = self._last_trained_model_dir
        self._last_trained_model_path = ""
        self._testing_model_path = ""
        self._testing_mode_enabled = False
        self._testing_temp_image_path = ""
        self._pending_expected_label = ""
        self._pending_candidate_names: list[str] = []
        self._pending_candidate_points: dict[str, np.ndarray] = {}
        self._pending_candidate_sources: dict[str, str] = {}
        self._auto_read_results_dialog: AutoReadResultsDialog | None = None
        self._digit_diagnosis_dialog: DigitDiagnosisDialog | None = None
        self._pending_digit_candidates: list[dict[str, object]] = []
        self._pending_digit_limiter_boxes: list[list[float]] = []
        self._last_yolo_model_dir = str(Path.cwd() / "trained_yolo_models")
        self._yolo_testing_model_path = ""
        self._auto_find_strip_enabled = False
        self._auto_read_enabled = False
        self._pending_auto_read_after_detection = False
        self._persist_rotation_enabled = False
        self._persistent_rotation_angle = 0
        self._pending_readjust_framing_template: GuideboxFramingTemplate | None = None

        self._build_ui()
        self._build_menu()
        self._connect_signals()
        self._restore_app_state()

    # -- UI construction -----------------------------------------------------

    def _build_ui(self):
        # Central splitter
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)
        self.setCentralWidget(splitter)
        self._main_splitter = splitter

        # --- Left: file list sidebar ---
        sidebar = QWidget()
        sidebar.setMinimumWidth(180)
        sidebar.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding
        )
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(4, 4, 4, 4)

        btn_open = QPushButton("Open Folder…")
        btn_open.setObjectName("btnOpenFolder")
        sb_layout.addWidget(btn_open)

        self._file_list = QListWidget()
        self._file_list.setMinimumWidth(180)
        self._file_list.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        sb_layout.addWidget(self._file_list)
        splitter.addWidget(sidebar)

        # --- Centre: viewer + bottom panel ---
        centre = QWidget()
        c_layout = QVBoxLayout(centre)
        c_layout.setContentsMargins(0, 0, 0, 0)

        self._viewer = ImageViewer()
        c_layout.addWidget(self._viewer, stretch=5)

        # Bottom controls
        ctrl = QWidget()
        ctrl_layout = QGridLayout(ctrl)
        ctrl_layout.setContentsMargins(6, 2, 6, 2)
        ctrl_layout.setHorizontalSpacing(8)
        ctrl_layout.setVerticalSpacing(6)

        self._btn_select = QPushButton("Legacy 4-Point Select")
        self._btn_select.setEnabled(False)
        self._btn_select.setToolTip("Optional compatibility workflow. The main extraction path uses the fixed 5:1 guidebox.")
        ctrl_layout.addWidget(self._btn_select, 0, 0)

        self._btn_extract = QPushButton("Extract && Preview")
        self._btn_extract.setEnabled(False)
        ctrl_layout.addWidget(self._btn_extract, 0, 1)

        self._label_entry = QLineEdit()
        self._label_entry.setPlaceholderText("5-char label (e.g. 011X9, X=Unreadable)")
        self._label_entry.setMaxLength(5)
        self._label_entry.setMinimumWidth(160)
        self._label_entry.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed
        )
        ctrl_layout.addWidget(self._label_entry, 0, 2)

        self._batch_checkbox = QCheckBox("Batch Processing")
        self._batch_checkbox.setToolTip(
            "Apply the current guidebox framing (rotation, zoom, and pan) to all images in this folder."
        )
        ctrl_layout.addWidget(self._batch_checkbox, 0, 3)

        self._auto_find_checkbox = QCheckBox("Auto Find Strip")
        self._auto_find_checkbox.setToolTip(
            "Use the selected YOLOv8 model to auto-detect the digit strip on file load."
        )
        ctrl_layout.addWidget(self._auto_find_checkbox, 0, 4)

        self._btn_find_digits = QPushButton("Find Digits")
        self._btn_find_digits.setEnabled(False)
        self._btn_find_digits.setToolTip(
            "Diagnose individual digits across the whole image using contour proposals and LeNet."
        )
        ctrl_layout.addWidget(self._btn_find_digits, 0, 5)

        self._btn_read_guidebox = QPushButton("READ")
        self._btn_read_guidebox.setEnabled(False)
        self._btn_read_guidebox.setToolTip(
            "Read the digits currently aligned inside the fixed 5:1 guidebox."
        )
        ctrl_layout.addWidget(self._btn_read_guidebox, 0, 6)

        self._btn_save = QPushButton("Save Segments")
        self._btn_save.setEnabled(False)
        ctrl_layout.addWidget(self._btn_save, 0, 7)

        self._btn_output = QPushButton("Set Output Dir…")
        ctrl_layout.addWidget(self._btn_output, 0, 8)

        self._rotation_label = QLabel("Rotate:")
        ctrl_layout.addWidget(self._rotation_label, 1, 0)

        self._persist_rotation_checkbox = QCheckBox("Persist Rotation")
        self._persist_rotation_checkbox.setToolTip(
            "Keep using the current rotation angle when switching to other images."
        )
        ctrl_layout.addWidget(self._persist_rotation_checkbox, 1, 1)

        self._min_reading_entry = QLineEdit()
        self._min_reading_entry.setPlaceholderText("Min reading")
        self._min_reading_entry.setMaxLength(5)
        self._min_reading_entry.setMaximumWidth(100)
        ctrl_layout.addWidget(self._min_reading_entry, 1, 2)

        self._max_reading_entry = QLineEdit()
        self._max_reading_entry.setPlaceholderText("Max reading")
        self._max_reading_entry.setMaxLength(5)
        self._max_reading_entry.setMaximumWidth(100)
        ctrl_layout.addWidget(self._max_reading_entry, 1, 3)

        self._rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self._rotation_slider.setRange(0, 359)
        self._rotation_slider.setSingleStep(1)
        self._rotation_slider.setPageStep(15)
        self._rotation_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._rotation_slider.setTickInterval(30)
        self._rotation_slider.setEnabled(False)
        self._rotation_slider.setMinimumWidth(180)
        self._rotation_slider.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed
        )
        self._rotation_slider.setToolTip("Rotate image (0°-359°)")
        ctrl_layout.addWidget(self._rotation_slider, 1, 4, 1, 4)

        self._rotation_value = QLabel("0°")
        self._rotation_value.setFixedWidth(40)
        ctrl_layout.addWidget(self._rotation_value, 1, 8)

        self._status_info = QLabel("")
        self._status_info.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred
        )
        self._status_info.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        ctrl_layout.addWidget(self._status_info, 1, 9)
        ctrl_layout.setColumnStretch(4, 1)
        ctrl_layout.setColumnStretch(9, 2)

        c_layout.addWidget(ctrl)

        # Preview panel
        self._preview = PreviewWidget()
        c_layout.addWidget(self._preview, stretch=1)

        self._prediction_label = QLabel("")
        self._prediction_label.setWordWrap(True)
        self._prediction_label.setStyleSheet("color: #9fc5e8; padding: 4px 8px;")
        c_layout.addWidget(self._prediction_label)

        splitter.addWidget(centre)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([260, 1020])

        # Status bar
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("Ready — open a folder to begin.")

        # Styling
        self.setStyleSheet("""
            QMainWindow { background: #2b2b2b; }
            QLabel, QGroupBox, QListWidget, QPushButton, QLineEdit, QStatusBar {
                color: #ddd;
                font-size: 13px;
            }
            QGroupBox { border: 1px solid #555; border-radius: 4px;
                        margin-top: 6px; padding-top: 14px; }
            QGroupBox::title { subcontrol-origin: margin;
                               left: 10px; padding: 0 4px; }
            QPushButton {
                background: #3a3f47; border: 1px solid #555;
                border-radius: 4px; padding: 5px 14px;
            }
            QPushButton:hover { background: #50565e; }
            QPushButton:disabled { color: #666; }
            QLineEdit {
                background: #222; border: 1px solid #555;
                border-radius: 3px; padding: 4px 8px;
            }
            QListWidget {
                background: #1e1e1e; border: 1px solid #444;
            }
            QListWidget::item:selected {
                background: #264f78;
            }
        """)

    def _build_menu(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")

        act_open = QAction("Open Folder…", self)
        act_open.setShortcut(QKeySequence("Ctrl+O"))
        act_open.triggered.connect(self._on_open_folder)
        file_menu.addAction(act_open)

        file_menu.addSeparator()

        act_quit = QAction("Quit", self)
        act_quit.setShortcut(QKeySequence("Ctrl+Q"))
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        view_menu = menu.addMenu("&View")
        act_fit = QAction("Fit Image", self)
        act_fit.setShortcut(QKeySequence("F"))
        act_fit.triggered.connect(self._viewer.fit_to_view)
        view_menu.addAction(act_fit)

        training_menu = menu.addMenu("&Training")
        act_train_lenet = QAction("Train LeNet-5 Digit Modelâ€¦", self)
        act_train_lenet.triggered.connect(self._on_train_lenet)
        training_menu.addAction(act_train_lenet)
        act_train_yolo = QAction("Train YOLOv8 Finder...", self)
        act_train_yolo.triggered.connect(self._on_train_yolo)
        training_menu.addAction(act_train_yolo)

        testing_menu = menu.addMenu("&Testing")
        act_test_lenet = QAction("Test LeNet-5 Digit Modelâ€¦", self)
        act_test_lenet.triggered.connect(self._on_test_lenet)
        testing_menu.addAction(act_test_lenet)
        act_select_yolo = QAction("Select YOLOv8 Finder Model...", self)
        act_select_yolo.triggered.connect(self._on_select_yolo_model)
        testing_menu.addAction(act_select_yolo)
        act_select_model = QAction("Select LeNet-5 Model...", self)
        act_select_model.triggered.connect(self._on_select_test_model)
        testing_menu.addAction(act_select_model)

        self._act_testing_mode = QAction("Enable Viewer Testing Mode", self)
        self._act_testing_mode.setCheckable(True)
        self._act_testing_mode.toggled.connect(self._on_testing_mode_toggled)
        testing_menu.addAction(self._act_testing_mode)

        tool_menu = menu.addMenu("&Tool")
        act_invert_colors = QAction("Invert Colors", self)
        act_invert_colors.triggered.connect(self._on_invert_colors)
        tool_menu.addAction(act_invert_colors)

        # Diversify
        act_diversify = QAction("Diversify Data (Augment)", self)
        act_diversify.triggered.connect(self._on_diversify_data)
        tool_menu.addAction(act_diversify)
        
        # --- NEW CODE TO ADD BELOW DIVERSIFY ---
        # Balance Dataset
        act_balance = QAction("Balance Dataset (Auto Up/Downsample)", self)
        act_balance.triggered.connect(self._on_balance_dataset)
        tool_menu.addAction(act_balance)
        # --- END NEW CODE ---

    def _connect_signals(self):
        self.findChild(QPushButton, "btnOpenFolder").clicked.connect(
            self._on_open_folder
        )
        self._file_list.currentRowChanged.connect(self._on_file_selected)
        self._btn_select.clicked.connect(self._on_start_select)
        self._btn_extract.clicked.connect(self._on_extract)
        self._btn_read_guidebox.clicked.connect(self._on_read_guidebox)
        self._btn_save.clicked.connect(self._on_save_segments)
        self._btn_output.clicked.connect(self._on_set_output)
        self._batch_checkbox.toggled.connect(self._on_batch_toggled)
        self._auto_find_checkbox.toggled.connect(self._on_auto_find_toggled)
        self._btn_find_digits.clicked.connect(self._on_find_digits_clicked)
        self._persist_rotation_checkbox.toggled.connect(self._on_persist_rotation_toggled)
        self._rotation_slider.valueChanged.connect(self._on_rotation_changed)
        self._viewer.points_ready.connect(self._on_points_ready)

    def _restore_app_state(self):
        last_lenet_model = str(self._settings.value("models/lenet_path", "", type=str) or "")
        last_yolo_model = str(self._settings.value("models/yolo_path", "", type=str) or "")
        last_backend_python = str(
            self._settings.value("models/backend_python", self._ml_backend_python, type=str) or self._ml_backend_python
        )

        if last_backend_python and Path(last_backend_python).exists():
            self._ml_backend_python = last_backend_python

        if last_lenet_model and Path(last_lenet_model).is_file():
            self._testing_model_path = last_lenet_model
            self._last_trained_model_path = last_lenet_model
            self._last_trained_model_dir = str(Path(last_lenet_model).parent)

        if last_yolo_model and Path(last_yolo_model).is_file():
            self._yolo_testing_model_path = last_yolo_model
            self._last_yolo_model_dir = str(Path(last_yolo_model).parent)

        info_lines = []
        if self._testing_model_path:
            info_lines.append(f"LeNet: {Path(self._testing_model_path).name}")
        if self._yolo_testing_model_path:
            info_lines.append(f"YOLO: {Path(self._yolo_testing_model_path).name}")
        if info_lines:
            self._prediction_label.setText("Restored models:\n" + "\n".join(info_lines))
            self._statusbar.showMessage("Restored last selected ML models.")

    def _save_app_state(self):
        self._settings.setValue("models/lenet_path", self._testing_model_path)
        self._settings.setValue("models/yolo_path", self._yolo_testing_model_path)
        self._settings.setValue("models/backend_python", self._ml_backend_python)
        self._settings.sync()

    def _cleanup_warp_worker(self, *_args):
        self._worker = None

    def closeEvent(self, event: QCloseEvent):
        self._save_app_state()

        running_tasks: list[str] = []
        if self._worker is not None and self._worker.isRunning():
            running_tasks.append("image processing")
        if self._ml_worker is not None and self._ml_worker.isRunning():
            running_tasks.append("ML task")

        if running_tasks:
            QMessageBox.information(
                self,
                "Task Still Running",
                "Please wait for the current background task to finish before closing the app.\n\n"
                f"Running: {', '.join(running_tasks)}",
            )
            event.ignore()
            return

        if self._auto_read_results_dialog is not None:
            self._auto_read_results_dialog.close()
        if self._digit_diagnosis_dialog is not None:
            self._digit_diagnosis_dialog.close()

        super().closeEvent(event)

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
                "Please wait for the current training/testing task to finish first.",
            )
            return

        progress = QProgressDialog(title, "", 0, 0, self)
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.show()

        self._ml_progress = progress
        self._ml_log_lines = []
        self._ml_worker = ExternalMlCommandWorker(command, str(Path(__file__).resolve().parent))
        self._ml_worker.result_ready.connect(success_handler)
        self._ml_worker.error.connect(self._on_ml_worker_error)
        self._ml_worker.log.connect(self._on_ml_worker_log)
        self._ml_worker.finished.connect(self._cleanup_ml_worker)
        self._statusbar.showMessage(status_message)
        print(f"[ML] Starting: {' '.join(command)}")
        self._ml_worker.start()

    def _cleanup_ml_worker(self, *_args):
        if self._ml_progress is not None:
            self._ml_progress.close()
            self._ml_progress = None
        self._ml_worker = None

    def _on_ml_worker_log(self, message: str):
        text = str(message).strip()
        if not text:
            return
        self._ml_log_lines.append(text)
        if self._ml_progress is not None:
            display_text = text if len(text) <= 140 else (text[:137] + "...")
            self._ml_progress.setLabelText(display_text)
        self._statusbar.showMessage(text)
        print(f"[ML] {text}")

    def _on_ml_worker_error(self, message: str):
        self._pending_auto_read_after_detection = False
        self._statusbar.showMessage("ML task failed.")
        details = "\n".join(self._ml_log_lines[-25:])
        full_message = str(message)
        if details:
            full_message = f"{full_message}\n\nRecent log:\n{details}"
        print(f"[ML][ERROR] {full_message}")
        QMessageBox.critical(self, "ML Task Failed", full_message)

    # -- Slots ---------------------------------------------------------------

    def _on_train_lenet(self):
        dataset_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Digit Dataset Folder (must contain 0-9 folders)"
        )
        if not dataset_dir:
            return

        valid, message, category_folders = self._validate_digit_category_parent(dataset_dir)
        if not valid:
            QMessageBox.warning(self, "Invalid Dataset Folder", message)
            return

        digit_folders = {str(i) for i in range(10)}
        missing_digits = sorted(digit_folders - set(category_folders), key=int)
        if missing_digits:
            QMessageBox.warning(
                self,
                "Incomplete Dataset",
                "Training requires folders 0 through 9.\n\n"
                f"Missing folder(s): {', '.join(missing_digits)}"
            )
            return

        keras_output_dir = self._last_trained_model_dir or str(Path.cwd() / LENET_MODEL_DIR_NAME)
        tflite_output_dir = self._last_tflite_model_dir or keras_output_dir
        dialog = ExternalLeNetTrainingDialog(
            dataset_dir=dataset_dir,
            backend_python=self._ml_backend_python,
            keras_output_dir=keras_output_dir,
            tflite_output_dir=tflite_output_dir,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        config = dialog.get_config()
        self._ml_backend_python = str(config["backend_python"])
        version = get_python_version(self._ml_backend_python)
        if not is_supported_tensorflow_backend(version):
            QMessageBox.warning(
                self,
                "Unsupported Python Version",
                "TensorFlow training backend should use Python 3.10 to 3.13.\n\n"
                f"Selected version: {version[0]}.{version[1]}" if version else
                "The selected Python executable could not be queried.",
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
            validation_split=float(config["validation_split"]),
            seed=int(config["seed"]),
        )
        self._start_ml_worker(
            command,
            "Training LeNet-5 Model",
            "Training LeNet-5 digit modelâ€¦",
            self._on_lenet_training_finished,
        )

    def _on_lenet_training_finished(self, result: dict[str, object]):
        keras_path = str(result.get("keras_model_path", ""))
        tflite_path = str(result.get("tflite_model_path", ""))
        self._last_trained_model_path = tflite_path or keras_path
        if tflite_path:
            self._last_trained_model_dir = str(Path(tflite_path).parent)

        train_acc = float(result.get("train_accuracy", 0.0)) * 100.0
        val_acc = float(result.get("val_accuracy", 0.0)) * 100.0
        test_acc = float(result.get("test_accuracy", 0.0)) * 100.0
        dataset_size = int(result.get("dataset_size", 0))

        self._statusbar.showMessage(
            f"LeNet training complete. Test accuracy: {test_acc:.2f}%"
        )
        QMessageBox.information(
            self,
            "LeNet Training Complete",
            f"Dataset size: {dataset_size}\n"
            f"Train accuracy: {train_acc:.2f}%\n"
            f"Validation accuracy: {val_acc:.2f}%\n"
            f"Test accuracy: {test_acc:.2f}%\n\n"
            f"Keras model: {keras_path or '(not saved)'}\n"
            f"TFLite model: {tflite_path or '(not saved)'}"
        )

    def _on_test_lenet(self):
        model_path = self._last_trained_model_path
        if not model_path:
            default_tflite = Path(self._last_trained_model_dir) / LENET_TFLITE_FILENAME
            default_keras = Path(self._last_trained_model_dir) / LENET_KERAS_FILENAME
            if default_tflite.exists():
                model_path = str(default_tflite)
            elif default_keras.exists():
                model_path = str(default_keras)

        dialog = LeNetTestingDialog(
            backend_python=self._ml_backend_python,
            model_path=model_path,
            parent=self,
        )
        self._active_test_dialog = dialog
        dialog.run_requested.connect(self._run_lenet_test)
        dialog.exec()
        self._active_test_dialog = None

    def _run_lenet_test(self, config: dict[str, str]):
        self._ml_backend_python = config["backend_python"]
        self._last_trained_model_path = config["model_path"]

        script_path = get_ml_backend_script_path()
        if not script_path.exists():
            QMessageBox.critical(
                self,
                "Missing Backend Script",
                f"Cannot find testing backend script:\n{script_path}"
            )
            return

        if self._active_test_dialog is not None:
            self._active_test_dialog.set_busy(True)

        command = [
            config["backend_python"],
            str(script_path),
            "predict",
            "--model-path", config["model_path"],
            "--image-path", config["image_path"],
        ]
        if config["expected_label"]:
            command.extend(["--expected-label", config["expected_label"]])

        self._start_ml_worker(
            command,
            "Testing LeNet-5 Model",
            "Running LeNet-5 prediction on the strip imageâ€¦",
            self._on_lenet_test_finished,
        )

    def _on_lenet_test_finished(self, result: dict[str, object]):
        self._statusbar.showMessage(
            f"LeNet prediction: {result.get('predicted_label', '')}"
        )
        if self._active_test_dialog is not None:
            self._active_test_dialog.apply_result(result)
            self._active_test_dialog.raise_()
            self._active_test_dialog.activateWindow()

    def _on_train_lenet(self):
        dataset_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Digit Dataset Folder (must contain 0-9 folders)"
        )
        if not dataset_dir:
            return

        valid, message, category_folders = self._validate_digit_category_parent(dataset_dir)
        if not valid:
            QMessageBox.warning(self, "Invalid Dataset Folder", message)
            return

        digit_folders = {str(i) for i in range(10)}
        missing_digits = sorted(digit_folders - set(category_folders), key=int)
        if missing_digits:
            QMessageBox.warning(
                self,
                "Incomplete Dataset",
                "Training requires folders 0 through 9.\n\n"
                f"Missing folder(s): {', '.join(missing_digits)}"
            )
            return

        dialog = ExternalLeNetTrainingDialog(
            dataset_dir=dataset_dir,
            backend_python=self._ml_backend_python,
            keras_output_dir=self._last_trained_model_dir,
            tflite_output_dir=self._last_tflite_model_dir,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        config = dialog.get_config()
        self._ml_backend_python = str(config["backend_python"])
        version = get_python_version(self._ml_backend_python)
        if not is_supported_tensorflow_backend(version):
            selected_version = (
                f"{version[0]}.{version[1]}" if version else "(could not detect version)"
            )
            QMessageBox.warning(
                self,
                "Unsupported Python Version",
                "TensorFlow training backend should use Python 3.10 to 3.13.\n\n"
                f"Selected version: {selected_version}",
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
            validation_split=float(config["validation_split"]),
            seed=int(config["seed"]),
        )
        self._start_ml_worker(
            command,
            "Training LeNet-5 Model",
            "Training LeNet-5 digit model...",
            self._on_lenet_training_finished,
        )

    def _on_lenet_training_finished(self, result: dict[str, object]):
        keras_path = str(result.get("keras_model_path", ""))
        tflite_path = str(result.get("tflite_model_path", ""))
        self._last_trained_model_path = tflite_path or keras_path
        if self._last_trained_model_path:
            self._testing_model_path = self._last_trained_model_path
        if keras_path:
            self._last_trained_model_dir = str(Path(keras_path).parent)
        if tflite_path:
            self._last_tflite_model_dir = str(Path(tflite_path).parent)
        self._save_app_state()

        train_acc = float(result.get("train_accuracy", 0.0)) * 100.0
        val_acc = float(result.get("val_accuracy", 0.0)) * 100.0
        test_acc = float(result.get("test_accuracy", 0.0)) * 100.0
        dataset_size = int(result.get("dataset_size", 0))

        self._prediction_label.setText(
            "Training output saved.\n"
            f"TensorFlow/Keras: {keras_path or '(not saved)'}\n"
            f"TFLite: {tflite_path or '(not saved)'}"
        )
        self._statusbar.showMessage(
            f"LeNet training complete. Test accuracy: {test_acc:.2f}%"
        )
        QMessageBox.information(
            self,
            "LeNet Training Complete",
            f"Dataset size: {dataset_size}\n"
            f"Train accuracy: {train_acc:.2f}%\n"
            f"Validation accuracy: {val_acc:.2f}%\n"
            f"Test accuracy: {test_acc:.2f}%\n\n"
            f"TensorFlow/Keras model: {keras_path or '(not saved)'}\n"
            f"TFLite model: {tflite_path or '(not saved)'}"
        )

    def _on_test_lenet(self):
        self._on_select_test_model()

    def _on_select_test_model(self):
        start_dir = self._last_tflite_model_dir or self._last_trained_model_dir
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select LeNet-5 Model",
            start_dir,
            "Models (*.tflite *.keras);;All Files (*)",
        )
        if not model_path:
            return

        self._testing_model_path = model_path
        self._last_trained_model_path = model_path
        self._save_app_state()
        self._prediction_label.setText(
            f"Selected testing model:\n{self._testing_model_path}"
        )
        self._statusbar.showMessage(f"Testing model selected: {Path(model_path).name}")
        if hasattr(self, "_act_testing_mode") and not self._act_testing_mode.isChecked():
            self._act_testing_mode.setChecked(True)
        else:
            self._update_testing_ui_state()

    def _on_testing_mode_toggled(self, enabled: bool):
        if enabled and not self._testing_model_path:
            QMessageBox.information(
                self,
                "Select a Model First",
                "Choose a LeNet model first, then enable viewer testing mode."
            )
            self._act_testing_mode.blockSignals(True)
            self._act_testing_mode.setChecked(False)
            self._act_testing_mode.blockSignals(False)
            return

        self._testing_mode_enabled = enabled
        self._update_testing_ui_state()

    def _on_persist_rotation_toggled(self, enabled: bool):
        self._persist_rotation_enabled = enabled
        self._persistent_rotation_angle = int(self._rotation_slider.value()) % 360
        if enabled:
            self._statusbar.showMessage(
                f"Persist Rotation enabled at {self._persistent_rotation_angle}°."
            )
            return
        self._statusbar.showMessage("Persist Rotation disabled.")

    def _update_testing_ui_state(self):
        if self._testing_mode_enabled:
            self._batch_checkbox.setChecked(False)
            self._batch_checkbox.setEnabled(False)
            self._btn_extract.setText("Extract && Predict")
            self._btn_save.setEnabled(False)
            self._label_entry.setPlaceholderText(
                "Optional expected 5-digit label for comparison"
            )
            model_name = Path(self._testing_model_path).name if self._testing_model_path else "(none)"
            self._status_info.setText(f"Testing Mode ON | Model: {model_name}")
            self._statusbar.showMessage(
                "Viewer testing mode enabled. Align the strip inside the fixed 5:1 guidebox, then click Extract."
            )
            return

        self._batch_checkbox.setEnabled(True)
        self._btn_extract.setText("Extract && Preview")
        self._label_entry.setPlaceholderText("5-char label (e.g. 011X9, X=Unreadable)")
        self._status_info.setText("")
        self._prediction_label.setText("")
        if self._file_list.count() > 0:
            self._statusbar.showMessage("Testing mode disabled.")

    def _run_viewer_test_on_strip(self, strip: np.ndarray):
        if not self._testing_model_path:
            QMessageBox.warning(
                self,
                "No Testing Model",
                "Select a LeNet model before using viewer testing mode.",
            )
            return

        version = get_python_version(self._ml_backend_python)
        if not is_supported_tensorflow_backend(version):
            selected_version = (
                f"{version[0]}.{version[1]}" if version else "(could not detect version)"
            )
            QMessageBox.warning(
                self,
                "Unsupported Python Version",
                "TensorFlow inference backend should use Python 3.10 to 3.13.\n\n"
                f"Selected version: {selected_version}",
            )
            return

        expected_label = self._label_entry.text().strip()
        if expected_label and (len(expected_label) != NUM_SEGMENTS or not expected_label.isdigit()):
            expected_label = ""

        candidates = generate_strip_candidates(strip)
        candidate_images = [entry["image"] for entry in candidates]
        self._testing_temp_image_path = write_temp_images(candidate_images, prefix="lenet_candidates_")
        self._pending_expected_label = expected_label
        self._pending_candidate_names = [str(entry["name"]) for entry in candidates]
        command = build_lenet_predict_batch_command(
            backend_python=self._ml_backend_python,
            model_path=self._testing_model_path,
            images_dir=self._testing_temp_image_path,
            invert_input=True,
        )
        self._start_ml_worker(
            command,
            "Testing LeNet-5 Model",
            "Running LeNet-5 candidate search on the current 4-point selection...",
            self._on_lenet_batch_test_finished,
        )

    def _on_read_guidebox(self):
        current_img = self._viewer.get_cv_image()
        if current_img is None:
            return

        guide_crop = self._viewer.get_guidebox_crop()
        if guide_crop is None:
            self._statusbar.showMessage("Guidebox crop is empty. Align an image inside the guidebox first.")
            return

        try:
            strip = prepare_guidebox_strip(guide_crop)
        except Exception as exc:
            QMessageBox.warning(self, "Guidebox Read Error", str(exc))
            return

        self._preview.set_strip(strip)
        self._btn_save.setEnabled(True)

        if self._testing_model_path:
            self._run_viewer_test_on_strip(strip)
            return

        self._prediction_label.setText(
            "Guidebox crop prepared. Select a LeNet model to run READ inference on this strip."
        )
        self._statusbar.showMessage("Guidebox crop prepared. Select a LeNet model to enable reading.")

    def _show_auto_read_results_dialog(self, payload: dict[str, object]):
        if self._auto_read_results_dialog is None:
            self._auto_read_results_dialog = AutoReadResultsDialog(self)
        self._auto_read_results_dialog.apply_results(payload)
        self._auto_read_results_dialog.show()
        self._auto_read_results_dialog.raise_()
        self._auto_read_results_dialog.activateWindow()

    def _show_digit_diagnosis_dialog(self, payload: dict[str, object]):
        if self._digit_diagnosis_dialog is None:
            self._digit_diagnosis_dialog = DigitDiagnosisDialog(self)
        self._digit_diagnosis_dialog.apply_results(payload)
        self._digit_diagnosis_dialog.show()
        self._digit_diagnosis_dialog.raise_()
        self._digit_diagnosis_dialog.activateWindow()

    def _build_digit_diagnosis_overlay(
        self,
        image: np.ndarray,
        detections: list[dict[str, object]],
        selected_digits: list[dict[str, object]] | None = None,
        reading_roi_xyxy: list[float] | None = None,
    ) -> np.ndarray:
        if len(image.shape) == 2:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            overlay = image.copy()

        for limiter_index, limiter in enumerate(self._pending_digit_limiter_boxes, start=1):
            if not isinstance(limiter, list) or len(limiter) != 4:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in limiter]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 170, 0), 2, cv2.LINE_AA)
            cv2.putText(
                overlay,
                f"strip {limiter_index}",
                (x1, max(y1 - 6, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 170, 0),
                2,
                cv2.LINE_AA,
            )

        if isinstance(reading_roi_xyxy, list) and len(reading_roi_xyxy) == 4:
            x1, y1, x2, y2 = [int(round(float(v))) for v in reading_roi_xyxy]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (180, 60, 255), 2, cv2.LINE_AA)
            cv2.putText(
                overlay,
                "5-digit ROI",
                (x1, min(max(y2 + 18, 18), overlay.shape[0] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 60, 255),
                2,
                cv2.LINE_AA,
            )

        highlighted_keys = set()
        for digit in (selected_digits or []):
            bbox_xywh = digit.get("bbox_xywh", [])
            if isinstance(bbox_xywh, list) and len(bbox_xywh) == 4:
                highlighted_keys.add(tuple(int(v) for v in bbox_xywh))

        ranked = sorted(
            detections,
            key=lambda item: (
                float(item.get("center", [0.0, 0.0])[0]),
                float(item.get("center", [0.0, 0.0])[1]),
            ),
        )

        for index, detection in enumerate(ranked, start=1):
            points = np.asarray(detection.get("points"), dtype=np.float32)
            if points.shape != (4, 2):
                continue
            quad = np.round(points).astype(np.int32)
            confidence = float(detection.get("confidence", 0.0))
            digit = str(detection.get("predicted_label", "?"))

            bbox_key = tuple(int(v) for v in detection.get("bbox_xywh", [0, 0, 0, 0]))
            if bbox_key in highlighted_keys:
                color = (255, 90, 200)
                thickness = 2
            else:
                color = (120, 120, 120)
                thickness = 1

            cv2.polylines(overlay, [quad], True, color, thickness, cv2.LINE_AA)
            x, y, _w, _h = [int(v) for v in detection.get("bbox_xywh", [0, 0, 0, 0])]
            if bbox_key in highlighted_keys:
                label = f"{index}:{digit} {confidence * 100.0:.0f}%"
                text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                text_x = max(x, 0)
                text_y = max(y - 6, text_size[1] + 6)
                cv2.rectangle(
                    overlay,
                    (text_x - 2, text_y - text_size[1] - 4),
                    (text_x + text_size[0] + 4, text_y + baseline),
                    color,
                    thickness=-1,
                )
                cv2.putText(
                    overlay,
                    label,
                    (text_x, text_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (20, 20, 20),
                    1,
                    cv2.LINE_AA,
                )

        return overlay

    def _extract_yolo_limiter_boxes(
        self,
        result: dict[str, object],
        image_shape: tuple[int, ...],
        max_boxes: int = 3,
    ) -> list[list[float]]:
        raw_candidates = result.get("candidates", [])
        if not isinstance(raw_candidates, list) or not raw_candidates:
            bbox = result.get("bbox_xyxy", [])
            if isinstance(bbox, list) and len(bbox) == 4:
                raw_candidates = [{
                    "bbox_xyxy": bbox,
                    "confidence": result.get("confidence", 0.0),
                    "rank_score": result.get("rank_score", result.get("confidence", 0.0)),
                }]

        image_h, image_w = image_shape[:2]
        image_area = max(float(image_h * image_w), 1.0)
        filtered: list[dict[str, object]] = []
        seen: set[tuple[int, int, int, int]] = set()
        for candidate in raw_candidates:
            if not isinstance(candidate, dict):
                continue
            bbox = candidate.get("bbox_xyxy", [])
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            bw = max(x2 - x1, 0.0)
            bh = max(y2 - y1, 0.0)
            area = bw * bh
            if bw < max(20.0, image_w * 0.08):
                continue
            if bh < max(10.0, image_h * 0.02):
                continue
            if area < image_area * 0.0025:
                continue
            key = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
            if key in seen:
                continue
            seen.add(key)
            filtered.append(dict(candidate))

        filtered.sort(
            key=lambda item: (
                float(item.get("rank_score", item.get("confidence", 0.0))),
                float(item.get("confidence", 0.0)),
            ),
            reverse=True,
        )
        return [
            [float(v) for v in item["bbox_xyxy"]]
            for item in filtered[:max(max_boxes, 1)]
        ]

    def _run_digit_diagnosis_with_limiters(
        self,
        image: np.ndarray,
        limiter_boxes: list[list[float]],
    ):
        detections = detect_digit_candidates(image, limiter_boxes=limiter_boxes)
        if not detections:
            self._prediction_label.setText(
                "Digit diagnosis could not find any plausible digit shapes inside the YOLO strip limiter."
            )
            self._statusbar.showMessage("Digit diagnosis found no plausible digit candidates inside the strip limiter.")
            return

        self._pending_digit_limiter_boxes = [list(box) for box in limiter_boxes]
        self._pending_digit_candidates = detections
        candidate_images = [np.asarray(item["image"], dtype=np.uint8) for item in detections]
        self._testing_temp_image_path = write_temp_images(candidate_images, prefix="digit_candidates_")

        command = build_lenet_predict_digits_command(
            backend_python=self._ml_backend_python,
            model_path=self._testing_model_path,
            images_dir=self._testing_temp_image_path,
            invert_input=True,
        )
        self._start_ml_worker(
            command,
            "Finding Digits",
            "Detecting individual digit shapes inside the YOLO strip limiter and classifying them with LeNet...",
            self._on_find_digits_finished,
        )

    def _run_yolo_digit_limiter_for_current_image(self, current_img: np.ndarray):
        if not self._yolo_testing_model_path:
            QMessageBox.information(
                self,
                "Select YOLO Model First",
                "Choose a YOLOv8 finder model before running digit diagnosis.",
            )
            return

        version = get_python_version(self._ml_backend_python)
        if not is_supported_tensorflow_backend(version):
            self._statusbar.showMessage("YOLO backend Python is not configured.")
            return

        temp_image_path = write_temp_image(current_img, prefix="yolo_digit_limit_")
        command = build_yolo_predict_windows_command(
            backend_python=self._ml_backend_python,
            model_path=self._yolo_testing_model_path,
            image_path=temp_image_path,
            image_size=640,
            conf_threshold=0.10,
        )
        self._start_ml_worker(
            command,
            "Finding Digit Strip",
            "Running YOLOv8 strip limiter before digit diagnosis...",
            self._on_yolo_digit_limiter_finished,
        )

    def _on_yolo_digit_limiter_finished(self, result: dict[str, object]):
        current_img = self._viewer.get_cv_image()
        if current_img is None:
            return

        limiter_boxes = self._extract_yolo_limiter_boxes(result, current_img.shape)
        if not limiter_boxes:
            self._prediction_label.setText(
                "Digit diagnosis stopped because YOLO could not find a usable strip limiter."
            )
            self._statusbar.showMessage("YOLO could not find a usable strip limiter for digit diagnosis.")
            return

        self._run_digit_diagnosis_with_limiters(current_img, limiter_boxes)

    def _refine_digit_diagnosis_detections(
        self,
        detections: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if not detections:
            return []

        confident = [item for item in detections if float(item.get("confidence", 0.0)) >= 0.80]
        height_pool = confident or detections
        dominant_height = float(np.median([float(item.get("height", 1.0)) for item in height_pool]))
        dominant_height = max(dominant_height, 1.0)

        by_limiter: dict[int, list[dict[str, object]]] = {}
        for item in detections:
            limiter_index = int(item.get("limiter_index", -1))
            by_limiter.setdefault(limiter_index, []).append(item)

        kept: list[dict[str, object]] = []
        for limiter_index, items in by_limiter.items():
            ordered = sorted(items, key=lambda item: float(item.get("center", [0.0, 0.0])[0]))
            for idx, item in enumerate(ordered):
                confidence = float(item.get("confidence", 0.0))
                width = max(float(item.get("width", 1.0)), 1.0)
                height = max(float(item.get("height", 1.0)), 1.0)
                size_ratio = max(height, dominant_height) / max(min(height, dominant_height), 1.0)
                size_ok = size_ratio <= 1.45

                left_support = False
                right_support = False
                if idx > 0:
                    left = ordered[idx - 1]
                    if abs(float(left.get("center", [0.0, 0.0])[1]) - float(item.get("center", [0.0, 0.0])[1])) <= height * 0.70:
                        if abs(float(item.get("center", [0.0, 0.0])[0]) - float(left.get("center", [0.0, 0.0])[0])) <= width * 2.8:
                            left_support = True
                if idx + 1 < len(ordered):
                    right = ordered[idx + 1]
                    if abs(float(right.get("center", [0.0, 0.0])[1]) - float(item.get("center", [0.0, 0.0])[1])) <= height * 0.70:
                        if abs(float(right.get("center", [0.0, 0.0])[0]) - float(item.get("center", [0.0, 0.0])[0])) <= width * 2.8:
                            right_support = True

                merged = dict(item)
                merged["left_neighbor"] = bool(left_support)
                merged["right_neighbor"] = bool(right_support)
                merged["size_ratio_to_dominant"] = float(size_ratio)
                neighbor_support = int(left_support) + int(right_support)

                keep = False
                if confidence >= 0.80:
                    keep = True
                elif confidence >= 0.62 and size_ok and neighbor_support >= 1:
                    keep = True
                elif confidence >= 0.55 and size_ok and neighbor_support >= 2:
                    keep = True

                if keep:
                    kept.append(merged)

        kept.sort(
            key=lambda item: (
                float(item.get("confidence", 0.0)),
                float(item.get("final_score", 0.0)),
            ),
            reverse=True,
        )
        return kept

    def _parse_restriction_bounds(self) -> tuple[int | None, int | None]:
        def _parse(entry: QLineEdit) -> int | None:
            raw = entry.text().strip()
            if not raw:
                return None
            if not raw.isdigit():
                return None
            return int(raw)

        min_value = _parse(self._min_reading_entry)
        max_value = _parse(self._max_reading_entry)
        if min_value is not None and max_value is not None and min_value > max_value:
            min_value, max_value = max_value, min_value
        return min_value, max_value

    def _derive_allowed_digits_by_position(
        self,
        min_reading: int | None,
        max_reading: int | None,
    ) -> list[set[str]]:
        if min_reading is None and max_reading is None:
            return [set(str(d) for d in range(10)) for _ in range(NUM_SEGMENTS)]

        low = min_reading if min_reading is not None else 0
        high = max_reading if max_reading is not None else 99999
        low = max(0, min(low, 99999))
        high = max(0, min(high, 99999))
        if low > high:
            low, high = high, low

        allowed = [set() for _ in range(NUM_SEGMENTS)]
        for value in range(low, high + 1):
            reading = f"{value:05d}"
            for idx, digit in enumerate(reading):
                allowed[idx].add(digit)

        for idx in range(NUM_SEGMENTS):
            if not allowed[idx]:
                allowed[idx] = set(str(d) for d in range(10))
        return allowed

    def _filtered_slot_candidates(
        self,
        slot: list[dict[str, object]],
        allowed_digits: set[str],
    ) -> list[dict[str, object]]:
        filtered = [
            item for item in slot
            if str(item.get("predicted_label", "?")) in allowed_digits
        ]
        filtered.sort(
            key=lambda item: (
                float(item.get("confidence", 0.0)),
                float(item.get("final_score", 0.0)),
            ),
            reverse=True,
        )
        return filtered

    def _build_range_constrained_strip_candidates(
        self,
        candidates: list[dict[str, object]],
        top_k: int = 5,
    ) -> list[dict[str, object]]:
        min_reading, max_reading = self._parse_restriction_bounds()
        if min_reading is None and max_reading is None:
            return sorted(
                candidates,
                key=lambda item: float(item.get("score", 0.0)),
                reverse=True,
            )[:max(int(top_k), 1)]

        allowed_digits_by_position = self._derive_allowed_digits_by_position(min_reading, max_reading)
        ranked = sorted(
            candidates,
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )[:max(int(top_k), 1)]

        position_supports: list[dict[str, float]] = [dict() for _ in range(NUM_SEGMENTS)]
        exact_reading_scores: dict[str, float] = {}
        for candidate in ranked:
            label = str(candidate.get("predicted_label", ""))
            score = float(candidate.get("score", 0.0))
            confidences = candidate.get("confidences", [])
            if not isinstance(confidences, list):
                confidences = []
            if len(label) != NUM_SEGMENTS or not label.isdigit():
                continue

            for idx, digit in enumerate(label):
                support = score
                if idx < len(confidences):
                    support += 0.35 * float(confidences[idx])
                position_supports[idx][digit] = position_supports[idx].get(digit, 0.0) + support

            numeric_value = int(label)
            if (min_reading is None or numeric_value >= min_reading) and (
                max_reading is None or numeric_value <= max_reading
            ):
                exact_reading_scores[label] = max(exact_reading_scores.get(label, -1e9), score)

        low = min_reading if min_reading is not None else 0
        high = max_reading if max_reading is not None else 99999
        low = max(0, min(low, 99999))
        high = max(0, min(high, 99999))
        if low > high:
            low, high = high, low

        constrained_candidates: list[dict[str, object]] = []
        for value in range(low, high + 1):
            reading = f"{value:05d}"
            score = 0.0
            confidences: list[float] = []
            for idx, digit in enumerate(reading):
                digit_support = float(position_supports[idx].get(digit, 0.0))
                if allowed_digits_by_position[idx] == {digit}:
                    digit_support += 0.50
                    confidences.append(1.0)
                else:
                    total_support = sum(position_supports[idx].values())
                    confidence = digit_support / total_support if total_support > 1e-6 else 0.0
                    confidences.append(confidence)
                score += digit_support

            score += 1.25 * float(exact_reading_scores.get(reading, 0.0))
            constrained_candidates.append(
                {
                    "image_name": f"range:{reading}",
                    "predicted_label": reading,
                    "confidences": confidences,
                    "score": score,
                    "source_name": "range-constrained-vote",
                }
            )

        constrained_candidates.sort(
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        return constrained_candidates[:max(int(top_k), 1)]

    def _cluster_reading_slots(
        self,
        detections: list[dict[str, object]],
    ) -> list[list[dict[str, object]]]:
        if not detections:
            return []

        ordered = sorted(detections, key=lambda item: float(item.get("center", [0.0, 0.0])[0]))
        slots: list[list[dict[str, object]]] = []
        for detection in ordered:
            x, _y, w, _h = [float(v) for v in detection.get("bbox_xywh", [0, 0, 0, 0])]
            center_x = float(detection.get("center", [0.0, 0.0])[0])
            placed = False
            for slot in slots:
                slot_xs = [float(item.get("bbox_xywh", [0, 0, 0, 0])[0]) for item in slot]
                slot_ws = [float(item.get("bbox_xywh", [0, 0, 0, 0])[2]) for item in slot]
                slot_centers = [float(item.get("center", [0.0, 0.0])[0]) for item in slot]
                slot_left = min(slot_xs)
                slot_right = max(slot_xs[idx] + slot_ws[idx] for idx in range(len(slot)))
                slot_center = float(np.mean(slot_centers))
                overlap = max(0.0, min(x + w, slot_right) - max(x, slot_left))
                min_width = max(min(w, slot_right - slot_left), 1.0)
                same_slot = overlap / min_width >= 0.22
                same_slot = same_slot or abs(center_x - slot_center) <= max(w, slot_right - slot_left) * 0.55
                if same_slot:
                    slot.append(detection)
                    placed = True
                    break
            if not placed:
                slots.append([detection])

        normalized_slots: list[list[dict[str, object]]] = []
        for slot in slots:
            deduped_by_digit: dict[str, dict[str, object]] = {}
            for detection in sorted(
                slot,
                key=lambda item: (
                    float(item.get("confidence", 0.0)),
                    float(item.get("final_score", 0.0)),
                ),
                reverse=True,
            ):
                digit = str(detection.get("predicted_label", "?"))
                if digit not in deduped_by_digit:
                    deduped_by_digit[digit] = detection
            slot_items = list(deduped_by_digit.values())
            slot_items.sort(
                key=lambda item: (
                    float(item.get("confidence", 0.0)),
                    float(item.get("final_score", 0.0)),
                ),
                reverse=True,
            )
            normalized_slots.append(slot_items)

        normalized_slots.sort(
            key=lambda slot: float(np.mean([item.get("center", [0.0, 0.0])[0] for item in slot]))
        )
        return normalized_slots

    def _score_slot_window(
        self,
        slots: list[list[dict[str, object]]],
        allowed_digits_by_position: list[set[str]] | None = None,
    ) -> float:
        if len(slots) != 5:
            return -1e9

        constrained_slots: list[list[dict[str, object]]] = []
        for idx, slot in enumerate(slots):
            allowed_digits = (
                allowed_digits_by_position[idx]
                if allowed_digits_by_position is not None and idx < len(allowed_digits_by_position)
                else set(str(d) for d in range(10))
            )
            filtered = self._filtered_slot_candidates(slot, allowed_digits)
            if not filtered:
                return -1e9
            constrained_slots.append(filtered)

        top_items = [slot[0] for slot in constrained_slots if slot]
        if len(top_items) != 5:
            return -1e9

        confidences = [float(item.get("confidence", 0.0)) for item in top_items]
        heights = [max(float(item.get("height", 1.0)), 1.0) for item in top_items]
        widths = [max(float(item.get("width", 1.0)), 1.0) for item in top_items]
        center_ys = [float(item.get("center", [0.0, 0.0])[1]) for item in top_items]
        centers = [float(item.get("center", [0.0, 0.0])[0]) for item in top_items]
        spacing = [centers[idx + 1] - centers[idx] for idx in range(len(centers) - 1)]
        height_penalty = float(np.std(heights) / max(np.mean(heights), 1.0))
        width_penalty = float(np.std(widths) / max(np.mean(widths), 1.0))
        spacing_penalty = float(np.std(spacing) / max(np.mean(spacing), 1.0)) if spacing else 0.0
        slant_penalty = float(np.std(center_ys) / max(np.mean(heights), 1.0))
        prefix_bonus = 0.0
        for idx, item in enumerate(top_items[:3]):
            allowed_digits = allowed_digits_by_position[idx] if allowed_digits_by_position is not None else set()
            if allowed_digits == {"0"} and str(item.get("predicted_label", "")) == "0":
                prefix_bonus += 0.40 * float(item.get("confidence", 0.0))
        return (
            float(np.mean(confidences))
            + (0.25 * float(np.min(confidences)))
            + prefix_bonus
            - (0.20 * height_penalty)
            - (0.15 * width_penalty)
            - (0.15 * spacing_penalty)
            - (0.12 * slant_penalty)
        )

    def _choose_five_digit_window(
        self,
        detections: list[dict[str, object]],
        allowed_digits_by_position: list[set[str]],
    ) -> tuple[list[list[dict[str, object]]], list[float] | None, int]:
        by_limiter: dict[int, list[dict[str, object]]] = {}
        for item in detections:
            limiter_index = int(item.get("limiter_index", -1))
            by_limiter.setdefault(limiter_index, []).append(item)

        best_slots: list[list[dict[str, object]]] = []
        best_roi: list[float] | None = None
        best_limiter_index = -1
        best_score = -1e9

        for limiter_index, items in by_limiter.items():
            slots = self._cluster_reading_slots(items)
            if len(slots) < 5:
                continue
            for start in range(0, len(slots) - 4):
                window = slots[start:start + 5]
                score = self._score_slot_window(window, allowed_digits_by_position)
                if score <= best_score:
                    continue
                constrained_window = [
                    self._filtered_slot_candidates(window[idx], allowed_digits_by_position[idx])
                    for idx in range(5)
                ]
                top_items = [slot[0] for slot in constrained_window if slot]
                if len(top_items) != 5:
                    continue
                xs = [float(item.get("bbox_xywh", [0, 0, 0, 0])[0]) for item in top_items]
                ys = [float(item.get("bbox_xywh", [0, 0, 0, 0])[1]) for item in top_items]
                rights = [
                    float(item.get("bbox_xywh", [0, 0, 0, 0])[0]) + float(item.get("bbox_xywh", [0, 0, 0, 0])[2])
                    for item in top_items
                ]
                bottoms = [
                    float(item.get("bbox_xywh", [0, 0, 0, 0])[1]) + float(item.get("bbox_xywh", [0, 0, 0, 0])[3])
                    for item in top_items
                ]
                best_score = score
                best_slots = constrained_window
                best_roi = [min(xs), min(ys), max(rights), max(bottoms)]
                best_limiter_index = limiter_index

        return best_slots, best_roi, best_limiter_index

    def _build_reading_candidates(
        self,
        slots: list[list[dict[str, object]]],
        allowed_digits_by_position: list[set[str]],
        min_reading: int | None,
        max_reading: int | None,
        top_k: int = 5,
    ) -> list[dict[str, object]]:
        if len(slots) != 5:
            return []

        trimmed_slots: list[list[dict[str, object]]] = []
        for idx, slot in enumerate(slots):
            ranked = self._filtered_slot_candidates(slot, allowed_digits_by_position[idx])[:3]
            if not ranked:
                return []
            trimmed_slots.append(ranked)

        candidates: list[dict[str, object]] = []
        for combo in product(*trimmed_slots):
            reading = "".join(str(item.get("predicted_label", "?")) for item in combo)
            if len(reading) != 5 or not reading.isdigit():
                continue
            numeric_value = int(reading)
            if min_reading is not None and numeric_value < min_reading:
                continue
            if max_reading is not None and numeric_value > max_reading:
                continue

            score = 0.0
            for idx, item in enumerate(combo):
                confidence = max(float(item.get("confidence", 0.0)), 1e-4)
                score += math.log(confidence)
                score += 0.12 * float(item.get("final_score", 0.0))
                if allowed_digits_by_position[idx] == {"0"} and str(item.get("predicted_label", "")) == "0":
                    score += 0.40
            center_xs = [float(item.get("center", [0.0, 0.0])[0]) for item in combo]
            center_ys = [float(item.get("center", [0.0, 0.0])[1]) for item in combo]
            heights = [max(float(item.get("height", 1.0)), 1.0) for item in combo]
            spacings = [center_xs[i + 1] - center_xs[i] for i in range(len(center_xs) - 1)]
            if spacings:
                score -= 0.20 * float(np.std(spacings) / max(np.mean(spacings), 1.0))
            score -= 0.12 * float(np.std(center_ys) / max(np.mean(heights), 1.0))

            candidates.append(
                {
                    "reading": reading,
                    "score": float(score),
                    "digits": [dict(item) for item in combo],
                    "confidences": [float(item.get("confidence", 0.0)) for item in combo],
                }
            )

        if not candidates and (min_reading is not None or max_reading is not None):
            unconstrained_digits = [set(str(d) for d in range(10)) for _ in range(NUM_SEGMENTS)]
            return self._build_reading_candidates(
                slots,
                unconstrained_digits,
                None,
                None,
                top_k=top_k,
            )

        deduped: dict[str, dict[str, object]] = {}
        for candidate in sorted(candidates, key=lambda item: float(item.get("score", -1e9)), reverse=True):
            reading = str(candidate.get("reading", ""))
            if reading not in deduped:
                deduped[reading] = candidate
        return list(deduped.values())[:max(int(top_k), 1)]

    def _on_find_digits_clicked(self):
        current_img = self._viewer.get_cv_image()
        if current_img is None:
            QMessageBox.information(
                self,
                "Open an Image First",
                "Load an image before running digit diagnosis.",
            )
            return

        if not self._testing_model_path:
            QMessageBox.information(
                self,
                "Select LeNet Model First",
                "Choose a LeNet model before running digit diagnosis.",
            )
            return

        version = get_python_version(self._ml_backend_python)
        if not is_supported_tensorflow_backend(version):
            selected_version = (
                f"{version[0]}.{version[1]}" if version else "(could not detect version)"
            )
            QMessageBox.warning(
                self,
                "Unsupported Python Version",
                "TensorFlow inference backend should use Python 3.10 to 3.13.\n\n"
                f"Selected version: {selected_version}",
            )
            return

        self._run_yolo_digit_limiter_for_current_image(current_img)

    def _on_find_digits_finished(self, result: dict[str, object]):
        raw_candidates = result.get("candidates", [])
        if not isinstance(raw_candidates, list):
            raw_candidates = []

        prediction_map: dict[str, dict[str, object]] = {}
        for candidate in raw_candidates:
            if not isinstance(candidate, dict):
                continue
            prediction_map[str(candidate.get("image_name", ""))] = candidate

        detections: list[dict[str, object]] = []
        for index, candidate in enumerate(self._pending_digit_candidates):
            name = f"{index:03d}"
            prediction = prediction_map.get(name)
            if prediction is None:
                continue
            merged = dict(candidate)
            merged["predicted_label"] = str(prediction.get("predicted_label", "?"))
            merged["confidence"] = float(prediction.get("confidence", 0.0))
            merged["classification_score"] = float(prediction.get("score", 0.0))
            detections.append(merged)

        if not detections:
            self._prediction_label.setText("Digit diagnosis did not receive any usable LeNet predictions.")
            self._statusbar.showMessage("Digit diagnosis failed to classify the detected candidates.")
            return

        detections = self._refine_digit_diagnosis_detections(detections)
        if not detections:
            self._prediction_label.setText(
                "Digit diagnosis found candidates, but none survived the confidence, size, and neighbor checks."
            )
            self._statusbar.showMessage(
                "Digit diagnosis rejected all candidates after confidence, size, and neighbor checks."
            )
            return

        current_img = self._viewer.get_cv_image()
        if current_img is None:
            return

        min_reading, max_reading = self._parse_restriction_bounds()
        allowed_digits_by_position = self._derive_allowed_digits_by_position(min_reading, max_reading)
        best_slots, reading_roi_xyxy, limiter_index = self._choose_five_digit_window(
            detections,
            allowed_digits_by_position,
        )
        if len(best_slots) != 5 or reading_roi_xyxy is None:
            self._prediction_label.setText(
                "Digit diagnosis kept digits, but could not form one clean 5-digit ROI from left to right."
            )
            self._statusbar.showMessage(
                "Digit diagnosis could not form a clean 5-digit ROI from the kept digits."
            )
            return

        reading_candidates = self._build_reading_candidates(
            best_slots,
            allowed_digits_by_position,
            min_reading=min_reading,
            max_reading=max_reading,
            top_k=5,
        )
        if not reading_candidates:
            self._prediction_label.setText(
                "Digit diagnosis found a 5-digit ROI, but no reading survived the current restrictions."
            )
            self._statusbar.showMessage(
                "Digit diagnosis found a 5-digit ROI, but no reading survived the current restrictions."
            )
            return

        best_reading = reading_candidates[0]
        selected_digits = [dict(item) for item in best_reading.get("digits", [])]
        top_readings_text = ", ".join(str(item.get("reading", "")) for item in reading_candidates[:5])
        slot_summaries: list[dict[str, object]] = []
        for idx, slot in enumerate(best_slots):
            allowed_digits = allowed_digits_by_position[idx]
            filtered = self._filtered_slot_candidates(slot, allowed_digits)
            chosen = selected_digits[idx] if idx < len(selected_digits) else None
            slot_summaries.append(
                {
                    "slot_index": idx + 1,
                    "allowed_digits": "".join(sorted(allowed_digits)),
                    "chosen_digit": str(chosen.get("predicted_label", "")) if isinstance(chosen, dict) else "",
                    "chosen_confidence": float(chosen.get("confidence", 0.0)) if isinstance(chosen, dict) else 0.0,
                    "candidate_digits": [
                        {
                            "digit": str(item.get("predicted_label", "")),
                            "confidence": float(item.get("confidence", 0.0)),
                        }
                        for item in filtered[:3]
                    ],
                }
            )
        restriction_text = "none"
        if min_reading is not None or max_reading is not None:
            min_text = f"{min_reading:05d}" if min_reading is not None else "-----"
            max_text = f"{max_reading:05d}" if max_reading is not None else "-----"
            restriction_text = f"{min_text} to {max_text}"

        overlay = self._build_digit_diagnosis_overlay(
            current_img,
            detections,
            selected_digits=selected_digits,
            reading_roi_xyxy=reading_roi_xyxy,
        )
        overlay_path = write_temp_image(overlay, prefix="digit_diagnosis_")
        confident_count = sum(1 for item in detections if float(item.get("confidence", 0.0)) >= 0.70)

        self._prediction_label.setText(
            f"Digit diagnosis: best reading {best_reading.get('reading', '')} | "
            f"Top 5: {top_readings_text}"
        )
        self._statusbar.showMessage(
            f"Digit diagnosis complete: best reading {best_reading.get('reading', '')}."
        )
        self._show_digit_diagnosis_dialog(
            {
                "overlay_path": overlay_path,
                "candidate_count": len(self._pending_digit_candidates),
                "confident_count": confident_count,
                "detections": selected_digits,
                "best_reading": str(best_reading.get("reading", "")),
                "top_readings": [str(item.get("reading", "")) for item in reading_candidates[:5]],
                "restriction_text": restriction_text,
                "reading_roi_xyxy": [float(v) for v in reading_roi_xyxy],
                "selected_slot_count": len(best_slots),
                "selected_limiter_index": int(limiter_index),
                "slot_summaries": slot_summaries,
            }
        )

    def _run_autoread_candidate_search(
        self,
        image: np.ndarray,
        base_points: np.ndarray | None = None,
        base_sources: list[dict[str, object]] | None = None,
    ):
        if not self._testing_model_path:
            return
        print("[AutoRead] Starting candidate search.", flush=True)

        version = get_python_version(self._ml_backend_python)
        if not is_supported_tensorflow_backend(version):
            self._statusbar.showMessage("LeNet backend Python is not configured.")
            return

        expected_label = self._label_entry.text().strip()
        if expected_label and (len(expected_label) != NUM_SEGMENTS or not expected_label.isdigit()):
            expected_label = ""

        source_entries = base_sources or []
        if not source_entries and base_points is not None:
            source_entries = [{
                "name": "opencv",
                "points": np.asarray(base_points, dtype=np.float32),
            }]

        if not source_entries:
            self._statusbar.showMessage("Auto-Read could not build ROI candidates.")
            print("[AutoRead] No source entries available.", flush=True)
            return

        candidate_images: list[np.ndarray] = []
        candidate_names: list[str] = []
        candidate_points: dict[str, np.ndarray] = {}
        candidate_sources: dict[str, str] = {}

        for source_entry in source_entries:
            source_name = str(source_entry.get("name", "candidate"))
            source_points = source_entry.get("points")
            search_bbox = source_entry.get("search_bbox_xyxy")

            resolved_points: np.ndarray | None = None
            if isinstance(search_bbox, (list, tuple)) and len(search_bbox) == 4:
                refined = find_digit_strip_quad(
                    image,
                    search_bbox_xyxy=(
                        int(search_bbox[0]),
                        int(search_bbox[1]),
                        int(search_bbox[2]),
                        int(search_bbox[3]),
                    ),
                )
                if refined is not None:
                    resolved_points = np.asarray(refined["points"], dtype=np.float32)
            elif source_points is not None:
                resolved_points = np.asarray(source_points, dtype=np.float32)

            if resolved_points is None:
                continue

            point_candidates = generate_quad_candidates(resolved_points, image.shape)
            for candidate in point_candidates:
                short_name = str(candidate["name"])
                name = f"{source_name}:{short_name}"
                points = np.asarray(candidate["points"], dtype=np.float32)
                try:
                    strip, _preview_strip = extract_processed_strips(image, points)
                except Exception:
                    continue
                candidate_images.append(strip)
                candidate_names.append(name)
                candidate_points[name] = points
                candidate_sources[name] = source_name

        if not candidate_images:
            self._statusbar.showMessage("Auto-Read could not extract any valid ROI candidates.")
            print("[AutoRead] No valid ROI candidate strips extracted.", flush=True)
            return

        print(
            f"[AutoRead] Prepared {len(candidate_images)} ROI candidate strips from "
            f"{len(source_entries)} base sources.",
            flush=True,
        )
        self._testing_temp_image_path = write_temp_images(candidate_images, prefix="autoread_candidates_")
        self._pending_expected_label = expected_label
        self._pending_candidate_names = candidate_names
        self._pending_candidate_points = candidate_points
        self._pending_candidate_sources = candidate_sources
        command = build_lenet_predict_batch_command(
            backend_python=self._ml_backend_python,
            model_path=self._testing_model_path,
            images_dir=self._testing_temp_image_path,
            invert_input=True,
        )
        self._start_ml_worker(
            command,
            "Auto-Read (LeNet)",
            "Scoring multiple ROI candidates with LeNet...",
            self._on_lenet_batch_test_finished,
        )

    def _on_lenet_test_finished(self, result: dict[str, object]):
        predicted_label = str(result.get("predicted_label", ""))
        expected_label = str(result.get("expected_label", ""))
        confidences = result.get("confidences", [])
        confidence_summary = ", ".join(f"{float(score) * 100.0:.1f}%" for score in confidences)

        if expected_label:
            outcome = "MATCH" if expected_label == predicted_label else "MISMATCH"
            self._prediction_label.setText(
                f"Prediction: {predicted_label}\n"
                f"Expected: {expected_label}\n"
                f"Result: {outcome}\n"
                f"Per-digit confidence: {confidence_summary}"
            )
        else:
            self._prediction_label.setText(
                f"Prediction: {predicted_label}\n"
                f"Per-digit confidence: {confidence_summary}"
            )

        self._status_info.setText(f"Predicted: {predicted_label}")
        self._statusbar.showMessage(f"LeNet prediction: {predicted_label}")

    def _on_lenet_batch_test_finished(self, result: dict[str, object]):
        print("[AutoRead] LeNet batch result received.", flush=True)
        best = result.get("best", {})
        candidates = result.get("candidates", [])
        if not isinstance(best, dict):
            best = {}
        if not isinstance(candidates, list):
            candidates = []

        def _resolve_candidate_name(raw_name: str) -> str:
            try:
                idx = int(raw_name)
            except Exception:
                return raw_name
            if 0 <= idx < len(self._pending_candidate_names):
                return self._pending_candidate_names[idx]
            return raw_name

        normalized_candidates: list[dict[str, object]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            normalized = dict(candidate)
            resolved_name = _resolve_candidate_name(str(candidate.get("image_name", "")))
            normalized["image_name"] = resolved_name
            normalized["source_name"] = self._pending_candidate_sources.get(resolved_name, resolved_name)
            normalized_candidates.append(normalized)

        normalized_best = dict(best)
        normalized_best_name = _resolve_candidate_name(str(best.get("image_name", "base")))
        normalized_best["image_name"] = normalized_best_name
        normalized_best["source_name"] = self._pending_candidate_sources.get(
            normalized_best_name,
            normalized_best_name,
        )

        constrained_top = self._build_range_constrained_strip_candidates(normalized_candidates, top_k=5)
        if constrained_top:
            predicted_label = str(constrained_top[0].get("predicted_label", ""))
            confidences = constrained_top[0].get("confidences", [])
            best_name = str(constrained_top[0].get("image_name", "range"))
            best_score = float(constrained_top[0].get("score", 0.0)) * 100.0
            top_five = constrained_top
        else:
            vote_result = vote_prediction_candidates(normalized_candidates, top_k=5)
            voted_label = str(vote_result.get("voted_label", ""))
            top_five = vote_result.get("top_candidates", [])
            if not isinstance(top_five, list):
                top_five = []
            predicted_label = voted_label or str(normalized_best.get("predicted_label", ""))
            confidences = normalized_best.get("confidences", [])
            best_name = str(normalized_best.get("image_name", "base"))
            best_score = float(normalized_best.get("score", 0.0)) * 100.0

        confidence_summary = ", ".join(f"{float(score) * 100.0:.1f}%" for score in confidences)
        expected_label = str(getattr(self, "_pending_expected_label", "") or "")

        top_labels = []
        for candidate in top_five:
            if not isinstance(candidate, dict):
                continue
            top_labels.append(
                f"{candidate.get('image_name', '?')}={candidate.get('predicted_label', '')}"
            )
        top_summary = ", ".join(top_labels)

        best_points = self._pending_candidate_points.get(best_name)
        if best_points is not None:
            self._viewer.set_points(best_points)

        if expected_label:
            outcome = "MATCH" if expected_label == predicted_label else "MISMATCH"
            self._prediction_label.setText(
                f"Prediction: {predicted_label}\n"
                f"Expected: {expected_label}\n"
                f"Result: {outcome}\n"
                f"Voted from top 5 candidates\n"
                f"Best ROI candidate: {best_name} ({best_score:.1f})\n"
                f"Per-digit confidence: {confidence_summary}\n"
                f"Top candidates: {top_summary}"
            )
        else:
            self._prediction_label.setText(
                f"Prediction: {predicted_label}\n"
                f"Voted from top 5 candidates\n"
                f"Best ROI candidate: {best_name} ({best_score:.1f})\n"
                f"Per-digit confidence: {confidence_summary}\n"
                f"Top candidates: {top_summary}"
            )

        self._status_info.setText(f"Predicted: {predicted_label}")
        self._statusbar.showMessage(f"LeNet top-5 voted prediction: {predicted_label}")
        print(
            f"[AutoRead] Final voted prediction={predicted_label} "
            f"best={best_name} score={best_score:.1f}",
            flush=True,
        )
        self._show_auto_read_results_dialog({
            "voted_label": predicted_label,
            "expected_label": expected_label,
            "best_name": best_name,
            "best_score": best_score,
            "candidate_count": len(normalized_candidates),
            "top_candidates": top_five,
        })

    def _on_train_yolo(self):
        images_dir = QFileDialog.getExistingDirectory(
            self,
            "Select ROI_640 Folder"
        )
        if not images_dir:
            return

        labels_dir = QFileDialog.getExistingDirectory(
            self,
            "Select ROI_640_labels Folder"
        )
        if not labels_dir:
            return

        dialog = ExternalYoloTrainingDialog(
            images_dir=images_dir,
            labels_dir=labels_dir,
            backend_python=self._ml_backend_python,
            output_dir=self._last_yolo_model_dir,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        config = dialog.get_config()
        self._ml_backend_python = str(config["backend_python"])
        version = get_python_version(self._ml_backend_python)
        if not is_supported_tensorflow_backend(version):
            selected_version = (
                f"{version[0]}.{version[1]}" if version else "(could not detect version)"
            )
            QMessageBox.warning(
                self,
                "Unsupported Python Version",
                "YOLO backend should use Python 3.10 to 3.13.\n\n"
                f"Selected version: {selected_version}",
            )
            return

        self._last_yolo_model_dir = str(config["output_dir"])
        command = build_yolo_train_command(
            backend_python=str(config["backend_python"]),
            images_dir=str(config["images_dir"]),
            labels_dir=str(config["labels_dir"]),
            output_dir=str(config["output_dir"]),
            epochs=int(config["epochs"]),
            image_size=int(config["image_size"]),
            batch_size=int(config["batch_size"]),
        )
        self._start_ml_worker(
            command,
            "Training YOLOv8 Finder",
            "Training YOLOv8 digit-strip finder...",
            self._on_yolo_training_finished,
        )

    def _on_yolo_training_finished(self, result: dict[str, object]):
        best_model_path = str(result.get("best_model_path", ""))
        tflite_model_path = str(result.get("tflite_model_path", ""))
        export_warning = str(result.get("export_warning", ""))
        train_images = int(result.get("train_images", 0))
        val_images = int(result.get("val_images", 0))

        if best_model_path:
            self._last_yolo_model_dir = str(Path(best_model_path).parent.parent)
            self._yolo_testing_model_path = best_model_path
            self._save_app_state()

        info_lines = [
            "YOLO training output saved.",
            f"Best model: {best_model_path or '(not saved)'}",
            f"TFLite export: {tflite_model_path or '(not exported)'}",
        ]
        if export_warning:
            info_lines.append(f"Export warning: {export_warning}")
        self._prediction_label.setText("\n".join(info_lines))
        self._statusbar.showMessage("YOLOv8 training complete.")
        QMessageBox.information(
            self,
            "YOLOv8 Training Complete",
            f"Train images: {train_images}\n"
            f"Validation images: {val_images}\n\n"
            f"Best model: {best_model_path or '(not saved)'}\n"
            f"TFLite export: {tflite_model_path or '(not exported)'}"
            + (f"\n\nExport warning:\n{export_warning}" if export_warning else "")
        )

    def _on_select_yolo_model(self):
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLOv8 Finder Model",
            self._last_yolo_model_dir,
            "YOLO Models (*.pt *.onnx *.tflite);;All Files (*)",
        )
        if not model_path:
            return

        self._yolo_testing_model_path = model_path
        self._save_app_state()
        self._prediction_label.setText(
            f"Selected YOLO finder model:\n{self._yolo_testing_model_path}"
        )
        self._statusbar.showMessage(f"YOLO finder model selected: {Path(model_path).name}")

    def _on_auto_find_toggled(self, enabled: bool):
        if enabled and not self._yolo_testing_model_path:
            QMessageBox.information(
                self,
                "Select YOLO Model First",
                "Choose a YOLOv8 finder model before enabling auto find."
            )
            self._auto_find_checkbox.blockSignals(True)
            self._auto_find_checkbox.setChecked(False)
            self._auto_find_checkbox.blockSignals(False)
            return

        self._auto_find_strip_enabled = enabled
        if enabled:
            self._statusbar.showMessage("Auto Find Strip enabled.")

    def _on_auto_read_toggled(self, enabled: bool):
        if not hasattr(self, "_auto_read_checkbox"):
            self._auto_read_enabled = False
            self._statusbar.showMessage("Auto-Read (LeNet) is currently disabled in the UI.")
            return

        if enabled and not self._testing_model_path:
            QMessageBox.information(
                self,
                "Select LeNet Model First",
                "Choose a LeNet-5 model before enabling Auto-Read (LeNet)."
            )
            self._auto_read_checkbox.blockSignals(True)
            self._auto_read_checkbox.setChecked(False)
            self._auto_read_checkbox.blockSignals(False)
            return

        self._auto_read_enabled = enabled
        if enabled:
            if self._auto_find_checkbox.isChecked() and self._yolo_testing_model_path:
                self._statusbar.showMessage(
                    "Auto-Read (LeNet) enabled. YOLO will suggest the region, OpenCV will refine the 4 points, then LeNet will read."
                )
            else:
                self._statusbar.showMessage(
                    "Auto-Read (LeNet) enabled. OpenCV will try to find the strip, then LeNet will read."
                )
            return

        self._statusbar.showMessage("Auto-Read (LeNet) disabled.")

    def _apply_detected_points_and_maybe_read(
        self,
        points: np.ndarray,
        source_label: str,
        confidence_percent: float | None = None,
        auto_read: bool = False,
    ) -> bool:
        if not self._viewer.set_points(points):
            self._statusbar.showMessage(f"{source_label} returned an invalid 4-point region.")
            return False

        label = source_label
        if confidence_percent is not None:
            label = f"{label} ({confidence_percent:.1f}%)"

        self._prediction_label.setText(f"{label}\n4-point region prepared for extraction.")
        self._statusbar.showMessage(f"{label}. Drag points if needed, or extract.")

        if auto_read and self._testing_model_path:
            current_img = self._viewer.get_cv_image()
            if current_img is not None:
                self._testing_mode_enabled = True
                self._update_testing_ui_state()
                self._run_autoread_candidate_search(current_img, points)
        return True

    def _run_opencv_autoread_for_current_image(self):
        current_img = self._viewer.get_cv_image()
        if current_img is None or not self._testing_model_path:
            return

        detection = find_digit_strip_quad(current_img)
        if detection is None:
            self._statusbar.showMessage(
                "OpenCV could not isolate a digit strip for Auto-Read (LeNet)."
            )
            self._prediction_label.setText(
                "Auto-Read (LeNet): OpenCV could not find a usable digit strip."
            )
            return

        points = detection["points"]
        score = float(detection.get("score", 0.0)) * 100.0
        self._apply_detected_points_and_maybe_read(
            points,
            "OpenCV Auto-Read (LeNet)",
            confidence_percent=score,
            auto_read=True,
        )

    def _run_yolo_detection_for_current_image(self, auto_read: bool = False):
        current_img = self._viewer.get_cv_image()
        if current_img is None or not self._yolo_testing_model_path:
            return
        print("[YOLO] Starting auto-find for current image.", flush=True)

        version = get_python_version(self._ml_backend_python)
        if not is_supported_tensorflow_backend(version):
            self._statusbar.showMessage("YOLO backend Python is not configured.")
            return

        self._pending_auto_read_after_detection = auto_read and bool(self._testing_model_path)
        temp_image_path = write_temp_image(current_img, prefix="yolo_")
        command = build_yolo_predict_windows_command(
            backend_python=self._ml_backend_python,
            model_path=self._yolo_testing_model_path,
            image_path=temp_image_path,
            image_size=640,
            conf_threshold=0.10,
        )
        self._start_ml_worker(
            command,
            "Finding Digit Strip",
            "Running YOLOv8 sliding-window finder on the current image...",
            self._on_yolo_detection_finished,
        )

    def _on_yolo_detection_finished(self, result: dict[str, object]):
        print("[YOLO] Detection result received.", flush=True)
        self._pending_auto_read_after_detection = bool(self._pending_auto_read_after_detection)
        found = bool(result.get("found", False))
        if not found:
            self._statusbar.showMessage("YOLOv8 could not find a digit strip in this image.")
            self._prediction_label.setText("YOLOv8 finder: no digit strip detected.")
            self._pending_auto_read_after_detection = False
            return

        current_img = self._viewer.get_cv_image()
        if current_img is None:
            self._pending_auto_read_after_detection = False
            return

        limiter_boxes = self._extract_yolo_limiter_boxes(result, current_img.shape, max_boxes=5)
        if not limiter_boxes:
            self._statusbar.showMessage("YOLOv8 returned only tiny or invalid strip boxes.")
            self._prediction_label.setText("YOLOv8 finder: only tiny or invalid strip boxes were returned.")
            self._pending_auto_read_after_detection = False
            return

        base_sources: list[dict[str, object]] = []
        preview_points: np.ndarray | None = None
        preview_confidence = 0.0

        raw_candidates = result.get("candidates", [])
        if not isinstance(raw_candidates, list):
            raw_candidates = []

        for det_index, bbox in enumerate(limiter_boxes[:5]):
            confidence = 0.0
            for raw_candidate in raw_candidates:
                if not isinstance(raw_candidate, dict):
                    continue
                raw_bbox = raw_candidate.get("bbox_xyxy", [])
                if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
                    if all(abs(float(raw_bbox[i]) - float(bbox[i])) < 1.5 for i in range(4)):
                        confidence = float(raw_candidate.get("confidence", 0.0))
                        break
            bbox_variants = generate_bbox_candidates(
                (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                current_img.shape,
            )
            for variant in bbox_variants:
                search_bbox = variant.get("bbox_xyxy", [])
                if not isinstance(search_bbox, list) or len(search_bbox) != 4:
                    continue
                refined = find_digit_strip_quad(
                    current_img,
                    search_bbox_xyxy=(
                        int(search_bbox[0]),
                        int(search_bbox[1]),
                        int(search_bbox[2]),
                        int(search_bbox[3]),
                    ),
                )
                source_name = f"yolo{det_index + 1}:{variant.get('name', 'base')}"
                base_sources.append({
                    "name": source_name,
                    "search_bbox_xyxy": [
                        int(search_bbox[0]),
                        int(search_bbox[1]),
                        int(search_bbox[2]),
                        int(search_bbox[3]),
                    ],
                    "confidence": confidence,
                })
                if preview_points is None:
                    if refined is not None:
                        preview_points = np.asarray(refined["points"], dtype=np.float32)
                    else:
                        preview_points = bbox_to_quad((
                            float(search_bbox[0]),
                            float(search_bbox[1]),
                            float(search_bbox[2]),
                            float(search_bbox[3]),
                        ))
                    preview_confidence = confidence * 100.0

        if not base_sources or preview_points is None:
            self._statusbar.showMessage("YOLOv8 could not build usable ROI candidates.")
            self._prediction_label.setText("YOLOv8 finder: no usable ROI candidates.")
            self._pending_auto_read_after_detection = False
            return

        if not self._viewer.set_points(preview_points):
            self._pending_auto_read_after_detection = False
            return

        self._prediction_label.setText(
            f"YOLOv8 generated {len(base_sources)} ROI candidates.\n"
            f"Previewing strongest candidate ({preview_confidence:.1f}%)."
        )
        self._statusbar.showMessage(
            f"YOLOv8 generated {len(base_sources)} ROI candidates."
        )

        if self._pending_auto_read_after_detection and self._testing_model_path:
            self._testing_mode_enabled = True
            self._update_testing_ui_state()
            self._run_autoread_candidate_search(current_img, base_sources=base_sources)

        self._pending_auto_read_after_detection = False

    def _on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        self._file_list.clear()
        files = sorted(
            (
                p for p in Path(folder).iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda p: (p.stat().st_mtime, p.name.lower())
        )
        if not files:
            QMessageBox.information(self, "No Images",
                                    "No supported image files found.")
            return
        for f in files:
            item = QListWidgetItem(f.name)
            item.setData(Qt.ItemDataRole.UserRole, str(f))
            self._file_list.addItem(item)
        self._statusbar.showMessage(f"Loaded {len(files)} images from {folder}")
        self._file_list.setCurrentRow(0)

    def _on_file_selected(self, row: int):
        if row < 0:
            return
        item = self._file_list.item(row)
        path = item.data(Qt.ItemDataRole.UserRole)
        self._viewer.load_image(path)
        rotation_to_apply = (
            int(self._persistent_rotation_angle) % 360
            if self._persist_rotation_enabled
            else 0
        )
        self._rotation_slider.blockSignals(True)
        self._rotation_slider.setValue(rotation_to_apply)
        self._rotation_slider.blockSignals(False)
        self._rotation_slider.setEnabled(True)
        self._rotation_value.setText(f"{rotation_to_apply}°")
        self._viewer.set_rotation(rotation_to_apply)
        self._btn_select.setEnabled(True)
        self._btn_extract.setEnabled(True)
        self._btn_read_guidebox.setEnabled(True)
        self._btn_save.setEnabled(False)
        self._btn_find_digits.setEnabled(True)
        self._preview.clear()
        self._prediction_label.setText("")
        if self._apply_pending_readjust_template_if_any():
            self._statusbar.showMessage(
                "Readjust template loaded. Fine-tune the guidebox framing, preview it, then continue batch."
            )
            return
        self._statusbar.showMessage(
            f"Viewing: {item.text()} | Align the digits inside the 5:1 guidebox, then press READ."
        )
        if self._auto_find_strip_enabled and self._yolo_testing_model_path:
            self._run_yolo_detection_for_current_image(auto_read=False)
            return

    def _apply_pending_readjust_template_if_any(self) -> bool:
        if self._pending_readjust_framing_template is None:
            return False

        template = self._pending_readjust_framing_template
        self._pending_readjust_framing_template = None

        self._rotation_slider.blockSignals(True)
        self._rotation_slider.setValue(int(template.rotation_angle) % 360)
        self._rotation_slider.blockSignals(False)
        self._rotation_value.setText(f"{int(template.rotation_angle) % 360}°")
        self._viewer.set_rotation(int(template.rotation_angle) % 360)
        return self._viewer.apply_framing_template(template)

    def _set_pending_readjust_template(
        self,
        template: GuideboxFramingTemplate,
    ):
        self._pending_readjust_framing_template = GuideboxFramingTemplate(
            rotation_angle=int(template.rotation_angle) % 360,
            zoom_factor=float(template.zoom_factor),
            center_x_norm=float(template.center_x_norm),
            center_y_norm=float(template.center_y_norm),
            width_norm=float(template.width_norm),
            height_norm=float(template.height_norm),
        )

    def _on_rotation_changed(self, angle: int):
        self._persistent_rotation_angle = int(angle) % 360
        self._rotation_value.setText(f"{angle}°")
        changed = self._viewer.set_rotation(angle)
        if not changed:
            return
        self._btn_extract.setEnabled(True)
        self._btn_save.setEnabled(False)
        self._preview.clear()
        self._statusbar.showMessage(
            f"Rotation set to {angle}°. Re-align the strip inside the 5:1 guidebox, then extract."
        )

    def _on_start_select(self):
        self._viewer.start_selection()
        self._btn_extract.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._preview.clear()
        self._statusbar.showMessage(
            "Legacy 4-point mode enabled. The main workflow uses the fixed 5:1 guidebox."
        )

    def _on_batch_toggled(self, enabled: bool):
        self._btn_save.setText("Batch Save All" if enabled else "Save Segments")
        self._label_entry.setEnabled(not enabled)

        if enabled:
            self._statusbar.showMessage(
                "Batch mode enabled. Align the first image inside the 5:1 guidebox, preview it, then click Batch Save All."
            )
            self._btn_save.setEnabled(len(self._preview.get_segments()) == NUM_SEGMENTS)
            return

        self._statusbar.showMessage("Batch mode disabled.")
        self._btn_save.setEnabled(len(self._preview.get_segments()) == NUM_SEGMENTS)

    def _on_points_ready(self):
        self._btn_extract.setEnabled(self._viewer.get_cv_image() is not None)
        if self._testing_mode_enabled and self._testing_model_path:
            self._statusbar.showMessage(
                "Legacy 4-point selection updated. Main extraction still uses the guidebox."
            )
            return
        if self._batch_checkbox.isChecked():
            self._statusbar.showMessage(
                "Legacy 4-point selection updated. Batch saving still uses the guidebox framing template."
            )
            return

        self._statusbar.showMessage(
            "Legacy 4-point selection updated. Use the fixed 5:1 guidebox for the main extract-and-save flow."
        )

    def _on_extract(self):
        guide_crop = self._viewer.get_guidebox_crop()
        if guide_crop is None:
            self._statusbar.showMessage("Guidebox crop is empty. Align the strip inside the fixed 5:1 guidebox first.")
            return
        self._btn_extract.setEnabled(False)
        if self._testing_mode_enabled and self._testing_model_path:
            self._statusbar.showMessage("Processing guidebox crop and running model prediction...")
        else:
            self._statusbar.showMessage("Processing guidebox crop...")
        self._worker = WarpWorker(guide_crop)
        self._worker.signals.finished.connect(self._on_warp_done)
        self._worker.signals.error.connect(self._on_warp_error)
        self._worker.finished.connect(self._cleanup_warp_worker)
        self._worker.start()

    def _on_warp_done(self, strip: np.ndarray):
        self._preview.set_strip(strip)
        self._btn_extract.setEnabled(True)
        if self._testing_mode_enabled and self._testing_model_path:
            self._btn_save.setEnabled(False)
            self._run_viewer_test_on_strip(strip)
            return
        self._btn_save.setEnabled(True)
        if self._batch_checkbox.isChecked():
            self._statusbar.showMessage(
                "Guidebox preview ready. Batch mode is active: click Batch Save All."
            )
            return
        self._statusbar.showMessage(
            "Guidebox extraction complete. Enter a 5-char label and save."
        )

    def _on_warp_error(self, msg: str):
        QMessageBox.critical(self, "Processing Error", msg)
        self._btn_extract.setEnabled(True)
        self._statusbar.showMessage("Error during processing.")

    def _on_set_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self._output_dir = folder
            self._statusbar.showMessage(f"Output directory: {folder}")

    def _on_save_segments(self):
        if self._batch_checkbox.isChecked():
            self._on_batch_save_segments()
            return

        label = self._label_entry.text().strip().upper()
        if not is_digit_or_unreadable_label(label):
            QMessageBox.warning(
                self, "Invalid Label",
                "Please enter exactly 5 characters using digits (0-9) and X."
            )
            return
        segments = self._preview.get_segments()
        if len(segments) != NUM_SEGMENTS:
            QMessageBox.warning(self, "No Segments",
                                "Extract an image first.")
            return
        if not self._output_dir:
            self._on_set_output()
            if not self._output_dir:
                return

        saved, folders_used, write_errors = self._save_segments_with_label(segments, label)

        roi_raw_saved = 0
        roi_640_saved = 0
        roi_errors = 0
        guide_crop = self._viewer.get_guidebox_crop()
        if guide_crop is not None:
            roi_raw_saved, roi_640_saved, roi_errors, _roi_base = self._save_roi_exports(
                guide_crop,
                label
            )
        else:
            roi_errors = 1

        self._statusbar.showMessage(
            f"Saved {saved} segments for label '{label}' → {self._output_dir}"
        )
        QMessageBox.information(
            self, "Saved",
            f"Saved {saved} segment(s) into:\n{self._output_dir}\n\n"
            f"Folders: {', '.join(sorted(folders_used))}\n"
            f"ROI raw saved: {roi_raw_saved}\n"
            f"ROI 640 saved: {roi_640_saved}\n"
            f"Write errors: {write_errors + roi_errors}"
        )

    def _on_batch_save_segments(self):
        if self._file_list.count() == 0:
            QMessageBox.warning(self, "No Images", "Open a folder with images first.")
            return

        framing_template = self._viewer.capture_framing_template()
        if framing_template is None:
            QMessageBox.warning(
                self,
                "Missing Template",
                "Batch mode needs a guidebox framing template. Align the current image inside the fixed 5:1 guidebox first."
            )
            return

        if not self._output_dir:
            self._on_set_output()
            if not self._output_dir:
                return

        rotation_angle = int(framing_template.rotation_angle) % 360

        batch_items: list[tuple[str, str]] = []
        for row in range(self._file_list.count()):
            item = self._file_list.item(row)
            batch_items.append((item.data(Qt.ItemDataRole.UserRole), item.text()))

        total_images = len(batch_items)
        previous_label = self._label_entry.text().strip().upper()
        if previous_label and not is_digit_or_unreadable_label(previous_label):
            previous_label = ""

        processed_images = 0
        skipped_images = 0
        error_count = 0
        saved_segments = 0
        roi_raw_saved_count = 0
        roi_640_saved_count = 0
        used_folders: set[str] = set()
        canceled = False
        readjust_target_path = ""
        processed_paths: list[str] = []

        self._statusbar.showMessage(f"Batch processing started for {total_images} images…")

        for i, (image_path, image_name) in enumerate(batch_items):

            src_img = read_image_any(image_path, cv2.IMREAD_COLOR)
            if src_img is None:
                skipped_images += 1
                continue

            if rotation_angle != 0:
                src_img = ImageViewer._rotate_image(src_img, rotation_angle)

            guide_crop = crop_image_with_normalized_rect(
                src_img,
                framing_template.center_x_norm - (framing_template.width_norm / 2.0),
                framing_template.center_y_norm - (framing_template.height_norm / 2.0),
                framing_template.width_norm,
                framing_template.height_norm,
            )
            if guide_crop is None:
                error_count += 1
                continue

            try:
                binary_strip = prepare_guidebox_strip(guide_crop)
            except Exception:
                error_count += 1
                continue

            binary_segments = split_strip_segments(binary_strip)
            preview_segments = split_strip_segments(binary_strip)

            dialog = BatchLabelDialog(
                preview_segments=preview_segments,
                image_name=image_name,
                image_index=i + 1,
                total_images=total_images,
                previous_label=previous_label,
                parent=self,
            )
            if dialog.exec() != QDialog.DialogCode.Accepted:
                if dialog.readjust_requested():
                    readjust_target_path = image_path
                else:
                    canceled = True
                break

            label = dialog.get_label()
            previous_label = label
            self._label_entry.setText(label)

            saved_now, folders_now, write_errors = self._save_segments_with_label(
                binary_segments,
                label
            )
            saved_segments += saved_now
            used_folders.update(folders_now)
            error_count += write_errors

            roi_raw_saved, roi_640_saved, roi_write_errors, _roi_base = self._save_roi_exports(
                guide_crop,
                label
            )
            roi_raw_saved_count += roi_raw_saved
            roi_640_saved_count += roi_640_saved
            error_count += roi_write_errors

            processed_images += 1
            processed_paths.append(image_path)

            self._statusbar.showMessage(
                f"Batch progress: {processed_images}/{total_images} image(s) saved."
            )
            QApplication.processEvents()

        if readjust_target_path:
            self._set_pending_readjust_template(framing_template)
            self._remove_processed_and_focus_image(processed_paths, readjust_target_path)
            result_title = "Batch Paused for Readjust"
            self._statusbar.showMessage(
                f"{result_title} — processed: {processed_images}, "
                f"remaining in list: {self._file_list.count()}"
            )
            QMessageBox.information(
                self,
                result_title,
                f"Processed images removed from list: {processed_images}\n"
                f"ROI raw saved so far: {roi_raw_saved_count}\n"
                f"ROI 640 saved so far: {roi_640_saved_count}\n"
                f"Now focused on: {Path(readjust_target_path).name}\n\n"
                "Rotation, zoom, and framing were inherited. "
                "Readjust the guidebox framing, preview it, then click Batch Save All again to continue."
            )
            return

        result_title = "Batch Stopped" if canceled else "Batch Complete"
        self._statusbar.showMessage(
            f"{result_title} — images: {processed_images}/{total_images}, "
            f"saved segments: {saved_segments}, errors: {error_count}"
        )
        QMessageBox.information(
            self,
            result_title,
            f"Processed images: {processed_images}/{total_images}\n"
            f"Skipped images (load failed): {skipped_images}\n"
            f"Saved segments: {saved_segments}\n"
            f"ROI raw saved: {roi_raw_saved_count}\n"
            f"ROI 640 saved: {roi_640_saved_count}\n"
            f"Errors: {error_count}\n"
            f"Output: {self._output_dir}\n"
            f"Folders: {', '.join(sorted(used_folders)) if used_folders else '(none)'}"
        )

    def _remove_processed_and_focus_image(
        self,
        processed_paths: list[str],
        focus_path: str
    ):
        if processed_paths:
            processed_set = set(processed_paths)
            for row in reversed(range(self._file_list.count())):
                item = self._file_list.item(row)
                path = item.data(Qt.ItemDataRole.UserRole)
                if path in processed_set:
                    self._file_list.takeItem(row)

        for row in range(self._file_list.count()):
            item = self._file_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == focus_path:
                self._file_list.setCurrentRow(row)
                return

        if self._file_list.count() > 0:
            self._file_list.setCurrentRow(0)

    def _save_roi_exports(
        self,
        guide_crop: np.ndarray,
        label: str
    ) -> tuple[int, int, int, str]:
        if guide_crop is None or guide_crop.size == 0:
            return 0, 0, 1, ""

        raw_crop = guide_crop.copy()
        roi_640 = prepare_guidebox_dataset_image(raw_crop)

        raw_dir = Path(self._output_dir) / ROI_RAW_DIR_NAME
        size_dir = Path(self._output_dir) / ROI_640_DIR_NAME
        raw_dir.mkdir(parents=True, exist_ok=True)
        size_dir.mkdir(parents=True, exist_ok=True)

        uid = uuid.uuid4().hex[:10]
        base_name = f"{label}_{uid}"
        raw_path = raw_dir / f"{base_name}_raw.png"
        size_path = size_dir / f"{base_name}_640.png"

        raw_ok = cv2.imwrite(str(raw_path), raw_crop)
        size_ok = cv2.imwrite(str(size_path), roi_640)
        errors = int(not raw_ok) + int(not size_ok)
        return int(raw_ok), int(size_ok), errors, base_name

    def _save_segments_with_label(
        self,
        segments: list[np.ndarray],
        label: str
    ) -> tuple[int, set[str], int]:
        saved = 0
        folders_used: set[str] = set()
        write_errors = 0

        for i, seg in enumerate(segments):
            char = label[i]
            category_folder = self._label_char_to_category_folder(char)
            folders_used.add(category_folder)
            char_dir = os.path.join(self._output_dir, category_folder)
            os.makedirs(char_dir, exist_ok=True)
            fname = f"segment_{uuid.uuid4().hex[:8]}.png"
            save_path = os.path.join(char_dir, fname)
            if cv2.imwrite(save_path, seg):
                saved += 1
            else:
                write_errors += 1

        return saved, folders_used, write_errors

    @staticmethod
    def _label_char_to_category_folder(label_char: str) -> str:
        if label_char.upper() == UNREADABLE_LABEL_CHAR:
            return UNREADABLE_FOLDER_NAME
        return label_char
    def _on_diversify_data(self):
        input_parent = QFileDialog.getExistingDirectory(
            self, "Select Source Folder (0-9 / Unreadable)"
        )
        if not input_parent:
            return

        valid, message, category_folders = self._validate_digit_category_parent(input_parent)
        if not valid:
            QMessageBox.warning(self, "Invalid Input Folder", message)
            return

        output_parent = QFileDialog.getExistingDirectory(self, "Select Output Folder for Augmented Data")
        if not output_parent:
            return

        # Ask user how many variations they want per image
        num_variants, ok = QInputDialog.getInt(
            self, "Augmentation Density", 
            "How many augmented versions to create per image?", 2, 1, 10
        )
        if not ok:
            return

        self._statusbar.showMessage("Diversifying dataset... please wait.")
        processed, errors = self._diversify_category_images(
            input_parent, output_parent, category_folders, num_variants
        )

        QMessageBox.information(
            self, "Diversification Complete",
            f"Finished generating {processed} new images.\nErrors encountered: {errors}"
        )
        self._statusbar.showMessage(f"Generated {processed} augmented images.")

    def _diversify_category_images(self, input_parent, output_parent, categories, variants):
        processed_count = 0
        error_count = 0

        for cat in categories:
            src_dir = Path(input_parent) / cat
            dst_dir = Path(output_parent) / cat
            dst_dir.mkdir(parents=True, exist_ok=True)

            for entry in src_dir.iterdir():
                if not entry.is_file() or entry.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                img = read_image_any(str(entry), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                for v in range(variants):
                    augmented = self._apply_augmentation_pipeline(img)
                    new_name = f"aug_{v}_{uuid.uuid4().hex[:6]}_{entry.name}"
                    save_path = dst_dir / new_name
                    
                    if cv2.imwrite(str(save_path), augmented):
                        processed_count += 1
                    else:
                        error_count += 1
        
        return processed_count, error_count

    def _apply_augmentation_pipeline(self, img: np.ndarray) -> np.ndarray:
        """Applies 3D perspective warping and smart thresholding."""
        aug = img.copy()
        h, w = aug.shape[:2]
        
        # 1. Perspective 'Perspective Squeeze' (3D Tilt)
        # We define source points (the 28x28 square)
        
        # Choose a squeeze amount (2 to 5 pixels)
        # Squeezing more than 6-7 pixels on a 28px canvas usually ruins legibility
        squeeze = np.random.randint(2, 6)
        side = np.random.choice(['left', 'right', 'none'], p=[0.2, 0.2, 0.6])
        src_pts = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        dst_pts = src_pts.copy()
        if side == 'right':
            # Squeeze the right edge inward (top down, bottom up)
            dst_pts[1] = [w-1, squeeze]        # TR moves down
            dst_pts[2] = [w-1, h-1-squeeze]    # BR moves up
        elif side == 'left':
            # Squeeze the left edge inward
            dst_pts[0] = [0, squeeze]          # TL moves down
            dst_pts[3] = [0, h-1-squeeze]      # BL moves up

        M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
        aug = cv2.warpPerspective(aug, M_persp, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # 2. Subtle Rotation & Translation 
        # (Reduced ranges because perspective already adds 'tilt')
        angle = np.random.uniform(-2, 2)
        M_rot = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        aug = cv2.warpAffine(aug, M_rot, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # 3. Smart Thresholding (Otsu's Method)
        blur = cv2.GaussianBlur(aug, (3, 3), 0)
        optimal_thresh, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Nudge the threshold to vary thickness
        nudge = np.random.randint(-10, 10)
        smart_thresh_val = np.clip(optimal_thresh + nudge, 10, 245)
        _, aug = cv2.threshold(aug, smart_thresh_val, 255, cv2.THRESH_BINARY)

        # 4. White Oblivion Safeguard
        non_zero = cv2.countNonZero(aug)
        coverage = non_zero / (h * w)
        if coverage < 0.05 or coverage > 0.95:
            # Revert to a clean adaptive threshold if perspective + Otsu killed the image
            aug = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )

        # 5. Fine-Grained Noise
        # Instead of large dots, we use a very light 'salt and pepper' 
        # that only affects the dark pixels.
        noise = np.random.randint(-12, 13, (h, w), dtype='int16')
        aug_noise = np.clip(aug.astype('int16') + noise, 0, 255).astype('uint8')
        aug = np.where(aug < 255, aug_noise, 255)

        return aug

    def _on_balance_dataset(self):
        input_parent = QFileDialog.getExistingDirectory(
            self, "Select Imbalanced Source Folder (0-9 / Unreadable)"
        )
        if not input_parent:
            return

        valid, message, category_folders = self._validate_digit_category_parent(input_parent)
        if not valid:
            QMessageBox.warning(self, "Invalid Input Folder", message)
            return

        # PRE-SCAN: Count the images so the UI is smart
        category_counts = {}
        for cat in category_folders:
            cat_path = Path(input_parent) / cat
            count = len([p for p in cat_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
            category_counts[cat] = count

        # Open the Smart Dialog
        dialog = BalanceDialog(category_counts, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
            
        target_count = dialog.get_target()

        output_parent = QFileDialog.getExistingDirectory(self, "Select Output Folder for Balanced Data")
        if not output_parent:
            return

        self._statusbar.showMessage(f"Balancing dataset to {target_count} per class... This may take a moment.")
        QApplication.processEvents() 

        log_messages = self._balance_category_images(
            input_parent, output_parent, category_folders, target_count
        )

        summary = "\n".join(log_messages)
        QMessageBox.information(
            self, "Balancing Complete",
            f"Finished balancing dataset to {target_count} images per class.\n\nSummary:\n{summary}"
        )
        self._statusbar.showMessage("Dataset balancing complete.")

    def _balance_category_images(self, input_parent, output_parent, categories, target_count):
        log = []
        for cat in categories:
            src_dir = Path(input_parent) / cat
            dst_dir = Path(output_parent) / cat
            dst_dir.mkdir(parents=True, exist_ok=True)

            # Gather all valid images in this category
            image_paths = [
                p for p in src_dir.iterdir() 
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            current_count = len(image_paths)

            if current_count == 0:
                log.append(f"Category '{cat}': Skipped (0 images)")
                continue

            if current_count == target_count:
                # Perfect already, just copy them
                self._copy_images(image_paths, dst_dir)
                log.append(f"Category '{cat}': Copied {current_count} (Already balanced)")

            elif current_count > target_count:
                # DOWNSAMPLING (Too many images, e.g., '0's)
                # Use smart deduplication to remove the most similar look-alikes
                kept_paths = self._smart_downsample(image_paths, target_count)
                self._copy_images(kept_paths, dst_dir)
                log.append(f"Category '{cat}': Downsampled {current_count} -> {target_count} (Removed look-alikes)")

            else:
                # OVERSAMPLING (Too few images, e.g., '3's)
                # Copy all originals first
                self._copy_images(image_paths, dst_dir)
                
                # Generate new augmented ones to make up the difference
                shortfall = target_count - current_count
                self._generate_targeted_augmentations(image_paths, dst_dir, shortfall)
                log.append(f"Category '{cat}': Oversampled {current_count} -> {target_count} (+{shortfall} augmented)")

        return log

    def _copy_images(self, paths, dst_dir):
        """Helper to copy original images to the balanced folder."""
        for p in paths:
            img = read_image_any(str(p), cv2.IMREAD_UNCHANGED)
            if img is not None:
                cv2.imwrite(str(dst_dir / p.name), img)

    def _generate_targeted_augmentations(self, base_paths, dst_dir, shortfall):
        """Randomly picks base images and augments them until the shortfall is met."""
        for i in range(shortfall):
            # Randomly select a source image to augment
            src_path = np.random.choice(base_paths)
            img = read_image_any(str(src_path), cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                augmented = self._apply_augmentation_pipeline(img)
                new_name = f"bal_aug_{uuid.uuid4().hex[:6]}.png"
                cv2.imwrite(str(dst_dir / new_name), augmented)

    def _smart_downsample(self, image_paths, target_count):
        """
        Keeps images that are visually distinct.
        For a 28x28 image, comparing raw pixels using Mean Squared Error (MSE)
        is a very fast and effective way to find near-duplicates.
        """
        # Load all images into memory for fast comparison (they are tiny, so this is safe)
        loaded_images = []
        for p in image_paths:
            img = read_image_any(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Blur slightly to focus on structure rather than noise
                blur = cv2.GaussianBlur(img, (3,3), 0)
                loaded_images.append((p, blur.astype('float32')))

        # Shuffle to ensure we don't just keep the first N images chronologically
        np.random.shuffle(loaded_images)

        kept = [loaded_images[0]] # Always keep the first one
        
        # We need to find `target_count` images.
        # We check each candidate against our `kept` list. If it's too similar 
        # (MSE is too low) to an already kept image, we discard it.
        
        # A dynamic threshold. We start strict, and loosen it if we are running out of images
        mse_threshold = 800.0 

        while len(kept) < target_count and mse_threshold >= 0:
            for item in loaded_images:
                if len(kept) >= target_count:
                    break
                if item in kept:
                    continue
                
                path, img_data = item
                
                # Check MSE against all currently kept images
                # If the minimum distance to ANY kept image is greater than threshold, it's unique enough
                is_unique = True
                for _, kept_data in kept:
                    mse = np.mean((img_data - kept_data) ** 2)
                    if mse < mse_threshold:
                        is_unique = False
                        break
                
                if is_unique:
                    kept.append(item)
            
            # If we looped through all images and still need more, lower our standards (accept more similar images)
            mse_threshold -= 100.0

        # If we STILL don't have enough (very rare, means images are literal perfect clones),
        # just randomly sample the rest.
        if len(kept) < target_count:
            remaining = [i for i in loaded_images if i not in kept]
            np.random.shuffle(remaining)
            needed = target_count - len(kept)
            kept.extend(remaining[:needed])

        return [item[0] for item in kept]

    def _on_invert_colors(self):
        input_parent = QFileDialog.getExistingDirectory(
            self,
            "Select a Folder with 0-9 (+ optional Unreadable) Categories"
        )
        if not input_parent:
            return

        valid, message, category_folders = self._validate_digit_category_parent(
            input_parent
        )
        if not valid:
            QMessageBox.warning(self, "Invalid Input Folder", message)
            return

        output_parent = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder"
        )
        if not output_parent:
            return

        processed_count, skipped_count, error_count = self._invert_category_images(
            input_parent,
            output_parent,
            category_folders
        )

        self._statusbar.showMessage(
            f"Invert complete — processed: {processed_count}, "
            f"skipped: {skipped_count}, errors: {error_count}"
        )
        QMessageBox.information(
            self,
            "Invert Colors Complete",
            "Processing finished.\n\n"
            f"Input: {input_parent}\n"
            f"Output: {output_parent}\n"
            f"Category folders: {', '.join(category_folders)}\n"
            f"Processed images: {processed_count}\n"
            f"Skipped non-images/unreadable: {skipped_count}\n"
            f"Errors while saving: {error_count}"
        )

    def _validate_digit_category_parent(
        self,
        parent_folder: str
    ) -> tuple[bool, str, list[str]]:
        try:
            children = [p for p in Path(parent_folder).iterdir() if p.is_dir()]
        except OSError as exc:
            return False, f"Cannot access selected folder:\n{exc}", []

        if not children:
            return (
                False,
                "Selected folder has no subfolders. "
                "It must contain at least one category subfolder (0-9 or Unreadable).",
                []
            )

        invalid_names = [
            p.name for p in children
            if not ((p.name.isdigit() and len(p.name) == 1) or p.name.lower() == "unreadable")
        ]
        if invalid_names:
            return (
                False,
                "All direct subfolders must be a single digit name (0-9) "
                "or Unreadable.\n\n"
                f"Invalid subfolder(s): {', '.join(sorted(invalid_names))}",
                []
            )

        digit_folders = sorted(
            [p.name for p in children if p.name.isdigit() and len(p.name) == 1],
            key=int
        )
        unreadable_folders = sorted(
            [p.name for p in children if p.name.lower() == "unreadable"],
            key=str.lower
        )
        category_folders = digit_folders + unreadable_folders
        return True, "", category_folders

    def _invert_category_images(
        self,
        input_parent: str,
        output_parent: str,
        digit_folders: list[str]
    ) -> tuple[int, int, int]:
        processed_count = 0
        skipped_count = 0
        error_count = 0

        for digit in digit_folders:
            src_dir = Path(input_parent) / digit
            dst_dir = Path(output_parent) / digit
            dst_dir.mkdir(parents=True, exist_ok=True)

            for entry in src_dir.iterdir():
                if not entry.is_file() or entry.suffix.lower() not in IMAGE_EXTENSIONS:
                    skipped_count += 1
                    continue

                src_img = read_image_any(str(entry), cv2.IMREAD_UNCHANGED)
                if src_img is None:
                    skipped_count += 1
                    continue

                inverted = cv2.bitwise_not(src_img)
                save_path = dst_dir / entry.name
                if cv2.imwrite(str(save_path), inverted):
                    processed_count += 1
                else:
                    error_count += 1

        return processed_count, skipped_count, error_count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "com.digitextractor.app"
            )
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    icon = QIcon()
    for icon_path in (base_dir / "icon.ico", base_dir / "icon.png"):
        if icon_path.exists():
            candidate = QIcon(str(icon_path))
            if not candidate.isNull():
                icon = candidate
                break
    if not icon.isNull():
        app.setWindowIcon(icon)

    # Dark palette for Fusion
    from PyQt6.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(43, 43, 43))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(50, 50, 50))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    win = MainWindow()
    if not icon.isNull():
        win.setWindowIcon(icon)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
