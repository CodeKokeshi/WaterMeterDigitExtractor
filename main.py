"""
High-Precision Image Dataset Extractor
PyQt6 + OpenCV application for extracting and segmenting digit images.
"""

import sys
import os
import uuid
import importlib
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from digit_ml_commands import (
    MlCommandWorker as ExternalMlCommandWorker,
    build_lenet_predict_command,
    build_lenet_train_command,
    get_python_version,
    is_supported_tensorflow_backend,
    write_temp_strip_image,
)
from digit_ml_dialogs import LeNetTrainingDialog as ExternalLeNetTrainingDialog


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


# ---------------------------------------------------------------------------
# CV Processing Worker (runs on a QThread to keep UI responsive)
# ---------------------------------------------------------------------------
class ProcessingSignals(QObject):
    finished = pyqtSignal(object)   # emits the 140x28 strip (numpy)
    error = pyqtSignal(str)


class WarpWorker(QThread):
    """Perspective-warp → binarize → downscale in a background thread."""

    def __init__(self, image: np.ndarray, points: np.ndarray):
        super().__init__()
        self.signals = ProcessingSignals()
        self.image = image
        self.points = points

    def run(self):
        try:
            strip, _preview_strip = extract_processed_strips(self.image, self.points)
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
    """Zoomable / pannable image viewer with 4-point polygon selection."""

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
        self.scale(ratio, ratio)
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
        self._output_dir = ""
        self._worker: WarpWorker | None = None
        self._ml_worker: ExternalMlCommandWorker | None = None
        self._ml_progress: QProgressDialog | None = None
        self._ml_backend_python = sys.executable
        self._last_trained_model_dir = str(Path.cwd() / LENET_MODEL_DIR_NAME)
        self._last_tflite_model_dir = self._last_trained_model_dir
        self._last_trained_model_path = ""
        self._testing_model_path = ""
        self._testing_mode_enabled = False
        self._testing_temp_image_path = ""
        self._pending_readjust_rotation: int | None = None
        self._pending_readjust_points_normalized: np.ndarray | None = None
        self._pending_readjust_zoom_factor: float | None = None

        self._build_ui()
        self._build_menu()
        self._connect_signals()

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

        self._btn_select = QPushButton("Select 4 Points")
        self._btn_select.setEnabled(False)
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
            "Apply current rotation and 4-point selection to all images in this folder."
        )
        ctrl_layout.addWidget(self._batch_checkbox, 0, 3)

        self._btn_save = QPushButton("Save Segments")
        self._btn_save.setEnabled(False)
        ctrl_layout.addWidget(self._btn_save, 0, 4)

        self._btn_output = QPushButton("Set Output Dir…")
        ctrl_layout.addWidget(self._btn_output, 0, 5)

        self._rotation_label = QLabel("Rotate:")
        ctrl_layout.addWidget(self._rotation_label, 1, 0)

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
        ctrl_layout.addWidget(self._rotation_slider, 1, 1, 1, 4)

        self._rotation_value = QLabel("0°")
        self._rotation_value.setFixedWidth(40)
        ctrl_layout.addWidget(self._rotation_value, 1, 5)

        self._status_info = QLabel("")
        self._status_info.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred
        )
        self._status_info.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        ctrl_layout.addWidget(self._status_info, 1, 6)
        ctrl_layout.setColumnStretch(2, 1)
        ctrl_layout.setColumnStretch(6, 2)

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

        testing_menu = menu.addMenu("&Testing")
        act_test_lenet = QAction("Test LeNet-5 Digit Modelâ€¦", self)
        act_test_lenet.triggered.connect(self._on_test_lenet)
        testing_menu.addAction(act_test_lenet)
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
        self._btn_save.clicked.connect(self._on_save_segments)
        self._btn_output.clicked.connect(self._on_set_output)
        self._batch_checkbox.toggled.connect(self._on_batch_toggled)
        self._rotation_slider.valueChanged.connect(self._on_rotation_changed)
        self._viewer.points_ready.connect(self._on_points_ready)

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
        self._ml_worker = ExternalMlCommandWorker(command, str(Path(__file__).resolve().parent))
        self._ml_worker.finished.connect(success_handler)
        self._ml_worker.error.connect(self._on_ml_worker_error)
        self._ml_worker.finished.connect(self._cleanup_ml_worker)
        self._ml_worker.error.connect(self._cleanup_ml_worker)
        self._statusbar.showMessage(status_message)
        self._ml_worker.start()

    def _cleanup_ml_worker(self, *_args):
        if self._ml_progress is not None:
            self._ml_progress.close()
            self._ml_progress = None
        self._ml_worker = None

    def _on_ml_worker_error(self, message: str):
        self._statusbar.showMessage("ML task failed.")
        QMessageBox.critical(self, "ML Task Failed", message)

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
        if keras_path:
            self._last_trained_model_dir = str(Path(keras_path).parent)
        if tflite_path:
            self._last_tflite_model_dir = str(Path(tflite_path).parent)

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
                "Viewer testing mode enabled. Select 4 points, then click Extract."
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

        self._testing_temp_image_path = write_temp_strip_image(strip)
        command = build_lenet_predict_command(
            backend_python=self._ml_backend_python,
            model_path=self._testing_model_path,
            image_path=self._testing_temp_image_path,
            expected_label=expected_label,
        )
        self._start_ml_worker(
            command,
            "Testing LeNet-5 Model",
            "Running LeNet-5 prediction on the current 4-point selection...",
            self._on_lenet_test_finished,
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
        self._rotation_slider.blockSignals(True)
        self._rotation_slider.setValue(0)
        self._rotation_slider.blockSignals(False)
        self._rotation_slider.setEnabled(True)
        self._rotation_value.setText("0°")
        self._btn_select.setEnabled(True)
        self._btn_extract.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._preview.clear()
        self._prediction_label.setText("")
        if self._apply_pending_readjust_template_if_any():
            self._statusbar.showMessage(
                "Readjust template loaded. Drag points to adjust, then continue batch."
            )
            return
        self._statusbar.showMessage(f"Viewing: {item.text()}")

    def _apply_pending_readjust_template_if_any(self) -> bool:
        if (
            self._pending_readjust_rotation is None
            or self._pending_readjust_points_normalized is None
        ):
            return False

        rotation = int(self._pending_readjust_rotation) % 360
        points_normalized = self._pending_readjust_points_normalized.copy()
        zoom_factor = self._pending_readjust_zoom_factor or 1.0

        self._pending_readjust_rotation = None
        self._pending_readjust_points_normalized = None
        self._pending_readjust_zoom_factor = None

        self._rotation_slider.blockSignals(True)
        self._rotation_slider.setValue(rotation)
        self._rotation_slider.blockSignals(False)
        self._rotation_value.setText(f"{rotation}°")
        self._viewer.set_rotation(rotation)

        current_img = self._viewer.get_cv_image()
        if current_img is None:
            return False

        h, w = current_img.shape[:2]
        inherited_points = denormalize_points(points_normalized, w, h)
        points_set = self._viewer.set_points(inherited_points)
        self._viewer.set_zoom_factor(zoom_factor)
        return points_set

    def _set_pending_readjust_template(
        self,
        rotation_angle: int,
        normalized_points: np.ndarray,
        zoom_factor: float,
    ):
        self._pending_readjust_rotation = int(rotation_angle) % 360
        self._pending_readjust_points_normalized = normalized_points.copy()
        self._pending_readjust_zoom_factor = float(zoom_factor)

    def _on_rotation_changed(self, angle: int):
        self._rotation_value.setText(f"{angle}°")
        changed = self._viewer.set_rotation(angle)
        if not changed:
            return
        self._btn_extract.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._preview.clear()
        self._statusbar.showMessage(
            f"Rotation set to {angle}°. Re-select 4 points, then extract."
        )

    def _on_start_select(self):
        self._viewer.start_selection()
        self._btn_extract.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._preview.clear()
        self._statusbar.showMessage(
            "Click 4 corners on the image. Press Esc to cancel."
        )

    def _on_batch_toggled(self, enabled: bool):
        self._btn_save.setText("Batch Save All" if enabled else "Save Segments")
        self._label_entry.setEnabled(not enabled)

        if enabled:
            self._statusbar.showMessage(
                "Batch mode enabled. Rotate + set 4 points once, then click Batch Save All."
            )
            self._btn_save.setEnabled(self._viewer.get_points() is not None)
            return

        self._statusbar.showMessage("Batch mode disabled.")
        self._btn_save.setEnabled(len(self._preview.get_segments()) == NUM_SEGMENTS)

    def _on_points_ready(self):
        self._btn_extract.setEnabled(True)
        if self._testing_mode_enabled and self._testing_model_path:
            self._statusbar.showMessage(
                "4 points placed. Testing mode is enabled: click Extract to predict."
            )
            return
        if self._batch_checkbox.isChecked():
            self._btn_save.setEnabled(True)
            self._statusbar.showMessage(
                "4 points placed (auto-sorted). Batch mode ready: click Batch Save All."
            )
            return

        self._statusbar.showMessage(
            "4 points placed (auto-sorted). "
            "Drag handles to fine-tune, then click Extract."
        )

    def _on_extract(self):
        pts = self._viewer.get_points()
        img = self._viewer.get_cv_image()
        if pts is None or img is None:
            return
        self._btn_extract.setEnabled(False)
        if self._testing_mode_enabled and self._testing_model_path:
            self._statusbar.showMessage("Processing selection and running model prediction...")
        else:
            self._statusbar.showMessage("Processing...")
        self._worker = WarpWorker(img, pts)
        self._worker.signals.finished.connect(self._on_warp_done)
        self._worker.signals.error.connect(self._on_warp_error)
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
                "Extraction complete. Batch mode is active: click Batch Save All."
            )
            return
        self._statusbar.showMessage(
            "Extraction complete — enter a 5-char label and save."
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
        roi_label_saved = 0
        roi_errors = 0
        current_img = self._viewer.get_cv_image()
        current_pts = self._viewer.get_points()
        if current_img is not None and current_pts is not None:
            roi_raw_saved, roi_640_saved, roi_label_saved, roi_errors, _roi_base = self._save_roi_exports(
                current_img,
                current_pts,
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
            f"ROI label saved: {roi_label_saved}\n"
            f"Write errors: {write_errors + roi_errors}"
        )

    def _on_batch_save_segments(self):
        if self._file_list.count() == 0:
            QMessageBox.warning(self, "No Images", "Open a folder with images first.")
            return

        template_points = self._viewer.get_points()
        template_image = self._viewer.get_cv_image()
        if template_points is None or template_image is None:
            QMessageBox.warning(
                self,
                "Missing Template",
                "Batch mode needs a template. Please set 4 points on the current image first."
            )
            return

        if not self._output_dir:
            self._on_set_output()
            if not self._output_dir:
                return

        template_h, template_w = template_image.shape[:2]
        normalized_points = normalize_points(template_points, template_w, template_h)
        rotation_angle = int(self._rotation_slider.value()) % 360
        zoom_factor = self._viewer.get_zoom_factor()

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
        roi_label_saved_count = 0
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

            img_h, img_w = src_img.shape[:2]
            target_points = denormalize_points(normalized_points, img_w, img_h)

            try:
                binary_strip, preview_strip = extract_processed_strips(src_img, target_points)
            except Exception:
                error_count += 1
                continue

            binary_segments = split_strip_segments(binary_strip)
            preview_segments = split_strip_segments(preview_strip)

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

            roi_raw_saved, roi_640_saved, roi_label_saved, roi_write_errors, _roi_base = self._save_roi_exports(
                src_img,
                target_points,
                label
            )
            roi_raw_saved_count += roi_raw_saved
            roi_640_saved_count += roi_640_saved
            roi_label_saved_count += roi_label_saved
            error_count += roi_write_errors

            processed_images += 1
            processed_paths.append(image_path)

            self._statusbar.showMessage(
                f"Batch progress: {processed_images}/{total_images} image(s) saved."
            )
            QApplication.processEvents()

        if readjust_target_path:
            self._set_pending_readjust_template(rotation_angle, normalized_points, zoom_factor)
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
                f"ROI labels saved so far: {roi_label_saved_count}\n"
                f"Now focused on: {Path(readjust_target_path).name}\n\n"
                "Rotation, zoom, and points were inherited. "
                "Readjust the points, then click Batch Save All again to continue."
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
            f"ROI labels saved: {roi_label_saved_count}\n"
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
        rotated_image: np.ndarray,
        points: np.ndarray,
        label: str
    ) -> tuple[int, int, int, int, str]:
        crops = extract_roi_crops_with_context(rotated_image, points)
        if crops is None:
            return 0, 0, 0, 1, ""

        raw_crop, context_crop, bbox_in_context = crops
        roi_640, scale, pad_left, pad_top = letterbox_to_square_with_meta(context_crop, ROI_SIZE)

        bx, by, bw, bh = bbox_in_context
        bx_640 = (bx * scale) + pad_left
        by_640 = (by * scale) + pad_top
        bw_640 = bw * scale
        bh_640 = bh * scale
        yolo_line = build_yolo_bbox_line(
            bx_640,
            by_640,
            bw_640,
            bh_640,
            ROI_SIZE,
            ROI_SIZE,
        )

        raw_dir = Path(self._output_dir) / ROI_RAW_DIR_NAME
        size_dir = Path(self._output_dir) / ROI_640_DIR_NAME
        label_dir = Path(self._output_dir) / ROI_640_LABELS_DIR_NAME
        raw_dir.mkdir(parents=True, exist_ok=True)
        size_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        uid = uuid.uuid4().hex[:10]
        base_name = f"{label}_{uid}"
        raw_path = raw_dir / f"{base_name}_raw.png"
        size_path = size_dir / f"{base_name}_640.png"
        label_path = label_dir / f"{base_name}_640.txt"

        raw_ok = cv2.imwrite(str(raw_path), raw_crop)
        size_ok = cv2.imwrite(str(size_path), roi_640)
        label_ok = False
        if size_ok:
            try:
                label_path.write_text(yolo_line + "\n", encoding="utf-8")
                label_ok = True
            except OSError:
                label_ok = False

        errors = int(not raw_ok) + int(not size_ok) + int(not label_ok)
        return int(raw_ok), int(size_ok), int(label_ok), errors, base_name

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
