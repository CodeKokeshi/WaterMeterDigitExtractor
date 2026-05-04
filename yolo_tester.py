"""
YOLO Model Tester - DigitExtractor
==================================
Standalone viewer for testing YOLOv8 models trained with DigitExtractor.

Controls:
  Left / Right      previous / next image
  Space             run inference (Read)
  Scroll wheel      zoom in / out
  Click + drag      pan

Requirements:
  pip install ultralytics opencv-python PyQt6 numpy
  pip install pillow pillow-heif   # for HEIC/HEIF support
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import cv2
import numpy as np

import threading

from PyQt6.QtCore import QObject, QRectF, QRunnable, QSettings, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QPixmap,
    QShortcut,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from auto_read_pipeline import SlidingWindow, build_sliding_windows

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None  # type: ignore[assignment,misc]
    ULTRALYTICS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Persistent settings (QSettings keys — mirrors main.py pattern)
# ---------------------------------------------------------------------------
SETTINGS_ORG            = "DigitExtractor"
SETTINGS_APP            = "YoloTester"
SETTINGS_LAST_IMAGE_DIR = "paths/last_image_dir"
SETTINGS_LAST_MODEL_DIR = "paths/last_model_dir"

# ---------------------------------------------------------------------------
# Supported image extensions (includes HEIC/HEIF)
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".heif"
}

STRIP_CLASS_ID = 0
DIGIT_OFFSET = 1
UNREADABLE_CLS = 11
NUM_DIGITS = 5
MODEL_IMAGE_SIZE = 640

_COL_STRIP = (0, 200, 255)
_COL_DIGIT = (60, 220, 60)
_COL_UNREAD = (60, 60, 230)
_COL_TEXT_BG = (20, 20, 20)
_COL_WINNER = (255, 210, 0)

# ---------------------------------------------------------------------------
# HEIC / HEIF decoder (lazy-loaded, mirrors main.py)
# ---------------------------------------------------------------------------
HEIF_DECODER_AVAILABLE = False
_PIL_IMAGE_MODULE = None


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
    """Read an image from disk, falling back to pillow-heif for HEIC/HEIF files."""
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


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def _rotate_cv(img: np.ndarray, angle: int) -> np.ndarray:
    a = angle % 360
    if a == 0:
        return img
    if a == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if a == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if a == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), -a, 1.0)
    cos_v, sin_v = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(h * sin_v + w * cos_v)
    new_h = int(h * cos_v + w * sin_v)
    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(img, matrix, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT)


def _cv_to_pixmap(img: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    image = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def _box_center_x(box: tuple[int, int, int, int]) -> float:
    return (box[0] + box[2]) / 2.0


def _box_center_y(box: tuple[int, int, int, int]) -> float:
    return (box[1] + box[3]) / 2.0


def _box_width(box: tuple[int, int, int, int]) -> float:
    return max(float(box[2] - box[0]), 1.0)


def _box_height(box: tuple[int, int, int, int]) -> float:
    return max(float(box[3] - box[1]), 1.0)


def _box_area(box: tuple[int, int, int, int]) -> float:
    return _box_width(box) * _box_height(box)


def _intersection_area(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(x2 - x1) * float(y2 - y1)


def _digit_char(cls_id: int) -> str:
    if cls_id == UNREADABLE_CLS:
        return "X"
    return str(cls_id - DIGIT_OFFSET)


def _is_digit_class(cls_id: int) -> bool:
    return DIGIT_OFFSET <= cls_id <= DIGIT_OFFSET + 9 or cls_id == UNREADABLE_CLS


def _style_name(cls_id: int) -> str:
    if cls_id == STRIP_CLASS_ID:
        return "Strip"
    if cls_id == UNREADABLE_CLS:
        return "Unreadable"
    if _is_digit_class(cls_id):
        return f"Digit {_digit_char(cls_id)}"
    return f"Class {cls_id}"


def _draw_detections(
    img: np.ndarray,
    dets: list[dict],
    selected_strip: dict | None = None,
    selected_digits: list[dict] | None = None,
) -> np.ndarray:
    out = img.copy()
    selected_ids = {id(item) for item in (selected_digits or [])}
    selected_strip_id = id(selected_strip) if selected_strip is not None else None

    for det in dets:
        x1, y1, x2, y2 = det["box"]
        cls_id = det["cls"]
        conf = det["conf"]
        text = f"{det['label']} {conf:.0%}"

        if id(det) == selected_strip_id or id(det) in selected_ids:
            colour, thick = _COL_WINNER, 4
            text = f"* {text}"
        elif cls_id == STRIP_CLASS_ID:
            colour, thick = _COL_STRIP, 3
        elif cls_id == UNREADABLE_CLS:
            colour, thick = _COL_UNREAD, 2
        else:
            colour, thick = _COL_DIGIT, 2

        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thick)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_top = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, label_top), (x1 + tw + 4, y1), _COL_TEXT_BG, -1)
        cv2.putText(
            out,
            text,
            (x1 + 2, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            1,
            cv2.LINE_AA,
        )
    return out


def _map_window_box_to_image(
    xyxy: np.ndarray,
    window: SlidingWindow,
    image_size: int,
) -> tuple[int, int, int, int]:
    scale = window.size / float(image_size)
    mapped = [
        int(round(window.x + float(xyxy[0]) * scale)),
        int(round(window.y + float(xyxy[1]) * scale)),
        int(round(window.x + float(xyxy[2]) * scale)),
        int(round(window.y + float(xyxy[3]) * scale)),
    ]
    x1, y1, x2, y2 = mapped
    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)
    return x1, y1, x2, y2


def _score_digit_for_strip(strip_det: dict, digit_det: dict) -> float:
    strip_box = strip_det["box"]
    digit_box = digit_det["box"]
    strip_w = _box_width(strip_box)
    strip_h = _box_height(strip_box)
    margin_x = strip_w * 0.18
    margin_y = strip_h * 0.55
    sx1 = strip_box[0] - margin_x
    sy1 = strip_box[1] - margin_y
    sx2 = strip_box[2] + margin_x
    sy2 = strip_box[3] + margin_y
    cx = _box_center_x(digit_box)
    cy = _box_center_y(digit_box)
    if not (sx1 <= cx <= sx2 and sy1 <= cy <= sy2):
        return -1.0

    overlap = _intersection_area(strip_box, digit_box) / max(_box_area(digit_box), 1.0)
    vertical_distance = abs(cy - _box_center_y(strip_box)) / max(strip_h, 1.0)
    horizontal_inset = min(abs(cx - strip_box[0]), abs(cx - strip_box[2])) / max(strip_w, 1.0)
    return float(digit_det["conf"]) + (0.35 * overlap) - (0.30 * vertical_distance) - (0.05 * horizontal_inset)


def _dedupe_digits_by_position(digit_dets: list[dict]) -> list[dict]:
    if not digit_dets:
        return []
    ordered = sorted(digit_dets, key=lambda det: _box_center_x(det["box"]))
    groups: list[list[dict]] = [[ordered[0]]]
    for det in ordered[1:]:
        previous = groups[-1][-1]
        center_gap = abs(_box_center_x(det["box"]) - _box_center_x(previous["box"]))
        width_threshold = 0.55 * min(_box_width(det["box"]), _box_width(previous["box"]))
        overlap = _intersection_area(det["box"], previous["box"])
        min_area = min(_box_area(det["box"]), _box_area(previous["box"]))
        same_slot = center_gap <= max(width_threshold, 6.0)
        if min_area > 0.0 and overlap / min_area > 0.45:
            same_slot = True
        if same_slot:
            groups[-1].append(det)
        else:
            groups.append([det])
    collapsed = [max(group, key=lambda det: det["conf"]) for group in groups]
    return sorted(collapsed, key=lambda det: _box_center_x(det["box"]))


def _pick_best_digit_subset(digit_dets: list[dict], desired: int = NUM_DIGITS) -> list[dict]:
    deduped = _dedupe_digits_by_position(digit_dets)
    if len(deduped) <= desired:
        return deduped

    best_window = deduped[:desired]
    best_score = -1e9
    for start in range(0, len(deduped) - desired + 1):
        window = deduped[start:start + desired]
        conf_score = sum(float(det["conf"]) for det in window)
        span = _box_center_x(window[-1]["box"]) - _box_center_x(window[0]["box"])
        widths = sum(_box_width(det["box"]) for det in window) / max(len(window), 1)
        spread_bonus = span / max(widths * desired, 1.0)
        score = conf_score + (0.15 * spread_bonus)
        if score > best_score:
            best_score = score
            best_window = window
    return best_window


def _build_reading_text(digit_dets: list[dict], strip_det: dict | None = None) -> str:
    """Build a reading string, inserting '?' for any positionally missing digit slots."""
    if not digit_dets:
        return "?" * NUM_DIGITS

    # ── Strip-based slot assignment (most accurate) ────────────────────────
    if strip_det is not None:
        strip_box = strip_det["box"]
        strip_x1 = float(strip_box[0])
        strip_x2 = float(strip_box[2])
        strip_w   = max(strip_x2 - strip_x1, 1.0)
        slot_w    = strip_w / NUM_DIGITS

        # For each slot keep the highest-confidence digit
        slot_best: dict[int, dict] = {}
        for det in digit_dets:
            cx       = _box_center_x(det["box"])
            slot_idx = int((cx - strip_x1) / slot_w)
            slot_idx = max(0, min(NUM_DIGITS - 1, slot_idx))
            if slot_idx not in slot_best or det["conf"] > slot_best[slot_idx]["conf"]:
                slot_best[slot_idx] = det

        return "".join(
            _digit_char(slot_best[i]["cls"]) if i in slot_best else "?"
            for i in range(NUM_DIGITS)
        )

    # ── Gap-based slot assignment (fallback when no strip detected) ────────
    sorted_dets = sorted(digit_dets, key=lambda d: _box_center_x(d["box"]))

    if len(sorted_dets) >= NUM_DIGITS:
        # All (or more) digits found — no gaps to fill
        return "".join(_digit_char(d["cls"]) for d in sorted_dets[:NUM_DIGITS])

    avg_width = sum(_box_width(d["box"]) for d in sorted_dets) / len(sorted_dets)
    avg_width = max(avg_width, 1.0)

    chars: list[str] = [_digit_char(sorted_dets[0]["cls"])]
    for i in range(1, len(sorted_dets)):
        prev_cx = _box_center_x(sorted_dets[i - 1]["box"])
        curr_cx = _box_center_x(sorted_dets[i]["box"])
        gap     = curr_cx - prev_cx
        # Round to nearest number of digit-widths; subtract 1 for the current digit itself
        missing = max(0, round(gap / avg_width) - 1)
        # Cap so we never exceed the total expected digit count
        missing = min(missing, NUM_DIGITS - len(chars) - (len(sorted_dets) - i))
        chars.extend(["?"] * missing)
        chars.append(_digit_char(sorted_dets[i]["cls"]))

    # Pad any remaining missing digits at the end
    while len(chars) < NUM_DIGITS:
        chars.append("?")

    return "".join(chars[:NUM_DIGITS])


def _status_from_candidate(candidate: dict) -> str:
    if candidate["valid"]:
        return "Valid"
    if candidate["digit_count"] > 0 or candidate["strip_det"] is not None:
        return "Partial"
    return "No Detection"


def _candidate_rank(candidate: dict) -> float:
    digit_score = float(candidate["digit_count"]) * 3.0
    strip_score = 1.5 if candidate["strip_det"] is not None else 0.0
    conf_score = float(candidate["digit_conf_avg"]) + float(candidate["strip_conf"])
    return digit_score + strip_score + conf_score


def _evaluate_window_candidate(dets: list[dict], window_index: int, total_windows: int, window: SlidingWindow) -> dict:
    strip_dets = sorted(
        [det for det in dets if det["cls"] == STRIP_CLASS_ID],
        key=lambda det: det["conf"],
        reverse=True,
    )
    digit_dets = [det for det in dets if _is_digit_class(det["cls"])]

    best_strip: dict | None = None
    best_digits: list[dict] = []
    best_score = -1e9

    for strip_det in strip_dets:
        associated = [det for det in digit_dets if _score_digit_for_strip(strip_det, det) >= 0.0]
        chosen = _pick_best_digit_subset(associated, NUM_DIGITS)
        chosen_score = sum(_score_digit_for_strip(strip_det, det) for det in chosen) + float(strip_det["conf"]) * 1.8
        if len(chosen) == NUM_DIGITS:
            chosen_score += 2.5
        if chosen_score > best_score:
            best_score = chosen_score
            best_strip = strip_det
            best_digits = chosen

    if best_strip is None:
        best_digits = _pick_best_digit_subset(digit_dets, NUM_DIGITS)

    best_digits = sorted(best_digits, key=lambda det: _box_center_x(det["box"]))
    reading = _build_reading_text(best_digits, best_strip)
    strip_conf = float(best_strip["conf"]) if best_strip is not None else 0.0
    digit_count = len(best_digits)
    digit_conf_avg = (
        sum(float(det["conf"]) for det in best_digits) / digit_count if digit_count else 0.0
    )
    valid = best_strip is not None and len(strip_dets) == 1 and digit_count == NUM_DIGITS

    notes: list[str] = []
    if best_strip is None:
        notes.append("No strip box in this window.")
    elif len(strip_dets) > 1:
        notes.append(f"{len(strip_dets)} strip boxes detected; using strongest match.")
    else:
        notes.append("Single strip box matched.")
    if digit_count == 0:
        notes.append("No digit boxes matched the strip.")
    elif digit_count < NUM_DIGITS:
        notes.append(f"Only {digit_count}/{NUM_DIGITS} digit slots matched the strip.")
    elif digit_count > NUM_DIGITS:
        notes.append("Extra digit boxes collapsed to the strongest 5 slots.")
    else:
        notes.append("Recovered all 5 digit slots.")

    return {
        "window": window,
        "window_index": window_index,
        "total_windows": total_windows,
        "all_dets": dets,
        "strip_dets": strip_dets,
        "strip_det": best_strip,
        "digit_dets": best_digits,
        "reading": reading,
        "digit_count": digit_count,
        "digit_conf_avg": digit_conf_avg,
        "strip_conf": strip_conf,
        "valid": valid,
        "status": _status_from_candidate(
            {
                "valid": valid,
                "digit_count": digit_count,
                "strip_det": best_strip,
            }
        ),
        "notes": notes,
    }


def _parse_result_detections(result, window: SlidingWindow) -> list[dict]:
    dets: list[dict] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return dets

    xyxy_values = boxes.xyxy.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    names = result.names

    for xyxy, cls_id, conf in zip(xyxy_values, classes, confs):
        mapped_box = _map_window_box_to_image(xyxy, window, MODEL_IMAGE_SIZE)
        dets.append(
            {
                "box": mapped_box,
                "cls": int(cls_id),
                "conf": float(conf),
                "label": names.get(int(cls_id), _style_name(int(cls_id))),
            }
        )
    return dets


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

def _theme_stylesheet() -> str:
    return """
    QWidget {
        background: #14181d;
        color: #eef2f8;
        font-family: "Segoe UI";
        font-size: 11pt;
    }
    QMainWindow {
        background: #14181d;
    }
    QDialog {
        background: #0d1117;
    }
    QGroupBox {
        background: #1b2129;
        border: 1px solid #2d3744;
        border-radius: 16px;
        margin-top: 14px;
        padding: 14px 16px 16px 16px;
        font-weight: 600;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
        color: #9fb2c8;
    }
    QPushButton {
        background: #27313d;
        border: 1px solid #334151;
        border-radius: 10px;
        padding: 9px 14px;
        font-weight: 600;
    }
    QPushButton:hover {
        background: #2f3b49;
    }
    QPushButton:pressed {
        background: #1d2630;
    }
    QPushButton:disabled {
        background: #1a2027;
        color: #6a7787;
        border-color: #222b35;
    }
    QLabel#titleLabel {
        font-size: 18pt;
        font-weight: 700;
        color: #ffffff;
    }
    QLabel#subtitleLabel {
        color: #8ea1b7;
        font-size: 10pt;
    }
    QLabel#readingValue {
        font-family: "Consolas";
        font-size: 34pt;
        font-weight: 700;
        color: #ffffff;
        background: #10151b;
        border: 1px solid #2d3744;
        border-radius: 14px;
        padding: 14px 18px;
    }
    QLabel#statusBadge {
        font-weight: 700;
        border-radius: 12px;
        padding: 6px 12px;
        min-width: 90px;
    }
    QLabel#metaLabel {
        color: #8ea1b7;
        font-size: 9pt;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    QLabel#metaValue {
        color: #ffffff;
        font-size: 14pt;
        font-weight: 600;
    }
    QLabel#detailPanel {
        background: #10151b;
        border: 1px solid #28313d;
        border-radius: 12px;
        padding: 12px;
        color: #d7e0ea;
    }
    QGraphicsView {
        border: 1px solid #2d3744;
        border-radius: 16px;
        background: #0d1117;
    }
    QStatusBar {
        background: #0f1319;
        color: #aab8c7;
        border-top: 1px solid #212833;
    }
    QProgressBar {
        background: #1b2129;
        border: none;
        border-radius: 4px;
    }
    QProgressBar::chunk {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #2563eb, stop:1 #7c3aed);
        border-radius: 4px;
    }
    """


# ---------------------------------------------------------------------------
# Process Preview Dialog — timelapse of the reading pipeline
# ---------------------------------------------------------------------------

class ProcessPreviewDialog(QDialog):
    """
    Non-blocking modal shown during the READING phase.

    Displays a live "timelapse" of each pipeline step:
      Original image → window sampling → crop extraction →
      grayscale preprocessing → YOLO inference → detections → final result.

    Call push_frame() from the inference loop to update each step.
    Call finish() when done — auto-closes after a short delay.
    """

    # Step-name → accent colour (BGR for OpenCV overlays, hex for Qt labels)
    _STEP_COLOURS: dict[str, str] = {
        "original":    "#3b82f6",   # blue
        "sampling":    "#06b6d4",   # cyan
        "crop":        "#10b981",   # green
        "preprocess":  "#f59e0b",   # amber
        "inference":   "#8b5cf6",   # violet
        "detections":  "#ec4899",   # pink
        "result":      "#22c55e",   # bright green
        "complete":    "#22c55e",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reading in Progress…")
        self.setModal(False)
        self.resize(880, 640)
        self.setStyleSheet(_theme_stylesheet())

        self._close_timer = QTimer(self)
        self._close_timer.setSingleShot(True)
        self._close_timer.timeout.connect(self.accept)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(10)

        # ── Step title ──────────────────────────────────────────────────
        self._step_label = QLabel("Initialising…")
        self._step_label.setObjectName("titleLabel")
        self._step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._step_label)

        # ── Subtitle / description ───────────────────────────────────────
        self._sub_label = QLabel("")
        self._sub_label.setObjectName("subtitleLabel")
        self._sub_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._sub_label)

        # ── Live frame canvas ────────────────────────────────────────────
        self._canvas = QLabel()
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas.setMinimumSize(600, 380)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._canvas.setStyleSheet(
            "background: #060a0f; border: 1px solid #2d3744; border-radius: 14px;"
        )
        layout.addWidget(self._canvas, stretch=1)

        # ── Progress bar ────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(8)
        layout.addWidget(self._progress)

        # ── Window counter ───────────────────────────────────────────────
        self._window_label = QLabel("")
        self._window_label.setObjectName("metaLabel")
        self._window_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._window_label)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_frame(
        self,
        title: str,
        image_bgr: np.ndarray,
        subtitle: str = "",
        progress: float = 0.0,
        window_text: str = "",
        accent_key: str = "original",
    ) -> None:
        """
        Display a new frame with updated step title and progress.
        Calls processEvents() internally so the UI stays live.
        """
        accent = self._STEP_COLOURS.get(accent_key, "#3b82f6")
        self._step_label.setText(title)
        self._step_label.setStyleSheet(
            f"font-size: 16pt; font-weight: 700; color: {accent};"
        )
        self._sub_label.setText(subtitle)
        self._progress.setValue(int(min(max(progress, 0.0), 1.0) * 1000))
        self._window_label.setText(window_text)

        # Render image into the canvas, scaled to fit
        if image_bgr is not None and image_bgr.size > 0:
            pix = _cv_to_pixmap(image_bgr)
            avail = self._canvas.size()
            if avail.width() > 10 and avail.height() > 10:
                pix = pix.scaled(
                    avail,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            self._canvas.setPixmap(pix)

        QApplication.processEvents()

    def finish(self, delay_ms: int = 2000) -> None:
        """Mark pipeline as complete and schedule auto-close."""
        accent = self._STEP_COLOURS["complete"]
        self._step_label.setText("✓  Reading Complete")
        self._step_label.setStyleSheet(
            f"font-size: 16pt; font-weight: 700; color: {accent};"
        )
        self._progress.setValue(1000)
        self._window_label.setText("Closing in a moment…")
        QApplication.processEvents()
        self._close_timer.start(delay_ms)


# ---------------------------------------------------------------------------
# Standalone batch inference (module-level, safe to call from worker threads)
# ---------------------------------------------------------------------------

def _infer_image_batch(
    path: Path,
    model,
    rotation: int,
    model_lock: threading.Lock,
) -> dict:
    """
    Run the full sliding-window inference pipeline on a single image.
    Image loading, resizing, and pre/post-processing run freely in the
    calling thread.  model.predict() is serialised via model_lock so that
    only one thread drives the YOLO model at a time (safe for CPU & GPU).
    Returns a result dict compatible with BatchReportDialog.add_result().
    """
    try:
        image = read_image_any(str(path))
        if image is None:
            return {
                "filename": path.name, "reading": "-----", "status": "Error",
                "strip_conf": "--", "digit_count": "--", "window_label": "--",
            }

        rotated = _rotate_cv(image, rotation)
        windows = build_sliding_windows(rotated.shape[1], rotated.shape[0])
        total_windows = len(windows)

        candidates: list[dict] = []
        best_candidate: dict | None = None

        for idx, window in enumerate(windows, start=1):
            crop = rotated[window.y:window.y + window.size, window.x:window.x + window.size]
            if crop.size == 0:
                continue

            resized = cv2.resize(crop, (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            _gray  = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            _infer = cv2.cvtColor(_gray,   cv2.COLOR_GRAY2BGR)

            # Serialise only the GPU/CPU inference call
            with model_lock:
                results = model.predict(source=_infer, imgsz=MODEL_IMAGE_SIZE, verbose=False)

            dets = [] if not results else _parse_result_detections(results[0], window)
            candidate = _evaluate_window_candidate(dets, idx, total_windows, window)
            candidates.append(candidate)

            if best_candidate is None or _candidate_rank(candidate) > _candidate_rank(best_candidate):
                best_candidate = candidate

            if candidate["valid"]:   # Early exit on first clean reading
                break

        if best_candidate is None:
            return {
                "filename": path.name, "reading": "?" * NUM_DIGITS, "status": "No Detection",
                "strip_conf": "--", "digit_count": "0",
                "window_label": f"-- / {total_windows}",
            }

        strip_conf_str = (
            f"{best_candidate['strip_conf']:.0%}"
            if best_candidate["strip_det"] is not None else "--"
        )
        return {
            "filename":     path.name,
            "reading":      best_candidate["reading"],
            "status":       best_candidate["status"],
            "strip_conf":   strip_conf_str,
            "digit_count":  str(best_candidate["digit_count"]),
            "window_label": f"{best_candidate['window_index']} / {best_candidate['total_windows']}",
        }

    except Exception as exc:
        return {
            "filename": path.name, "reading": "-----", "status": "Error",
            "strip_conf": "--", "digit_count": f"err: {exc}", "window_label": "--",
        }


# ---------------------------------------------------------------------------
# Batch worker (QRunnable + signals for thread → main-thread communication)
# ---------------------------------------------------------------------------

class _BatchWorkerSignals(QObject):
    """Signals emitted by BatchImageWorker back to the main thread."""
    result = pyqtSignal(int, dict)   # (original_index, result_dict)


class BatchImageWorker(QRunnable):
    """Processes one image on a QThreadPool thread and emits a result signal."""

    def __init__(
        self,
        index: int,
        path: Path,
        model,
        rotation: int,
        model_lock: threading.Lock,
    ) -> None:
        super().__init__()
        self.index      = index
        self.path       = path
        self.model      = model
        self.rotation   = rotation
        self.model_lock = model_lock
        self.signals    = _BatchWorkerSignals()
        self.setAutoDelete(True)

    def run(self) -> None:
        result = _infer_image_batch(self.path, self.model, self.rotation, self.model_lock)
        self.signals.result.emit(self.index, result)


# ---------------------------------------------------------------------------
# Batch Report Dialog
# ---------------------------------------------------------------------------

class BatchReportDialog(QDialog):
    """
    Shown after a batch run over a folder of images.

    Phase 1 (progress):  live progress bar while images are processed.
    Phase 2 (report):    summary stats + per-image scrollable table + CSV export.
    """

    _STATUS_COLOURS: dict[str, tuple[str, str]] = {
        "Valid":         ("#1f4d38", "#e8fff1"),
        "Partial":       ("#5a3d12", "#fff1d6"),
        "No Detection":  ("#5a2020", "#ffe1e1"),
        "Error":         ("#5a2020", "#ffe1e1"),
    }

    def __init__(self, total_images: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Read")
        self.setModal(True)
        self.resize(1000, 720)
        self.setStyleSheet(_theme_stylesheet())

        self._total = total_images
        self._results: list[dict] = []       # filled via add_result()
        self._folder_path: str = ""

        self._build_ui()
        self._show_progress_phase()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(12)

        # ── Title ───────────────────────────────────────────────────────
        self._title_label = QLabel("Batch Read in Progress…")
        self._title_label.setObjectName("titleLabel")
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._title_label)

        # ── Subtitle ────────────────────────────────────────────────────
        self._sub_label = QLabel("")
        self._sub_label.setObjectName("subtitleLabel")
        self._sub_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._sub_label)

        # ── Progress bar (shown during phase 1) ─────────────────────────
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, max(self._total, 1))
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFixedHeight(20)
        layout.addWidget(self._progress_bar)

        # ── Summary stats row (shown during phase 2) ─────────────────────
        self._summary_widget = QWidget()
        summary_layout = QHBoxLayout(self._summary_widget)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(10)

        self._stat_labels: dict[str, QLabel] = {}
        for key in ("Total", "Valid", "Partial", "No Detection", "Error"):
            box = QGroupBox(key)
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(10, 10, 10, 10)
            lbl = QLabel("0")
            lbl.setObjectName("metaValue")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pct = QLabel("--")
            pct.setObjectName("subtitleLabel")
            pct.setAlignment(Qt.AlignmentFlag.AlignCenter)
            box_layout.addWidget(lbl)
            box_layout.addWidget(pct)
            summary_layout.addWidget(box)
            self._stat_labels[key] = lbl
            self._stat_labels[key + "_pct"] = pct

        layout.addWidget(self._summary_widget)

        # ── Per-image table ──────────────────────────────────────────────
        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels([
            "#", "Filename", "Reading", "Status",
            "Strip Conf", "Digits Found", "Source Window",
        ])
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.verticalHeader().setVisible(False)
        self._table.setStyleSheet(
            "QTableWidget { background: #0f1319; alternate-background-color: #141a22; "
            "gridline-color: #2a3440; border: 1px solid #2d3744; border-radius: 10px; }"
            "QHeaderView::section { background: #1b2129; color: #9fb2c8; "
            "padding: 6px; border: none; font-weight: 600; }"
        )
        layout.addWidget(self._table, stretch=1)

        # ── Bottom row: export + close ───────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        self._btn_export = QPushButton("Export CSV")
        self._btn_export.setEnabled(False)
        self._btn_close = QPushButton("Close")
        self._btn_close.clicked.connect(self.accept)
        self._btn_export.clicked.connect(self._export_csv)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_export)
        btn_row.addWidget(self._btn_close)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def _show_progress_phase(self) -> None:
        self._summary_widget.setVisible(False)
        self._table.setVisible(False)
        self._btn_export.setVisible(False)
        self._progress_bar.setVisible(True)

    def _show_report_phase(self) -> None:
        self._progress_bar.setVisible(False)
        self._summary_widget.setVisible(True)
        self._table.setVisible(True)
        self._btn_export.setVisible(True)

    # ------------------------------------------------------------------
    # Public API — called from the main window while processing
    # ------------------------------------------------------------------

    def set_folder(self, folder_path: str) -> None:
        self._folder_path = folder_path

    def update_progress(self, done: int, filename: str) -> None:
        """Called from the main thread each time a worker emits its result."""
        self._progress_bar.setValue(done)
        self._sub_label.setText(
            f"Completed {done} / {self._total}  —  last finished: {filename}"
        )
        QApplication.processEvents()

    def add_result(self, result: dict) -> None:
        """
        result keys expected:
          filename, reading, status, strip_conf, digit_count,
          window_index, total_windows, error (optional)
        """
        self._results.append(result)

    def finish(self) -> None:
        """Switch to the report phase and populate summary + table."""
        self._title_label.setText("Batch Read Complete")
        self._title_label.setStyleSheet(
            "font-size: 18pt; font-weight: 700; color: #22c55e;"
        )

        total = len(self._results)
        counts: dict[str, int] = {
            "Valid": 0, "Partial": 0, "No Detection": 0, "Error": 0,
        }
        for r in self._results:
            status = r.get("status", "Error")
            if status in counts:
                counts[status] += 1
            else:
                counts["Error"] += 1

        # ── Update summary stats ─────────────────────────────────────────
        self._stat_labels["Total"].setText(str(total))
        self._stat_labels["Total_pct"].setText("100%")
        for key in ("Valid", "Partial", "No Detection", "Error"):
            n = counts[key]
            pct = (n / total * 100) if total else 0.0
            self._stat_labels[key].setText(str(n))
            self._stat_labels[key + "_pct"].setText(f"{pct:.1f}%")

        valid_pct = (counts["Valid"] / total * 100) if total else 0.0
        self._sub_label.setText(
            f"{total} image(s) processed  —  "
            f"Success rate: {valid_pct:.1f}%  "
            f"({counts['Valid']} valid / {counts['Partial']} partial / "
            f"{counts['No Detection']} no-det / {counts['Error']} error)"
        )

        # ── Populate table ───────────────────────────────────────────────
        self._table.setRowCount(len(self._results))
        for row, r in enumerate(self._results):
            status = r.get("status", "Error")
            bg, fg = self._STATUS_COLOURS.get(status, ("#2a3440", "#dfe8f2"))

            cells = [
                str(row + 1),
                r.get("filename", ""),
                r.get("reading", "-----"),
                status,
                r.get("strip_conf", "--"),
                r.get("digit_count", "--"),
                r.get("window_label", "--"),
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(str(text))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col == 3:          # Status column — colour-coded
                    item.setBackground(QColor(bg))
                    item.setForeground(QColor(fg))
                if col == 1:          # Filename — left-aligned
                    item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(row, col, item)

        self._table.resizeColumnToContents(0)
        self._table.resizeColumnToContents(4)
        self._table.resizeColumnToContents(5)
        self._table.resizeColumnToContents(6)

        self._btn_export.setEnabled(bool(self._results))
        self._show_report_phase()
        QApplication.processEvents()

    # ------------------------------------------------------------------
    # CSV Export
    # ------------------------------------------------------------------

    def _export_csv(self) -> None:
        import csv as _csv

        default_name = "batch_report.csv"
        default_dir = self._folder_path if self._folder_path else str(Path.cwd())
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Batch Report", str(Path(default_dir) / default_name),
            "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = _csv.writer(f)
                writer.writerow([
                    "#", "Filename", "Reading", "Status",
                    "Strip Conf", "Digits Found", "Source Window",
                ])
                for i, r in enumerate(self._results, start=1):
                    writer.writerow([
                        i,
                        r.get("filename", ""),
                        r.get("reading", ""),
                        r.get("status", ""),
                        r.get("strip_conf", ""),
                        r.get("digit_count", ""),
                        r.get("window_label", ""),
                    ])
            self.parent().statusBar().showMessage(f"Report exported: {Path(path).name}") if self.parent() else None
        except Exception as exc:
            self._sub_label.setText(f"Export failed: {exc}")


# ---------------------------------------------------------------------------
# Image viewer widget
# ---------------------------------------------------------------------------

class _ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(16, 21, 27)))
        self._item: QGraphicsPixmapItem | None = None
        self._zoom = 1.0

    def show_pixmap(self, pix: QPixmap, reset: bool = True) -> None:
        self._scene.clear()
        self._item = self._scene.addPixmap(pix)
        self._scene.setSceneRect(QRectF(pix.rect().toRectF()))
        if reset:
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = 1.0

    def fit(self) -> None:
        if self._item:
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = 1.0

    def save_view(self) -> tuple:
        return (
            self.transform(),
            self._zoom,
            self.horizontalScrollBar().value(),
            self.verticalScrollBar().value(),
        )

    def restore_view(self, state: tuple) -> None:
        transform, zoom, h_value, v_value = state
        self.setTransform(transform)
        self._zoom = zoom
        self.horizontalScrollBar().setValue(h_value)
        self.verticalScrollBar().setValue(v_value)

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        self._zoom = max(0.05, min(self._zoom * factor, 50.0))


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class YoloTesterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Model Tester - DigitExtractor")
        self.resize(1320, 920)
        self.setStyleSheet(_theme_stylesheet())

        self._images: list[Path] = []
        self._idx = -1
        self._rotation = 0
        self._model = None
        self._raw_cv: np.ndarray | None = None
        self._annotated_cv: np.ndarray | None = None
        self._annotated_is_rotated = False

        # Persistent settings — remember last-used directories across restarts
        self._settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self._last_image_dir = self._read_existing_dir_setting(SETTINGS_LAST_IMAGE_DIR)
        self._last_model_dir = self._read_existing_dir_setting(SETTINGS_LAST_MODEL_DIR)

        self._build_ui()
        self._wire_signals()
        self._refresh_controls()
        self._reset_result_state()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 16, 16, 10)
        outer.setSpacing(14)

        hero = QGroupBox("Session")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setSpacing(10)

        title = QLabel("YOLO Meter Reader Tester")
        title.setObjectName("titleLabel")
        subtitle = QLabel(
            "Windowed 640×640 scanning for full-resolution images, with strip-local digit decoding."
        )
        subtitle.setObjectName("subtitleLabel")
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)

        top_controls = QHBoxLayout()
        top_controls.setSpacing(10)
        self._btn_folder = QPushButton("Open Folder")
        self._btn_prev = QPushButton("Prev")
        self._btn_next = QPushButton("Next")
        self._btn_fit = QPushButton("Fit View")
        self._lbl_name = QLabel("No folder loaded")
        self._lbl_name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        top_controls.addWidget(self._btn_folder)
        top_controls.addWidget(self._btn_prev)
        top_controls.addWidget(self._btn_next)
        top_controls.addWidget(self._btn_fit)
        top_controls.addWidget(self._lbl_name, stretch=1)
        hero_layout.addLayout(top_controls)

        config_row = QHBoxLayout()
        config_row.setSpacing(10)
        self._btn_model = QPushButton("Select Model (.pt)")
        self._lbl_model = QLabel("No model loaded")
        self._lbl_model.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        config_row.addWidget(self._btn_model)
        config_row.addWidget(self._lbl_model, stretch=1)
        config_row.addWidget(QLabel("Rotation"))
        self._btn_rotate_left = QPushButton("Rotate Left")
        self._btn_rotate_right = QPushButton("Rotate Right")
        self._btn_rotate_left.setFixedWidth(108)
        self._btn_rotate_right.setFixedWidth(108)
        config_row.addWidget(self._btn_rotate_left)
        config_row.addWidget(self._btn_rotate_right)
        hero_layout.addLayout(config_row)

        outer.addWidget(hero)

        content = QHBoxLayout()
        content.setSpacing(14)

        viewer_group = QGroupBox("Image Viewer")
        viewer_layout = QVBoxLayout(viewer_group)
        self._viewer = _ImageViewer()
        viewer_layout.addWidget(self._viewer)
        content.addWidget(viewer_group, stretch=3)

        results_group = QGroupBox("Read Result")
        results_layout = QVBoxLayout(results_group)
        results_layout.setSpacing(12)

        read_btn_row = QHBoxLayout()
        read_btn_row.setSpacing(8)
        self._btn_read = QPushButton("Run Read")
        self._btn_read.setFixedHeight(48)
        self._btn_read.setEnabled(False)
        self._btn_batch = QPushButton("Batch Read All")
        self._btn_batch.setFixedHeight(48)
        self._btn_batch.setEnabled(False)
        self._btn_batch.setToolTip("Run inference on every image in the loaded folder and generate a report.")
        read_btn_row.addWidget(self._btn_read)
        read_btn_row.addWidget(self._btn_batch)
        results_layout.addLayout(read_btn_row)

        self._lbl_status = QLabel("Idle")
        self._lbl_status.setObjectName("statusBadge")
        self._lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(self._lbl_status)

        self._lbl_reading = QLabel("-----")
        self._lbl_reading.setObjectName("readingValue")
        self._lbl_reading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(self._lbl_reading)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(14)
        metrics.setVerticalSpacing(10)
        self._lbl_strip_conf = QLabel("--")
        self._lbl_strip_conf.setObjectName("metaValue")
        self._lbl_digit_count = QLabel("--")
        self._lbl_digit_count.setObjectName("metaValue")
        self._lbl_window = QLabel("--")
        self._lbl_window.setObjectName("metaValue")
        self._lbl_avg_conf = QLabel("--")
        self._lbl_avg_conf.setObjectName("metaValue")

        for row, (label_text, value_label) in enumerate(
            [
                ("Strip confidence", self._lbl_strip_conf),
                ("Digits found", self._lbl_digit_count),
                ("Source window", self._lbl_window),
                ("Digit average", self._lbl_avg_conf),
            ]
        ):
            label = QLabel(label_text)
            label.setObjectName("metaLabel")
            metrics.addWidget(label, row, 0)
            metrics.addWidget(value_label, row, 1)
        results_layout.addLayout(metrics)

        detail_title = QLabel("Scan Summary")
        detail_title.setObjectName("metaLabel")
        results_layout.addWidget(detail_title)

        self._lbl_detail = QLabel("")
        self._lbl_detail.setObjectName("detailPanel")
        self._lbl_detail.setWordWrap(True)
        self._lbl_detail.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._lbl_detail.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        results_layout.addWidget(self._lbl_detail, stretch=1)

        help_text = QLabel(
            "Read behavior: class 0 = strip, classes 1..10 = digits 0..9, class 11 = unreadable."
        )
        help_text.setObjectName("subtitleLabel")
        help_text.setWordWrap(True)
        results_layout.addWidget(help_text)

        content.addWidget(results_group, stretch=2)
        outer.addLayout(content, stretch=1)

        self.setStatusBar(QStatusBar())

    def _wire_signals(self) -> None:
        self._btn_folder.clicked.connect(self._open_folder)
        self._btn_prev.clicked.connect(self._go_prev)
        self._btn_next.clicked.connect(self._go_next)
        self._btn_fit.clicked.connect(self._viewer.fit)
        self._btn_model.clicked.connect(self._select_model)
        self._btn_read.clicked.connect(self._run_read)
        self._btn_batch.clicked.connect(self._run_batch_read)
        self._btn_rotate_left.clicked.connect(self._rotate_left)
        self._btn_rotate_right.clicked.connect(self._rotate_right)

        QShortcut(QKeySequence(Qt.Key.Key_Left), self, self._go_prev)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self._go_next)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._run_read)

    def _reset_result_state(self) -> None:
        self._lbl_reading.setText("-----")
        self._lbl_detail.setText("Load an image and a trained YOLO model, then run a read.")
        self._set_status_badge("Idle")
        self._lbl_strip_conf.setText("--")
        self._lbl_digit_count.setText("--")
        self._lbl_window.setText("--")
        self._lbl_avg_conf.setText("--")

    def _set_status_badge(self, status: str) -> None:
        palette = {
            "Idle": ("#2a3440", "#dfe8f2"),
            "Scanning": ("#28435c", "#ebf6ff"),
            "Valid": ("#1f4d38", "#e8fff1"),
            "Partial": ("#5a3d12", "#fff1d6"),
            "No Detection": ("#5a2020", "#ffe1e1"),
            "Error": ("#5a2020", "#ffe1e1"),
        }
        background, foreground = palette.get(status, ("#2a3440", "#dfe8f2"))
        self._lbl_status.setText(status)
        self._lbl_status.setStyleSheet(
            f"background: {background}; color: {foreground}; font-weight: 700; border-radius: 12px; padding: 6px 12px;"
        )

    # ------------------------------------------------------------------
    # Settings helpers (mirrors main.py pattern)
    # ------------------------------------------------------------------

    def _read_existing_dir_setting(self, key: str) -> str:
        value = str(self._settings.value(key, "", type=str) or "")
        return value if value and Path(value).exists() else ""

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

    # ------------------------------------------------------------------

    def _open_folder(self) -> None:
        folder = self._pick_directory("Select Image Folder", self._last_image_dir)
        if not folder:
            return
        self._last_image_dir = folder
        self._settings.setValue(SETTINGS_LAST_IMAGE_DIR, folder)
        paths = sorted(
            (
                path
                for path in Path(folder).iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.name.lower(),
        )
        if not paths:
            self.statusBar().showMessage("No images found in that folder.")
            return
        self._images = paths
        self._idx = 0
        self._load_image(reset_view=True)
        self._refresh_controls()
        self.statusBar().showMessage(f"Loaded {len(paths)} image(s) from {folder}")

    def _load_image(self, reset_view: bool = False) -> None:
        if not self._images or self._idx < 0:
            return
        path = self._images[self._idx]
        # Use read_image_any so HEIC/HEIF files are handled via pillow-heif
        image = read_image_any(str(path))
        if image is None:
            self.statusBar().showMessage(f"Cannot read: {path.name}")
            return
        self._raw_cv = image
        self._annotated_cv = None
        self._annotated_is_rotated = False
        self._reset_result_state()
        self._display(reset_view=reset_view)
        total = len(self._images)
        self._lbl_name.setText(f"{path.name}  ({self._idx + 1} / {total})")

    def _display(self, reset_view: bool = False) -> None:
        if self._annotated_cv is not None:
            image = self._annotated_cv if self._annotated_is_rotated else _rotate_cv(self._annotated_cv, self._rotation)
        elif self._raw_cv is not None:
            image = _rotate_cv(self._raw_cv, self._rotation)
        else:
            return
        saved_view = None if reset_view else self._viewer.save_view()
        self._viewer.show_pixmap(_cv_to_pixmap(image), reset=reset_view)
        if saved_view is not None:
            self._viewer.restore_view(saved_view)

    def _go_prev(self) -> None:
        if self._idx > 0:
            saved = self._viewer.save_view()
            self._idx -= 1
            self._load_image(reset_view=False)
            self._viewer.restore_view(saved)
            self._refresh_controls()

    def _go_next(self) -> None:
        if self._idx < len(self._images) - 1:
            saved = self._viewer.save_view()
            self._idx += 1
            self._load_image(reset_view=False)
            self._viewer.restore_view(saved)
            self._refresh_controls()

    def _set_rotation(self, angle: int) -> None:
        self._rotation = angle
        self._annotated_cv = None
        self._annotated_is_rotated = False
        self._reset_result_state()
        self._display(reset_view=False)

    def _rotate_left(self) -> None:
        self._set_rotation((self._rotation - 90) % 360)

    def _rotate_right(self) -> None:
        self._set_rotation((self._rotation + 90) % 360)

    def _select_model(self) -> None:
        if not ULTRALYTICS_AVAILABLE:
            self.statusBar().showMessage("ultralytics not installed - run: pip install ultralytics")
            return
        start_dir = self._last_model_dir if self._last_model_dir else self._last_image_dir
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLOv8 Model",
            start_dir,
            "PyTorch Model (*.pt)",
        )
        if not path:
            return
        try:
            self._model = YOLO(path)
            self._lbl_model.setText(f"Model: {Path(path).name}")
            self.statusBar().showMessage(f"Model loaded: {Path(path).name}")
            # Remember the folder this model came from
            model_dir = str(Path(path).parent)
            self._last_model_dir = model_dir
            self._settings.setValue(SETTINGS_LAST_MODEL_DIR, model_dir)
        except Exception as exc:
            self._model = None
            self._lbl_model.setText("Failed to load model")
            self.statusBar().showMessage(f"Model load error: {exc}")
            self._set_status_badge("Error")
        self._refresh_controls()

    # ------------------------------------------------------------------
    # Inference pipeline (with optional live-preview callback)
    # ------------------------------------------------------------------

    def _infer_windows(
        self,
        image: np.ndarray,
        preview: ProcessPreviewDialog | None = None,
    ) -> tuple[dict | None, list[dict]]:
        windows = build_sliding_windows(image.shape[1], image.shape[0])
        if not windows:
            return None, []

        total_windows = len(windows)

        # ── Frame 0: show the full source image ──────────────────────────
        if preview:
            preview.push_frame(
                "Original Image",
                image,
                subtitle=f"Image size: {image.shape[1]}×{image.shape[0]} px  —  {total_windows} window(s) to scan",
                progress=0.0,
                window_text=f"0 / {total_windows} windows scanned",
                accent_key="original",
            )

        candidates: list[dict] = []
        best_candidate: dict | None = None

        for idx, window in enumerate(windows, start=1):
            base_progress = (idx - 1) / total_windows

            # ── Frame A: highlight the current sampling window ────────────
            if preview:
                overlay = image.copy()
                # Dim everything outside the window
                dim = (image * 0.25).astype(np.uint8)
                combined = dim.copy()
                wy1 = max(window.y, 0)
                wy2 = min(window.y + window.size, image.shape[0])
                wx1 = max(window.x, 0)
                wx2 = min(window.x + window.size, image.shape[1])
                combined[wy1:wy2, wx1:wx2] = image[wy1:wy2, wx1:wx2]
                # Bright border around the active window
                cv2.rectangle(
                    combined,
                    (window.x, window.y),
                    (window.x + window.size, window.y + window.size),
                    (0, 200, 255), 3,
                )
                # Corner label
                cv2.putText(
                    combined,
                    f"W{idx}/{total_windows}",
                    (window.x + 6, window.y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA,
                )
                preview.push_frame(
                    f"Sampling — Window {idx} / {total_windows}",
                    combined,
                    subtitle=f"Crop at  x={window.x},  y={window.y},  size={window.size}×{window.size}",
                    progress=base_progress + 0.1 / total_windows,
                    window_text=f"Window {idx} of {total_windows}",
                    accent_key="sampling",
                )

            crop = image[window.y:window.y + window.size, window.x:window.x + window.size]
            if crop.size == 0:
                continue

            # ── Frame B: raw crop extracted ───────────────────────────────
            if preview:
                preview.push_frame(
                    f"Crop Extracted — Window {idx} / {total_windows}",
                    crop,
                    subtitle=f"{window.size}×{window.size} px region isolated",
                    progress=base_progress + 0.25 / total_windows,
                    window_text=f"Window {idx} of {total_windows}",
                    accent_key="crop",
                )

            resized = cv2.resize(crop, (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), interpolation=cv2.INTER_AREA)

            # Convert to 3-channel greyscale for inference
            _gray  = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            _infer = cv2.cvtColor(_gray,   cv2.COLOR_GRAY2BGR)

            # ── Frame C: greyscale preprocessing ─────────────────────────
            if preview:
                preview.push_frame(
                    f"Preprocessing — Window {idx} / {total_windows}",
                    _infer,
                    subtitle=f"Resized to {MODEL_IMAGE_SIZE}×{MODEL_IMAGE_SIZE}  →  greyscale  →  3-channel",
                    progress=base_progress + 0.45 / total_windows,
                    window_text=f"Window {idx} of {total_windows}",
                    accent_key="preprocess",
                )

            # ── Frame D: inference running ────────────────────────────────
            if preview:
                preview.push_frame(
                    f"Running Inference — Window {idx} / {total_windows}",
                    _infer,
                    subtitle="YOLO model scanning for strip + digits…",
                    progress=base_progress + 0.55 / total_windows,
                    window_text=f"Window {idx} of {total_windows}",
                    accent_key="inference",
                )

            results = self._model.predict(  # type: ignore[union-attr]
                source=_infer,
                imgsz=MODEL_IMAGE_SIZE,
                verbose=False,
            )
            dets = [] if not results else _parse_result_detections(results[0], window)

            # ── Frame E: show detections mapped onto the crop ─────────────
            if preview:
                if dets:
                    det_vis = _infer.copy()
                    scale = MODEL_IMAGE_SIZE / float(window.size)
                    for det in dets:
                        dx1 = int((det["box"][0] - window.x) * scale)
                        dy1 = int((det["box"][1] - window.y) * scale)
                        dx2 = int((det["box"][2] - window.x) * scale)
                        dy2 = int((det["box"][3] - window.y) * scale)
                        cls_id = det["cls"]
                        if cls_id == STRIP_CLASS_ID:
                            col = _COL_STRIP
                        elif cls_id == UNREADABLE_CLS:
                            col = _COL_UNREAD
                        else:
                            col = _COL_DIGIT
                        cv2.rectangle(det_vis, (dx1, dy1), (dx2, dy2), col, 2)
                        cv2.putText(
                            det_vis, det["label"],
                            (dx1 + 2, max(12, dy1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA,
                        )
                    det_subtitle = f"{len(dets)} detection(s)  —  strip={'yes' if any(d['cls'] == STRIP_CLASS_ID for d in dets) else 'no'}"
                else:
                    det_vis = _infer.copy()
                    cv2.putText(
                        det_vis, "No detections", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2, cv2.LINE_AA,
                    )
                    det_subtitle = "No detections in this window"

                preview.push_frame(
                    f"Detections — Window {idx} / {total_windows}",
                    det_vis,
                    subtitle=det_subtitle,
                    progress=base_progress + 0.85 / total_windows,
                    window_text=f"Window {idx} of {total_windows}",
                    accent_key="detections",
                )

            candidate = _evaluate_window_candidate(dets, idx, total_windows, window)
            candidates.append(candidate)

            if best_candidate is None or _candidate_rank(candidate) > _candidate_rank(best_candidate):
                best_candidate = candidate

            # Valid hit — show annotated full image and exit early
            if candidate["valid"]:
                if preview:
                    final_vis = _draw_detections(
                        image,
                        candidate["all_dets"],
                        selected_strip=candidate["strip_det"],
                        selected_digits=candidate["digit_dets"],
                    )
                    preview.push_frame(
                        "✓  Valid Reading Found!",
                        final_vis,
                        subtitle=f"Reading: {candidate['reading']}  |  Strip: {candidate['strip_conf']:.0%}  |  Digits: {candidate['digit_count']}/{NUM_DIGITS}",
                        progress=1.0,
                        window_text=f"Window {idx} of {total_windows}  —  early exit",
                        accent_key="result",
                    )
                return candidate, candidates

        # ── All windows scanned — show best candidate ────────────────────
        if preview and best_candidate is not None:
            final_vis = _draw_detections(
                image,
                best_candidate["all_dets"],
                selected_strip=best_candidate["strip_det"],
                selected_digits=best_candidate["digit_dets"],
            )
            preview.push_frame(
                "Best Candidate Selected",
                final_vis,
                subtitle=f"Reading: {best_candidate['reading']}  |  Status: {best_candidate['status']}",
                progress=1.0,
                window_text=f"All {total_windows} window(s) scanned",
                accent_key="result",
            )

        return best_candidate, candidates

    # ------------------------------------------------------------------
    # Result UI update
    # ------------------------------------------------------------------

    def _update_result_ui(self, candidate: dict | None, scanned_windows: int, total_windows: int) -> None:
        if candidate is None:
            self._annotated_cv = None
            self._reset_result_state()
            self._set_status_badge("No Detection")
            self._lbl_detail.setText(
                f"Scanned {scanned_windows}/{total_windows} windows and found no usable strip or digit detections."
            )
            self.statusBar().showMessage("No digit strip or digit boxes detected after window scan.")
            return

        source_image = _rotate_cv(self._raw_cv, self._rotation) if self._raw_cv is not None else None
        if source_image is not None:
            self._annotated_cv = _draw_detections(
                source_image,
                candidate["all_dets"],
                selected_strip=candidate["strip_det"],
                selected_digits=candidate["digit_dets"],
            )
            self._annotated_is_rotated = True
            self._display(reset_view=False)

        self._lbl_reading.setText(candidate["reading"])
        self._set_status_badge(candidate["status"])
        self._lbl_strip_conf.setText(f"{candidate['strip_conf']:.0%}" if candidate["strip_det"] else "--")
        self._lbl_digit_count.setText(f"{candidate['digit_count']} / {NUM_DIGITS}")
        self._lbl_window.setText(f"{candidate['window_index']} / {candidate['total_windows']}")
        self._lbl_avg_conf.setText(
            f"{candidate['digit_conf_avg']:.0%}" if candidate["digit_count"] else "--"
        )

        detail_lines = [
            f"Scanned windows: {scanned_windows} of {total_windows}",
            f"Chosen crop: x={candidate['window'].x}, y={candidate['window'].y}, size={candidate['window'].size}",
            f"Decoded reading: {candidate['reading']}",
        ]
        detail_lines.extend(candidate["notes"])
        self._lbl_detail.setText("\n".join(detail_lines))

        self.statusBar().showMessage(
            f"{candidate['status']}: {candidate['reading']} | digits {candidate['digit_count']}/{NUM_DIGITS} | window {candidate['window_index']}/{candidate['total_windows']}"
        )

    # ------------------------------------------------------------------
    # Run Read — opens process preview, runs inference, closes on finish
    # ------------------------------------------------------------------

    def _run_read(self) -> None:
        if self._model is None or self._raw_cv is None:
            return

        visible_image = _rotate_cv(self._raw_cv, self._rotation)
        self._set_status_badge("Scanning")
        self._lbl_detail.setText("Running center-first 640×640 window scan…")
        self.statusBar().showMessage("Running center-first 640×640 window scan…")
        QApplication.processEvents()

        # Open the timelapse process-preview dialog
        preview = ProcessPreviewDialog(self)
        preview.show()
        QApplication.processEvents()

        try:
            best_candidate, candidates = self._infer_windows(visible_image, preview=preview)
        except Exception as exc:
            preview.close()
            self._set_status_badge("Error")
            self._lbl_detail.setText(f"Inference error: {exc}")
            self.statusBar().showMessage(f"Inference error: {exc}")
            return

        # Trigger the auto-close countdown on the preview
        preview.finish(delay_ms=2000)

        total_windows = len(build_sliding_windows(visible_image.shape[1], visible_image.shape[0]))
        scanned_windows = len(candidates)
        self._update_result_ui(best_candidate, scanned_windows, total_windows)

    # ------------------------------------------------------------------
    # Batch Read — dispatch all images to QThreadPool, collect via signals
    # ------------------------------------------------------------------

    def _run_batch_read(self) -> None:
        if self._model is None or not self._images:
            return

        total        = len(self._images)
        folder_str   = str(self._images[0].parent)
        model_lock   = threading.Lock()   # serialises model.predict() calls

        report_dialog = BatchReportDialog(total, self)
        report_dialog.set_folder(folder_str)
        report_dialog.show()
        QApplication.processEvents()

        self._btn_batch.setEnabled(False)
        self._btn_read.setEnabled(False)

        # Preallocate slots so results are stored in original folder order
        results_store: list[dict | None] = [None] * total
        completed      = [0]   # mutable counter (closure-friendly)

        def _on_result(index: int, result: dict) -> None:
            """Slot — always called on the main thread via Qt's queued connection."""
            results_store[index] = result
            completed[0] += 1
            report_dialog.update_progress(completed[0], result.get("filename", ""))

            if completed[0] == total:
                # All workers done — add results in original order then show report
                for r in results_store:
                    if r is not None:
                        report_dialog.add_result(r)
                report_dialog.finish()
                self._btn_batch.setEnabled(True)
                self._btn_read.setEnabled(True)
                self.statusBar().showMessage(
                    f"Batch read complete — {total} image(s) processed."
                )

        # Use QThreadPool with a sensible thread cap:
        #   • enough threads to overlap I/O + preprocessing
        #   • not so many that we thrash CPU or VRAM
        pool = QThreadPool.globalInstance()
        thread_cap = max(2, min(pool.maxThreadCount(), 6))
        pool.setMaxThreadCount(thread_cap)

        for idx, path in enumerate(self._images):
            worker = BatchImageWorker(idx, path, self._model, self._rotation, model_lock)
            # Qt queued connection ensures _on_result runs on the main thread
            worker.signals.result.connect(_on_result, Qt.ConnectionType.QueuedConnection)
            pool.start(worker)

    def _refresh_controls(self) -> None:
        has_images = bool(self._images)
        has_model = self._model is not None
        self._btn_prev.setEnabled(has_images and self._idx > 0)
        self._btn_next.setEnabled(has_images and self._idx < len(self._images) - 1)
        self._btn_read.setEnabled(has_images and has_model)
        self._btn_batch.setEnabled(has_images and has_model)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not ULTRALYTICS_AVAILABLE:
        print("[WARNING] ultralytics not installed.")
        print("          Model testing will not work.")
        print("          Install with: pip install ultralytics")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = YoloTesterWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
