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
  pip install tensorflow   # optional, required for .tflite model support
  pip install pillow pillow-heif   # for HEIC/HEIF support
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
from typing import Any
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
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStatusBar,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from auto_read_pipeline import SlidingWindow, build_sliding_windows
from yolo_review_backend import (
    CLASS_NAMES,
    ReviewStore,
    build_reading_from_detections,
    char_to_class_id,
    class_id_to_char,
    normalize_box,
    sanitize_brand,
)

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
DEFAULT_CONF_THRESHOLD = 0.25
TFLITE_IOU_THRESHOLD = 0.45

_COL_STRIP = (0, 200, 255)
_COL_DIGIT = (60, 220, 60)
_COL_UNREAD = (60, 60, 230)
_COL_TEXT_BG = (20, 20, 20)
_COL_WINNER = (255, 210, 0)

_TENSORFLOW_MODULE = None


@dataclass
class LoadedModel:
    name: str
    path: str
    backend: str
    runtime: Any


def _load_tensorflow():
    global _TENSORFLOW_MODULE
    if _TENSORFLOW_MODULE is None:
        # Keras 3.14+ eagerly imports matplotlib at the top level.
        # On Windows / Python 3.13 this triggers a GUI-backend scan that hangs.
        # Force the headless Agg backend before TF is imported so the scan is skipped.
        import os
        os.environ.setdefault("MPLBACKEND", "Agg")
        _TENSORFLOW_MODULE = importlib.import_module("tensorflow")
    return _TENSORFLOW_MODULE

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


def _clone_detection(det: dict | None) -> dict | None:
    if det is None:
        return None
    cloned = dict(det)
    cloned["box"] = list(det["box"])
    return cloned


def _clone_detection_list(detections: list[dict] | None) -> list[dict]:
    return [_clone_detection(det) for det in (detections or []) if det is not None]


def _selected_review_detections(candidate: dict | None) -> list[dict]:
    if candidate is None:
        return []
    detections: list[dict] = []
    if candidate.get("strip_det") is not None:
        detections.append(_clone_detection(candidate["strip_det"]))
    detections.extend(_clone_detection_list(candidate.get("digit_dets", [])))
    return [det for det in detections if det is not None]


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


def _normalize_tflite_output(output: np.ndarray) -> np.ndarray:
    data = np.asarray(output)
    data = np.squeeze(data)
    if data.ndim == 1:
        data = np.expand_dims(data, 0)
    if data.ndim != 2:
        raise ValueError(f"Unexpected TFLite output shape: {tuple(np.asarray(output).shape)}")

    if data.shape[0] >= 6 and data.shape[0] < data.shape[1]:
        data = data.T
    if data.shape[1] < 6:
        raise ValueError(f"TFLite output does not look like YOLO detections: {tuple(data.shape)}")
    return data.astype(np.float32, copy=False)


def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    cx, cy, w, h = [float(v) for v in xywh[:4]]
    half_w = w / 2.0
    half_h = h / 2.0
    return np.array([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dtype=np.float32)


def _dequantize_array(values: np.ndarray, quantization: tuple[float, int] | list[Any] | None) -> np.ndarray:
    if not quantization:
        return values.astype(np.float32, copy=False)
    scale = float(quantization[0]) if len(quantization) >= 1 else 0.0
    zero_point = float(quantization[1]) if len(quantization) >= 2 else 0.0
    if scale == 0.0:
        return values.astype(np.float32, copy=False)
    return (values.astype(np.float32) - zero_point) * scale


def _quantize_input(image: np.ndarray, input_detail: dict) -> np.ndarray:
    input_dtype = np.dtype(input_detail["dtype"])
    array = np.asarray(image, dtype=np.float32)
    if np.issubdtype(input_dtype, np.floating):
        return array.astype(input_dtype)

    quantization = input_detail.get("quantization", (0.0, 0))
    scale = float(quantization[0]) if quantization else 0.0
    zero_point = float(quantization[1]) if quantization else 0.0
    if scale == 0.0:
        return array.astype(input_dtype)

    quantized = np.round(array / scale + zero_point)
    limits = np.iinfo(input_dtype)
    quantized = np.clip(quantized, limits.min, limits.max)
    return quantized.astype(input_dtype)


def _nms_xyxy(detections: list[dict], iou_threshold: float = TFLITE_IOU_THRESHOLD) -> list[dict]:
    if not detections:
        return []

    boxes = []
    scores = []
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        boxes.append([float(x1), float(y1), float(max(1, x2 - x1)), float(max(1, y2 - y1))])
        scores.append(float(det["conf"]))

    indices = cv2.dnn.NMSBoxes(boxes, scores, DEFAULT_CONF_THRESHOLD, iou_threshold)
    if indices is None or len(indices) == 0:
        return []

    kept: list[dict] = []
    for idx in np.array(indices).reshape(-1):
        kept.append(detections[int(idx)])
    kept.sort(key=lambda det: float(det["conf"]), reverse=True)
    return kept


def _parse_tflite_output(output: np.ndarray, window: SlidingWindow, conf_threshold: float = DEFAULT_CONF_THRESHOLD) -> list[dict]:
    preds = _normalize_tflite_output(output)
    detections: list[dict] = []

    for row in preds:
        xyxy = _xywh_to_xyxy(row[:4])
        class_scores = row[4:]
        if class_scores.size == 0:
            continue
        cls_id = int(np.argmax(class_scores))
        conf = float(class_scores[cls_id])
        if conf < conf_threshold:
            continue
        mapped_box = _map_window_box_to_image(xyxy, window, MODEL_IMAGE_SIZE)
        detections.append(
            {
                "box": mapped_box,
                "cls": cls_id,
                "conf": conf,
                "label": _style_name(cls_id),
            }
        )

    return _nms_xyxy(detections)


def _create_loaded_model(path: str) -> LoadedModel:
    suffix = Path(path).suffix.lower()
    name = Path(path).name
    if suffix == ".pt":
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("Ultralytics is not installed. Run: pip install ultralytics")
        return LoadedModel(name=name, path=path, backend="pt", runtime=YOLO(path))
    if suffix == ".tflite":
        try:
            tf = _load_tensorflow()
        except Exception as exc:
            raise RuntimeError(
                "TensorFlow is required for .tflite models. Use the project backend environment with TensorFlow installed."
            ) from exc
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        return LoadedModel(name=name, path=path, backend="tflite", runtime=interpreter)
    raise RuntimeError("Unsupported model format. Select a .pt or .tflite model.")


def _infer_window_detections(model_entry: LoadedModel, infer_image: np.ndarray, window: SlidingWindow) -> list[dict]:
    if model_entry.backend == "pt":
        results = model_entry.runtime.predict(source=infer_image, imgsz=MODEL_IMAGE_SIZE, verbose=False)
        return [] if not results else _parse_result_detections(results[0], window)

    interpreter = model_entry.runtime
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    batch = np.expand_dims(infer_image, axis=0)
    interpreter.set_tensor(input_detail["index"], _quantize_input(batch, input_detail))
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"])
    dequantized = _dequantize_array(output, output_detail.get("quantization"))
    return _parse_tflite_output(dequantized, window)


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
    model_entry: LoadedModel,
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

            # Serialise only the model inference call
            with model_lock:
                dets = _infer_window_detections(model_entry, _infer, window)
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
            "model_name":   model_entry.name,
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
        model_entry: LoadedModel,
        rotation: int,
        model_lock: threading.Lock,
    ) -> None:
        super().__init__()
        self.index      = index
        self.path       = path
        self.model_entry = model_entry
        self.rotation   = rotation
        self.model_lock = model_lock
        self.signals    = _BatchWorkerSignals()
        self.setAutoDelete(True)

    def run(self) -> None:
        result = _infer_image_batch(self.path, self.model_entry, self.rotation, self.model_lock)
        self.signals.result.emit(self.index, result)


# ---------------------------------------------------------------------------
# Batch Report Dialog
# ---------------------------------------------------------------------------

class BatchReportDialog(QDialog):
    open_image_requested = pyqtSignal(str)
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
        self._review_lookup: dict[str, dict] = {}

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
        filter_row = QHBoxLayout()
        filter_row.setSpacing(10)
        filter_row.addWidget(QLabel("Status Filter"))
        self._status_filter = QComboBox()
        self._status_filter.addItems(["All", "Valid", "Partial", "No Detection", "Error"])
        filter_row.addWidget(self._status_filter)
        filter_row.addWidget(QLabel("Review Filter"))
        self._review_filter = QComboBox()
        self._review_filter.addItems(["All", "Unreviewed", "marked_correct", "reading_only", "detection_fixed", "unreadable_or_skip"])
        filter_row.addWidget(self._review_filter)
        filter_row.addStretch(1)
        layout.addLayout(filter_row)

        # ── Per-image table ──────────────────────────────────────────────
        self._table = QTableWidget()
        self._table.setColumnCount(8)
        self._table.setHorizontalHeaderLabels([
            "#", "Filename", "Reading", "Status",
            "Strip Conf", "Digits Found", "Source Window", "Review State",
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
        self._status_filter.currentTextChanged.connect(self._apply_filters)
        self._review_filter.currentTextChanged.connect(self._apply_filters)
        self._table.cellDoubleClicked.connect(self._handle_row_open)

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

    def set_review_lookup(self, review_lookup: dict[str, dict]) -> None:
        self._review_lookup = dict(review_lookup)

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
            image_path = str(Path(self._folder_path) / str(r.get("filename", "")))
            review = self._review_lookup.get(image_path)
            review_state = str(review.get("review_type", "Unreviewed")) if review else "Unreviewed"
            r["review_state"] = review_state

            cells = [
                str(row + 1),
                r.get("filename", ""),
                r.get("reading", "-----"),
                status,
                r.get("strip_conf", "--"),
                r.get("digit_count", "--"),
                r.get("window_label", "--"),
                review_state,
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
        self._table.resizeColumnToContents(7)

        self._btn_export.setEnabled(bool(self._results))
        self._show_report_phase()
        self._apply_filters()
        QApplication.processEvents()

    def _apply_filters(self) -> None:
        status_filter = self._status_filter.currentText()
        review_filter = self._review_filter.currentText()
        for row, result in enumerate(self._results):
            status_ok = status_filter == "All" or str(result.get("status", "")) == status_filter
            review_state = str(result.get("review_state", "Unreviewed"))
            review_ok = review_filter == "All" or review_state == review_filter
            self._table.setRowHidden(row, not (status_ok and review_ok))

    def _handle_row_open(self, row: int, _column: int) -> None:
        if row < 0 or row >= len(self._results):
            return
        filename = str(self._results[row].get("filename", ""))
        if filename:
            self.open_image_requested.emit(filename)

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
                    "Strip Conf", "Digits Found", "Source Window", "Review State",
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
                        r.get("review_state", "Unreviewed"),
                    ])
            self.parent().statusBar().showMessage(f"Report exported: {Path(path).name}") if self.parent() else None
        except Exception as exc:
            self._sub_label.setText(f"Export failed: {exc}")


# ---------------------------------------------------------------------------
# Review / correction dialogs
# ---------------------------------------------------------------------------

class ExportReviewsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Reviewed Cases")
        self.setModal(True)
        self.resize(420, 170)
        self.setStyleSheet(_theme_stylesheet())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        layout.addWidget(QLabel("Brand filter"))
        self._brand_combo = QComboBox()
        self._brand_combo.setEditable(True)
        self._brand_combo.addItems(["", "AsiaM", "Maxwinner", "Unknown"])
        self._brand_combo.setCurrentText("")
        layout.addWidget(self._brand_combo)

        self._include_correct = QCheckBox("Include 'marked correct' reviews")
        self._include_correct.setChecked(False)
        layout.addWidget(self._include_correct)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> tuple[str, bool]:
        return self._brand_combo.currentText().strip(), self._include_correct.isChecked()


class DetectionAnnotationCanvas(QWidget):
    selection_changed = pyqtSignal(int)
    boxes_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(760, 460)
        self._image: np.ndarray | None = None
        self._boxes: list[dict] = []
        self._selected_index = -1
        self._mode = "select"
        self._draft_start: tuple[int, int] | None = None
        self._draft_end: tuple[int, int] | None = None
        self._pending_class_id = DIGIT_OFFSET
        self._moving = False
        self._move_offset = (0, 0)

    def set_image(self, image_bgr: np.ndarray) -> None:
        self._image = image_bgr.copy()
        self.update()

    def set_boxes(self, boxes: list[dict]) -> None:
        self._boxes = _clone_detection_list(boxes)
        self._selected_index = -1
        self.selection_changed.emit(-1)
        self.boxes_changed.emit()
        self.update()

    def boxes(self) -> list[dict]:
        return _clone_detection_list(self._boxes)

    def selected_index(self) -> int:
        return self._selected_index

    def selected_box(self) -> dict | None:
        if 0 <= self._selected_index < len(self._boxes):
            return self._boxes[self._selected_index]
        return None

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self._draft_start = None
        self._draft_end = None
        self._moving = False
        self.update()

    def set_pending_class_id(self, class_id: int) -> None:
        self._pending_class_id = class_id

    def set_selected_class_id(self, class_id: int) -> None:
        if 0 <= self._selected_index < len(self._boxes):
            self._boxes[self._selected_index]["cls"] = class_id
            self.boxes_changed.emit()
            self.update()

    def delete_selected(self) -> None:
        if 0 <= self._selected_index < len(self._boxes):
            del self._boxes[self._selected_index]
            self._selected_index = -1
            self.selection_changed.emit(-1)
            self.boxes_changed.emit()
            self.update()

    def _image_rect(self) -> tuple[QRectF, float, float]:
        if self._image is None:
            return QRectF(), 1.0, 1.0
        h, w = self._image.shape[:2]
        available = self.rect().adjusted(10, 10, -10, -10)
        if available.width() <= 0 or available.height() <= 0:
            return QRectF(), 1.0, 1.0
        scale = min(available.width() / w, available.height() / h)
        draw_w = w * scale
        draw_h = h * scale
        left = available.left() + (available.width() - draw_w) / 2.0
        top = available.top() + (available.height() - draw_h) / 2.0
        return QRectF(left, top, draw_w, draw_h), scale, scale

    def _widget_to_image(self, pos) -> tuple[int, int] | None:
        if self._image is None:
            return None
        image_rect, scale_x, scale_y = self._image_rect()
        if not image_rect.contains(pos):
            return None
        x = int((pos.x() - image_rect.left()) / scale_x)
        y = int((pos.y() - image_rect.top()) / scale_y)
        return normalize_box([x, y, x + 1, y + 1], self._image.shape[1], self._image.shape[0])[:2]

    def _find_box_at(self, x: int, y: int) -> int:
        for idx in range(len(self._boxes) - 1, -1, -1):
            x1, y1, x2, y2 = self._boxes[idx]["box"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return idx
        return -1

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._image is None or event.button() != Qt.MouseButton.LeftButton:
            return
        image_pos = self._widget_to_image(event.position())
        if image_pos is None:
            return
        x, y = image_pos
        hit_index = self._find_box_at(x, y)

        if self._mode == "select":
            self._selected_index = hit_index
            self.selection_changed.emit(hit_index)
            if hit_index >= 0:
                box = self._boxes[hit_index]["box"]
                self._moving = True
                self._move_offset = (x - box[0], y - box[1])
            self.update()
            return

        self._selected_index = hit_index if self._mode == "replace" else self._selected_index
        self.selection_changed.emit(self._selected_index)
        self._draft_start = (x, y)
        self._draft_end = (x, y)
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._image is None:
            return
        image_pos = self._widget_to_image(event.position())
        if image_pos is None:
            return
        x, y = image_pos
        if self._moving and 0 <= self._selected_index < len(self._boxes):
            box = self._boxes[self._selected_index]["box"]
            width = box[2] - box[0]
            height = box[3] - box[1]
            new_x1 = x - self._move_offset[0]
            new_y1 = y - self._move_offset[1]
            self._boxes[self._selected_index]["box"] = normalize_box(
                [new_x1, new_y1, new_x1 + width, new_y1 + height],
                self._image.shape[1],
                self._image.shape[0],
            )
            self.boxes_changed.emit()
            self.update()
            return
        if self._draft_start is not None:
            self._draft_end = (x, y)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._image is None or event.button() != Qt.MouseButton.LeftButton:
            return
        if self._moving:
            self._moving = False
            return
        if self._draft_start is None or self._draft_end is None:
            return

        x1 = min(self._draft_start[0], self._draft_end[0])
        y1 = min(self._draft_start[1], self._draft_end[1])
        x2 = max(self._draft_start[0], self._draft_end[0])
        y2 = max(self._draft_start[1], self._draft_end[1])
        box = normalize_box([x1, y1, x2, y2], self._image.shape[1], self._image.shape[0])
        if box[2] - box[0] < 5 or box[3] - box[1] < 5:
            self._draft_start = None
            self._draft_end = None
            self.update()
            return

        class_id = STRIP_CLASS_ID if self._mode == "add_strip" else self._pending_class_id
        det = {
            "box": box,
            "cls": class_id,
            "conf": 1.0,
            "label": CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else f"class_{class_id}",
        }
        if self._mode == "replace" and 0 <= self._selected_index < len(self._boxes):
            self._boxes[self._selected_index] = det
        else:
            self._boxes.append(det)
            self._selected_index = len(self._boxes) - 1
            self.selection_changed.emit(self._selected_index)
        self._draft_start = None
        self._draft_end = None
        self.boxes_changed.emit()
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0d1117"))
        if self._image is None:
            painter.setPen(QColor("#9fb2c8"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image")
            return

        image_rect, scale_x, scale_y = self._image_rect()
        pix = _cv_to_pixmap(self._image)
        painter.drawPixmap(image_rect.toRect(), pix)

        def _box_rect(box: list[int]) -> QRectF:
            return QRectF(
                image_rect.left() + (box[0] * scale_x),
                image_rect.top() + (box[1] * scale_y),
                max((box[2] - box[0]) * scale_x, 1.0),
                max((box[3] - box[1]) * scale_y, 1.0),
            )

        for idx, det in enumerate(self._boxes):
            cls_id = int(det["cls"])
            if cls_id == STRIP_CLASS_ID:
                colour = QColor(0, 200, 255)
            elif cls_id == UNREADABLE_CLS:
                colour = QColor(60, 60, 230)
            else:
                colour = QColor(60, 220, 60)
            pen = QPen(colour, 3 if idx == self._selected_index else 2)
            painter.setPen(pen)
            rect = _box_rect(det["box"])
            painter.drawRect(rect)
            painter.fillRect(QRectF(rect.left(), max(rect.top() - 18, image_rect.top()), 120, 18), QColor(20, 20, 20, 220))
            painter.setPen(QColor("#f3f7fb"))
            painter.drawText(
                QRectF(rect.left() + 3, max(rect.top() - 18, image_rect.top()), 116, 18),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"class_{cls_id}",
            )

        if self._draft_start is not None and self._draft_end is not None:
            x1 = min(self._draft_start[0], self._draft_end[0])
            y1 = min(self._draft_start[1], self._draft_end[1])
            x2 = max(self._draft_start[0], self._draft_end[0])
            y2 = max(self._draft_start[1], self._draft_end[1])
            painter.setPen(QPen(QColor("#f59e0b"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(_box_rect([x1, y1, x2, y2]))


class DetectionCorrectionDialog(QDialog):
    def __init__(self, image_bgr: np.ndarray, detections: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fix Detection")
        self.setModal(True)
        self.resize(1180, 760)
        self.setStyleSheet(_theme_stylesheet())
        self._result_detections: list[dict] = []
        self._image = image_bgr.copy()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        self._canvas = DetectionAnnotationCanvas()
        self._canvas.set_image(self._image)
        self._canvas.set_boxes(detections)
        layout.addWidget(self._canvas, stretch=3)

        sidebar = QVBoxLayout()
        sidebar.setSpacing(10)
        sidebar.addWidget(QLabel("Correction Tools"))

        self._btn_select = QPushButton("Select / Move")
        self._btn_add_strip = QPushButton("Draw Strip Box")
        self._btn_add_digit = QPushButton("Draw Digit Box")
        self._btn_replace = QPushButton("Replace Selected")
        self._btn_delete = QPushButton("Delete Selected")
        for btn in (self._btn_select, self._btn_add_strip, self._btn_add_digit, self._btn_replace, self._btn_delete):
            sidebar.addWidget(btn)

        sidebar.addWidget(QLabel("Digit class"))
        self._digit_combo = QComboBox()
        for class_id in range(DIGIT_OFFSET, UNREADABLE_CLS + 1):
            self._digit_combo.addItem(CLASS_NAMES[class_id], class_id)
        sidebar.addWidget(self._digit_combo)

        sidebar.addWidget(QLabel("Selected box class"))
        self._selected_combo = QComboBox()
        for class_id, name in enumerate(CLASS_NAMES):
            self._selected_combo.addItem(name, class_id)
        sidebar.addWidget(self._selected_combo)

        self._reading_label = QLabel("Reading: -----")
        self._reading_label.setObjectName("readingValue")
        self._reading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._reading_label.setFixedHeight(56)
        sidebar.addWidget(self._reading_label)

        self._summary_label = QLabel("")
        self._summary_label.setWordWrap(True)
        self._summary_label.setObjectName("detailPanel")
        sidebar.addWidget(self._summary_label, stretch=1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        sidebar.addWidget(buttons)

        layout.addLayout(sidebar, stretch=1)

        self._btn_select.clicked.connect(lambda: self._canvas.set_mode("select"))
        self._btn_add_strip.clicked.connect(lambda: self._canvas.set_mode("add_strip"))
        self._btn_add_digit.clicked.connect(lambda: self._canvas.set_mode("add_digit"))
        self._btn_replace.clicked.connect(lambda: self._canvas.set_mode("replace"))
        self._btn_delete.clicked.connect(self._canvas.delete_selected)
        self._digit_combo.currentIndexChanged.connect(self._sync_pending_digit_class)
        self._selected_combo.currentIndexChanged.connect(self._sync_selected_box_class)
        self._canvas.selection_changed.connect(self._on_selection_changed)
        self._canvas.boxes_changed.connect(self._refresh_summary)
        self._sync_pending_digit_class()
        self._refresh_summary()

    def _sync_pending_digit_class(self) -> None:
        self._canvas.set_pending_class_id(int(self._digit_combo.currentData()))

    def _sync_selected_box_class(self) -> None:
        selected = self._canvas.selected_box()
        if selected is None:
            return
        self._canvas.set_selected_class_id(int(self._selected_combo.currentData()))

    def _on_selection_changed(self, index: int) -> None:
        selected = self._canvas.selected_box()
        self._selected_combo.setEnabled(selected is not None)
        if selected is not None:
            combo_index = self._selected_combo.findData(int(selected["cls"]))
            if combo_index >= 0:
                self._selected_combo.setCurrentIndex(combo_index)
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        detections = self._canvas.boxes()
        digits = [det for det in detections if int(det["cls"]) != STRIP_CLASS_ID]
        strips = [det for det in detections if int(det["cls"]) == STRIP_CLASS_ID]
        reading = build_reading_from_detections(detections, expected_digits=NUM_DIGITS)
        self._reading_label.setText(f"Reading: {reading}")
        self._summary_label.setText(
            f"Strip boxes: {len(strips)}\n"
            f"Digit boxes: {len(digits)} / {NUM_DIGITS}\n"
            "Tip: use Select / Move for quick repositioning, or Replace Selected to redraw a box."
        )

    def _accept_if_valid(self) -> None:
        detections = self._canvas.boxes()
        strips = [det for det in detections if int(det["cls"]) == STRIP_CLASS_ID]
        digits = [det for det in detections if int(det["cls"]) != STRIP_CLASS_ID]
        if len(strips) != 1:
            QMessageBox.warning(self, "Invalid Annotation", "Please keep exactly one strip box.")
            return
        if len(digits) == 0 or len(digits) > NUM_DIGITS:
            QMessageBox.warning(self, "Invalid Annotation", f"Please keep between 1 and {NUM_DIGITS} digit boxes.")
            return
        self._result_detections = detections
        self.accept()

    def result_detections(self) -> list[dict]:
        return _clone_detection_list(self._result_detections)


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
        self.resize(1480, 960)
        self.setStyleSheet(_theme_stylesheet())

        self._images: list[Path] = []
        self._idx = -1
        self._rotation = 0
        self._models: list[LoadedModel] = []
        self._active_model_index = -1
        self._model_path = ""
        self._raw_cv: np.ndarray | None = None
        self._annotated_cv: np.ndarray | None = None
        self._annotated_is_rotated = False
        self._current_candidate: dict | None = None
        self._current_candidates: list[dict] = []
        self._last_compare_results: list[dict] = []
        self._review_store = ReviewStore(Path(__file__).resolve().parent / "corrections")
        self._latest_review_map = self._review_store.load_latest_review_map()
        self._current_review: dict | None = None

        # Persistent settings — remember last-used directories across restarts
        self._settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self._last_image_dir = self._read_existing_dir_setting(SETTINGS_LAST_IMAGE_DIR)
        self._last_model_dir = self._read_existing_dir_setting(SETTINGS_LAST_MODEL_DIR)

        self._build_ui()
        self._wire_signals()
        self._sync_model_widgets()
        self._populate_compare_results([])
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
        self._btn_add_model = QPushButton("Add Model (.pt / .tflite)")
        self._lbl_active_model = QLabel("Active model: none")
        self._lbl_active_model.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        config_row.addWidget(self._btn_add_model)
        config_row.addWidget(self._lbl_active_model, stretch=1)
        config_row.addWidget(QLabel("Rotation"))
        self._btn_rotate_left = QPushButton("Rotate Left")
        self._btn_rotate_right = QPushButton("Rotate Right")
        self._btn_rotate_left.setFixedWidth(108)
        self._btn_rotate_right.setFixedWidth(108)
        config_row.addWidget(self._btn_rotate_left)
        config_row.addWidget(self._btn_rotate_right)
        hero_layout.addLayout(config_row)

        outer.addWidget(hero)

        content = QSplitter(Qt.Orientation.Horizontal)
        content.setChildrenCollapsible(False)

        viewer_group = QGroupBox("Image Viewer")
        viewer_layout = QVBoxLayout(viewer_group)
        self._viewer = _ImageViewer()
        viewer_layout.addWidget(self._viewer)
        content.addWidget(viewer_group)

        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_scroll.setMinimumWidth(430)
        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(14)

        models_group = QGroupBox("Loaded Models")
        models_layout = QVBoxLayout(models_group)
        models_layout.setSpacing(10)
        self._tbl_models = QTableWidget(0, 3)
        self._tbl_models.setHorizontalHeaderLabels(["Active", "Model", "Backend"])
        self._tbl_models.verticalHeader().setVisible(False)
        self._tbl_models.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._tbl_models.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._tbl_models.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._tbl_models.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._tbl_models.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._tbl_models.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._tbl_models.setMinimumHeight(170)
        models_layout.addWidget(self._tbl_models)

        model_btn_row = QHBoxLayout()
        model_btn_row.setSpacing(8)
        self._btn_remove_model = QPushButton("Remove Model")
        self._btn_set_active_model = QPushButton("Set Active")
        self._btn_compare = QPushButton("Compare Models")
        model_btn_row.addWidget(self._btn_remove_model)
        model_btn_row.addWidget(self._btn_set_active_model)
        model_btn_row.addWidget(self._btn_compare)
        models_layout.addLayout(model_btn_row)

        self._lbl_model_hint = QLabel("Load one or more YOLO models. Batch reads use the active model only.")
        self._lbl_model_hint.setObjectName("subtitleLabel")
        self._lbl_model_hint.setWordWrap(True)
        models_layout.addWidget(self._lbl_model_hint)
        side_layout.addWidget(models_group)

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

        review_controls = QGridLayout()
        review_controls.setHorizontalSpacing(10)
        review_controls.setVerticalSpacing(8)

        self._brand_combo = QComboBox()
        self._brand_combo.setEditable(True)
        self._brand_combo.addItems(["Unknown", "AsiaM", "Maxwinner"])
        self._brand_combo.setCurrentText("Unknown")

        self._btn_mark_correct = QPushButton("Mark Correct")
        self._btn_correct_reading = QPushButton("Correct Reading")
        self._btn_fix_detection = QPushButton("Fix Detection")
        self._btn_needs_review = QPushButton("Needs Review")
        self._btn_export_reviewed = QPushButton("Export Reviewed Cases")

        review_controls.addWidget(QLabel("Meter Brand"), 0, 0)
        review_controls.addWidget(self._brand_combo, 0, 1, 1, 2)
        review_controls.addWidget(self._btn_mark_correct, 1, 0)
        review_controls.addWidget(self._btn_correct_reading, 1, 1)
        review_controls.addWidget(self._btn_fix_detection, 1, 2)
        review_controls.addWidget(self._btn_needs_review, 2, 0)
        review_controls.addWidget(self._btn_export_reviewed, 2, 1, 1, 2)
        results_layout.addLayout(review_controls)

        self._lbl_review_state = QLabel("Review state: none")
        self._lbl_review_state.setObjectName("detailPanel")
        self._lbl_review_state.setWordWrap(True)
        results_layout.addWidget(self._lbl_review_state)

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
            "Read behavior: class 0 = strip, classes 1..10 = digits 0..9, class 11 = unreadable. "
            ".pt models use Ultralytics; .tflite models use TensorFlow Lite."
        )
        help_text.setObjectName("subtitleLabel")
        help_text.setWordWrap(True)
        results_layout.addWidget(help_text)

        side_layout.addWidget(results_group)

        compare_group = QGroupBox("Comparison Results")
        compare_layout = QVBoxLayout(compare_group)
        compare_layout.setSpacing(10)
        self._tbl_compare = QTableWidget(0, 7)
        self._tbl_compare.setHorizontalHeaderLabels(
            ["Model", "Backend", "Reading", "Status", "Strip", "Digits", "Window"]
        )
        self._tbl_compare.verticalHeader().setVisible(False)
        self._tbl_compare.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._tbl_compare.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._tbl_compare.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._tbl_compare.setMinimumHeight(220)
        self._tbl_compare.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, 7):
            self._tbl_compare.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        compare_layout.addWidget(self._tbl_compare)

        self._lbl_compare_summary = QLabel("Run Compare Models to see side-by-side outputs for the current image.")
        self._lbl_compare_summary.setObjectName("detailPanel")
        self._lbl_compare_summary.setWordWrap(True)
        compare_layout.addWidget(self._lbl_compare_summary)
        side_layout.addWidget(compare_group)
        side_layout.addStretch(1)

        side_scroll.setWidget(side_panel)
        content.addWidget(side_scroll)
        content.setStretchFactor(0, 4)
        content.setStretchFactor(1, 3)
        outer.addWidget(content, stretch=1)

        self.setStatusBar(QStatusBar())

    def _wire_signals(self) -> None:
        self._btn_folder.clicked.connect(self._open_folder)
        self._btn_prev.clicked.connect(self._go_prev)
        self._btn_next.clicked.connect(self._go_next)
        self._btn_fit.clicked.connect(self._viewer.fit)
        self._btn_add_model.clicked.connect(self._select_model)
        self._btn_remove_model.clicked.connect(self._remove_selected_model)
        self._btn_set_active_model.clicked.connect(self._set_selected_model_active)
        self._btn_compare.clicked.connect(self._compare_models)
        self._tbl_models.itemSelectionChanged.connect(self._refresh_controls)
        self._tbl_models.itemDoubleClicked.connect(lambda _item: self._set_selected_model_active())
        self._btn_read.clicked.connect(self._run_read)
        self._btn_batch.clicked.connect(self._run_batch_read)
        self._btn_rotate_left.clicked.connect(self._rotate_left)
        self._btn_rotate_right.clicked.connect(self._rotate_right)
        self._btn_mark_correct.clicked.connect(self._mark_current_correct)
        self._btn_correct_reading.clicked.connect(self._correct_current_reading)
        self._btn_fix_detection.clicked.connect(self._fix_current_detection)
        self._btn_needs_review.clicked.connect(self._mark_needs_review)
        self._btn_export_reviewed.clicked.connect(self._export_reviewed_cases)

        QShortcut(QKeySequence(Qt.Key.Key_Left), self, self._go_prev)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self._go_next)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._run_read)

    def _reset_result_state(self) -> None:
        self._current_candidate = None
        self._current_candidates = []
        self._lbl_reading.setText("-----")
        self._lbl_detail.setText("Load an image and a trained YOLO model, then run a read.")
        self._set_status_badge("Idle")
        self._lbl_strip_conf.setText("--")
        self._lbl_digit_count.setText("--")
        self._lbl_window.setText("--")
        self._lbl_avg_conf.setText("--")
        self._update_review_state_label()
        if hasattr(self, "_btn_mark_correct"):
            self._refresh_controls()

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

    def _current_image_path(self) -> Path | None:
        if not self._images or self._idx < 0 or self._idx >= len(self._images):
            return None
        return self._images[self._idx]

    def _current_visible_image(self) -> np.ndarray | None:
        if self._raw_cv is None:
            return None
        return _rotate_cv(self._raw_cv, self._rotation)

    def _current_review_brand(self) -> str:
        return sanitize_brand(self._brand_combo.currentText())

    def _load_existing_review_state(self) -> None:
        image_path = self._current_image_path()
        self._current_review = self._latest_review_map.get(str(image_path)) if image_path else None
        if self._current_review is not None:
            brand = str(self._current_review.get("brand", "")).strip()
            if brand:
                self._brand_combo.setCurrentText(brand)
        self._update_review_state_label()

    def _update_review_state_label(self) -> None:
        if self._current_review is None:
            self._lbl_review_state.setText("Review state: none yet for this image.")
            return
        review_type = str(self._current_review.get("review_type", "unknown"))
        review_status = str(self._current_review.get("review_status", ""))
        corrected = str(self._current_review.get("corrected_reading", ""))
        brand = str(self._current_review.get("brand", "Unknown"))
        self._lbl_review_state.setText(
            f"Review state: {review_type} | status: {review_status or '--'} | brand: {brand} | corrected: {corrected or '--'}"
        )

    def _build_review_metadata(self) -> dict:
        candidate = self._current_candidate
        metadata = {
            "rotation": self._rotation,
            "tester_status": candidate["status"] if candidate else "No Detection",
            "strip_confidence": float(candidate["strip_conf"]) if candidate else 0.0,
            "digit_count": int(candidate["digit_count"]) if candidate else 0,
            "digit_avg_confidence": float(candidate["digit_conf_avg"]) if candidate else 0.0,
            "window_index": int(candidate["window_index"]) if candidate else 0,
            "total_windows": int(candidate["total_windows"]) if candidate else 0,
            "notes": candidate.get("notes", []) if candidate else [],
        }
        return metadata

    def _save_review(
        self,
        *,
        review_type: str,
        review_status: str,
        corrected_reading: str,
        corrected_detections: list[dict],
        copy_image: bool = False,
    ) -> None:
        image_path = self._current_image_path()
        visible_image = self._current_visible_image()
        if image_path is None or visible_image is None:
            QMessageBox.warning(self, "No Image", "Load an image before saving a review.")
            return
        predicted_reading = self._current_candidate["reading"] if self._current_candidate else ""
        original_detections = _selected_review_detections(self._current_candidate)
        payload = self._review_store.save_review(
            source_image_path=image_path,
            source_image_bgr=visible_image,
            review_type=review_type,
            review_status=review_status,
            brand=self._current_review_brand(),
            model_path=self._model_path,
            predicted_reading=predicted_reading,
            corrected_reading=corrected_reading,
            original_detections=original_detections,
            corrected_detections=_clone_detection_list(corrected_detections),
            metadata=self._build_review_metadata(),
            copy_image=copy_image,
        )
        self._latest_review_map[str(image_path)] = payload
        self._current_review = payload
        self._update_review_state_label()

    def _review_type_for_image(self, image_path: Path | None) -> str:
        if image_path is None:
            return "Unreviewed"
        review = self._latest_review_map.get(str(image_path))
        if review is None:
            return "Unreviewed"
        return str(review.get("review_type", "Unreviewed"))

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

    def _active_model(self) -> LoadedModel | None:
        if 0 <= self._active_model_index < len(self._models):
            return self._models[self._active_model_index]
        return None

    def _selected_model_index(self) -> int:
        row = self._tbl_models.currentRow() if hasattr(self, "_tbl_models") else -1
        return row if 0 <= row < len(self._models) else -1

    def _sync_model_widgets(self) -> None:
        active = self._active_model()
        self._model_path = active.path if active else ""
        self._lbl_active_model.setText(
            f"Active model: {active.name} ({active.backend})" if active else "Active model: none"
        )

        self._tbl_models.setRowCount(len(self._models))
        for row, model_entry in enumerate(self._models):
            active_text = "Yes" if row == self._active_model_index else ""
            values = [active_text, model_entry.name, model_entry.backend.upper()]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                self._tbl_models.setItem(row, col, item)

        selected = self._selected_model_index()
        if selected == -1 and self._models:
            self._tbl_models.selectRow(self._active_model_index if self._active_model_index >= 0 else 0)

    def _select_model(self) -> None:
        start_dir = self._last_model_dir if self._last_model_dir else self._last_image_dir
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model",
            start_dir,
            "YOLO Models (*.pt *.tflite);;All Files (*)",
        )
        if not path:
            return
        if any(existing.path == path for existing in self._models):
            self.statusBar().showMessage(f"Model already loaded: {Path(path).name}")
            return
        try:
            model_entry = _create_loaded_model(path)
            self._models.append(model_entry)
            if self._active_model_index < 0:
                self._active_model_index = 0
            self._last_model_dir = str(Path(path).parent)
            self._settings.setValue(SETTINGS_LAST_MODEL_DIR, self._last_model_dir)
            self._sync_model_widgets()
            self.statusBar().showMessage(f"Model loaded: {model_entry.name}")
        except Exception as exc:
            self.statusBar().showMessage(f"Model load error: {exc}")
            self._set_status_badge("Error")
        self._refresh_controls()

    def _set_selected_model_active(self) -> None:
        selected = self._selected_model_index()
        if selected == -1:
            return
        self._active_model_index = selected
        self._sync_model_widgets()
        self._reset_result_state()
        active = self._active_model()
        if active is not None:
            self.statusBar().showMessage(f"Active model set: {active.name}")
        self._refresh_controls()

    def _remove_selected_model(self) -> None:
        selected = self._selected_model_index()
        if selected == -1:
            return
        removed = self._models.pop(selected)
        if not self._models:
            self._active_model_index = -1
        elif selected == self._active_model_index:
            self._active_model_index = min(selected, len(self._models) - 1)
        elif selected < self._active_model_index:
            self._active_model_index -= 1
        self._last_compare_results = [r for r in self._last_compare_results if r.get("path") != removed.path]
        self._populate_compare_results(self._last_compare_results)
        self._sync_model_widgets()
        self._reset_result_state()
        self.statusBar().showMessage(f"Removed model: {removed.name}")
        self._refresh_controls()

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
        self._load_existing_review_state()
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

    # ------------------------------------------------------------------
    # Inference pipeline (with optional live-preview callback)
    # ------------------------------------------------------------------

    def _infer_windows(
        self,
        model_entry: LoadedModel,
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

            dets = _infer_window_detections(model_entry, _infer, window)

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

    def _candidate_summary_row(self, model_entry: LoadedModel, candidate: dict | None, scanned_windows: int, total_windows: int) -> dict:
        if candidate is None:
            return {
                "name": model_entry.name,
                "path": model_entry.path,
                "backend": model_entry.backend,
                "reading": "-----",
                "status": "No Detection",
                "strip_conf": "--",
                "digit_count": "0 / 5",
                "window": f"{scanned_windows} / {total_windows}",
            }
        return {
            "name": model_entry.name,
            "path": model_entry.path,
            "backend": model_entry.backend,
            "reading": candidate["reading"],
            "status": candidate["status"],
            "strip_conf": f"{candidate['strip_conf']:.0%}" if candidate["strip_det"] else "--",
            "digit_count": f"{candidate['digit_count']} / {NUM_DIGITS}",
            "window": f"{candidate['window_index']} / {candidate['total_windows']}",
        }

    def _populate_compare_results(self, rows: list[dict]) -> None:
        self._tbl_compare.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            values = [
                row.get("name", ""),
                str(row.get("backend", "")).upper(),
                row.get("reading", ""),
                row.get("status", ""),
                row.get("strip_conf", ""),
                row.get("digit_count", ""),
                row.get("window", ""),
            ]
            for col_idx, value in enumerate(values):
                self._tbl_compare.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))

        if not rows:
            self._lbl_compare_summary.setText("Run Compare Models to see side-by-side outputs for the current image.")
            return

        summary = ", ".join(f"{row['name']}: {row['reading']} ({row['status']})" for row in rows)
        self._lbl_compare_summary.setText(summary)

    def _compare_models(self) -> None:
        if self._raw_cv is None or not self._models:
            return

        visible_image = _rotate_cv(self._raw_cv, self._rotation)
        total_windows = len(build_sliding_windows(visible_image.shape[1], visible_image.shape[0]))
        comparison_rows: list[dict] = []
        errors: list[str] = []
        active_candidate: dict | None = None
        active_candidates: list[dict] = []

        self._set_status_badge("Scanning")
        self._lbl_detail.setText("Comparing loaded models on the current image…")
        self.statusBar().showMessage("Comparing loaded models on the current image…")
        QApplication.processEvents()

        for idx, model_entry in enumerate(self._models):
            preview: ProcessPreviewDialog | None = None
            try:
                preview = ProcessPreviewDialog(self) if idx == self._active_model_index else None
                if preview is not None:
                    preview.show()
                    QApplication.processEvents()
                candidate, candidates = self._infer_windows(model_entry, visible_image, preview=preview)
                if preview is not None:
                    preview.finish(delay_ms=1500)
                if idx == self._active_model_index:
                    active_candidate = candidate
                    active_candidates = candidates
                comparison_rows.append(
                    self._candidate_summary_row(model_entry, candidate, len(candidates), total_windows)
                )
            except Exception as exc:
                errors.append(f"{model_entry.name}: {exc}")
                comparison_rows.append(
                    {
                        "name": model_entry.name,
                        "path": model_entry.path,
                        "backend": model_entry.backend,
                        "reading": "-----",
                        "status": "Error",
                        "strip_conf": "--",
                        "digit_count": "--",
                        "window": "--",
                    }
                )
                if preview is not None:
                    preview.close()

        self._last_compare_results = comparison_rows
        self._populate_compare_results(comparison_rows)
        if active_candidate is not None or active_candidates:
            self._current_candidates = active_candidates
            self._update_result_ui(active_candidate, len(active_candidates), total_windows)
        elif comparison_rows:
            self._update_result_ui(None, total_windows, total_windows)
        if errors:
            self.statusBar().showMessage("Compare finished with errors: " + " | ".join(errors))

    # ------------------------------------------------------------------
    # Result UI update
    # ------------------------------------------------------------------

    def _update_result_ui(self, candidate: dict | None, scanned_windows: int, total_windows: int) -> None:
        self._current_candidate = candidate
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
        self._update_review_state_label()
        self._refresh_controls()

    # ------------------------------------------------------------------
    # Run Read — opens process preview, runs inference, closes on finish
    # ------------------------------------------------------------------

    def _run_read(self) -> None:
        active_model = self._active_model()
        if active_model is None or self._raw_cv is None:
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
            best_candidate, candidates = self._infer_windows(active_model, visible_image, preview=preview)
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
        self._current_candidates = candidates
        self._update_result_ui(best_candidate, scanned_windows, total_windows)
        self._last_compare_results = [
            self._candidate_summary_row(active_model, best_candidate, scanned_windows, total_windows)
        ]
        self._populate_compare_results(self._last_compare_results)

    def _mark_current_correct(self) -> None:
        if self._current_candidate is None:
            QMessageBox.information(self, "No Read Yet", "Run a read first, then mark the result.")
            return
        detections = _selected_review_detections(self._current_candidate)
        self._save_review(
            review_type="marked_correct",
            review_status="Correct",
            corrected_reading=self._current_candidate["reading"],
            corrected_detections=detections,
        )
        self.statusBar().showMessage("Saved review: marked correct.")

    def _correct_current_reading(self) -> None:
        if self._current_candidate is None:
            QMessageBox.information(self, "No Read Yet", "Run a read first, then correct the reading.")
            return
        if self._current_candidate.get("strip_det") is None or len(self._current_candidate.get("digit_dets", [])) != NUM_DIGITS:
            QMessageBox.warning(
                self,
                "Use Detection Fix",
                "This image does not have a trustworthy strip + 5 digits yet. Use 'Fix Detection' instead.",
            )
            return
        current = str(self._current_candidate["reading"])
        text, ok = QInputDialog.getText(
            self,
            "Correct Reading",
            "Enter the true 5-character reading (digits or X):",
            text=current,
        )
        if not ok:
            return
        corrected = text.strip().upper()
        if len(corrected) != NUM_DIGITS or any((not ch.isdigit()) and ch != "X" for ch in corrected):
            QMessageBox.warning(self, "Invalid Reading", "Use exactly 5 characters consisting of digits or X.")
            return

        detections = _selected_review_detections(self._current_candidate)
        digit_dets = [det for det in detections if int(det["cls"]) != STRIP_CLASS_ID]
        digit_dets.sort(key=lambda det: _box_center_x(det["box"]))
        for idx, ch in enumerate(corrected):
            digit_dets[idx]["cls"] = char_to_class_id(ch)
            digit_dets[idx]["label"] = _style_name(digit_dets[idx]["cls"])
        self._save_review(
            review_type="reading_only",
            review_status="Corrected Reading",
            corrected_reading=corrected,
            corrected_detections=detections,
        )
        self.statusBar().showMessage(f"Saved corrected reading: {corrected}")

    def _fix_current_detection(self) -> None:
        image = self._current_visible_image()
        if image is None:
            QMessageBox.information(self, "No Image", "Load an image first.")
            return
        detections = _selected_review_detections(self._current_candidate)
        dialog = DetectionCorrectionDialog(image, detections, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        corrected_dets = dialog.result_detections()
        corrected_reading = build_reading_from_detections(corrected_dets, expected_digits=NUM_DIGITS)
        self._save_review(
            review_type="detection_fixed",
            review_status="Detection Fixed",
            corrected_reading=corrected_reading,
            corrected_detections=corrected_dets,
        )
        self.statusBar().showMessage(f"Saved corrected boxes for reading {corrected_reading}")

    def _mark_needs_review(self) -> None:
        if self._current_image_path() is None:
            QMessageBox.information(self, "No Image", "Load an image first.")
            return
        corrected = self._current_candidate["reading"] if self._current_candidate else ""
        detections = _selected_review_detections(self._current_candidate)
        self._save_review(
            review_type="unreadable_or_skip",
            review_status="Needs Review",
            corrected_reading=corrected,
            corrected_detections=detections,
        )
        self.statusBar().showMessage("Saved review flag: needs review.")

    def _export_reviewed_cases(self) -> None:
        dialog = ExportReviewsDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        brand_filter, include_correct = dialog.values()
        default_dir = self._review_store.paths.exports_dir / "review_export"
        export_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Export Folder",
            str(default_dir),
        )
        if not export_dir:
            return
        result = self._review_store.export_reviews(
            export_dir,
            brand_filter=brand_filter,
            include_marked_correct=include_correct,
        )
        QMessageBox.information(
            self,
            "Export Complete",
            f"Exported {result['exported_count']} reviewed case(s).\nSkipped: {result['skipped_count']}\n\nFolder:\n{result['export_dir']}",
        )
        self.statusBar().showMessage(f"Reviewed cases exported: {result['exported_count']}")

    def _open_image_from_batch(self, filename: str) -> None:
        for idx, path in enumerate(self._images):
            if path.name == filename:
                self._idx = idx
                self._load_image(reset_view=True)
                self._refresh_controls()
                self.statusBar().showMessage(f"Loaded batch image for review: {filename}")
                return

    # ------------------------------------------------------------------
    # Batch Read — dispatch all images to QThreadPool, collect via signals
    # ------------------------------------------------------------------

    def _run_batch_read(self) -> None:
        active_model = self._active_model()
        if active_model is None or not self._images:
            return

        total        = len(self._images)
        folder_str   = str(self._images[0].parent)
        model_lock   = threading.Lock()   # serialises runtime inference calls

        report_dialog = BatchReportDialog(total, self)
        report_dialog.set_folder(folder_str)
        report_dialog.set_review_lookup(self._latest_review_map)
        report_dialog.open_image_requested.connect(self._open_image_from_batch)
        report_dialog.show()
        QApplication.processEvents()
        self.statusBar().showMessage(f"Batch read using active model: {active_model.name}")

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
            worker = BatchImageWorker(idx, path, active_model, self._rotation, model_lock)
            # Qt queued connection ensures _on_result runs on the main thread
            worker.signals.result.connect(_on_result, Qt.ConnectionType.QueuedConnection)
            pool.start(worker)

    def _refresh_controls(self) -> None:
        has_images = bool(self._images)
        has_model = self._active_model() is not None
        selected_model = self._selected_model_index() != -1
        self._btn_prev.setEnabled(has_images and self._idx > 0)
        self._btn_next.setEnabled(has_images and self._idx < len(self._images) - 1)
        self._btn_read.setEnabled(has_images and has_model)
        self._btn_batch.setEnabled(has_images and has_model)
        self._btn_remove_model.setEnabled(selected_model)
        self._btn_set_active_model.setEnabled(selected_model and self._selected_model_index() != self._active_model_index)
        self._btn_compare.setEnabled(has_images and len(self._models) > 1)
        self._btn_mark_correct.setEnabled(has_images and self._current_candidate is not None)
        self._btn_correct_reading.setEnabled(has_images and self._current_candidate is not None)
        self._btn_fix_detection.setEnabled(has_images)
        self._btn_needs_review.setEnabled(has_images)
        self._btn_export_reviewed.setEnabled(True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not ULTRALYTICS_AVAILABLE:
        print("[WARNING] ultralytics not installed.")
        print("          .pt model testing will not work.")
        print("          Install with: pip install ultralytics")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    wind