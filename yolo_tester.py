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
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from PyQt6.QtCore import QRectF, Qt
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
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStatusBar,
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


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
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


def _build_reading_text(digit_dets: list[dict]) -> str:
    if not digit_dets:
        return "-----"
    return "".join(_digit_char(det["cls"]) for det in digit_dets)


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
    reading = _build_reading_text(best_digits)
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
    """


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
            "Windowed 640x640 scanning for full-resolution images, with strip-local digit decoding."
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

        self._btn_read = QPushButton("Run Read")
        self._btn_read.setFixedHeight(48)
        self._btn_read.setEnabled(False)
        results_layout.addWidget(self._btn_read)

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

    def _open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        paths = sorted(
            (path for path in Path(folder).iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
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
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
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
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLOv8 Model",
            "",
            "PyTorch Model (*.pt)",
        )
        if not path:
            return
        try:
            self._model = YOLO(path)
            self._lbl_model.setText(f"Model: {Path(path).name}")
            self.statusBar().showMessage(f"Model loaded: {Path(path).name}")
        except Exception as exc:
            self._model = None
            self._lbl_model.setText("Failed to load model")
            self.statusBar().showMessage(f"Model load error: {exc}")
            self._set_status_badge("Error")
        self._refresh_controls()

    def _infer_windows(self, image: np.ndarray) -> tuple[dict | None, list[dict]]:
        windows = build_sliding_windows(image.shape[1], image.shape[0])
        if not windows:
            return None, []

        candidates: list[dict] = []
        best_candidate: dict | None = None
        total_windows = len(windows)

        for idx, window in enumerate(windows, start=1):
            crop = image[window.y:window.y + window.size, window.x:window.x + window.size]
            if crop.size == 0:
                continue
            resized = cv2.resize(crop, (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            # Convert to 3-channel grayscale for inference so the input
            # colour-space matches what the model was trained on.
            _gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            _infer   = cv2.cvtColor(_gray,   cv2.COLOR_GRAY2BGR)
            results = self._model.predict(  # type: ignore[union-attr]
                source=_infer,
                imgsz=MODEL_IMAGE_SIZE,
                verbose=False,
            )
            if not results:
                dets = []
            else:
                dets = _parse_result_detections(results[0], window)

            candidate = _evaluate_window_candidate(dets, idx, total_windows, window)
            candidates.append(candidate)
            if candidate["valid"]:
                return candidate, candidates
            if best_candidate is None or _candidate_rank(candidate) > _candidate_rank(best_candidate):
                best_candidate = candidate

        return best_candidate, candidates

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

    def _run_read(self) -> None:
        if self._model is None or self._raw_cv is None:
            return

        visible_image = _rotate_cv(self._raw_cv, self._rotation)
        self._set_status_badge("Scanning")
        self._lbl_detail.setText("Running center-first 640x640 window scan...")
        self.statusBar().showMessage("Running center-first 640x640 window scan...")
        QApplication.processEvents()

        try:
            best_candidate, candidates = self._infer_windows(visible_image)
        except Exception as exc:
            self._set_status_badge("Error")
            self._lbl_detail.setText(f"Inference error: {exc}")
            self.statusBar().showMessage(f"Inference error: {exc}")
            return

        total_windows = len(build_sliding_windows(visible_image.shape[1], visible_image.shape[0]))
        scanned_windows = len(candidates)
        self._update_result_ui(best_candidate, scanned_windows, total_windows)

    def _refresh_controls(self) -> None:
        has_images = bool(self._images)
        self._btn_prev.setEnabled(has_images and self._idx > 0)
        self._btn_next.setEnabled(has_images and self._idx < len(self._images) - 1)
        self._btn_read.setEnabled(has_images and self._model is not None)


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
