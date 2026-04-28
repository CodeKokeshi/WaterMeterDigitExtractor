"""
Streamlined guidebox-only DigitExtractor UI with canonical workspace framing.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QResizeEvent, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
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
    QPushButton,
    QSlider,
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
            f"Rotation: {int(round(self._batch_template.rotation_deg)) % 360}° | "
            f"Scale: {self._batch_template.scale:.3f} | "
            f"Pan: ({self._batch_template.translate_x:.1f}, {self._batch_template.translate_y:.1f})"
        )


def main():
    app = QApplication(sys.argv)
    window = StreamlinedMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
