"""Dialogs for configuring LeNet training in DigitExtractor."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QVBoxLayout,
    QWidget,
)


class LeNetTrainingDialog(QDialog):
    """Collect parameters for LeNet-style digit training."""

    def __init__(
        self,
        dataset_dir: str,
        backend_python: str,
        keras_output_dir: str,
        tflite_output_dir: str,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Train LeNet-5 Digit Model")
        self.setModal(True)
        self.setMinimumWidth(760)

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Train a LeNet-style digit classifier from your 0-9 folders and export "
            "both a TensorFlow/Keras model and a TFLite model."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        note = QLabel(
            "The app stays responsive during training. The actual TensorFlow work runs in "
            "a separate backend Python environment."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #f0d27a;")
        layout.addWidget(note)

        form = QFormLayout()

        dataset_row = QHBoxLayout()
        self._dataset_edit = QLineEdit(dataset_dir)
        dataset_btn = QPushButton("Browse...")
        dataset_btn.clicked.connect(self._browse_dataset)
        dataset_row.addWidget(self._dataset_edit)
        dataset_row.addWidget(dataset_btn)
        form.addRow("Dataset Folder:", self._wrap_layout(dataset_row))

        keras_row = QHBoxLayout()
        self._keras_output_edit = QLineEdit(keras_output_dir)
        keras_btn = QPushButton("Browse...")
        keras_btn.clicked.connect(self._browse_keras_output)
        keras_row.addWidget(self._keras_output_edit)
        keras_row.addWidget(keras_btn)
        form.addRow("TensorFlow Output:", self._wrap_layout(keras_row))

        tflite_row = QHBoxLayout()
        self._tflite_output_edit = QLineEdit(tflite_output_dir)
        tflite_btn = QPushButton("Browse...")
        tflite_btn.clicked.connect(self._browse_tflite_output)
        tflite_row.addWidget(self._tflite_output_edit)
        tflite_row.addWidget(tflite_btn)
        form.addRow("TFLite Output:", self._wrap_layout(tflite_row))

        backend_row = QHBoxLayout()
        self._backend_edit = QLineEdit(backend_python)
        backend_btn = QPushButton("Browse...")
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

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _wrap_layout(self, row_layout: QHBoxLayout) -> QWidget:
        widget = QWidget()
        widget.setLayout(row_layout)
        return widget

    def _browse_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Select 0-9 Dataset Folder")
        if folder:
            self._dataset_edit.setText(folder)

    def _browse_keras_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select TensorFlow/Keras Output Folder")
        if folder:
            self._keras_output_edit.setText(folder)

    def _browse_tflite_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select TFLite Output Folder")
        if folder:
            self._tflite_output_edit.setText(folder)

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
        keras_output_dir = self._keras_output_edit.text().strip()
        tflite_output_dir = self._tflite_output_edit.text().strip()
        backend_python = self._backend_edit.text().strip()

        if not dataset_dir or not Path(dataset_dir).is_dir():
            QMessageBox.warning(self, "Missing Dataset", "Select a valid dataset folder.")
            return

        if not keras_output_dir:
            QMessageBox.warning(
                self,
                "Missing TensorFlow Output",
                "Select where the TensorFlow/Keras model should be saved.",
            )
            return

        if not tflite_output_dir:
            QMessageBox.warning(
                self,
                "Missing TFLite Output",
                "Select where the TFLite model should be saved.",
            )
            return

        if not backend_python or not Path(backend_python).exists():
            QMessageBox.warning(
                self,
                "Missing Backend Python",
                "Select a compatible Python executable for TensorFlow training.",
            )
            return

        self.accept()

    def get_config(self) -> dict[str, object]:
        return {
            "dataset_dir": self._dataset_edit.text().strip(),
            "keras_output_dir": self._keras_output_edit.text().strip(),
            "tflite_output_dir": self._tflite_output_edit.text().strip(),
            "backend_python": self._backend_edit.text().strip(),
            "epochs": int(self._epochs_spin.value()),
            "batch_size": int(self._batch_size_spin.value()),
            "validation_split": float(self._validation_spin.value()),
            "seed": int(self._seed_spin.value()),
        }


class YoloTrainingDialog(QDialog):
    """Collect parameters for YOLOv8 digit-strip finder training."""

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        backend_python: str,
        output_dir: str,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Train YOLOv8 Digit Strip Finder")
        self.setModal(True)
        self.setMinimumWidth(760)

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Train a YOLOv8 detector from ROI_640 images and ROI_640_labels text files. "
            "This model finds the digit strip; it does not read digits."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()

        images_row = QHBoxLayout()
        self._images_edit = QLineEdit(images_dir)
        images_btn = QPushButton("Browse...")
        images_btn.clicked.connect(self._browse_images)
        images_row.addWidget(self._images_edit)
        images_row.addWidget(images_btn)
        form.addRow("ROI_640 Folder:", self._wrap_layout(images_row))

        labels_row = QHBoxLayout()
        self._labels_edit = QLineEdit(labels_dir)
        labels_btn = QPushButton("Browse...")
        labels_btn.clicked.connect(self._browse_labels)
        labels_row.addWidget(self._labels_edit)
        labels_row.addWidget(labels_btn)
        form.addRow("ROI_640_labels Folder:", self._wrap_layout(labels_row))

        output_row = QHBoxLayout()
        self._output_edit = QLineEdit(output_dir)
        output_btn = QPushButton("Browse...")
        output_btn.clicked.connect(self._browse_output)
        output_row.addWidget(self._output_edit)
        output_row.addWidget(output_btn)
        form.addRow("Output Folder:", self._wrap_layout(output_row))

        backend_row = QHBoxLayout()
        self._backend_edit = QLineEdit(backend_python)
        backend_btn = QPushButton("Browse...")
        backend_btn.clicked.connect(self._browse_backend)
        backend_row.addWidget(self._backend_edit)
        backend_row.addWidget(backend_btn)
        form.addRow("Backend Python:", self._wrap_layout(backend_row))

        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 500)
        self._epochs_spin.setValue(50)
        form.addRow("Epochs:", self._epochs_spin)

        self._image_size_spin = QSpinBox()
        self._image_size_spin.setRange(320, 1280)
        self._image_size_spin.setSingleStep(32)
        self._image_size_spin.setValue(640)
        form.addRow("Image Size:", self._image_size_spin)

        self._batch_size_spin = QSpinBox()
        self._batch_size_spin.setRange(1, 128)
        self._batch_size_spin.setValue(16)
        form.addRow("Batch Size:", self._batch_size_spin)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _wrap_layout(self, row_layout: QHBoxLayout) -> QWidget:
        widget = QWidget()
        widget.setLayout(row_layout)
        return widget

    def _browse_images(self):
        folder = QFileDialog.getExistingDirectory(self, "Select ROI_640 Folder")
        if folder:
            self._images_edit.setText(folder)

    def _browse_labels(self):
        folder = QFileDialog.getExistingDirectory(self, "Select ROI_640_labels Folder")
        if folder:
            self._labels_edit.setText(folder)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select YOLO Output Folder")
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
        images_dir = self._images_edit.text().strip()
        labels_dir = self._labels_edit.text().strip()
        output_dir = self._output_edit.text().strip()
        backend_python = self._backend_edit.text().strip()

        if not images_dir or not Path(images_dir).is_dir():
            QMessageBox.warning(self, "Missing Images", "Select a valid ROI_640 folder.")
            return
        if not labels_dir or not Path(labels_dir).is_dir():
            QMessageBox.warning(self, "Missing Labels", "Select a valid ROI_640_labels folder.")
            return
        if not output_dir:
            QMessageBox.warning(self, "Missing Output", "Select an output folder.")
            return
        if not backend_python or not Path(backend_python).exists():
            QMessageBox.warning(
                self,
                "Missing Backend Python",
                "Select a compatible Python executable for YOLO training.",
            )
            return

        self.accept()

    def get_config(self) -> dict[str, object]:
        return {
            "images_dir": self._images_edit.text().strip(),
            "labels_dir": self._labels_edit.text().strip(),
            "output_dir": self._output_edit.text().strip(),
            "backend_python": self._backend_edit.text().strip(),
            "epochs": int(self._epochs_spin.value()),
            "image_size": int(self._image_size_spin.value()),
            "batch_size": int(self._batch_size_spin.value()),
        }


class AutoReadResultsDialog(QDialog):
    """Show voted auto-read results and the strongest candidates in a roomy modal."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto-Read Results")
        self.setModal(True)
        self.resize(920, 620)

        layout = QVBoxLayout(self)

        self._summary_label = QLabel("No auto-read results yet.")
        self._summary_label.setWordWrap(True)
        self._summary_label.setStyleSheet("font-size: 15px; padding: 4px 0;")
        layout.addWidget(self._summary_label)

        self._details_label = QLabel("")
        self._details_label.setWordWrap(True)
        self._details_label.setStyleSheet("color: #c8d2dc;")
        layout.addWidget(self._details_label)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels([
            "Rank",
            "ROI Candidate",
            "Reading",
            "Score",
            "Confidence",
            "Source",
        ])
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._table)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def apply_results(self, payload: dict[str, object]):
        voted_label = str(payload.get("voted_label", ""))
        best_name = str(payload.get("best_name", ""))
        best_score = float(payload.get("best_score", 0.0))
        candidate_count = int(payload.get("candidate_count", 0))
        expected_label = str(payload.get("expected_label", ""))
        top_candidates = payload.get("top_candidates", [])
        if not isinstance(top_candidates, list):
            top_candidates = []

        if expected_label:
            outcome = "MATCH" if expected_label == voted_label else "MISMATCH"
            self._summary_label.setText(
                f"Voted prediction: {voted_label} | Expected: {expected_label} | Result: {outcome}"
            )
        else:
            self._summary_label.setText(f"Voted prediction: {voted_label}")

        self._details_label.setText(
            f"Best ROI candidate: {best_name} ({best_score:.1f}) | "
            f"Evaluated candidates: {candidate_count} | "
            f"Voting pool: top {len(top_candidates)}"
        )

        self._table.setRowCount(len(top_candidates))
        for row, candidate in enumerate(top_candidates):
            if not isinstance(candidate, dict):
                continue
            confidence_values = candidate.get("confidences", [])
            if isinstance(confidence_values, list):
                confidence_summary = ", ".join(f"{float(v) * 100.0:.1f}%" for v in confidence_values)
            else:
                confidence_summary = ""
            values = [
                str(row + 1),
                str(candidate.get("image_name", "")),
                str(candidate.get("predicted_label", "")),
                f"{float(candidate.get('score', 0.0)) * 100.0:.1f}",
                confidence_summary,
                str(candidate.get("source_name", "")),
            ]
            for col, value in enumerate(values):
                self._table.setItem(row, col, QTableWidgetItem(value))

        self._table.resizeRowsToContents()


class DigitDiagnosisDialog(QDialog):
    """Display individual digit detections and their LeNet predictions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Digit Diagnosis")
        self.setModal(True)
        self.resize(1180, 760)

        layout = QVBoxLayout(self)

        self._summary_label = QLabel("No digit diagnosis results yet.")
        self._summary_label.setWordWrap(True)
        self._summary_label.setStyleSheet("font-size: 15px; padding: 4px 0;")
        layout.addWidget(self._summary_label)

        self._details_label = QLabel("")
        self._details_label.setWordWrap(True)
        self._details_label.setStyleSheet("color: #c8d2dc;")
        layout.addWidget(self._details_label)

        self._image_label = QLabel("No diagnosis overlay yet.")
        self._image_label.setMinimumHeight(320)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("background: #1f1f1f; border: 1px solid #3a3a3a;")

        image_scroll = QScrollArea()
        image_scroll.setWidgetResizable(True)
        image_scroll.setWidget(self._image_label)
        layout.addWidget(image_scroll, stretch=1)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels([
            "Slot",
            "Allowed",
            "Chosen",
            "Confidence",
            "Top Candidates",
        ])
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._table, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def apply_results(self, payload: dict[str, object]):
        detections = payload.get("detections", [])
        if not isinstance(detections, list):
            detections = []

        overlay_path = str(payload.get("overlay_path", "") or "")
        total = int(payload.get("candidate_count", len(detections)))
        confident = int(payload.get("confident_count", 0))
        best_reading = str(payload.get("best_reading", "") or "")
        top_readings = payload.get("top_readings", [])
        if not isinstance(top_readings, list):
            top_readings = []
        restriction_text = str(payload.get("restriction_text", "none") or "none")
        selected_slot_count = int(payload.get("selected_slot_count", 0))
        selected_limiter_index = int(payload.get("selected_limiter_index", -1))
        slot_summaries = payload.get("slot_summaries", [])
        if not isinstance(slot_summaries, list):
            slot_summaries = []

        if best_reading:
            self._summary_label.setText(
                f"Best 5-digit reading: {best_reading} | Top readings: {', '.join(str(v) for v in top_readings[:5])}"
            )
        else:
            self._summary_label.setText(
                f"Digit diagnosis found {total} candidate shapes and classified {len(detections)} of them."
            )
        self._details_label.setText(
            f"Confident guesses (>= 70%): {confident} | "
            f"Restriction: {restriction_text} | "
            f"Chosen slots: {selected_slot_count} | "
            f"Limiter: {selected_limiter_index if selected_limiter_index >= 0 else 'n/a'}"
        )

        pixmap = QPixmap(overlay_path) if overlay_path else QPixmap()
        if not pixmap.isNull():
            self._image_label.setPixmap(pixmap)
            self._image_label.adjustSize()
        else:
            self._image_label.setText("Could not load the diagnosis overlay image.")

        self._table.setRowCount(len(slot_summaries))
        for row, slot in enumerate(slot_summaries):
            if not isinstance(slot, dict):
                continue
            candidate_digits = slot.get("candidate_digits", [])
            if not isinstance(candidate_digits, list):
                candidate_digits = []
            candidate_summary = ", ".join(
                f"{str(item.get('digit', ''))}:{float(item.get('confidence', 0.0)) * 100.0:.0f}%"
                for item in candidate_digits[:3]
                if isinstance(item, dict)
            )
            values = [
                str(slot.get("slot_index", row + 1)),
                str(slot.get("allowed_digits", "")),
                str(slot.get("chosen_digit", "")),
                f"{float(slot.get('chosen_confidence', 0.0)) * 100.0:.1f}%",
                candidate_summary,
            ]
            for col, value in enumerate(values):
                self._table.setItem(row, col, QTableWidgetItem(value))

        self._table.resizeRowsToContents()
