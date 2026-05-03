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

from lenet_backend import count_dataset_images, compute_adaptive_params


class LeNetTrainingDialog(QDialog):
    """Collect parameters for LeNet-style digit training.

    Supports the new caption-based dataset layout::

        1 - Full           digit 1, full visibility
        1 - Going 2        digit 2 rising into view; label = 1
        1 - Rolling from 0 digit 0 fading out above; label = 1

    as well as the legacy plain-digit folder layout (0, 1, … 9).

    Use **Scan & Auto-Tune** to count all images in the selected folder and
    pre-fill Epochs, Batch Size, Dropout, and Learning Rate with values
    optimised for that dataset size.  All fields remain editable.
    """

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
        self.setMinimumWidth(800)

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Train a LeNet-style digit classifier and export both a TensorFlow/Keras "
            "model and a TFLite model. Dataset folders may use the new caption format "
            "('1 - Full', '1 - Going 2', '1 - Rolling from 0') or the legacy plain-digit "
            "format ('0' … '9')."
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

        # ── Auto-tune banner ───────────────────────────────────────────────
        autotune_row = QHBoxLayout()
        self._autotune_label = QLabel("Click 'Scan & Auto-Tune' after selecting a dataset folder.")
        self._autotune_label.setStyleSheet("color: #8ecae6;")
        self._autotune_label.setWordWrap(True)
        autotune_btn = QPushButton("Scan && Auto-Tune")
        autotune_btn.setToolTip(
            "Counts images in the dataset folder and fills Epochs, Batch Size, "
            "Dropout Rate, and Learning Rate with values optimised for that size."
        )
        autotune_btn.clicked.connect(self._auto_tune)
        autotune_row.addWidget(self._autotune_label, stretch=1)
        autotune_row.addWidget(autotune_btn)
        layout.addLayout(autotune_row)

        form = QFormLayout()

        # Dataset folder
        dataset_row = QHBoxLayout()
        self._dataset_edit = QLineEdit(dataset_dir)
        dataset_btn = QPushButton("Browse...")
        dataset_btn.clicked.connect(self._browse_dataset)
        dataset_row.addWidget(self._dataset_edit)
        dataset_row.addWidget(dataset_btn)
        form.addRow("Dataset Folder:", self._wrap_layout(dataset_row))

        # Output folders
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

        # Backend Python
        backend_row = QHBoxLayout()
        self._backend_edit = QLineEdit(backend_python)
        backend_btn = QPushButton("Browse...")
        backend_btn.clicked.connect(self._browse_backend)
        backend_row.addWidget(self._backend_edit)
        backend_row.addWidget(backend_btn)
        form.addRow("Backend Python:", self._wrap_layout(backend_row))

        # ── Adaptive hyperparameters ───────────────────────────────────────
        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 500)
        self._epochs_spin.setValue(20)
        self._epochs_spin.setToolTip("Number of full passes through the training set.")
        form.addRow("Epochs:", self._epochs_spin)

        self._batch_size_spin = QSpinBox()
        self._batch_size_spin.setRange(4, 512)
        self._batch_size_spin.setValue(32)
        self._batch_size_spin.setToolTip("Images processed per gradient update.")
        form.addRow("Batch Size:", self._batch_size_spin)

        self._dropout_spin = QDoubleSpinBox()
        self._dropout_spin.setRange(0.0, 0.9)
        self._dropout_spin.setSingleStep(0.05)
        self._dropout_spin.setDecimals(2)
        self._dropout_spin.setValue(0.30)
        self._dropout_spin.setToolTip(
            "Fraction of units randomly dropped during training to reduce overfitting. "
            "Smaller datasets benefit from higher dropout."
        )
        form.addRow("Dropout Rate:", self._dropout_spin)

        self._lr_spin = QDoubleSpinBox()
        self._lr_spin.setRange(0.000001, 0.1)
        self._lr_spin.setSingleStep(0.0001)
        self._lr_spin.setDecimals(6)
        self._lr_spin.setValue(0.001)
        self._lr_spin.setToolTip("Adam optimiser learning rate.")
        form.addRow("Learning Rate:", self._lr_spin)

        self._patience_spin = QSpinBox()
        self._patience_spin.setRange(0, 50)
        self._patience_spin.setValue(5)
        self._patience_spin.setToolTip(
            "Epochs without val_accuracy improvement before stopping early. 0 = disabled."
        )
        form.addRow("Early Stop Patience:", self._patience_spin)

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

    # ── Helpers ────────────────────────────────────────────────────────────

    def _wrap_layout(self, row_layout: QHBoxLayout) -> QWidget:
        widget = QWidget()
        widget.setLayout(row_layout)
        return widget

    def _auto_tune(self):
        """Scan the selected dataset folder and fill in adaptive hyperparameters."""
        dataset_dir = self._dataset_edit.text().strip()
        if not dataset_dir or not Path(dataset_dir).is_dir():
            QMessageBox.warning(
                self, "No Dataset", "Select a valid dataset folder first, then scan."
            )
            return

        try:
            n_total = count_dataset_images(Path(dataset_dir))
        except Exception as exc:
            QMessageBox.warning(self, "Scan Failed", str(exc))
            return

        if n_total == 0:
            QMessageBox.warning(
                self, "Empty Dataset",
                "No recognised image files found in the dataset folder."
            )
            return

        validation_split = float(self._validation_spin.value())
        params = compute_adaptive_params(n_total, validation_split)

        self._epochs_spin.setValue(params["epochs"])
        self._batch_size_spin.setValue(params["batch_size"])
        self._dropout_spin.setValue(round(params["dropout_rate"], 2))
        self._lr_spin.setValue(params["learning_rate"])
        self._patience_spin.setValue(params["early_stopping_patience"])

        self._autotune_label.setText(
            f"Auto-tuned for {n_total:,} images — "
            f"epochs={params['epochs']}, batch={params['batch_size']}, "
            f"dropout={params['dropout_rate']:.2f}, "
            f"patience={params['early_stopping_patience']}."
        )

    # ── Browse callbacks ───────────────────────────────────────────────────

    def _browse_dataset(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Dataset Folder (caption-based or plain 0-9)"
        )
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

    # ── Validation & result ────────────────────────────────────────────────

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
            "dropout_rate": float(self._dropout_spin.value()),
            "learning_rate": float(self._lr_spin.value()),
            "early_stopping_patience": int(self._patience_spin.value()),
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
