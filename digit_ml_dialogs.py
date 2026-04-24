"""Dialogs for configuring LeNet training in DigitExtractor."""

from __future__ import annotations

from pathlib import Path

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
    QSpinBox,
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
