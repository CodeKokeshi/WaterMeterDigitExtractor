import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from pillow_heif import register_heif_opener
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


register_heif_opener()

SUPPORTED_EXTENSIONS = {".heic", ".heif"}


@dataclass(frozen=True)
class ConversionJob:
    source_path: str
    output_path: str
    output_format: str
    jpeg_quality: int


def discover_heic_files(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_output_path(
    source_path: Path,
    input_dir: Path,
    output_dir: Path,
    output_extension: str,
    preserve_structure: bool,
) -> Path:
    if preserve_structure:
        relative_dir = source_path.relative_to(input_dir).parent
        target_dir = output_dir / relative_dir
    else:
        target_dir = output_dir

    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = target_dir / f"{source_path.stem}{output_extension}"
    if not target_path.exists():
        return target_path

    counter = 1
    while True:
        candidate = target_dir / f"{source_path.stem}_{counter}{output_extension}"
        if not candidate.exists():
            return candidate
        counter += 1


def convert_one(job: ConversionJob) -> tuple[bool, str]:
    try:
        source_path = Path(job.source_path)
        output_path = Path(job.output_path)

        with Image.open(source_path) as image:
            image.load()

            if job.output_format == "JPEG":
                if image.mode not in ("RGB", "L"):
                    image = image.convert("RGB")
                image.save(
                    output_path,
                    format="JPEG",
                    quality=job.jpeg_quality,
                    optimize=True,
                )
            else:
                image.save(output_path, format="PNG", optimize=True)

        return True, f"OK  | {source_path.name} -> {output_path.name}"
    except Exception:
        error_text = traceback.format_exc(limit=3)
        return False, f"ERR | {job.source_path}\n{error_text}"


class ConversionWorker(QThread):
    progress_changed = pyqtSignal(int, int)
    log_message = pyqtSignal(str)
    finished_summary = pyqtSignal(int, int)

    def __init__(self, jobs: list[ConversionJob], workers: int):
        super().__init__()
        self.jobs = jobs
        self.workers = workers

    def run(self) -> None:
        success_count = 0
        total = len(self.jobs)

        if total == 0:
            self.finished_summary.emit(0, 0)
            return

        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(convert_one, job): job for job in self.jobs}

            completed = 0
            for future in as_completed(futures):
                try:
                    ok, message = future.result()
                except Exception:
                    ok = False
                    failed_job = futures[future]
                    message = (
                        f"ERR | {failed_job.source_path}\n"
                        f"{traceback.format_exc(limit=3)}"
                    )
                completed += 1
                if ok:
                    success_count += 1
                self.log_message.emit(message)
                self.progress_changed.emit(completed, total)

        self.finished_summary.emit(success_count, total)


class HeicBatchConverterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker: ConversionWorker | None = None
        self.setWindowTitle("HEIC Batch Converter")
        self.resize(860, 560)
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(14)

        title = QLabel("Batch HEIC / HEIF Converter")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        main_layout.addWidget(title)

        subtitle = QLabel(
            "Choose an input folder with HEIC images, pick an output folder, then process them in parallel."
        )
        subtitle.setWordWrap(True)
        main_layout.addWidget(subtitle)

        folder_layout = QGridLayout()
        folder_layout.setHorizontalSpacing(10)
        folder_layout.setVerticalSpacing(10)

        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()

        input_button = QPushButton("Open Input Folder")
        input_button.clicked.connect(self.choose_input_folder)

        output_button = QPushButton("Set Output Folder")
        output_button.clicked.connect(self.choose_output_folder)

        folder_layout.addWidget(QLabel("Input Folder"), 0, 0)
        folder_layout.addWidget(self.input_edit, 0, 1)
        folder_layout.addWidget(input_button, 0, 2)

        folder_layout.addWidget(QLabel("Output Folder"), 1, 0)
        folder_layout.addWidget(self.output_edit, 1, 1)
        folder_layout.addWidget(output_button, 1, 2)

        main_layout.addLayout(folder_layout)

        options_layout = QFormLayout()

        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPEG"])
        self.format_combo.currentTextChanged.connect(self._sync_quality_state)

        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(95)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, max(1, os.cpu_count() or 1))
        self.workers_spin.setValue(max(1, os.cpu_count() or 1))

        self.recursive_checkbox = QCheckBox("Include subfolders")
        self.recursive_checkbox.setChecked(False)

        self.preserve_structure_checkbox = QCheckBox("Preserve folder structure in output")
        self.preserve_structure_checkbox.setChecked(True)

        options_layout.addRow("Output Format", self.format_combo)
        options_layout.addRow("JPEG Quality", self.quality_spin)
        options_layout.addRow("Parallel Workers", self.workers_spin)
        options_layout.addRow("", self.recursive_checkbox)
        options_layout.addRow("", self.preserve_structure_checkbox)

        main_layout.addLayout(options_layout)

        stats_row = QHBoxLayout()
        self.file_count_label = QLabel("Files found: 0")
        self.cpu_hint_label = QLabel(
            f"Detected CPU cores: {max(1, os.cpu_count() or 1)}"
        )
        stats_row.addWidget(self.file_count_label)
        stats_row.addStretch(1)
        stats_row.addWidget(self.cpu_hint_label)
        main_layout.addLayout(stats_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        button_row = QHBoxLayout()
        self.scan_button = QPushButton("Scan Folder")
        self.scan_button.clicked.connect(self.scan_folder)
        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.process_files)

        button_row.addWidget(self.scan_button)
        button_row.addWidget(self.process_button)
        button_row.addStretch(1)
        main_layout.addLayout(button_row)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        main_layout.addWidget(self.log_output, stretch=1)

        self._sync_quality_state(self.format_combo.currentText())

    def _sync_quality_state(self, selected_format: str) -> None:
        self.quality_spin.setEnabled(selected_format == "JPEG")

    def choose_input_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.input_edit.setText(folder)
            self.scan_folder()

    def choose_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_edit.setText(folder)

    def _get_input_dir(self) -> Path | None:
        text = self.input_edit.text().strip()
        if not text:
            return None
        return Path(text)

    def _get_output_dir(self) -> Path | None:
        text = self.output_edit.text().strip()
        if not text:
            return None
        return Path(text)

    def scan_folder(self) -> None:
        input_dir = self._get_input_dir()
        if input_dir is None or not input_dir.is_dir():
            self.file_count_label.setText("Files found: 0")
            return

        files = discover_heic_files(input_dir, self.recursive_checkbox.isChecked())
        self.file_count_label.setText(f"Files found: {len(files)}")
        self.log_output.appendPlainText(
            f"Scanned {input_dir} and found {len(files)} HEIC/HEIF file(s)."
        )

    def _collect_jobs(self) -> list[ConversionJob]:
        input_dir = self._get_input_dir()
        output_dir = self._get_output_dir()
        if input_dir is None or not input_dir.is_dir():
            raise ValueError("Please choose a valid input folder.")
        if output_dir is None:
            raise ValueError("Please choose an output folder.")
        if input_dir.resolve() == output_dir.resolve():
            raise ValueError("Input and output folders must be different.")

        output_dir.mkdir(parents=True, exist_ok=True)

        files = discover_heic_files(input_dir, self.recursive_checkbox.isChecked())
        self.file_count_label.setText(f"Files found: {len(files)}")
        if not files:
            raise ValueError("No HEIC/HEIF files were found in the selected folder.")

        output_format = self.format_combo.currentText()
        extension = ".jpg" if output_format == "JPEG" else ".png"

        jobs: list[ConversionJob] = []
        for file_path in files:
            output_path = build_output_path(
                source_path=file_path,
                input_dir=input_dir,
                output_dir=output_dir,
                output_extension=extension,
                preserve_structure=self.preserve_structure_checkbox.isChecked(),
            )
            jobs.append(
                ConversionJob(
                    source_path=str(file_path),
                    output_path=str(output_path),
                    output_format=output_format,
                    jpeg_quality=self.quality_spin.value(),
                )
            )
        return jobs

    def process_files(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Processing", "Conversion is already running.")
            return

        try:
            jobs = self._collect_jobs()
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot Start", str(exc))
            return

        workers = self.workers_spin.value()
        self.progress_bar.setValue(0)
        self.log_output.clear()
        self.log_output.appendPlainText(
            f"Starting conversion for {len(jobs)} file(s) with {workers} worker(s)..."
        )

        self._set_controls_enabled(False)
        self.worker = ConversionWorker(jobs=jobs, workers=workers)
        self.worker.progress_changed.connect(self._on_progress_changed)
        self.worker.log_message.connect(self.log_output.appendPlainText)
        self.worker.finished_summary.connect(self._on_finished_summary)
        self.worker.start()

    def _on_progress_changed(self, completed: int, total: int) -> None:
        percent = int((completed / total) * 100) if total else 0
        self.progress_bar.setValue(percent)

    def _on_finished_summary(self, success_count: int, total: int) -> None:
        self._set_controls_enabled(True)
        failed_count = total - success_count
        self.progress_bar.setValue(100 if total else 0)
        self.log_output.appendPlainText("")
        self.log_output.appendPlainText(
            f"Finished. Success: {success_count} | Failed: {failed_count} | Total: {total}"
        )

        if failed_count == 0:
            QMessageBox.information(
                self,
                "Conversion Complete",
                f"Converted {success_count} file(s) successfully.",
            )
        else:
            QMessageBox.warning(
                self,
                "Conversion Finished With Errors",
                f"Converted {success_count} file(s). Failed: {failed_count}. Check the log for details.",
            )

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.input_edit.setEnabled(enabled)
        self.output_edit.setEnabled(enabled)
        self.format_combo.setEnabled(enabled)
        self.quality_spin.setEnabled(enabled and self.format_combo.currentText() == "JPEG")
        self.workers_spin.setEnabled(enabled)
        self.recursive_checkbox.setEnabled(enabled)
        self.preserve_structure_checkbox.setEnabled(enabled)
        self.scan_button.setEnabled(enabled)
        self.process_button.setEnabled(enabled)


def main() -> int:
    app = QApplication(sys.argv)
    window = HeicBatchConverterWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
