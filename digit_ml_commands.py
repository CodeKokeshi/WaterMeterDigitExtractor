"""Helpers for running external LeNet training and inference backends."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


# ---------------------------------------------------------------------------
# Output parsers used by MlCommandWorker to extract real progress
# ---------------------------------------------------------------------------
# Matches Keras / TensorFlow training output:
#   "Epoch 3/10"
_EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)")
# Matches the Keras step-progress prefix at the start of a progress line:
#   "  42/200 [..." or "200/200 [..."  ->  current=42, total=200
_STEP_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\b")
# Stage hints that the backend prints (look for substring, case-insensitive).
# Each stage maps to a friendly label and a coarse percentage milestone so the
# bar moves forward sensibly even before any epoch line shows up.
_STAGE_HINTS = (
    ("Loading dataset",              "Loading dataset…",            2),
    ("Loaded dataset",               "Dataset loaded",              5),
    ("Building model",               "Building model…",             7),
    ("Compiling model",              "Compiling model…",            8),
    ("Starting training",            "Starting training…",          10),
    ("Evaluating",                   "Evaluating model…",           94),
    ("Saving",                       "Saving model…",               96),
    ("Exporting",                    "Exporting TFLite…",           98),
    ("Prediction complete",          "Prediction complete",         100),
    ("training complete",            "Training complete",           100),
)


class MlCommandWorker(QThread):
    """Run an external ML helper command without freezing the UI.

    Signals
    -------
    result_ready : object
        Emitted with the decoded JSON dict from the final stdout line.
    error : str
        Emitted with the worker's accumulated stdout/stderr on failure.
    log : str
        Emitted once per output line, useful for status-bar tickers.
    progress : (int current, int total, str message)
        Emitted whenever the worker can extract real progress from the
        backend output.  ``total == 0`` means "indeterminate" — keep the
        spinner.  ``total == 100`` means "treat current as a percentage".
    """

    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)

    def __init__(self, command: list[str], cwd: str):
        super().__init__()
        self._command = command
        self._cwd = cwd

    # -- run loop ------------------------------------------------------------

    def run(self):
        completed_stdout: list[str] = []
        last_json_line = ""

        # Per-run progress state.
        current_epoch = 0
        total_epochs = 0
        last_pct = -1

        try:
            process = subprocess.Popen(
                self._command,
                cwd=self._cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                # CRITICAL: TensorFlow / Keras on Windows occasionally emit
                # bytes in cp1252 (em-dashes, curly quotes, …) which break
                # a strict utf-8 decode.  Replace bad bytes instead of
                # raising — the worker is a status pipe, not a parser.
                errors="replace",
                bufsize=1,
            )
        except Exception as exc:
            self.error.emit(str(exc))
            return

        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            completed_stdout.append(line)
            self.log.emit(line)

            # 1) Stage hints — coarse milestones independent of epoch info.
            stage_match = self._match_stage(line)
            if stage_match is not None:
                stage_label, stage_pct = stage_match
                if stage_pct > last_pct:
                    last_pct = stage_pct
                    self.progress.emit(stage_pct, 100, stage_label)

            # 2) Epoch headers — "Epoch X/Y".
            m_epoch = _EPOCH_RE.search(line)
            if m_epoch is not None:
                current_epoch = int(m_epoch.group(1))
                total_epochs = int(m_epoch.group(2))
                # Whole-epochs progress: 10..90 % range so we leave room
                # for the loading and saving stages on either side.
                if total_epochs > 0:
                    pct = 10 + int(
                        80.0 * (current_epoch - 1) / max(total_epochs, 1)
                    )
                    pct = max(min(pct, 90), last_pct)
                    last_pct = pct
                    self.progress.emit(
                        pct,
                        100,
                        f"Epoch {current_epoch} of {total_epochs}",
                    )
                continue

            # 3) Per-step inside the current epoch — "42/200 [...]".
            #    Only count it if we already know the epoch context; this
            #    avoids picking up unrelated "N/M" patterns elsewhere.
            if total_epochs > 0 and current_epoch > 0:
                m_step = _STEP_RE.match(line)
                if m_step is not None:
                    step = int(m_step.group(1))
                    total_steps = int(m_step.group(2))
                    if total_steps > 0:
                        epoch_frac = (current_epoch - 1) / max(total_epochs, 1)
                        within = step / total_steps / max(total_epochs, 1)
                        pct = 10 + int(80.0 * (epoch_frac + within))
                        pct = max(min(pct, 90), last_pct)
                        if pct > last_pct:
                            last_pct = pct
                            self.progress.emit(
                                pct,
                                100,
                                f"Epoch {current_epoch}/{total_epochs}"
                                f" — step {step}/{total_steps}",
                            )

            # 4) JSON capture for the final result dictionary.
            try:
                json.loads(line)
                last_json_line = line
            except json.JSONDecodeError:
                pass

        process.stdout.close()
        return_code = process.wait()
        stdout = "\n".join(completed_stdout).strip()

        if return_code != 0:
            self.error.emit(stdout or "Unknown ML backend error.")
            return

        # Final progress nudge so the bar always reaches 100 on success.
        self.progress.emit(100, 100, "Done")

        if not last_json_line:
            self.result_ready.emit({})
            return

        try:
            self.result_ready.emit(json.loads(last_json_line))
        except json.JSONDecodeError:
            self.error.emit(stdout)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _match_stage(line: str) -> tuple[str, int] | None:
        """Return (friendly_label, percentage_milestone) for a stage hint."""
        lower = line.lower()
        for needle, label, pct in _STAGE_HINTS:
            if needle.lower() in lower:
                return label, pct
        return None


def get_ml_backend_script_path() -> Path:
    return Path(__file__).resolve().parent / "lenet_backend.py"


def get_yolo_backend_script_path() -> Path:
    return Path(__file__).resolve().parent / "yolo_backend.py"


def get_python_version(executable: str) -> tuple[int, int] | None:
    try:
        completed = subprocess.run(
            [
                executable,
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
    except Exception:
        return None

    if completed.returncode != 0:
        return None

    raw = completed.stdout.strip()
    try:
        major_str, minor_str = raw.split(".", 1)
        return int(major_str), int(minor_str)
    except Exception:
        return None


def is_supported_tensorflow_backend(version: tuple[int, int] | None) -> bool:
    if version is None:
        return False
    return (3, 10) <= version <= (3, 13)


def build_lenet_train_command(
    backend_python: str,
    dataset_dir: str,
    keras_output_dir: str,
    tflite_output_dir: str,
    epochs: int = 0,
    batch_size: int = 0,
    validation_split: float = 0.2,
    seed: int = 42,
    dropout_rate: float = -1.0,
    learning_rate: float = 0.0,
    early_stopping_patience: int = -1,
) -> list[str]:
    """Build the CLI command for LeNet training.

    Pass 0 for *epochs* or *batch_size* to let the backend auto-adapt from
    the dataset size.  Pass negative values for *dropout_rate* or
    *early_stopping_patience* to auto-adapt those as well.
    """
    return [
        backend_python,
        str(get_ml_backend_script_path()),
        "train",
        "--dataset-dir", dataset_dir,
        "--keras-output-dir", keras_output_dir,
        "--tflite-output-dir", tflite_output_dir,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--dropout-rate", str(dropout_rate),
        "--learning-rate", str(learning_rate),
        "--early-stopping-patience", str(early_stopping_patience),
        "--validation-split", str(validation_split),
        "--seed", str(seed),
    ]


def build_lenet_predict_command(
    backend_python: str,
    model_path: str,
    image_path: str,
    expected_label: str = "",
    invert_input: bool = False,
) -> list[str]:
    command = [
        backend_python,
        str(get_ml_backend_script_path()),
        "predict",
        "--model-path", model_path,
        "--image-path", image_path,
    ]
    if expected_label:
        command.extend(["--expected-label", expected_label])
    if invert_input:
        command.append("--invert-input")
    return command


def build_lenet_predict_batch_command(
    backend_python: str,
    model_path: str,
    images_dir: str,
    invert_input: bool = False,
) -> list[str]:
    command = [
        backend_python,
        str(get_ml_backend_script_path()),
        "predict-batch",
        "--model-path", model_path,
        "--images-dir", images_dir,
    ]
    if invert_input:
        command.append("--invert-input")
    return command


def build_lenet_predict_digits_command(
    backend_python: str,
    model_path: str,
    images_dir: str,
    invert_input: bool = False,
) -> list[str]:
    command = [
        backend_python,
        str(get_ml_backend_script_path()),
        "predict-digits",
        "--model-path", model_path,
        "--images-dir", images_dir,
    ]
    if invert_input:
        command.append("--invert-input")
    return command


def build_yolo_train_command(
    backend_python: str,
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    epochs: int,
    image_size: int,
    batch_size: int,
) -> list[str]:
    return [
        backend_python,
        str(get_yolo_backend_script_path()),
        "train",
        "--images-dir", images_dir,
        "--labels-dir", labels_dir,
        "--output-dir", output_dir,
        "--epochs", str(epochs),
        "--image-size", str(image_size),
        "--batch-size", str(batch_size),
    ]


def build_yolo_predict_command(
    backend_python: str,
    model_path: str,
    image_path: str,
    image_size: int = 640,
    conf_threshold: float = 0.25,
) -> list[str]:
    return [
        backend_python,
        str(get_yolo_backend_script_path()),
        "predict",
        "--model-path", model_path,
        "--image-path", image_path,
        "--image-size", str(image_size),
        "--conf-threshold", str(conf_threshold),
    ]


def build_yolo_predict_windows_command(
    backend_python: str,
    model_path: str,
    image_path: str,
    image_size: int = 640,
    conf_threshold: float = 0.25,
) -> list[str]:
    return [
        backend_python,
        str(get_yolo_backend_script_path()),
        "predict-windows",
        "--model-path", model_path,
        "--image-path", image_path,
        "--image-size", str(image_size),
        "--conf-threshold", str(conf_threshold),
    ]


def write_temp_image(image: np.ndarray, prefix: str = "img_") -> str:
    temp_dir = Path(tempfile.gettempdir()) / "digit_extractor_testing"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".png",
        prefix=prefix,
        dir=temp_dir,
        delete=False,
    )
    temp_file.close()
    cv2.imwrite(temp_file.name, image)
    return temp_file.name


def write_temp_strip_image(strip: np.ndarray) -> str:
    return write_temp_image(strip, prefix="strip_")


def write_temp_images(images: list[np.ndarray], prefix: str = "img_") -> str:
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=tempfile.gettempdir()))
    for idx, image in enumerate(images):
        image_path = temp_dir / f"{idx:03d}.png"
        cv2.imwrite(str(image_path), image)
    return str(temp_dir)
