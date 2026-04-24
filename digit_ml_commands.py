"""Helpers for running external LeNet training and inference backends."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


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
            self.error.emit(stderr or stdout or "Unknown ML backend error.")
            return

        if not stdout:
            self.finished.emit({})
            return

        try:
            self.finished.emit(json.loads(stdout))
        except json.JSONDecodeError:
            self.error.emit(stdout)


def get_ml_backend_script_path() -> Path:
    return Path(__file__).resolve().parent / "lenet_backend.py"


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
    epochs: int,
    batch_size: int,
    validation_split: float,
    seed: int,
) -> list[str]:
    return [
        backend_python,
        str(get_ml_backend_script_path()),
        "train",
        "--dataset-dir", dataset_dir,
        "--keras-output-dir", keras_output_dir,
        "--tflite-output-dir", tflite_output_dir,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--validation-split", str(validation_split),
        "--seed", str(seed),
    ]


def build_lenet_predict_command(
    backend_python: str,
    model_path: str,
    image_path: str,
    expected_label: str = "",
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
    return command


def write_temp_strip_image(strip: np.ndarray) -> str:
    temp_dir = Path(tempfile.gettempdir()) / "digit_extractor_testing"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".png",
        prefix="strip_",
        dir=temp_dir,
        delete=False,
    )
    temp_file.close()
    cv2.imwrite(temp_file.name, strip)
    return temp_file.name
