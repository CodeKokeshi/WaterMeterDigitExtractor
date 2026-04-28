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

    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, command: list[str], cwd: str):
        super().__init__()
        self._command = command
        self._cwd = cwd

    def run(self):
        completed_stdout: list[str] = []
        last_json_line = ""
        try:
            process = subprocess.Popen(
                self._command,
                cwd=self._cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
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

        if not last_json_line:
            self.result_ready.emit({})
            return

        try:
            self.result_ready.emit(json.loads(last_json_line))
        except json.JSONDecodeError:
            self.error.emit(stdout)


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
