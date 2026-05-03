"""Train and run a LeNet-style digit classifier for DigitExtractor."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np


SEGMENT_SIZE = 28
NUM_SEGMENTS = 5
FINAL_W = 140
FINAL_H = 28
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".heif"
}

# ── Folder naming ────────────────────────────────────────────────────────────
# New caption-based format:  "1 - Full"  /  "1 - Going 2"  /  "1 - Rolling from 0"
# Legacy format:             plain digit folders "0" … "9"
_CAPTION_FOLDER_RE = re.compile(r"^(\d)\s*-\s*(.+)$")


def log(message: str):
    print(message, flush=True)


# ── Dataset folder helpers ───────────────────────────────────────────────────

def _iter_digit_folders(dataset_dir: Path):
    """Yield (digit, folder_path, caption) for every recognised dataset folder.

    Caption types (new format)
    --------------------------
    Full            - the digit is completely visible (a rolling digit may peek
                      above or below but the label digit dominates the frame).
    Going <X>       - digit X is rising into view from below; the label digit
                      still dominates but X's top is visible.
    Rolling from <X>- digit X above is fading out; the label digit is emerging
                      and dominates the frame.
    """
    for entry in sorted(dataset_dir.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name

        # New format: "1 - Full", "1 - Going 2", "1 - Rolling from 0"
        m = _CAPTION_FOLDER_RE.match(name)
        if m:
            digit = int(m.group(1))
            caption = m.group(2).strip()
            yield digit, entry, caption
            continue

        # Legacy format: plain single-digit folder name
        if name.isdigit() and len(name) == 1:
            yield int(name), entry, "Full"


def count_dataset_images(dataset_dir: Path) -> int:
    """Quickly count images in a dataset directory without loading them."""
    total = 0
    for _digit, folder, _caption in _iter_digit_folders(dataset_dir):
        for img_entry in folder.iterdir():
            if img_entry.is_file() and img_entry.suffix.lower() in IMAGE_EXTENSIONS:
                total += 1
    return total


# ── Adaptive hyperparameter engine ──────────────────────────────────────────

def compute_adaptive_params(n_total: int, validation_split: float) -> dict:
    """Return adaptive training hyperparameters scaled to dataset size.

    The goal is ~100 000 gradient-update steps per training run so the model
    always receives a consistent amount of *work*, whether you have 5 000
    images or 500 000.

    Parameters
    ----------
    n_total:          total image count (before splitting)
    validation_split: fraction reserved for validation (e.g. 0.20)

    Returns
    -------
    dict with keys: batch_size, epochs, dropout_rate,
                    early_stopping_patience, learning_rate
    """
    n_train = max(1, int(n_total * (1.0 - validation_split)))

    # Batch size — larger datasets tolerate bigger batches
    if n_train < 5_000:
        batch_size = 32
    elif n_train < 25_000:
        batch_size = 64
    else:
        batch_size = 128

    # Epochs: target ~100 000 gradient steps total
    TARGET_STEPS = 100_000
    steps_per_epoch = max(1, n_train // batch_size)
    epochs = max(5, min(50, round(TARGET_STEPS / steps_per_epoch)))

    # Dropout — larger datasets provide implicit regularisation
    if n_train < 5_000:
        dropout_rate = 0.50
    elif n_train < 25_000:
        dropout_rate = 0.35
    else:
        dropout_rate = 0.20

    # Early stopping patience (epochs without improvement before halting)
    patience = max(3, min(10, epochs // 5))

    # Adam learning rate — default 1e-3 works well at all scales
    learning_rate = 1e-3

    return {
        "batch_size": batch_size,
        "epochs": epochs,
        "dropout_rate": dropout_rate,
        "early_stopping_patience": patience,
        "learning_rate": learning_rate,
    }


# ── Image / strip helpers ────────────────────────────────────────────────────

def split_strip_segments(strip: np.ndarray) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for i in range(NUM_SEGMENTS):
        x0 = i * SEGMENT_SIZE
        segments.append(strip[:, x0:x0 + SEGMENT_SIZE].copy())
    return segments


def prepare_digit_image(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Empty digit image.")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    interpolation = cv2.INTER_AREA if gray.shape[0] >= SEGMENT_SIZE else cv2.INTER_LINEAR
    resized = cv2.resize(gray, (SEGMENT_SIZE, SEGMENT_SIZE), interpolation=interpolation)
    return resized.astype(np.float32) / 255.0


def prepare_strip_image(path: str) -> tuple[np.ndarray, list[np.ndarray]]:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read strip image: {path}")
    return prepare_strip_array(image)


def prepare_strip_array(image: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    if image is None or image.size == 0:
        raise ValueError("Cannot read strip image.")

    interpolation = cv2.INTER_AREA if image.shape[1] >= FINAL_W else cv2.INTER_LINEAR
    strip = cv2.resize(image, (FINAL_W, FINAL_H), interpolation=interpolation)
    segments = split_strip_segments(strip)
    return strip, segments


# ── Dataset loader ───────────────────────────────────────────────────────────

def load_digit_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Load images from a dataset directory.

    Supports two folder-naming conventions (they can coexist in the same root):

    **New caption-based** (one or more subfolders per digit)::

        1 - Full           digit 1 is clearly visible
        1 - Going 2        digit 2 is rising into view from below; label = 1
        1 - Rolling from 0 digit 0 is fading out above;      label = 1

    **Legacy** (plain digit folders)::

        0   1   2   …   9

    All folders whose name starts with the same digit are merged into that
    digit's class so the label is always the *dominant* digit.
    """
    # Collect folder paths per digit
    digit_folders: dict[int, list[tuple[Path, str]]] = {d: [] for d in range(10)}
    for digit, folder, caption in _iter_digit_folders(dataset_dir):
        digit_folders[digit].append((folder, caption))

    # Every digit class must be represented
    missing = [str(d) for d in range(10) if not digit_folders[d]]
    if missing:
        raise ValueError(
            f"Missing folders for digits: {', '.join(missing)}. "
            "Create at least one folder per digit, e.g. '3 - Full' or plain '3'."
        )

    images: list[np.ndarray] = []
    labels: list[int] = []
    counts: dict[str, int] = {}

    for digit in range(10):
        count = 0
        for folder, _caption in digit_folders[digit]:
            for entry in sorted(folder.iterdir()):
                if not entry.is_file() or entry.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                img = cv2.imread(str(entry), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                images.append(prepare_digit_image(img))
                labels.append(digit)
                count += 1
        counts[str(digit)] = count

    if not images:
        raise ValueError("No readable training images found in the dataset.")

    too_small = [d for d, c in counts.items() if c < 2]
    if too_small:
        raise ValueError(
            "Each digit class needs at least 2 readable images for training. "
            f"Too small: {', '.join(too_small)}"
        )

    x = np.asarray(images, dtype=np.float32)[..., np.newaxis]
    y = np.asarray(labels, dtype=np.int64)
    return x, y, counts


def stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    validation_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for digit in range(10):
        digit_indices = np.where(y == digit)[0]
        if len(digit_indices) == 0:
            continue
        shuffled = rng.permutation(digit_indices)
        val_count = max(1, int(round(len(shuffled) * validation_split)))
        if val_count >= len(shuffled):
            val_count = max(1, len(shuffled) - 1)
        if val_count <= 0:
            train_indices.extend(shuffled.tolist())
            continue
        val_indices.extend(shuffled[:val_count].tolist())
        train_indices.extend(shuffled[val_count:].tolist())

    if not train_indices or not val_indices:
        raise ValueError(
            "Dataset is too small for a train/validation split. "
            "Add more images per digit or reduce the validation split."
        )

    return x[train_indices], y[train_indices], x[val_indices], y[val_indices]


# ── Model builder ────────────────────────────────────────────────────────────

def build_lenet5_model(tf, dropout_rate: float = 0.3):
    """Build LeNet-5 with a configurable dropout layer for regularisation.

    Dropout sits between the two fully-connected layers (Dense 120 → Dense 84)
    where overfitting most often appears in small-to-medium datasets.
    """
    keras = tf.keras
    return keras.Sequential([
        keras.layers.Input(shape=(SEGMENT_SIZE, SEGMENT_SIZE, 1)),
        keras.layers.Conv2D(6, kernel_size=5, padding="same", activation="relu"),
        keras.layers.AveragePooling2D(pool_size=2),
        keras.layers.Conv2D(16, kernel_size=5, activation="relu"),
        keras.layers.AveragePooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(120, activation="relu"),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(84, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])


# ── Training command ─────────────────────────────────────────────────────────

def command_train(args: argparse.Namespace):
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is not available in the selected backend Python. "
            "Use a Python 3.10-3.13 environment with TensorFlow installed."
        ) from exc

    dataset_dir = Path(args.dataset_dir)
    keras_output_dir = Path(args.keras_output_dir)
    tflite_output_dir = Path(args.tflite_output_dir)
    keras_output_dir.mkdir(parents=True, exist_ok=True)
    tflite_output_dir.mkdir(parents=True, exist_ok=True)

    x, y, counts = load_digit_dataset(dataset_dir)
    n_total = len(x)
    log(f"[LeNet] Loaded dataset with {n_total} images.")

    # ── Resolve adaptive hyperparameters ────────────────────────────────────
    adaptive = compute_adaptive_params(n_total, args.validation_split)

    # A value of 0 (or negative for floats) means "let the engine decide"
    epochs        = args.epochs        if args.epochs        > 0    else adaptive["epochs"]
    batch_size    = args.batch_size    if args.batch_size    > 0    else adaptive["batch_size"]
    dropout_rate  = args.dropout_rate  if args.dropout_rate  >= 0.0 else adaptive["dropout_rate"]
    learning_rate = args.learning_rate if args.learning_rate > 0.0  else adaptive["learning_rate"]
    patience      = (
        args.early_stopping_patience
        if args.early_stopping_patience >= 0
        else adaptive["early_stopping_patience"]
    )

    log(
        f"[LeNet] Hyperparams — epochs={epochs}, batch={batch_size}, "
        f"dropout={dropout_rate:.2f}, lr={learning_rate:.5f}, "
        f"early_stop_patience={patience}"
    )

    x_train, y_train, x_val, y_val = stratified_split(
        x, y, validation_split=args.validation_split, seed=args.seed,
    )
    log(
        f"[LeNet] Train/val split ready. Train={len(x_train)}, "
        f"Val={len(x_val)}, Epochs={epochs}, Batch={batch_size}."
    )

    tf.keras.utils.set_random_seed(args.seed)

    model = build_lenet5_model(tf, dropout_rate=dropout_rate)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    callbacks: list = []

    class EpochLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            log(
                "[LeNet] "
                f"Epoch {epoch + 1}/{epochs} "
                f"loss={float(logs.get('loss', 0.0)):.4f} "
                f"acc={float(logs.get('accuracy', 0.0)):.4f} "
                f"val_loss={float(logs.get('val_loss', 0.0)):.4f} "
                f"val_acc={float(logs.get('val_accuracy', 0.0)):.4f}"
            )

    callbacks.append(EpochLogger())

    if patience > 0:
        # Stop early when val_accuracy plateaus
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=patience,
                restore_best_weights=True,
                verbose=0,
            )
        )
        # Halve the learning rate when val_loss stalls
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(2, patience // 2),
                min_lr=1e-6,
                verbose=0,
            )
        )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    actual_epochs = len(history.history["accuracy"])

    test_loss, test_accuracy = model.evaluate(x_val, y_val, verbose=0)
    log(
        f"[LeNet] Evaluation complete. "
        f"test_loss={float(test_loss):.4f} test_acc={float(test_accuracy):.4f} "
        f"(ran {actual_epochs}/{epochs} epochs)"
    )

    keras_path = keras_output_dir / "lenet5_digits.keras"
    model.save(keras_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = tflite_output_dir / "lenet5_digits.tflite"
    tflite_path.write_bytes(tflite_model)

    labels_path = tflite_output_dir / "labels.json"
    labels_path.write_text(json.dumps([str(i) for i in range(10)], indent=2), encoding="utf-8")

    metrics = {
        "dataset_size": int(n_total),
        "class_counts": counts,
        "epochs_planned": int(epochs),
        "epochs_actual": int(actual_epochs),
        "batch_size": int(batch_size),
        "dropout_rate": float(dropout_rate),
        "learning_rate": float(learning_rate),
        "early_stopping_patience": int(patience),
        "train_accuracy": float(history.history["accuracy"][-1]),
        "val_accuracy": float(history.history["val_accuracy"][-1]),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "keras_model_path": str(keras_path),
        "tflite_model_path": str(tflite_path),
        "labels_path": str(labels_path),
    }
    metrics_path = tflite_output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    log("[LeNet] Training complete. Writing final metrics JSON.")
    print(json.dumps(metrics))


# ── Inference commands ───────────────────────────────────────────────────────

def predict_with_keras(model_path: Path, samples: np.ndarray):
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    probs = model.predict(samples, verbose=0)
    return probs.astype(np.float32)


def predict_with_tflite(model_path: Path, samples: np.ndarray):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_index = input_details["index"]
    output_index = output_details["index"]

    outputs = []
    for sample in samples:
        batch = sample[np.newaxis, ...].astype(np.float32)

        if input_details["dtype"] == np.uint8:
            scale, zero_point = input_details["quantization"]
            if scale == 0:
                quantized = np.clip(batch, 0.0, 1.0).astype(np.uint8)
            else:
                quantized = np.clip(np.round(batch / scale + zero_point), 0, 255).astype(np.uint8)
            interpreter.set_tensor(input_index, quantized)
        else:
            interpreter.set_tensor(input_index, batch.astype(input_details["dtype"]))

        interpreter.invoke()
        output = interpreter.get_tensor(output_index)[0]

        if output_details["dtype"] == np.uint8:
            scale, zero_point = output_details["quantization"]
            output = (output.astype(np.float32) - zero_point) * scale
        outputs.append(output.astype(np.float32))

    return np.asarray(outputs, dtype=np.float32)


def command_predict(args: argparse.Namespace):
    try:
        import tensorflow as tf  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is not available in the selected backend Python. "
            "Use a Python 3.10-3.13 environment with TensorFlow installed."
        ) from exc

    model_path = Path(args.model_path)
    _strip, segments = prepare_strip_image(args.image_path)
    log(f"[LeNet] Predicting 5-digit strip from {args.image_path}")
    if args.invert_input:
        segments = [cv2.bitwise_not(seg) for seg in segments]
    samples = np.asarray([prepare_digit_image(seg) for seg in segments], dtype=np.float32)[..., np.newaxis]

    if model_path.suffix.lower() == ".tflite":
        probs = predict_with_tflite(model_path, samples)
    else:
        probs = predict_with_keras(model_path, samples)

    predicted_digits = np.argmax(probs, axis=1)
    predicted_label = "".join(str(int(digit)) for digit in predicted_digits)
    confidences = [float(probs[i, predicted_digits[i]]) for i in range(len(predicted_digits))]

    # Full per-segment softmax (5 x 10) so the UI can do top-2 analysis
    # for the rolling-context reader (Full / Going X / Rolling from X).
    # Rounded to 6 decimals to keep the JSON small.
    probs_list = [
        [round(float(v), 6) for v in row]
        for row in probs
    ]

    result = {
        "predicted_label": predicted_label,
        "expected_label": args.expected_label or "",
        "confidences": confidences,
        "probs": probs_list,
    }
    log(f"[LeNet] Prediction complete: {predicted_label}")
    print(json.dumps(result))


def command_predict_batch(args: argparse.Namespace):
    try:
        import tensorflow as tf  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is not available in the selected backend Python. "
            "Use a Python 3.10-3.13 environment with TensorFlow installed."
        ) from exc

    model_path = Path(args.model_path)
    images_dir = Path(args.images_dir)
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError("No strip candidate images were found.")
    log(f"[LeNet] Predict-batch on {len(image_paths)} strip candidates from {images_dir}")

    samples: list[np.ndarray] = []
    sample_ranges: list[tuple[str, int, int]] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        strip, segments = prepare_strip_array(image)
        if args.invert_input:
            segments = [cv2.bitwise_not(seg) for seg in segments]
        start = len(samples)
        samples.extend(prepare_digit_image(seg) for seg in segments)
        end = len(samples)
        sample_ranges.append((image_path.stem, start, end))

    sample_array = np.asarray(samples, dtype=np.float32)[..., np.newaxis]
    if model_path.suffix.lower() == ".tflite":
        probs = predict_with_tflite(model_path, sample_array)
    else:
        probs = predict_with_keras(model_path, sample_array)

    candidates = []
    best_candidate = None
    best_score = -1.0
    for image_name, start, end in sample_ranges:
        subset = probs[start:end]
        predicted_digits = np.argmax(subset, axis=1)
        predicted_label = "".join(str(int(digit)) for digit in predicted_digits)
        confidences = [
            float(subset[i, predicted_digits[i]])
            for i in range(len(predicted_digits))
        ]
        score = (0.65 * float(np.mean(confidences))) + (0.35 * float(np.min(confidences)))
        candidate = {
            "image_name": image_name,
            "predicted_label": predicted_label,
            "confidences": confidences,
            "score": score,
        }
        candidates.append(candidate)
        if score > best_score:
            best_score = score
            best_candidate = candidate

    candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    if best_candidate is not None:
        log(
            f"[LeNet] Batch prediction complete. Best={best_candidate['predicted_label']} "
            f"from {best_candidate['image_name']} score={float(best_candidate['score']):.4f}"
        )

    print(json.dumps({
        "best": best_candidate or {},
        "candidates": candidates,
    }))


def command_predict_digits(args: argparse.Namespace):
    try:
        import tensorflow as tf  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is not available in the selected backend Python. "
            "Use a Python 3.10-3.13 environment with TensorFlow installed."
        ) from exc

    model_path = Path(args.model_path)
    images_dir = Path(args.images_dir)
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError("No digit candidate images were found.")

    log(f"[LeNet] Predict-digits on {len(image_paths)} candidates from {images_dir}")

    samples: list[np.ndarray] = []
    valid_names: list[str] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        sample = cv2.bitwise_not(image) if args.invert_input else image
        samples.append(prepare_digit_image(sample))
        valid_names.append(image_path.stem)

    if not samples:
        raise ValueError("No readable digit candidate images were found.")

    sample_array = np.asarray(samples, dtype=np.float32)[..., np.newaxis]
    if model_path.suffix.lower() == ".tflite":
        probs = predict_with_tflite(model_path, sample_array)
    else:
        probs = predict_with_keras(model_path, sample_array)

    candidates = []
    for idx, image_name in enumerate(valid_names):
        prob = probs[idx]
        predicted_digit = int(np.argmax(prob))
        confidence = float(prob[predicted_digit])
        candidates.append({
            "image_name": image_name,
            "predicted_label": str(predicted_digit),
            "confidence": confidence,
            "score": confidence,
        })

    candidates.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    print(json.dumps({"candidates": candidates}))


# ── Argument parser ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset-dir", required=True)
    train_parser.add_argument("--keras-output-dir", required=True)
    train_parser.add_argument("--tflite-output-dir", required=True)
    train_parser.add_argument(
        "--epochs", type=int, default=0,
        help="Training epochs. 0 = auto-adapt based on dataset size.",
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=0,
        help="Mini-batch size. 0 = auto-adapt based on dataset size.",
    )
    train_parser.add_argument(
        "--dropout-rate", type=float, default=-1.0,
        help="Dropout rate 0-1. Negative = auto-adapt based on dataset size.",
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=0.0,
        help="Adam learning rate. 0 = auto-adapt (default 0.001).",
    )
    train_parser.add_argument(
        "--early-stopping-patience", type=int, default=-1,
        help="Epochs without improvement before stopping. Negative = auto-adapt. 0 = disabled.",
    )
    train_parser.add_argument("--validation-split", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model-path", required=True)
    predict_parser.add_argument("--image-path", required=True)
    predict_parser.add_argument("--expected-label", default="")
    predict_parser.add_argument("--invert-input", action="store_true")

    batch_predict_parser = subparsers.add_parser("predict-batch")
    batch_predict_parser.add_argument("--model-path", required=True)
    batch_predict_parser.add_argument("--images-dir", required=True)
    batch_predict_parser.add_argument("--invert-input", action="store_true")

    predict_digits_parser = subparsers.add_parser("predict-digits")
    predict_digits_parser.add_argument("--model-path", required=True)
    predict_digits_parser.add_argument("--images-dir", required=True)
    predict_digits_parser.add_argument("--invert-input", action="store_true")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "train":
            command_train(args)
            return
        if args.command == "predict":
            command_predict(args)
            return
        if args.command == "predict-batch":
            command_predict_batch(args)
            return
        if args.command == "predict-digits":
            command_predict_digits(args)
            return
        raise ValueError(f"Unsupported command: {args.command}")
    except Exception as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
