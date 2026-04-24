"""Train and run a LeNet-style digit classifier for DigitExtractor."""

from __future__ import annotations

import argparse
import json
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

    interpolation = cv2.INTER_AREA if image.shape[1] >= FINAL_W else cv2.INTER_LINEAR
    strip = cv2.resize(image, (FINAL_W, FINAL_H), interpolation=interpolation)
    segments = split_strip_segments(strip)
    return strip, segments


def load_digit_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    counts: dict[str, int] = {}

    for digit in range(10):
        class_dir = dataset_dir / str(digit)
        if not class_dir.is_dir():
            raise ValueError(f"Missing required class folder: {class_dir}")

        count = 0
        for entry in sorted(class_dir.iterdir()):
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

    too_small = [digit for digit, count in counts.items() if count < 2]
    if too_small:
        raise ValueError(
            "Each digit folder needs at least 2 readable images for training. "
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


def build_lenet5_model(tf):
    keras = tf.keras
    return keras.Sequential([
        keras.layers.Input(shape=(SEGMENT_SIZE, SEGMENT_SIZE, 1)),
        keras.layers.Conv2D(6, kernel_size=5, padding="same", activation="relu"),
        keras.layers.AveragePooling2D(pool_size=2),
        keras.layers.Conv2D(16, kernel_size=5, activation="relu"),
        keras.layers.AveragePooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(120, activation="relu"),
        keras.layers.Dense(84, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])


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
    x_train, y_train, x_val, y_val = stratified_split(
        x,
        y,
        validation_split=args.validation_split,
        seed=args.seed,
    )

    tf.keras.utils.set_random_seed(args.seed)

    model = build_lenet5_model(tf)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
    )

    test_loss, test_accuracy = model.evaluate(x_val, y_val, verbose=0)

    keras_path = keras_output_dir / "lenet5_digits.keras"
    model.save(keras_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = tflite_output_dir / "lenet5_digits.tflite"
    tflite_path.write_bytes(tflite_model)

    labels_path = tflite_output_dir / "labels.json"
    labels_path.write_text(json.dumps([str(i) for i in range(10)], indent=2), encoding="utf-8")

    metrics = {
        "dataset_size": int(len(x)),
        "class_counts": counts,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
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

    print(json.dumps(metrics))


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
    samples = np.asarray([prepare_digit_image(seg) for seg in segments], dtype=np.float32)[..., np.newaxis]

    if model_path.suffix.lower() == ".tflite":
        probs = predict_with_tflite(model_path, samples)
    else:
        probs = predict_with_keras(model_path, samples)

    predicted_digits = np.argmax(probs, axis=1)
    predicted_label = "".join(str(int(digit)) for digit in predicted_digits)
    confidences = [float(probs[i, predicted_digits[i]]) for i in range(len(predicted_digits))]

    result = {
        "predicted_label": predicted_label,
        "expected_label": args.expected_label or "",
        "confidences": confidences,
    }
    print(json.dumps(result))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset-dir", required=True)
    train_parser.add_argument("--keras-output-dir", required=True)
    train_parser.add_argument("--tflite-output-dir", required=True)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--validation-split", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model-path", required=True)
    predict_parser.add_argument("--image-path", required=True)
    predict_parser.add_argument("--expected-label", default="")

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
        raise ValueError(f"Unsupported command: {args.command}")
    except Exception as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
