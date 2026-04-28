"""Train and run a YOLOv8 detector for digit-strip finding."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from auto_read_pipeline import build_sliding_windows

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def log(message: str):
    print(message, flush=True)


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


def _bbox_area(box_xyxy: list[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _is_reasonable_strip_bbox(
    box_xyxy: list[float],
    image_shape: tuple[int, int],
) -> bool:
    image_h, image_w = image_shape
    bw = max(float(box_xyxy[2]) - float(box_xyxy[0]), 0.0)
    bh = max(float(box_xyxy[3]) - float(box_xyxy[1]), 0.0)
    area = bw * bh
    image_area = max(float(image_w * image_h), 1.0)
    aspect = bw / max(bh, 1.0)

    if bw < max(20.0, image_w * 0.08):
        return False
    if bh < max(10.0, image_h * 0.02):
        return False
    if area < image_area * 0.0025:
        return False
    if aspect < 1.6 or aspect > 20.0:
        return False
    return True


def _score_strip_candidate(
    box_xyxy: list[float],
    confidence: float,
    image_shape: tuple[int, int],
) -> float:
    image_h, image_w = image_shape
    bw = max(float(box_xyxy[2]) - float(box_xyxy[0]), 0.0)
    bh = max(float(box_xyxy[3]) - float(box_xyxy[1]), 0.0)
    area_ratio = (bw * bh) / max(float(image_w * image_h), 1.0)
    aspect = bw / max(bh, 1.0)
    aspect_score = 1.0 - min(abs(aspect - 5.0) / 8.0, 1.0)
    area_score = min(area_ratio / 0.04, 1.0)
    return float(confidence) + (0.35 * area_score) + (0.20 * aspect_score)


def build_dataset_workspace(images_dir: Path, labels_dir: Path) -> tuple[Path, dict[str, int]]:
    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise ValueError("No readable ROI_640 images were found.")

    workspace = Path(tempfile.mkdtemp(prefix="digit_yolo_dataset_"))
    train_images = workspace / "images" / "train"
    val_images = workspace / "images" / "val"
    train_labels = workspace / "labels" / "train"
    val_labels = workspace / "labels" / "val"
    for folder in (train_images, val_images, train_labels, val_labels):
        folder.mkdir(parents=True, exist_ok=True)

    split_index = max(1, int(round(len(image_files) * 0.8)))
    if split_index >= len(image_files):
        split_index = max(1, len(image_files) - 1)
    train_set = image_files[:split_index]
    val_set = image_files[split_index:] if split_index < len(image_files) else image_files[-1:]

    if not val_set:
        raise ValueError("Need at least 2 labeled images to train YOLO.")

    def copy_subset(files: list[Path], img_target: Path, label_target: Path):
        for image_path in files:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue
            shutil.copy2(image_path, img_target / image_path.name)
            shutil.copy2(label_path, label_target / label_path.name)

    copy_subset(train_set, train_images, train_labels)
    copy_subset(val_set, val_images, val_labels)

    counts = {
        "total_images": len(image_files),
        "train_images": len(list(train_images.iterdir())),
        "val_images": len(list(val_images.iterdir())),
    }
    if counts["train_images"] == 0 or counts["val_images"] == 0:
        raise ValueError("Training and validation splits both need labeled image/label pairs.")

    data_yaml = workspace / "dataset.yaml"
    data_yaml.write_text(
        "\n".join([
            f"path: {workspace.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: digit_strip",
            "",
        ]),
        encoding="utf-8",
    )
    return workspace, counts


def command_train(args: argparse.Namespace):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics is not available in the selected backend Python. "
            "Install training_requirements.txt in a compatible Python 3.10-3.13 environment."
        ) from exc

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace, counts = build_dataset_workspace(images_dir, labels_dir)
    log(
        f"[YOLO] Dataset workspace ready. Total={counts['total_images']} "
        f"Train={counts['train_images']} Val={counts['val_images']}"
    )

    model = YOLO("yolov8n.pt")
    log(
        f"[YOLO] Training start. Epochs={args.epochs} ImageSize={args.image_size} "
        f"Batch={args.batch_size}"
    )
    train_result = model.train(
        data=str(workspace / "dataset.yaml"),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        project=str(output_dir),
        name="yolov8_digit_strip",
        exist_ok=True,
        verbose=True,
    )

    best_model_path = Path(train_result.save_dir) / "weights" / "best.pt"
    log(f"[YOLO] Training complete. Best model at {best_model_path}")
    export_warning = ""
    tflite_model_path = ""
    try:
        log("[YOLO] Exporting TFLite...")
        exported_path = model.export(format="tflite", imgsz=args.image_size)
        if exported_path:
            tflite_model_path = str(Path(exported_path))
            log(f"[YOLO] TFLite export complete: {tflite_model_path}")
    except Exception as exc:
        export_warning = str(exc)
        log(f"[YOLO] Export warning: {export_warning}")

    metrics = {
        "train_images": counts["train_images"],
        "val_images": counts["val_images"],
        "total_images": counts["total_images"],
        "best_model_path": str(best_model_path),
        "tflite_model_path": tflite_model_path,
        "export_warning": export_warning,
    }
    (output_dir / "yolo_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics))


def command_predict(args: argparse.Namespace):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics is not available in the selected backend Python. "
            "Install training_requirements.txt in a compatible Python 3.10-3.13 environment."
        ) from exc

    log(f"[YOLO] Predicting on {args.image_path} with {args.model_path}")
    model = YOLO(args.model_path, task="detect")
    results = model.predict(
        source=args.image_path,
        imgsz=args.image_size,
        conf=args.conf_threshold,
        verbose=False,
    )
    if not results:
        print(json.dumps({"found": False}))
        return

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        print(json.dumps({"found": False}))
        return

    image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {args.image_path}")

    conf_values = boxes.conf.detach().cpu().numpy().astype(np.float32)
    xyxy_values = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    candidates = []
    for idx, confidence in enumerate(conf_values.tolist()):
        xyxy = [float(v) for v in xyxy_values[idx].tolist()]
        if not _is_reasonable_strip_bbox(xyxy, image.shape[:2]):
            continue
        candidates.append(
            {
                "bbox_xyxy": xyxy,
                "confidence": float(confidence),
                "rank_score": _score_strip_candidate(xyxy, float(confidence), image.shape[:2]),
            }
        )

    if not candidates:
        print(json.dumps({"found": False}))
        return

    candidates.sort(key=lambda item: float(item["rank_score"]), reverse=True)
    best = candidates[0]
    confidence = float(best["confidence"])
    log(f"[YOLO] Best detection confidence={confidence:.4f}")
    print(json.dumps({
        "found": True,
        "bbox_xyxy": best["bbox_xyxy"],
        "confidence": confidence,
        "candidates": candidates[:5],
    }))


def command_predict_windows(args: argparse.Namespace):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics is not available in the selected backend Python. "
            "Install training_requirements.txt in a compatible Python 3.10-3.13 environment."
        ) from exc

    image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {args.image_path}")

    model = YOLO(args.model_path, task="detect")
    h, w = image.shape[:2]
    windows = build_sliding_windows(w, h)
    if not windows:
        print(json.dumps({"found": False}))
        return
    log(f"[YOLO] Sliding-window detect on {len(windows)} windows for image {args.image_path}")

    raw_candidates = []
    for idx, window in enumerate(windows, start=1):
        log(f"[YOLO] Window {idx}/{len(windows)} x={window.x} y={window.y} size={window.size}")
        crop = image[window.y:window.y + window.size, window.x:window.x + window.size]
        if crop.size == 0:
            continue
        results = model.predict(
            source=cv2.resize(crop, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA),
            imgsz=args.image_size,
            conf=args.conf_threshold,
            verbose=False,
        )
        if not results:
            continue
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue

        conf_values = boxes.conf.detach().cpu().numpy().astype(np.float32)
        xyxy_values = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scale = window.size / float(args.image_size)
        for idx, confidence in enumerate(conf_values.tolist()):
            xyxy = xyxy_values[idx]
            mapped = [
                float(window.x + (xyxy[0] * scale)),
                float(window.y + (xyxy[1] * scale)),
                float(window.x + (xyxy[2] * scale)),
                float(window.y + (xyxy[3] * scale)),
            ]
            mapped_box = [float(v) for v in mapped]
            if not _is_reasonable_strip_bbox(mapped_box, image.shape[:2]):
                continue
            raw_candidates.append({
                "bbox_xyxy": mapped_box,
                "confidence": float(confidence),
                "rank_score": _score_strip_candidate(mapped_box, float(confidence), image.shape[:2]),
                "window": {
                    "x": int(window.x),
                    "y": int(window.y),
                    "size": int(window.size),
                },
            })

    if not raw_candidates:
        log("[YOLO] No detections from sliding-window search.")
        print(json.dumps({"found": False}))
        return

    raw_candidates.sort(key=lambda item: float(item["rank_score"]), reverse=True)
    merged: list[dict[str, object]] = []
    for candidate in raw_candidates:
        bbox = candidate["bbox_xyxy"]
        duplicate_index = -1
        for existing_index, existing in enumerate(merged):
            if _bbox_iou(bbox, existing["bbox_xyxy"]) > 0.65:
                duplicate_index = existing_index
                break
        if duplicate_index >= 0:
            if float(candidate["rank_score"]) > float(merged[duplicate_index]["rank_score"]):
                merged[duplicate_index] = candidate
            continue
        merged.append(candidate)
        if len(merged) >= 8:
            break

    merged.sort(key=lambda item: float(item["rank_score"]), reverse=True)
    best = merged[0]
    log(
        f"[YOLO] Sliding-window complete. Raw={len(raw_candidates)} Merged={len(merged)} "
        f"BestConf={float(best['confidence']):.4f} BestScore={float(best['rank_score']):.4f}"
    )
    print(json.dumps({
        "found": True,
        "bbox_xyxy": best["bbox_xyxy"],
        "confidence": best["confidence"],
        "window": best["window"],
        "candidates": merged,
    }))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--images-dir", required=True)
    train_parser.add_argument("--labels-dir", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--image-size", type=int, default=640)
    train_parser.add_argument("--batch-size", type=int, default=16)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model-path", required=True)
    predict_parser.add_argument("--image-path", required=True)
    predict_parser.add_argument("--image-size", type=int, default=640)
    predict_parser.add_argument("--conf-threshold", type=float, default=0.25)

    predict_windows_parser = subparsers.add_parser("predict-windows")
    predict_windows_parser.add_argument("--model-path", required=True)
    predict_windows_parser.add_argument("--image-path", required=True)
    predict_windows_parser.add_argument("--image-size", type=int, default=640)
    predict_windows_parser.add_argument("--conf-threshold", type=float, default=0.25)

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
        if args.command == "predict-windows":
            command_predict_windows(args)
            return
        raise ValueError(f"Unsupported command: {args.command}")
    except Exception as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
