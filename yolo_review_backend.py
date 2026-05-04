from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


STRIP_CLASS_ID = 0
DIGIT_OFFSET = 1
UNREADABLE_CLS = 11
YOLO_EXPORT_SIZE = 640
CLASS_NAMES = [
    "digit_strip",
    "digit_0",
    "digit_1",
    "digit_2",
    "digit_3",
    "digit_4",
    "digit_5",
    "digit_6",
    "digit_7",
    "digit_8",
    "digit_9",
    "digit_unreadable",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_brand(brand: str) -> str:
    raw = (brand or "").strip()
    return raw if raw else "Unknown"


def _safe_stem(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return cleaned.strip("_") or "image"


def _source_digest(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def class_id_to_char(cls_id: int) -> str:
    if cls_id == UNREADABLE_CLS:
        return "X"
    if DIGIT_OFFSET <= cls_id <= DIGIT_OFFSET + 9:
        return str(cls_id - DIGIT_OFFSET)
    return "?"


def char_to_class_id(ch: str) -> int:
    upper = ch.strip().upper()
    if upper == "X":
        return UNREADABLE_CLS
    if len(upper) == 1 and upper.isdigit():
        return DIGIT_OFFSET + int(upper)
    raise ValueError(f"Unsupported reading character: {ch!r}")


def normalize_box(box: list[int] | tuple[int, int, int, int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, max(width - 1, 0)))
    y1 = max(0, min(y1, max(height - 1, 0)))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return [x1, y1, x2, y2]


def sort_digit_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    digits = [det for det in detections if det.get("cls") != STRIP_CLASS_ID]
    return sorted(digits, key=lambda det: (det["box"][0] + det["box"][2]) / 2.0)


def build_reading_from_detections(detections: list[dict[str, Any]], expected_digits: int = 5) -> str:
    digits = sort_digit_detections(detections)
    text = "".join(class_id_to_char(int(det["cls"])) for det in digits[:expected_digits])
    return text.ljust(expected_digits, "?")[:expected_digits]


def detections_to_yolo_lines(
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> list[str]:
    lines: list[str] = []
    if image_width <= 0 or image_height <= 0:
        return lines
    for det in detections:
        x1, y1, x2, y2 = [float(v) for v in det["box"]]
        cx = ((x1 + x2) / 2.0) / image_width
        cy = ((y1 + y2) / 2.0) / image_height
        bw = max(x2 - x1, 1.0) / image_width
        bh = max(y2 - y1, 1.0) / image_height
        lines.append(
            f"{int(det['cls'])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        )
    return lines


def resize_detections(
    detections: list[dict[str, Any]],
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> list[dict[str, Any]]:
    if src_width <= 0 or src_height <= 0:
        return []
    scale_x = dst_width / float(src_width)
    scale_y = dst_height / float(src_height)
    out: list[dict[str, Any]] = []
    for det in detections:
        x1, y1, x2, y2 = [float(v) for v in det["box"]]
        out.append(
            {
                **det,
                "box": [
                    int(round(x1 * scale_x)),
                    int(round(y1 * scale_y)),
                    int(round(x2 * scale_x)),
                    int(round(y2 * scale_y)),
                ],
            }
        )
    return out


def build_class_map() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for idx, name in enumerate(CLASS_NAMES):
        if idx == STRIP_CLASS_ID:
            meaning = "full 5-digit strip bounding box"
        elif idx == UNREADABLE_CLS:
            meaning = "single digit box labeled unreadable"
        else:
            meaning = f"single digit box labeled {idx - DIGIT_OFFSET}"
        result[str(idx)] = {
            "id": idx,
            "name": name,
            "meaning": meaning,
        }
    return result


@dataclass
class ReviewPaths:
    root: Path
    images_dir: Path
    labels_dir: Path
    overlays_dir: Path
    exports_dir: Path
    reviews_jsonl: Path


class ReviewStore:
    def __init__(self, root_dir: str | Path):
        root = Path(root_dir)
        self.paths = ReviewPaths(
            root=root,
            images_dir=root / "images",
            labels_dir=root / "labels",
            overlays_dir=root / "overlays",
            exports_dir=root / "exports",
            reviews_jsonl=root / "reviews.jsonl",
        )
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.paths.images_dir.mkdir(parents=True, exist_ok=True)
        self.paths.labels_dir.mkdir(parents=True, exist_ok=True)
        self.paths.overlays_dir.mkdir(parents=True, exist_ok=True)
        self.paths.exports_dir.mkdir(parents=True, exist_ok=True)
        if not self.paths.reviews_jsonl.exists():
            self.paths.reviews_jsonl.touch()

    def load_reviews(self) -> list[dict[str, Any]]:
        reviews: list[dict[str, Any]] = []
        if not self.paths.reviews_jsonl.exists():
            return reviews
        for line in self.paths.reviews_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                reviews.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return reviews

    def load_latest_review_map(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for review in self.load_reviews():
            latest[str(review.get("source_image_path", ""))] = review
        return latest

    def get_latest_review(self, image_path: str | Path) -> dict[str, Any] | None:
        key = str(Path(image_path))
        return self.load_latest_review_map().get(key)

    def save_review(
        self,
        *,
        source_image_path: str | Path,
        source_image_bgr: np.ndarray,
        review_type: str,
        review_status: str,
        brand: str,
        model_path: str,
        predicted_reading: str,
        corrected_reading: str,
        original_detections: list[dict[str, Any]],
        corrected_detections: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        copy_image: bool = True,
    ) -> dict[str, Any]:
        source_path = Path(source_image_path)
        image_id = f"{_safe_stem(source_path.stem)}_{_source_digest(source_path)}"
        image_copy_path = self.paths.images_dir / f"{image_id}{source_path.suffix.lower() or '.png'}"
        label_copy_path = self.paths.labels_dir / f"{image_id}.txt"
        overlay_copy_path = self.paths.overlays_dir / f"{image_id}.png"

        if copy_image and source_path.exists():
            shutil.copy2(source_path, image_copy_path)
        elif source_image_bgr is not None and source_image_bgr.size > 0:
            cv2.imwrite(str(image_copy_path), source_image_bgr)

        label_lines = detections_to_yolo_lines(
            corrected_detections,
            image_width=source_image_bgr.shape[1],
            image_height=source_image_bgr.shape[0],
        )
        label_copy_path.write_text("\n".join(label_lines), encoding="utf-8")

        overlay = source_image_bgr.copy()
        for det in corrected_detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            colour = (0, 200, 255) if int(det["cls"]) == STRIP_CLASS_ID else (60, 220, 60)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                overlay,
                CLASS_NAMES[int(det["cls"])],
                (x1 + 3, max(16, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                colour,
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(overlay_copy_path), overlay)

        payload = {
            "schema_version": 1,
            "review_id": uuid.uuid4().hex,
            "reviewed_at": utc_now_iso(),
            "source_image_path": str(source_path),
            "copied_image_path": str(image_copy_path),
            "copied_label_path": str(label_copy_path),
            "overlay_path": str(overlay_copy_path),
            "brand": sanitize_brand(brand),
            "model_path": model_path,
            "predicted_reading": predicted_reading,
            "corrected_reading": corrected_reading,
            "review_type": review_type,
            "review_status": review_status,
            "original_detections": original_detections,
            "corrected_detections": corrected_detections,
            "metadata": metadata or {},
        }
        with self.paths.reviews_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        return payload

    def export_reviews(
        self,
        export_dir: str | Path,
        *,
        brand_filter: str = "",
        include_marked_correct: bool = False,
    ) -> dict[str, Any]:
        export_root = Path(export_dir)
        images_dir = export_root / "ROI_640"
        labels_dir = export_root / "ROI_640_labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        reviews = self.load_reviews()
        if brand_filter.strip():
            reviews = [
                item for item in reviews
                if str(item.get("brand", "")).strip().lower() == brand_filter.strip().lower()
            ]

        exported = 0
        skipped = 0
        manifest: list[dict[str, Any]] = []
        review_types_to_keep = {"detection_fixed", "reading_only"}
        if include_marked_correct:
            review_types_to_keep.add("marked_correct")

        for review in reviews:
            review_type = str(review.get("review_type", ""))
            if review_type not in review_types_to_keep:
                skipped += 1
                continue

            detections = review.get("corrected_detections") or []
            image_path = Path(str(review.get("copied_image_path") or review.get("source_image_path") or ""))
            if not image_path.exists():
                skipped += 1
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                skipped += 1
                continue

            if review_type == "reading_only":
                digit_count = sum(1 for det in detections if int(det.get("cls", -1)) != STRIP_CLASS_ID)
                has_strip = any(int(det.get("cls", -1)) == STRIP_CLASS_ID for det in detections)
                if not has_strip or digit_count != 5:
                    skipped += 1
                    continue

            resized = cv2.resize(image, (YOLO_EXPORT_SIZE, YOLO_EXPORT_SIZE), interpolation=cv2.INTER_AREA)
            resized_dets = resize_detections(
                detections,
                src_width=image.shape[1],
                src_height=image.shape[0],
                dst_width=YOLO_EXPORT_SIZE,
                dst_height=YOLO_EXPORT_SIZE,
            )
            lines = detections_to_yolo_lines(resized_dets, YOLO_EXPORT_SIZE, YOLO_EXPORT_SIZE)
            stem = f"{_safe_stem(Path(str(review.get('source_image_path', 'image'))).stem)}_{review.get('review_id', '')[:8]}"
            cv2.imwrite(str(images_dir / f"{stem}.png"), resized)
            (labels_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            manifest.append(
                {
                    "source_image_path": review.get("source_image_path", ""),
                    "brand": review.get("brand", ""),
                    "review_type": review_type,
                    "predicted_reading": review.get("predicted_reading", ""),
                    "corrected_reading": review.get("corrected_reading", ""),
                    "image_file": f"{stem}.png",
                    "label_file": f"{stem}.txt",
                }
            )
            exported += 1

        (export_root / "yolo_classes.txt").write_text("\n".join(CLASS_NAMES), encoding="utf-8")
        (export_root / "yolo_class_map.json").write_text(
            json.dumps(build_class_map(), indent=2),
            encoding="utf-8",
        )
        (export_root / "review_export_manifest.json").write_text(
            json.dumps(
                {
                    "generated_at": utc_now_iso(),
                    "brand_filter": sanitize_brand(brand_filter) if brand_filter.strip() else "",
                    "exported_count": exported,
                    "skipped_count": skipped,
                    "items": manifest,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "export_dir": str(export_root),
            "exported_count": exported,
            "skipped_count": skipped,
        }
