import base64
import io
import importlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


WARP_HI_W, WARP_HI_H = 500, 100
FINAL_W, FINAL_H = 140, 28
SEGMENT_SIZE = 28
NUM_SEGMENTS = 5

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
HEIF_DECODER_AVAILABLE = False
_PIL_IMAGE_MODULE = None
_PIL_IMAGE_OPS_MODULE = None
IMAGE_CACHE: dict[str, dict[str, Any]] = {}

app = FastAPI(title="DigitExtractor Web")
app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")


class SavePayload(BaseModel):
    label: str
    output_dir: str
    segments: list[str]


class ProcessRegisteredPayload(BaseModel):
    image_id: str
    points: list[dict[str, float]]


def ensure_heif_decoder() -> bool:
    global HEIF_DECODER_AVAILABLE, _PIL_IMAGE_MODULE, _PIL_IMAGE_OPS_MODULE
    if HEIF_DECODER_AVAILABLE:
        return True

    try:
        pillow_heif = importlib.import_module("pillow_heif")
        pil_image = importlib.import_module("PIL.Image")
        pil_image_ops = importlib.import_module("PIL.ImageOps")
        pillow_heif.register_heif_opener()
        _PIL_IMAGE_MODULE = pil_image
        _PIL_IMAGE_OPS_MODULE = pil_image_ops
        HEIF_DECODER_AVAILABLE = True
        return True
    except Exception:
        return False


def ensure_pillow_modules() -> bool:
    global _PIL_IMAGE_MODULE, _PIL_IMAGE_OPS_MODULE
    if _PIL_IMAGE_MODULE is not None and _PIL_IMAGE_OPS_MODULE is not None:
        return True

    try:
        _PIL_IMAGE_MODULE = importlib.import_module("PIL.Image")
        _PIL_IMAGE_OPS_MODULE = importlib.import_module("PIL.ImageOps")
        return True
    except Exception:
        return False


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def encode_png_base64(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def encode_webp_base64(image_bgr: np.ndarray, max_side: int = 1600, quality: int = 90) -> str:
    h, w = image_bgr.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        image_bgr = cv2.resize(
            image_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    ok, buf = cv2.imencode(".webp", image_bgr, [cv2.IMWRITE_WEBP_QUALITY, quality])
    if not ok:
        raise ValueError("Failed to encode WebP image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_base64_png(data: str) -> np.ndarray:
    raw = base64.b64decode(data)
    np_buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(np_buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid PNG payload")
    return img


def decode_upload_bytes(raw: bytes, filename: str = "") -> np.ndarray | None:
    np_buf = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    if image is not None:
        return image

    suffix = Path(filename).suffix.lower()
    if suffix in {".heic", ".heif"} and not ensure_heif_decoder():
        return None

    if not ensure_pillow_modules():
        return None

    try:
        with _PIL_IMAGE_MODULE.open(io.BytesIO(raw)) as pil_img:
            pil_img = _PIL_IMAGE_OPS_MODULE.exif_transpose(pil_img)
            rgb = np.array(pil_img.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def parse_points(points: list[dict[str, float]]) -> np.ndarray:
    if not isinstance(points, list) or len(points) != 4:
        raise ValueError("points must be a list of exactly 4 points")
    return np.array([[float(p["x"]), float(p["y"])] for p in points], dtype=np.float32)


def process_strip(image: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    src = order_points(points)
    dst = np.array(
        [
            [0, 0],
            [WARP_HI_W - 1, 0],
            [WARP_HI_W - 1, WARP_HI_H - 1],
            [0, WARP_HI_H - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (WARP_HI_W, WARP_HI_H))

    if len(warped.shape) == 3:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    else:
        gray = warped

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )
    binary = cv2.medianBlur(binary, 3)

    strip = cv2.resize(binary, (FINAL_W, FINAL_H), interpolation=cv2.INTER_AREA)
    segments: list[np.ndarray] = []
    for i in range(NUM_SEGMENTS):
        x0 = i * SEGMENT_SIZE
        segments.append(strip[:, x0:x0 + SEGMENT_SIZE])

    return strip, segments


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/manifest.webmanifest")
def manifest() -> FileResponse:
    return FileResponse(WEB_DIR / "manifest.webmanifest")


@app.get("/sw.js")
def service_worker() -> FileResponse:
    return FileResponse(WEB_DIR / "sw.js")


@app.get("/icon-192.png")
def icon_192() -> FileResponse:
    return FileResponse(BASE_DIR / "icon.png")


@app.get("/icon-512.png")
def icon_512() -> FileResponse:
    return FileResponse(BASE_DIR / "icon.png")


@app.post("/api/process")
async def api_process(image: UploadFile = File(...), points: str = Form(...)):
    try:
        parsed = json.loads(points)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid points JSON: {exc}") from exc

    if not isinstance(parsed, list) or len(parsed) != 4:
        raise HTTPException(status_code=400, detail="points must be a list of exactly 4 points")

    try:
        pts = np.array([[float(p["x"]), float(p["y"])] for p in parsed], dtype=np.float32)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Malformed points: {exc}") from exc

    raw = await image.read()
    cv_img = decode_upload_bytes(raw, image.filename or "")
    if cv_img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    try:
        strip, segments = process_strip(cv_img, pts)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "strip": encode_png_base64(strip),
        "segments": [encode_png_base64(seg) for seg in segments],
        "size": {"width": FINAL_W, "height": FINAL_H},
    }


@app.post("/api/preview")
async def api_preview(image: UploadFile = File(...)):
    raw = await image.read()
    cv_img = decode_upload_bytes(raw, image.filename or "")
    if cv_img is None:
        raise HTTPException(status_code=400, detail="Could not decode image for preview")

    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return {
        "image": encode_png_base64(rgb),
        "width": int(cv_img.shape[1]),
        "height": int(cv_img.shape[0]),
    }


@app.post("/api/reset-cache")
def api_reset_cache():
    IMAGE_CACHE.clear()
    return {"ok": True}


@app.post("/api/register-image")
async def api_register_image(
    image: UploadFile = File(...),
    relative_path: str = Form(default=""),
):
    raw = await image.read()
    cv_img = decode_upload_bytes(raw, image.filename or relative_path)
    if cv_img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    img_h, img_w = cv_img.shape[:2]
    try:
        preview_webp = encode_webp_base64(cv_img, max_side=1800, quality=90)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Preview encode failed: {exc}") from exc

    preview_np = cv2.imdecode(np.frombuffer(base64.b64decode(preview_webp), dtype=np.uint8), cv2.IMREAD_COLOR)
    if preview_np is None:
        raise HTTPException(status_code=500, detail="Preview decode failed after encode")

    preview_h, preview_w = preview_np.shape[:2]

    image_id = uuid.uuid4().hex
    del cv_img
    IMAGE_CACHE[image_id] = {
        "raw_bytes": raw,
        "filename": image.filename or "",
        "relative_path": relative_path,
        "original_width": int(img_w),
        "original_height": int(img_h),
    }

    return {
        "image_id": image_id,
        "filename": image.filename or "",
        "relative_path": relative_path,
        "original_width": int(img_w),
        "original_height": int(img_h),
        "preview_width": int(preview_w),
        "preview_height": int(preview_h),
        "preview_webp": preview_webp,
    }


@app.post("/api/process-registered")
def api_process_registered(payload: ProcessRegisteredPayload):
    cached = IMAGE_CACHE.get(payload.image_id)
    if cached is None:
        raise HTTPException(status_code=404, detail="image_id not found in cache")

    try:
        pts = parse_points(payload.points)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Malformed points: {exc}") from exc

    try:
        cv_img = decode_upload_bytes(cached["raw_bytes"], cached["filename"])
        if cv_img is None:
            raise Exception("decode_upload_bytes returned None")
        strip, segments = process_strip(cv_img, pts)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "strip": encode_png_base64(strip),
        "segments": [encode_png_base64(seg) for seg in segments],
        "size": {"width": FINAL_W, "height": FINAL_H},
    }


@app.post("/api/save")
def api_save(payload: SavePayload):
    label = payload.label.strip()
    if len(label) != NUM_SEGMENTS:
        raise HTTPException(status_code=400, detail="label must be exactly 5 characters")

    if len(payload.segments) != NUM_SEGMENTS:
        raise HTTPException(status_code=400, detail="segments must have exactly 5 images")

    output_dir = payload.output_dir.strip()
    if not output_dir:
        raise HTTPException(status_code=400, detail="output_dir is required")

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Cannot create output_dir: {exc}") from exc

    saved = 0
    for i, encoded in enumerate(payload.segments):
        char = label[i]
        char_dir = os.path.join(output_dir, char)
        os.makedirs(char_dir, exist_ok=True)

        try:
            segment = decode_base64_png(encoded)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid segment at index {i}: {exc}") from exc

        fname = f"segment_{uuid.uuid4().hex[:8]}.png"
        full_path = os.path.join(char_dir, fname)
        if not cv2.imwrite(full_path, segment):
            raise HTTPException(status_code=500, detail=f"Failed to write file: {full_path}")
        saved += 1

    return {"saved": saved, "output_dir": output_dir}
