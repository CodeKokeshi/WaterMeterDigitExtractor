from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class WorkspaceFrame:
    rotation_deg: float
    scale: float
    translate_x: float
    translate_y: float
    workspace_size: tuple[int, int]
    guidebox_rect_workspace: tuple[float, float, float, float]


@dataclass
class WorkspaceRenderResult:
    workspace_image: np.ndarray
    guidebox_crop: np.ndarray | None
    affine_matrix: np.ndarray
    guidebox_rect_workspace: tuple[int, int, int, int]


def build_guidebox_rect(workspace_size: tuple[int, int]) -> tuple[float, float, float, float]:
    width = max(int(workspace_size[0]), 1)
    height = max(int(workspace_size[1]), 1)
    margin_w = max(int(width * 0.08), 24)
    margin_h = max(int(height * 0.18), 24)
    max_width = max(width - (2 * margin_w), 40)
    max_height = max(height - (2 * margin_h), 20)
    guidebox_width = min(float(max_width), float(max_height) * 5.0)
    guidebox_height = guidebox_width / 5.0
    guidebox_x = (width - guidebox_width) / 2.0
    guidebox_y = (height - guidebox_height) / 2.0
    return guidebox_x, guidebox_y, guidebox_width, guidebox_height


def build_default_workspace_frame(
    workspace_size: tuple[int, int],
    rotation_deg: float = 0.0,
) -> WorkspaceFrame:
    return WorkspaceFrame(
        rotation_deg=float(rotation_deg),
        scale=1.0,
        translate_x=0.0,
        translate_y=0.0,
        workspace_size=(int(max(workspace_size[0], 1)), int(max(workspace_size[1], 1))),
        guidebox_rect_workspace=build_guidebox_rect(workspace_size),
    )


def clone_workspace_frame(frame: WorkspaceFrame) -> WorkspaceFrame:
    return WorkspaceFrame(
        rotation_deg=float(frame.rotation_deg),
        scale=float(frame.scale),
        translate_x=float(frame.translate_x),
        translate_y=float(frame.translate_y),
        workspace_size=(int(frame.workspace_size[0]), int(frame.workspace_size[1])),
        guidebox_rect_workspace=tuple(float(v) for v in frame.guidebox_rect_workspace),
    )


def serialize_workspace_frame(frame: WorkspaceFrame) -> dict[str, object]:
    return asdict(frame)


def deserialize_workspace_frame(data: dict[str, object]) -> WorkspaceFrame:
    workspace_size_raw = data.get("workspace_size", (1, 1))
    guidebox_raw = data.get("guidebox_rect_workspace", build_guidebox_rect((1, 1)))
    workspace_size = (
        int(workspace_size_raw[0]),
        int(workspace_size_raw[1]),
    )
    guidebox_rect = tuple(float(v) for v in guidebox_raw)
    return WorkspaceFrame(
        rotation_deg=float(data.get("rotation_deg", 0.0)),
        scale=float(data.get("scale", 1.0)),
        translate_x=float(data.get("translate_x", 0.0)),
        translate_y=float(data.get("translate_y", 0.0)),
        workspace_size=workspace_size,
        guidebox_rect_workspace=guidebox_rect,
    )


def update_workspace_frame_for_size(
    frame: WorkspaceFrame,
    workspace_size: tuple[int, int],
) -> WorkspaceFrame:
    updated = clone_workspace_frame(frame)
    updated.workspace_size = (int(max(workspace_size[0], 1)), int(max(workspace_size[1], 1)))
    updated.guidebox_rect_workspace = build_guidebox_rect(updated.workspace_size)
    return updated


def _clamp_crop_rect(
    rect: tuple[float, float, float, float],
    workspace_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    max_w = max(int(workspace_size[0]), 1)
    max_h = max(int(workspace_size[1]), 1)
    x1 = max(0, min(int(round(x)), max_w - 1))
    y1 = max(0, min(int(round(y)), max_h - 1))
    x2 = max(x1 + 1, min(int(round(x + w)), max_w))
    y2 = max(y1 + 1, min(int(round(y + h)), max_h))
    return x1, y1, x2, y2


def build_workspace_frame_from_ui(
    current_frame: WorkspaceFrame | None,
    workspace_size: tuple[int, int],
    rotation_deg: float | None = None,
    scale: float | None = None,
    translate_x: float | None = None,
    translate_y: float | None = None,
) -> WorkspaceFrame:
    base = (
        update_workspace_frame_for_size(current_frame, workspace_size)
        if current_frame is not None
        else build_default_workspace_frame(workspace_size)
    )
    if rotation_deg is not None:
        base.rotation_deg = float(rotation_deg)
    if scale is not None:
        base.scale = float(max(scale, 0.05))
    if translate_x is not None:
        base.translate_x = float(translate_x)
    if translate_y is not None:
        base.translate_y = float(translate_y)
    return base


def render_workspace_view(source_image: np.ndarray, workspace_frame: WorkspaceFrame) -> WorkspaceRenderResult:
    workspace_width = max(int(workspace_frame.workspace_size[0]), 1)
    workspace_height = max(int(workspace_frame.workspace_size[1]), 1)
    workspace_shape = (workspace_height, workspace_width)
    blank = np.zeros((workspace_height, workspace_width, 3), dtype=np.uint8)

    if source_image is None or source_image.size == 0:
        x1, y1, x2, y2 = _clamp_crop_rect(workspace_frame.guidebox_rect_workspace, workspace_frame.workspace_size)
        crop = blank[y1:y2, x1:x2].copy()
        return WorkspaceRenderResult(
            workspace_image=blank,
            guidebox_crop=crop if crop.size else None,
            affine_matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            guidebox_rect_workspace=(x1, y1, x2, y2),
        )

    image_height, image_width = source_image.shape[:2]
    base_scale = min(
        workspace_width / max(float(image_width), 1.0),
        workspace_height / max(float(image_height), 1.0),
    )
    total_scale = max(base_scale * float(workspace_frame.scale), 1e-6)
    image_center = (image_width / 2.0, image_height / 2.0)
    affine_matrix = cv2.getRotationMatrix2D(image_center, float(workspace_frame.rotation_deg), total_scale)

    mapped_center_x = (
        affine_matrix[0, 0] * image_center[0]
        + affine_matrix[0, 1] * image_center[1]
        + affine_matrix[0, 2]
    )
    mapped_center_y = (
        affine_matrix[1, 0] * image_center[0]
        + affine_matrix[1, 1] * image_center[1]
        + affine_matrix[1, 2]
    )
    affine_matrix[0, 2] += (workspace_width / 2.0) + float(workspace_frame.translate_x) - mapped_center_x
    affine_matrix[1, 2] += (workspace_height / 2.0) + float(workspace_frame.translate_y) - mapped_center_y

    workspace_image = cv2.warpAffine(
        source_image,
        affine_matrix,
        (workspace_width, workspace_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    x1, y1, x2, y2 = _clamp_crop_rect(workspace_frame.guidebox_rect_workspace, workspace_frame.workspace_size)
    crop = workspace_image[y1:y2, x1:x2].copy()
    return WorkspaceRenderResult(
        workspace_image=workspace_image,
        guidebox_crop=crop if crop.size else None,
        affine_matrix=affine_matrix.astype(np.float32),
        guidebox_rect_workspace=(x1, y1, x2, y2),
    )


def extract_guidebox_crop(source_image: np.ndarray, workspace_frame: WorkspaceFrame) -> np.ndarray | None:
    return render_workspace_view(source_image, workspace_frame).guidebox_crop
