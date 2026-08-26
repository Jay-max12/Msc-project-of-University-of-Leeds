"""
Explicit data preprocessing: resize (shortest/longest edge), fixed center crop,
and ImageNet normalization (delegated to HuggingFace image processor).

For object detection, geometric transforms update bounding boxes; evaluation maps
predictions back to the original image using stored metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from PIL import Image

# ImageNet statistics (same as torchvision / DETR defaults)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class PreprocessConfig:
    """When use_geometric is False, the dataset uses only the HF image processor (legacy path)."""

    use_geometric: bool = True
    shortest_edge: int = 800
    longest_edge: int = 1333
    center_crop_h: int = 800
    center_crop_w: int = 800


def _resize_shortest_longest(
    image: Image.Image, box_xyxy: List[float], shortest_edge: int, longest_edge: int
) -> Tuple[Image.Image, List[float], float]:
    """DETR-style aspect-preserving resize; returns uniform scale from original to resized."""
    w, h = image.size
    if min(h, w) <= 0:
        raise ValueError("Invalid image size")

    scale = shortest_edge / float(min(h, w))
    new_w = round(w * scale)
    new_h = round(h * scale)
    if max(new_w, new_h) > longest_edge:
        scale *= longest_edge / float(max(new_w, new_h))
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))

    resized = image.resize((new_w, new_h), Image.BILINEAR)
    x0, y0, x1, y1 = box_xyxy
    box = [x0 * scale, y0 * scale, x1 * scale, y1 * scale]
    return resized, box, scale


def _center_crop(
    image: Image.Image, box_xyxy: List[float], crop_h: int, crop_w: int
) -> Tuple[Image.Image, List[float], int, int]:
    """Center crop; returns crop offsets in the pre-crop coordinate system."""
    w, h = image.size
    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)
    left = max(0, (w - crop_w) // 2)
    top = max(0, (h - crop_h) // 2)
    cropped = image.crop((left, top, left + crop_w, top + crop_h))

    x0, y0, x1, y1 = box_xyxy
    box = [x0 - left, y0 - top, x1 - left, y1 - top]
    return cropped, box, left, top


def apply_geometric_preprocess(
    image: Image.Image, box_xyxy: List[float], config: PreprocessConfig
) -> Tuple[Image.Image, List[float], dict]:
    """
    Apply resize (shortest/longest edge), then fixed center crop, and build metadata
    for mapping predictions back to original image coordinates during evaluation.
    """
    if config.center_crop_h <= 0 or config.center_crop_w <= 0:
        raise ValueError("center_crop_h and center_crop_w must be positive.")

    orig_w, orig_h = image.size
    box = [float(x) for x in box_xyxy]

    image, box, resize_scale = _resize_shortest_longest(
        image, box, config.shortest_edge, config.longest_edge
    )
    image, box, crop_left, crop_top = _center_crop(
        image, box, config.center_crop_h, config.center_crop_w
    )

    geom_w, geom_h = image.size
    meta = {
        "enabled": True,
        "orig_h": int(orig_h),
        "orig_w": int(orig_w),
        "geom_h": int(geom_h),
        "geom_w": int(geom_w),
        "resize_scale": float(resize_scale),
        "crop_left": int(crop_left),
        "crop_top": int(crop_top),
    }
    return image, box, meta


def map_boxes_xyxy_to_original(boxes: torch.Tensor, meta: Optional[dict]) -> torch.Tensor:
    """
    Map predicted boxes from geometric (model input) space to original image pixels.
    `meta` must come from the same sample's `apply_geometric_preprocess`.
    """
    if meta is None or not meta.get("enabled"):
        return boxes
    if boxes.numel() == 0:
        return boxes

    out = boxes.clone().float()
    crop_left = float(meta["crop_left"])
    crop_top = float(meta["crop_top"])
    scale = float(meta["resize_scale"])
    if scale <= 0:
        return out

    out[:, 0] += crop_left
    out[:, 2] += crop_left
    out[:, 1] += crop_top
    out[:, 3] += crop_top
    out /= scale
    return out
