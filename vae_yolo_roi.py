"""YOLO-predicted ROI crops for VAE + classifier joint training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

BoxSelection = Literal["all", "highest_conf", "largest_area"]

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from train_resnet50_roi_classifier import sanitize_box_xyxy
from train_yolov11_onestep import _disable_ultralytics_neptune_callback, read_split_file


def list_pred_boxes_from_arrays(
    all_xyxy: np.ndarray,
    all_conf: np.ndarray,
    *,
    eval_score_threshold: float,
    max_detections: int,
) -> List[Dict[str, object]]:
    """Return all score-filtered xyxy boxes sorted by confidence (up to max_detections)."""
    if all_xyxy.shape[0] == 0 or all_conf.shape[0] == 0:
        return []

    keep = all_conf > float(eval_score_threshold)
    if not np.any(keep):
        return []

    xyxy = all_xyxy[keep]
    conf = all_conf[keep]
    order = np.argsort(-conf)
    if max_detections > 0:
        order = order[: int(max_detections)]

    out: List[Dict[str, object]] = []
    for idx in order.tolist():
        out.append(
            {
                "box_xyxy": [float(v) for v in xyxy[int(idx)].tolist()],
                "conf": float(conf[int(idx)]),
                "source": "yolo",
            }
        )
    return out


def box_area_xyxy(box: Sequence[float]) -> float:
    x0, y0, x1, y1 = [float(v) for v in box]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def select_single_box(
    box_items: Sequence[Dict[str, object]],
    box_selection: BoxSelection,
) -> List[Dict[str, object]]:
    """Reduce candidate boxes to one (or keep all for multi-ROI fusion)."""
    items = list(box_items)
    if not items or box_selection == "all":
        return items
    if box_selection == "highest_conf":
        return [max(items, key=lambda b: float(b.get("conf", 0.0)))]
    if box_selection == "largest_area":
        return [max(items, key=lambda b: box_area_xyxy(b["box_xyxy"]))]  # type: ignore[index]
    raise ValueError(f"Unknown box_selection: {box_selection}")


def select_pred_box_for_crop(
    all_xyxy: np.ndarray,
    all_conf: np.ndarray,
    *,
    eval_score_threshold: float,
    max_detections: int,
) -> Optional[List[float]]:
    """Pick one xyxy box for ROI crop: highest confidence among score-filtered detections."""
    boxes = list_pred_boxes_from_arrays(
        all_xyxy,
        all_conf,
        eval_score_threshold=eval_score_threshold,
        max_detections=max_detections,
    )
    if not boxes:
        return None
    return list(boxes[0]["box_xyxy"])  # type: ignore[arg-type]


def list_pred_boxes_for_image(
    yolo_model,
    image_path: Path,
    *,
    eval_score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    imgsz: int,
    box_selection: BoxSelection = "all",
) -> List[Dict[str, object]]:
    """Run YOLO on one image; return filtered boxes (optionally reduced to a single box)."""
    pool_size = int(max_detections)
    if box_selection in {"highest_conf", "largest_area"}:
        pool_size = max(pool_size, 20)

    raw_res = yolo_model.predict(
        source=str(image_path),
        conf=0.001,
        iou=float(nms_iou_threshold),
        max_det=max(pool_size, 1),
        verbose=False,
        imgsz=int(imgsz),
    )[0]
    boxes = raw_res.boxes
    all_xyxy = boxes.xyxy.cpu().numpy() if boxes is not None and boxes.xyxy is not None else np.zeros((0, 4))
    all_conf = boxes.conf.cpu().numpy() if boxes is not None and boxes.conf is not None else np.zeros((0,))
    candidates = list_pred_boxes_from_arrays(
        all_xyxy,
        all_conf,
        eval_score_threshold=eval_score_threshold,
        max_detections=pool_size,
    )
    return select_single_box(candidates, box_selection)


def predict_yolo_box_for_image(
    yolo_model,
    image_path: Path,
    *,
    eval_score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    imgsz: int,
) -> Optional[List[float]]:
    raw_res = yolo_model.predict(
        source=str(image_path),
        conf=0.001,
        iou=float(nms_iou_threshold),
        max_det=max(int(max_detections), 1),
        verbose=False,
        imgsz=int(imgsz),
    )[0]
    boxes = raw_res.boxes
    all_xyxy = boxes.xyxy.cpu().numpy() if boxes is not None and boxes.xyxy is not None else np.zeros((0, 4))
    all_conf = boxes.conf.cpu().numpy() if boxes is not None and boxes.conf is not None else np.zeros((0,))
    return select_pred_box_for_crop(
        all_xyxy,
        all_conf,
        eval_score_threshold=eval_score_threshold,
        max_detections=max_detections,
    )


def build_yolo_box_cache(
    *,
    dataset_root: Path,
    yolo_ckpt: Path,
    cache_path: Path,
    splits: Sequence[str] = ("train", "test"),
    eval_score_threshold: float = 0.5,
    nms_iou_threshold: float = 0.5,
    max_detections: int = 5,
    imgsz: int = 800,
    fallback: str = "full_image",
) -> Dict[str, dict]:
    """Run YOLO on all images in splits and cache predicted xyxy boxes."""
    if fallback not in {"full_image", "skip"}:
        raise ValueError("fallback must be 'full_image' or 'skip'")

    _disable_ultralytics_neptune_callback()
    from ultralytics import YOLO

    yolo_model = YOLO(str(yolo_ckpt))
    imgs_dir = dataset_root / "imgs"

    cache: Dict[str, dict] = {
        "meta": {
            "yolo_ckpt": str(yolo_ckpt.resolve()),
            "eval_score_threshold": float(eval_score_threshold),
            "nms_iou_threshold": float(nms_iou_threshold),
            "max_detections": int(max_detections),
            "imgsz": int(imgsz),
            "fallback": fallback,
            "splits": list(splits),
        },
        "boxes": {},
    }

    stats = {"total": 0, "yolo_hit": 0, "fallback_full": 0, "skipped": 0}

    for split in splits:
        split_file = dataset_root / f"{split}.txt"
        if not split_file.exists():
            continue
        for image_name, _cls in read_split_file(split_file):
            image_path = imgs_dir / image_name
            stats["total"] += 1
            if not image_path.exists():
                stats["skipped"] += 1
                continue

            pred_box = predict_yolo_box_for_image(
                yolo_model,
                image_path,
                eval_score_threshold=eval_score_threshold,
                nms_iou_threshold=nms_iou_threshold,
                max_detections=max_detections,
                imgsz=imgsz,
            )

            entry = {
                "split": split,
                "source": "yolo",
                "box_xyxy": pred_box,
            }
            if pred_box is None:
                if fallback == "full_image":
                    with Image.open(image_path) as img:
                        w, h = img.size
                    entry["box_xyxy"] = [0.0, 0.0, float(w), float(h)]
                    entry["source"] = "full_image_fallback"
                    stats["fallback_full"] += 1
                else:
                    stats["skipped"] += 1
                    continue
            else:
                stats["yolo_hit"] += 1

            cache["boxes"][image_name] = entry

    cache["meta"]["stats"] = stats
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    return cache


def load_yolo_box_cache(cache_path: Path) -> Dict[str, dict]:
    with cache_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_or_build_yolo_box_cache(
    *,
    dataset_root: Path,
    yolo_ckpt: Path,
    cache_path: Path,
    rebuild: bool = False,
    **kwargs,
) -> Dict[str, dict]:
    if cache_path.exists() and not rebuild:
        cache = load_yolo_box_cache(cache_path)
        meta = cache.get("meta", {})
        if Path(meta.get("yolo_ckpt", "")).resolve() == Path(yolo_ckpt).resolve():
            return cache
    return build_yolo_box_cache(
        dataset_root=dataset_root,
        yolo_ckpt=yolo_ckpt,
        cache_path=cache_path,
        **kwargs,
    )


class YoloRoiPatchDataset(Dataset):
    """ROI patches cropped from YOLO-predicted boxes (cached)."""

    def __init__(
        self,
        dataset_root: Path,
        samples: List[Tuple[str, int]],
        transform: transforms.Compose,
        box_cache: Dict[str, dict],
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.imgs_dir = self.dataset_root / "imgs"
        self.transform = transform
        self.box_cache = box_cache

        self.samples: List[Tuple[str, int]] = []
        for name, cls_id in samples:
            if name not in box_cache.get("boxes", {}):
                continue
            if not (self.imgs_dir / name).exists():
                continue
            self.samples.append((name, int(cls_id)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        name, label = self.samples[idx]
        img = Image.open(self.imgs_dir / name).convert("RGB")
        w, h = img.size
        entry = self.box_cache["boxes"][name]
        box = entry["box_xyxy"]
        if box is None:
            x0, y0, x1, y1 = 0.0, 0.0, float(w), float(h)
        else:
            x0, y0, x1, y1 = sanitize_box_xyxy(box, width=w, height=h)
        crop = img.crop((x0, y0, x1, y1))
        x = self.transform(crop)
        return x, int(label)
