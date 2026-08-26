import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from detr_roi.preprocess import PreprocessConfig, apply_geometric_preprocess


ROI_LABELS = {"nml", "abn"}


def read_split_file(split_file: Path) -> List[str]:
    image_names: List[str] = []
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_name = line.split(",")[0].strip()
            image_names.append(image_name)
    return image_names


def _area_xyxy(box: List[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def choose_roi_box(bbs: List[List]) -> List[float]:
    """Pick ROI box; prefer 'nml'/'abn', then fallback to largest area."""
    roi_candidates: List[List[float]] = []
    fallback_boxes: List[List[float]] = []

    for item in bbs:
        if len(item) != 2:
            continue
        label, box = item
        if not isinstance(box, list) or len(box) != 4:
            continue
        box = [float(v) for v in box]
        fallback_boxes.append(box)
        if str(label).lower() in ROI_LABELS:
            roi_candidates.append(box)

    target_pool = roi_candidates if roi_candidates else fallback_boxes
    if not target_pool:
        raise ValueError("No valid boxes found for this image.")
    return max(target_pool, key=_area_xyxy)


def load_roi_annotations(annotation_path: Path) -> Dict[str, List[float]]:
    with annotation_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    parsed: Dict[str, List[float]] = {}
    for image_name, info in raw.items():
        bbs = info.get("bbs", [])
        parsed[image_name] = choose_roi_box(bbs)
    return parsed


class GBCUDetrDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        split_name: str,
        image_processor,
        preprocess_config: Optional[PreprocessConfig] = None,
        image_names: Optional[List[str]] = None,
        augment: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.image_dir = self.dataset_root / "imgs"
        self.split_file = self.dataset_root / f"{split_name}.txt"
        self.annotation_file = self.dataset_root / "bbox_annot.json"
        self.image_processor = image_processor
        self.preprocess_config = preprocess_config
        self.augment = augment

        split_images = set(image_names) if image_names is not None else set(read_split_file(self.split_file))
        all_annotations = load_roi_annotations(self.annotation_file)

        self.samples: List[Tuple[str, List[float]]] = []
        for image_name in sorted(split_images):
            if image_name not in all_annotations:
                continue
            image_path = self.image_dir / image_name
            if image_path.exists():
                self.samples.append((image_name, all_annotations[image_name]))

        if not self.samples:
            raise RuntimeError(f"No samples found in split '{split_name}'.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        image_name, box_xyxy_orig = self.samples[idx]
        image_path = self.image_dir / image_name
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        box_xyxy = [float(x) for x in box_xyxy_orig]
        preprocess_meta = None

        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            x0, y0, x1, y1 = box_xyxy
            box_xyxy = [orig_w - x1, y0, orig_w - x0, y1]

        proc_extra: Dict = {}
        if self.preprocess_config is not None and self.preprocess_config.use_geometric:
            image, box_xyxy, preprocess_meta = apply_geometric_preprocess(
                image, box_xyxy, self.preprocess_config
            )
            proc_extra = {"do_resize": False, "do_normalize": True, "do_rescale": True, "do_pad": True}

        x_min, y_min, x_max, y_max = box_xyxy
        coco_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        target = {
            "image_id": idx,
            "annotations": [
                {
                    "category_id": 0,
                    "bbox": coco_bbox,
                    "area": coco_bbox[2] * coco_bbox[3],
                    "iscrowd": 0,
                }
            ],
        }
        encoded = self.image_processor(
            images=image, annotations=target, return_tensors="pt", **proc_extra
        )

        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"][0],
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
            "gt_box_xyxy": torch.tensor(box_xyxy_orig, dtype=torch.float32),
            "preprocess_meta": preprocess_meta,
            "image_name": image_name,
        }


def collate_fn(image_processor, batch: List[Dict]) -> Dict:
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": [item["labels"] for item in batch],
        "orig_sizes": torch.stack([item["orig_size"] for item in batch], dim=0),
        "gt_boxes_xyxy": torch.stack([item["gt_box_xyxy"] for item in batch], dim=0),
        "preprocess_metas": [item.get("preprocess_meta") for item in batch],
        "image_names": [item["image_name"] for item in batch],
    }


def build_dataloader(
    dataset_root: Path,
    split_name: str,
    image_processor,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    preprocess_config: Optional[PreprocessConfig] = None,
    image_names: Optional[List[str]] = None,
    augment: bool = False,
) -> DataLoader:
    dataset = GBCUDetrDataset(
        dataset_root=dataset_root,
        split_name=split_name,
        image_processor=image_processor,
        preprocess_config=preprocess_config,
        image_names=image_names,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda x: collate_fn(image_processor, x),
    )
