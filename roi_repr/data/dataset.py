"""Online GT ROI crop dataset."""

from __future__ import annotations

from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from roi_repr.config import ReprConfig
from train_resnet50_roi_classifier import choose_roi_box, load_bbox_annotations, sanitize_box_xyxy

Sample = Tuple[str, int]


class RoiPatchDataset(Dataset):
    def __init__(
        self,
        cfg: ReprConfig,
        samples: List[Sample],
        transform: transforms.Compose,
        return_raw_label: bool = False,
    ) -> None:
        self.cfg = cfg
        self.dataset_root = cfg.dataset_root
        self.imgs_dir = self.dataset_root / "imgs"
        self.bbox = load_bbox_annotations(self.dataset_root / "bbox_annot.json")
        self.samples = list(samples)
        self.transform = transform
        self.return_raw_label = return_raw_label

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        name, raw_label = self.samples[idx]
        img = Image.open(self.imgs_dir / name).convert("RGB")
        w, h = img.size
        box = choose_roi_box(self.bbox[name]["bbs"])
        x0, y0, x1, y1 = sanitize_box_xyxy(box, width=w, height=h)
        crop = img.crop((x0, y0, x1, y1))
        x = self.transform(crop)
        label = self.cfg.map_label(raw_label)
        if self.return_raw_label:
            return x, int(label), int(raw_label)
        return x, int(label)
