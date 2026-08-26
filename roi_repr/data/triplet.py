"""Dynamic triplet sampling."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional

from torch.utils.data import Dataset
from torchvision import transforms

from roi_repr.config import ReprConfig
from roi_repr.data.dataset import RoiPatchDataset, Sample


class TripletDataset(Dataset):
    def __init__(
        self,
        cfg: ReprConfig,
        samples: List[Sample],
        transform: transforms.Compose,
        triplets_per_epoch: Optional[int] = None,
    ) -> None:
        self.cfg = cfg
        self.base = RoiPatchDataset(cfg, samples, transform)
        self.triplets_per_epoch = triplets_per_epoch or max(len(samples), cfg.batch_size * 32)
        self._epoch = 0
        labels = [self.cfg.map_label(raw) for _, raw in self.base.samples]
        self.by_label: Dict[int, List[int]] = defaultdict(list)
        for i, lb in enumerate(labels):
            self.by_label[int(lb)].append(i)
        self.labels = labels
        self.all_indices = list(range(len(self.base.samples)))
        self.label_list = sorted(self.by_label.keys())
        if len(self.label_list) < 2:
            raise ValueError("TripletDataset requires at least 2 classes.")
        for lb, idxs in self.by_label.items():
            if len(idxs) < 2:
                print(
                    f"[TripletDataset] warning: class {self.cfg.class_tag(lb)} "
                    f"has only {len(idxs)} sample(s); positive may equal anchor."
                )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.triplets_per_epoch

    def _rng(self, idx: int) -> random.Random:
        return random.Random(self.cfg.seed + self._epoch * 1_000_003 + idx)

    def __getitem__(self, idx: int):
        rng = self._rng(idx)
        anchor_i = rng.choice(self.all_indices)
        anchor_label = self.labels[anchor_i]
        pos_pool = self.by_label[anchor_label]
        if len(pos_pool) > 1:
            pos_i = anchor_i
            while pos_i == anchor_i:
                pos_i = rng.choice(pos_pool)
        else:
            pos_i = pos_pool[0]
        neg_label = rng.choice([lb for lb in self.label_list if lb != anchor_label])
        neg_i = rng.choice(self.by_label[neg_label])
        anchor, lbl = self.base[anchor_i]
        positive, _ = self.base[pos_i]
        negative, _ = self.base[neg_i]
        return anchor, positive, negative, lbl
