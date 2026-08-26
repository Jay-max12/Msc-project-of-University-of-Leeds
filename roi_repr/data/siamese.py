"""Dynamic pair sampling for Siamese + contrastive training."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from roi_repr.config import ReprConfig
from roi_repr.data.dataset import RoiPatchDataset, Sample


class SiamesePairDataset(Dataset):
    def __init__(
        self,
        cfg: ReprConfig,
        samples: List[Sample],
        transform: transforms.Compose,
        pairs_per_epoch: Optional[int] = None,
    ) -> None:
        self.cfg = cfg
        self.base = RoiPatchDataset(cfg, samples, transform)
        self.pairs_per_epoch = pairs_per_epoch or max(len(samples), cfg.batch_size * 32)
        self._epoch = 0
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        labels = [self.cfg.map_label(raw) for _, raw in self.base.samples]
        self.by_label: Dict[int, List[int]] = defaultdict(list)
        for i, lb in enumerate(labels):
            self.by_label[int(lb)].append(i)
        self.labels = labels
        self.all_indices = list(range(len(self.base.samples)))
        self.label_list = sorted(self.by_label.keys())
        if len(self.label_list) < 2:
            raise ValueError("SiamesePairDataset requires at least 2 classes.")

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def _rng(self, idx: int) -> random.Random:
        return random.Random(self.cfg.seed + self._epoch * 1_000_003 + idx)

    def _sample_index(self, rng: random.Random, label: int, exclude: Optional[int] = None) -> int:
        pool = self.by_label[label]
        for _ in range(20):
            j = rng.choice(pool)
            if exclude is None or j != exclude:
                return j
        return rng.choice(pool)

    def __getitem__(self, idx: int):
        rng = self._rng(idx)
        anchor_i = rng.choice(self.all_indices)
        anchor_label = self.labels[anchor_i]
        same_class = rng.random() < 0.5
        if same_class:
            j = self._sample_index(rng, anchor_label, exclude=anchor_i)
            pair_label = 1
        else:
            neg_label = rng.choice([lb for lb in self.label_list if lb != anchor_label])
            j = self._sample_index(rng, neg_label)
            pair_label = 0
        img1, lbl1 = self.base[anchor_i]
        img2, lbl2 = self.base[j]
        return img1, img2, torch.tensor(pair_label, dtype=torch.float32), lbl1, lbl2
