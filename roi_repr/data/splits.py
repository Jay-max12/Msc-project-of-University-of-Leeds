"""GT ROI datasets and 3-fold CV splits (no separate hold-out val)."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from roi_repr.config import ReprConfig
from roi_repr.data.cv_splits import load_cv_folds, stratify_for_task_mode
from roi_repr.data.dataset import RoiPatchDataset
from roi_repr.data.siamese import SiamesePairDataset
from roi_repr.data.triplet import TripletDataset

Sample = Tuple[str, int]
FoldSplits = Dict[str, List[Sample]]


def seed_worker(worker_id: int) -> None:
    """Seed Python/NumPy from the deterministic seed assigned by DataLoader."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def build_roi_transform(cfg: ReprConfig, *, train: bool = False) -> transforms.Compose:
    """Eval transform by default; train=True adds configured augmentations only."""
    mean, std = cfg.imagenet_norm()
    steps: list = [transforms.Resize(cfg.roi_resize)]
    if train:
        if cfg.train_rotation_degrees > 0:
            steps.append(transforms.RandomRotation(degrees=cfg.train_rotation_degrees))
        if cfg.train_hflip_prob > 0:
            steps.append(transforms.RandomHorizontalFlip(p=cfg.train_hflip_prob))
        if cfg.train_jitter_brightness > 0 or cfg.train_jitter_contrast > 0:
            steps.append(
                transforms.ColorJitter(
                    brightness=cfg.train_jitter_brightness or 0.0,
                    contrast=cfg.train_jitter_contrast or 0.0,
                )
            )
    steps.extend(
        [
            transforms.CenterCrop(cfg.roi_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(steps)


def build_roi_train_transform(cfg: ReprConfig) -> transforms.Compose:
    return build_roi_transform(cfg, train=True)


def list_valid_samples(cfg: ReprConfig, split: str) -> List[Sample]:
    from train_resnet50_roi_classifier import load_bbox_annotations, read_split_file

    bbox = load_bbox_annotations(cfg.dataset_root / "bbox_annot.json")
    imgs_dir = cfg.dataset_root / "imgs"
    rows = read_split_file(cfg.dataset_root / f"{split}.txt")
    out: List[Sample] = []
    for name, cls_id in rows:
        if name not in bbox:
            continue
        if not (imgs_dir / name).exists():
            continue
        out.append((name, int(cls_id)))
    return out


def build_cv_folds(cfg: ReprConfig) -> List[FoldSplits]:
    """Load pre-defined CV folds; manifest lives under dataset_root/cv_splits/."""
    stratify = stratify_for_task_mode(cfg.task_mode)
    folds, split_path = load_cv_folds(
        cfg.dataset_root,
        n_cv_folds=cfg.n_cv_folds,
        seed=cfg.seed,
        stratify=stratify,
        cv_split_file=cfg.cv_split_file,
    )
    print(f"[CV splits] loaded fixed manifest: {split_path}")
    return folds


def raw_class_dist(rows: List[Sample]) -> str:
    cnt = np.bincount([r[1] for r in rows], minlength=3)
    return f"normal={int(cnt[0])}, benign={int(cnt[1])}, malignant={int(cnt[2])}"


def task_class_dist(cfg: ReprConfig, rows: List[Sample]) -> str:
    labels = [cfg.map_label(r[1]) for r in rows]
    cnt = np.bincount(labels, minlength=cfg.num_classes)
    return ", ".join(f"{cfg.class_tag(i)}={int(cnt[i])}" for i in range(cfg.num_classes))


def print_cv_summary(cfg: ReprConfig, folds: List[FoldSplits]) -> None:
    print(f"[TASK_MODE={cfg.task_mode}] num_classes={cfg.num_classes} malignant_id={cfg.malignant_id}")
    aug_parts: List[str] = []
    if cfg.train_hflip_prob > 0:
        aug_parts.append(f"hflip p={cfg.train_hflip_prob}")
    if cfg.train_rotation_degrees > 0:
        aug_parts.append(f"rotation ±{cfg.train_rotation_degrees}°")
    if cfg.train_jitter_brightness > 0 or cfg.train_jitter_contrast > 0:
        aug_parts.append(
            f"ColorJitter brightness={cfg.train_jitter_brightness} contrast={cfg.train_jitter_contrast}"
        )
    print(f"[Augment] train: {', '.join(aug_parts) if aug_parts else 'none'} (val/test: none)")
    print(f"[CV] {cfg.n_cv_folds}-fold stratified on train ({len(folds[0]['train']) + len(folds[0]['val'])} samples)")
    for fd in folds:
        fi = fd["fold"]
        print(f"  fold {fi}: train ({len(fd['train']):4d}) raw: {raw_class_dist(fd['train'])}")
        print(f"           task: {task_class_dist(cfg, fd['train'])}")
        print(f"           fold_val ({len(fd['val']):4d}) raw: {raw_class_dist(fd['val'])}")
        print(f"           task: {task_class_dist(cfg, fd['val'])}")
    test_rows = folds[0]["test"]
    print(f"[test ] held-out ({len(test_rows):4d}) raw: {raw_class_dist(test_rows)}")
    print(f"         task: {task_class_dist(cfg, test_rows)}")


def make_single_loader(
    cfg: ReprConfig,
    rows: List[Sample],
    shuffle: bool,
    return_raw_label: bool = False,
    train: bool = False,
    loader_seed: int | None = None,
) -> DataLoader:
    tfm = build_roi_train_transform(cfg) if train else build_roi_transform(cfg)
    ds = RoiPatchDataset(cfg, rows, tfm, return_raw_label=return_raw_label)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=make_loader_generator(cfg.seed if loader_seed is None else loader_seed),
    )


def make_train_loader(
    cfg: ReprConfig,
    rows: List[Sample],
    loader_seed: int | None = None,
) -> Tuple[DataLoader, object]:
    tfm = build_roi_train_transform(cfg)
    seed = cfg.seed if loader_seed is None else loader_seed
    if cfg.method == "siamese":
        ds = SiamesePairDataset(cfg, rows, tfm)
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=make_loader_generator(seed),
        ), ds
    if cfg.method == "triplet":
        ds = TripletDataset(cfg, rows, tfm)
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=make_loader_generator(seed),
        ), ds
    return make_single_loader(
        cfg,
        rows,
        shuffle=True,
        train=True,
        loader_seed=seed,
    ), None
