import argparse
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

ROI_LABELS = {"nml", "abn"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("resnet50_roi_classifier")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def read_split_file(split_file: Path) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, cls = [x.strip() for x in line.split(",")[:2]]
            rows.append((name, int(cls)))
    return rows


def load_bbox_annotations(annotation_file: Path) -> Dict[str, List[List]]:
    with annotation_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_roi_box(raw_bbs: Sequence[Sequence]) -> List[float]:
    roi, fallback = [], []
    for item in raw_bbs:
        if len(item) != 2:
            continue
        label, box = item
        if not isinstance(box, list) or len(box) != 4:
            continue
        b = [float(v) for v in box]
        fallback.append(b)
        if str(label).lower() in ROI_LABELS:
            roi.append(b)
    pool = roi if roi else fallback
    if not pool:
        raise ValueError("No valid ROI box found.")
    return max(pool, key=lambda x: max(0.0, x[2] - x[0]) * max(0.0, x[3] - x[1]))


def sanitize_box_xyxy(box: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(v) for v in box]
    x0 = max(0.0, min(x0, width - 1))
    y0 = max(0.0, min(y0, height - 1))
    x1 = max(0.0, min(x1, width))
    y1 = max(0.0, min(y1, height))
    if x1 <= x0:
        x1 = min(width, x0 + 1.0)
    if y1 <= y0:
        y1 = min(height, y0 + 1.0)
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


@dataclass
class CropStats:
    total: int = 0
    fallback_to_gt: int = 0
    failed: int = 0


def generate_roi_crops(
    dataset_root: Path,
    output_dir: Path,
    seed: int,
    logger: logging.Logger,
) -> Dict[str, List[Tuple[Path, int]]]:
    imgs_dir = dataset_root / "imgs"
    bbox_annot = load_bbox_annotations(dataset_root / "bbox_annot.json")

    train_all = read_split_file(dataset_root / "train.txt")
    test_all = read_split_file(dataset_root / "test.txt")
    split_map = {"train": train_all, "test": test_all}

    crops_root = output_dir / "roi_crops"
    results: Dict[str, List[Tuple[Path, int]]] = {"train": [], "test": []}
    stats = {"train": CropStats(), "test": CropStats()}

    for split_name, items in split_map.items():
        split_dir = crops_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for image_name, class_id in items:
            stats[split_name].total += 1
            image_path = imgs_dir / image_name
            if not image_path.exists():
                stats[split_name].failed += 1
                continue
            image = Image.open(image_path).convert("RGB")
            w, h = image.size
            try:
                box = choose_roi_box(bbox_annot[image_name]["bbs"])
                x0, y0, x1, y1 = sanitize_box_xyxy(box, width=w, height=h)
                crop = image.crop((x0, y0, x1, y1))
                out_path = split_dir / f"{Path(image_name).stem}_cls{class_id}.jpg"
                crop.save(out_path, quality=95)
                results[split_name].append((out_path, class_id))
            except Exception:
                stats[split_name].failed += 1
                continue

    logger.info("ROI crop stats | train=%d test=%d", len(results["train"]), len(results["test"]))
    logger.info(
        "Test crop fallback_to_gt=%d failed_test=%d",
        stats["test"].fallback_to_gt,
        stats["test"].failed,
    )
    return results


class RoiCropDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], transform: transforms.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), int(label)


def compute_malignant_metrics(preds: np.ndarray, labels: np.ndarray, malignant_id: int = 2) -> Tuple[float, float]:
    pred_pos = preds == malignant_id
    true_pos = labels == malignant_id
    tp = int(np.logical_and(pred_pos, true_pos).sum())
    fn = int(np.logical_and(~pred_pos, true_pos).sum())
    tn = int(np.logical_and(~pred_pos, ~true_pos).sum())
    fp = int(np.logical_and(pred_pos, ~true_pos).sum())
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return sensitivity, specificity


def run_epoch_train(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: SGD, device: torch.device) -> float:
    model.train()
    losses: List[float] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def run_epoch_val(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, float, float]:
    model.eval()
    losses: List[float] = []
    preds_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        losses.append(float(loss.item()))
        preds_all.append(torch.argmax(logits, dim=1).cpu().numpy())
        labels_all.append(labels.cpu().numpy())
    if not labels_all:
        return 0.0, 0.0, 0.0, 0.0
    preds_np = np.concatenate(preds_all, axis=0)
    labels_np = np.concatenate(labels_all, axis=0)
    acc = float((preds_np == labels_np).mean())
    sens, spec = compute_malignant_metrics(preds_np, labels_np, malignant_id=2)
    return float(np.mean(losses)), acc, sens, spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train ResNet50 classifier on ROI crops")
    parser.add_argument("--dataset_root", type=str, default="GBCU")
    parser.add_argument("--output_dir", type=str, default="outputs/resnet50_roi_cls")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--step_size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_tensorboard", action="store_true")
    return parser.parse_args()


def load_previous_best_metrics(best_ckpt: Path, metrics_json: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if best_ckpt.exists():
        try:
            raw = torch.load(best_ckpt, map_location="cpu")
            if isinstance(raw, dict):
                if "epoch" in raw:
                    metrics["best_epoch_by_malignant_sensitivity"] = int(raw["epoch"])
                if "test_acc" in raw:
                    metrics["best_accuracy"] = float(raw["test_acc"])
                if "malignant_sensitivity" in raw:
                    metrics["best_malignant_sensitivity"] = float(raw["malignant_sensitivity"])
                if "malignant_specificity" in raw:
                    metrics["best_malignant_specificity"] = float(raw["malignant_specificity"])
        except Exception:
            pass

    if metrics_json.exists():
        try:
            with metrics_json.open("r", encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                for key in [
                    "best_epoch_by_malignant_sensitivity",
                    "best_accuracy",
                    "best_malignant_sensitivity",
                    "best_malignant_specificity",
                ]:
                    if key in saved:
                        metrics[key] = saved[key]
        except Exception:
            pass
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    logs_dir = output_dir / "logs"
    tb_dir = output_dir / "tensorboard"
    ckpt_dir = output_dir / "checkpoints"
    for d in [output_dir, logs_dir, tb_dir, ckpt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(logs_dir / "train.log")
    writer = SummaryWriter(log_dir=tb_dir) if args.use_tensorboard else None
    logger.info("Run started at %s", datetime.now().isoformat(timespec="seconds"))
    logger.info("Args: %s", vars(args))

    split_samples = generate_roi_crops(
        dataset_root=Path(args.dataset_root),
        output_dir=output_dir,
        seed=args.seed,
        logger=logger,
    )

    weights = ResNet50_Weights.IMAGENET1K_V1
    tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    train_loader = DataLoader(
        RoiCropDataset(split_samples["train"], tf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        RoiCropDataset(split_samples["test"], tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(weights=weights)
    model.fc = nn.Linear(int(model.fc.in_features), 3)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_sens = -1.0
    best_acc = 0.0
    best_spec = 0.0
    best_epoch = -1
    best_ckpt = ckpt_dir / "best_resnet50.pth"
    metrics_json = output_dir / "final_test_metrics.json"

    previous_best = load_previous_best_metrics(best_ckpt=best_ckpt, metrics_json=metrics_json)
    if previous_best:
        logger.info(
            "Previous best (from existing outputs) | epoch=%s acc=%.4f malignant_sens=%.4f malignant_spec=%.4f",
            str(previous_best.get("best_epoch_by_malignant_sensitivity", "N/A")),
            float(previous_best.get("best_accuracy", 0.0)),
            float(previous_best.get("best_malignant_sensitivity", 0.0)),
            float(previous_best.get("best_malignant_specificity", 0.0)),
        )
    else:
        logger.info("Previous best (from existing outputs) | none found, start from fresh metrics.")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch_train(model, train_loader, criterion, optimizer, device)
        lr_now = float(optimizer.param_groups[0]["lr"])
        logger.info(
            "[Epoch %03d/%03d] train_loss=%.4f lr=%.6f",
            epoch,
            args.epochs,
            train_loss,
            lr_now,
        )

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/lr", lr_now, epoch)

        scheduler.step()

        test_loss, test_acc, test_sens, test_spec = run_epoch_val(model, test_loader, criterion, device)
        logger.info(
            "[Epoch %03d/%03d] test_loss=%.4f test_acc=%.4f malignant_sens=%.4f malignant_spec=%.4f",
            epoch,
            args.epochs,
            test_loss,
            test_acc,
            test_sens,
            test_spec,
        )
        if writer is not None:
            writer.add_scalar("test/loss", test_loss, epoch)
            writer.add_scalar("test/accuracy", test_acc, epoch)
            writer.add_scalar("test/malignant_sensitivity", test_sens, epoch)
            writer.add_scalar("test/malignant_specificity", test_spec, epoch)

        if test_sens > best_sens:
            best_sens = test_sens
            best_acc = test_acc
            best_spec = test_spec
            best_epoch = epoch
            try:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                        "malignant_sensitivity": test_sens,
                        "malignant_specificity": test_spec,
                        "args": vars(args),
                    },
                    best_ckpt,
                )
                logger.info("Saved best checkpoint by malignant_sensitivity: %s", best_ckpt)
            except Exception as e:
                logger.error("Checkpoint save failed (likely disk quota/space issue): %s", e)

    with (output_dir / "final_test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch_by_malignant_sensitivity": (
                    best_epoch
                ),
                "best_accuracy": best_acc,
                "best_malignant_sensitivity": best_sens,
                "best_malignant_specificity": best_spec,
            },
            f,
            indent=2,
        )

    if writer is not None:
        writer.close()
    logger.info("== Classifier best ==")
    logger.info(
        "accuracy=%.4f malignant_sensitivity=%.4f malignant_specificity=%.4f",
        best_acc,
        best_sens,
        best_spec,
    )
    logger.info(
        "Training finished. best_accuracy=%.4f best_malignant_sensitivity=%.4f best_malignant_specificity=%.4f",
        best_acc,
        best_sens,
        best_spec,
    )


if __name__ == "__main__":
    main()
