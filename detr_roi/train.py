import argparse
import csv
import json
import logging
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoImageProcessor,
    DeformableDetrForObjectDetection,
    DetrForObjectDetection,
)

from detr_roi.dataset import build_dataloader, read_split_file
from detr_roi.evaluate import evaluate_model, print_metric_table
from detr_roi.preprocess import PreprocessConfig

CLASS_ID_TO_NAME = {0: "normal", 1: "benign", 2: "malignant"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser("Train DETR ROI detector on GBCU")
    parser.add_argument("--dataset_root", type=str, default="GBCU")
    parser.add_argument("--model_name", type=str, default="facebook/detr-resnet-50")
    parser.add_argument(
        "--model_type",
        type=str,
        default="detr",
        choices=["detr", "deformable_detr"],
        help="Backbone family to use while keeping training/eval pipeline unchanged.",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/detr_roi")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio sampled from train split (0 <= val_ratio < 1).",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--nms_iou_threshold", type=float, default=0.5)
    parser.add_argument("--max_detections", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every_steps", type=int, default=20)
    parser.add_argument("--use_tensorboard", action="store_true")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--early_stopping_patience", type=int, default=40)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument(
        "--use_paper_preprocess",
        action="store_true",
        help="Resize (shortest/longest edge), fixed center crop, then HF rescale+normalize.",
    )
    parser.add_argument(
        "--prep_shortest_edge",
        type=int,
        default=256,
        help="Shortest edge after resize (ImageNet-style default 256 before 224 crop).",
    )
    parser.add_argument("--prep_longest_edge", type=int, default=1333)
    parser.add_argument(
        "--prep_center_crop_h",
        type=int,
        default=800,
        help="Fixed center crop height (pixels) after resize.",
    )
    parser.add_argument(
        "--prep_center_crop_w",
        type=int,
        default=800,
        help="Fixed center crop width (pixels) after resize.",
    )
    parser.add_argument(
        "--num_vis_per_class",
        type=int,
        default=3,
        help="Number of visualization samples per class at training end.",
    )
    return parser.parse_args()


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("detr_roi_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def build_warmup_cosine_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler, warmup_steps


def _read_split_labels(split_file: Path) -> dict:
    labels = {}
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 2:
                continue
            image_name = parts[0]
            try:
                class_id = int(parts[1])
            except ValueError:
                continue
            labels[image_name] = class_id
    return labels


def _draw_xyxy(draw_ctx: ImageDraw.ImageDraw, box_xyxy, color, width=3):
    x0, y0, x1, y1 = [float(v) for v in box_xyxy]
    draw_ctx.rectangle([x0, y0, x1, y1], outline=color, width=width)


def save_end_of_train_visualizations(
    dataset_root: Path,
    output_dir: Path,
    num_per_class: int,
    logger: logging.Logger,
) -> None:
    records_file = output_dir / "best_test_roi_pred.json"
    if not records_file.exists():
        logger.warning("Skip visualization: %s not found.", records_file)
        return

    split_labels = _read_split_labels(dataset_root / "test.txt")
    with records_file.open("r", encoding="utf-8") as f:
        pred_records = json.load(f)

    vis_dir = output_dir / "vis_samples_3class"
    vis_dir.mkdir(parents=True, exist_ok=True)
    image_dir = dataset_root / "imgs"

    by_class = {0: [], 1: [], 2: []}
    for image_name, rec in pred_records.items():
        class_id = split_labels.get(image_name)
        if class_id not in by_class:
            continue
        by_class[class_id].append((image_name, rec))

    saved = 0
    for class_id in [0, 1, 2]:
        class_name = CLASS_ID_TO_NAME[class_id]
        class_dir = vis_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        candidates = sorted(
            by_class[class_id],
            key=lambda x: float(x[1].get("best_iou", 0.0)),
            reverse=True,
        )
        selected = candidates[:num_per_class]
        for idx, (image_name, rec) in enumerate(selected, start=1):
            image_path = image_dir / image_name
            if not image_path.exists():
                continue
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)

            gt_boxes = rec.get("Gold", [])
            pred_boxes = rec.get("Boxes", [])
            pred_scores = rec.get("Scores", [])

            if gt_boxes:
                _draw_xyxy(draw, gt_boxes[0], color=(0, 255, 0), width=4)
            if pred_boxes:
                _draw_xyxy(draw, pred_boxes[0], color=(255, 0, 0), width=3)
                if pred_scores:
                    score_text = f"{float(pred_scores[0]):.3f}"
                    px0, py0 = [int(v) for v in pred_boxes[0][:2]]
                    draw.text((px0 + 4, max(py0 - 16, 2)), f"pred {score_text}", fill=(255, 0, 0))

            draw.text((8, 8), f"{class_name} | {image_name}", fill=(255, 255, 0))
            draw.text((8, 28), f"best_iou={float(rec.get('best_iou', 0.0)):.3f}", fill=(255, 255, 0))

            out_name = (
                f"{idx:02d}_{Path(image_name).stem}_"
                f"iou_{float(rec.get('best_iou', 0.0)):.3f}.jpg"
            )
            image.save(class_dir / out_name, quality=95)
            saved += 1

    logger.info("Saved %d visualization images to: %s", saved, vis_dir)


def _f(rec: dict, key: str, default: float = 0.0) -> float:
    v = rec.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(rec: dict, key: str, default: int = 0) -> int:
    v = rec.get(key, default)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


REGISTRY_FIELDNAMES = [
    "timestamp",
    "run_name",
    "output_dir",
    "dataset_root",
    "best_epoch",
    "best_mIoU",
    "best_mIoU_std",
    "best_precision",
    "best_recall",
    "best_TP",
    "best_FP",
    "best_FN",
    "best_num_images",
    "best_avg_pred_per_image",
    "final_epoch",
    "final_mIoU",
    "final_mIoU_std",
    "final_precision",
    "final_recall",
    "final_TP",
    "final_FP",
    "final_FN",
    "final_num_images",
    "final_avg_pred_per_image",
    "miou_rule",
    "matching_rule",
    "model_name",
    "epochs",
    "train_batch_size",
    "eval_batch_size",
    "num_workers",
    "lr",
    "lr_backbone",
    "weight_decay",
    "max_grad_norm",
    "warmup_ratio",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "seed",
    "score_threshold",
    "nms_iou_threshold",
    "max_detections",
    "use_paper_preprocess",
    "prep_shortest_edge",
    "prep_longest_edge",
    "prep_center_crop_h",
    "prep_center_crop_w",
    "num_vis_per_class",
]


def save_run_records(
    args,
    output_dir: Path,
    best_ckpt_path: Path,
    history: list,
    logger: logging.Logger,
) -> None:
    if not history:
        return

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    # Best epoch follows model-selection rule used in training:
    # prioritize recall, then mIoU as tie-breaker.
    best_record = max(
        history,
        key=lambda x: (
            float(x.get("recall", -1.0)),
            float(x.get("mIoU", -1.0)),
        ),
    )
    final_record = history[-1]

    train_config_file = logs_dir / "train_config.json"
    best_metrics_file = logs_dir / "best_metrics.json"
    run_summary_file = logs_dir / "run_summary.json"
    train_payload = {
        "hyperparameters": vars(args),
        "best_metrics": best_record,
        "final_metrics": final_record,
    }
    with train_config_file.open("w", encoding="utf-8") as f:
        json.dump(train_payload, f, indent=2)
    with best_metrics_file.open("w", encoding="utf-8") as f:
        json.dump(best_record, f, indent=2)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_name": output_dir.name,
        "output_dir": str(output_dir),
        "dataset_root": str(args.dataset_root),
        "best_checkpoint": str(best_ckpt_path),
        "best_epoch": int(best_record.get("epoch", -1)),
        "best_metrics": best_record,
        "final_metrics": final_record,
        "config": vars(args),
    }
    with run_summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Global experiment registry for model/hyper-parameter comparison.
    registry_jsonl = output_dir.parent / "experiment_registry.jsonl"
    with registry_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")

    registry_csv = output_dir.parent / "experiment_registry.csv"
    if registry_csv.exists() and registry_csv.stat().st_size > 0:
        first_line = registry_csv.read_text(encoding="utf-8").splitlines()[0]
        if "best_mIoU_std" not in first_line:
            backup = registry_csv.with_suffix(".csv.legacy_bak")
            registry_csv.rename(backup)
            logger.warning(
                "Renamed old experiment_registry.csv (incompatible columns) to %s; "
                "a new registry will be created on next write.",
                backup,
            )

    registry_row = {
        "timestamp": summary["timestamp"],
        "run_name": summary["run_name"],
        "output_dir": summary["output_dir"],
        "dataset_root": str(args.dataset_root),
        "best_epoch": _i(best_record, "epoch", -1),
        "best_mIoU": _f(best_record, "mIoU"),
        "best_mIoU_std": _f(best_record, "mIoU_std"),
        "best_precision": _f(best_record, "precision"),
        "best_recall": _f(best_record, "recall"),
        "best_TP": _i(best_record, "TP"),
        "best_FP": _i(best_record, "FP"),
        "best_FN": _i(best_record, "FN"),
        "best_num_images": _i(best_record, "num_images"),
        "best_avg_pred_per_image": _f(best_record, "avg_pred_per_image"),
        "final_epoch": _i(final_record, "epoch"),
        "final_mIoU": _f(final_record, "mIoU"),
        "final_mIoU_std": _f(final_record, "mIoU_std"),
        "final_precision": _f(final_record, "precision"),
        "final_recall": _f(final_record, "recall"),
        "final_TP": _i(final_record, "TP"),
        "final_FP": _i(final_record, "FP"),
        "final_FN": _i(final_record, "FN"),
        "final_num_images": _i(final_record, "num_images"),
        "final_avg_pred_per_image": _f(final_record, "avg_pred_per_image"),
        "miou_rule": str(best_record.get("miou_rule", "")),
        "matching_rule": str(best_record.get("matching_rule", "")),
        "model_name": args.model_name,
        "epochs": int(args.epochs),
        "train_batch_size": int(args.train_batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "num_workers": int(args.num_workers),
        "lr": float(args.lr),
        "lr_backbone": float(args.lr_backbone),
        "weight_decay": float(args.weight_decay),
        "max_grad_norm": float(args.max_grad_norm),
        "warmup_ratio": float(args.warmup_ratio),
        "early_stopping_patience": int(args.early_stopping_patience),
        "early_stopping_min_delta": float(args.early_stopping_min_delta),
        "seed": int(args.seed),
        "score_threshold": float(args.score_threshold),
        "nms_iou_threshold": float(args.nms_iou_threshold),
        "max_detections": int(args.max_detections),
        "use_paper_preprocess": bool(args.use_paper_preprocess),
        "prep_shortest_edge": int(args.prep_shortest_edge),
        "prep_longest_edge": int(args.prep_longest_edge),
        "prep_center_crop_h": int(args.prep_center_crop_h),
        "prep_center_crop_w": int(args.prep_center_crop_w),
        "num_vis_per_class": int(args.num_vis_per_class),
    }
    write_header = not registry_csv.exists()
    with registry_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(registry_row)

    logger.info("Saved run records: %s, %s, %s", train_config_file, best_metrics_file, run_summary_file)
    logger.info("Updated experiment registry: %s and %s", registry_csv, registry_jsonl)


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    tb_dir = output_dir / "tensorboard"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(logs_dir / "train.log")
    step_log_file = logs_dir / "train_steps.jsonl"
    epoch_csv_file = logs_dir / "epoch_metrics.csv"
    latest_metrics_file = logs_dir / "latest_metrics.json"
    writer = SummaryWriter(log_dir=tb_dir) if args.use_tensorboard else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Output directory: %s", output_dir)

    image_processor = AutoImageProcessor.from_pretrained(args.model_name)
    preprocess_config = None
    if args.use_paper_preprocess:
        preprocess_config = PreprocessConfig(
            use_geometric=True,
            shortest_edge=args.prep_shortest_edge,
            longest_edge=args.prep_longest_edge,
            center_crop_h=args.prep_center_crop_h,
            center_crop_w=args.prep_center_crop_w,
        )
        logger.info(
            "Paper preprocess: shortest_edge=%d longest_edge=%d center_crop=%dx%d",
            args.prep_shortest_edge,
            args.prep_longest_edge,
            args.prep_center_crop_h,
            args.prep_center_crop_w,
        )
    model_cls = DetrForObjectDetection if args.model_type == "detr" else DeformableDetrForObjectDetection
    model = model_cls.from_pretrained(
        args.model_name,
        num_labels=1,
        ignore_mismatched_sizes=True,
        id2label={0: "roi"},
        label2id={"roi": 0},
    )

    model.to(device)
    logger.info("Model type: %s | model name: %s", args.model_type, args.model_name)

    if args.val_ratio != 0.0:
        logger.info(
            "Ignore val_ratio=%.3f and evaluate on test split each epoch (patient-level split policy).",
            args.val_ratio,
        )
    train_image_names = read_split_file(Path(args.dataset_root) / "train.txt")
    logger.info(
        "Using full train split for optimization: train=%d; per-epoch evaluation uses test split.",
        len(train_image_names),
    )

    train_loader = build_dataloader(
        dataset_root=Path(args.dataset_root),
        split_name="train",
        image_processor=image_processor,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        preprocess_config=preprocess_config,
        image_names=train_image_names,
        augment=True,
    )
    test_loader = build_dataloader(
        dataset_root=Path(args.dataset_root),
        split_name="test",
        image_processor=image_processor,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        preprocess_config=preprocess_config,
        augment=False,
    )

    backbone_params = []
    non_backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            non_backbone_params.append(param)

    optimizer = AdamW(
        [
            {"params": non_backbone_params, "lr": args.lr},
            {"params": backbone_params, "lr": args.lr_backbone},
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    scheduler, warmup_steps = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
    )
    logger.info(
        "LR scheduler: warmup+cosine, total_steps=%d warmup_steps=%d",
        total_steps,
        warmup_steps,
    )

    history = []
    best_recall = -1.0
    best_miou_tiebreak = -1.0
    best_ckpt_path = ckpt_dir / "best_recall_then_miou.pth"
    global_step = 0
    epochs_without_improve = 0

    with epoch_csv_file.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(
            f,
            extrasaction="ignore",
            fieldnames=[
                "epoch",
                "train_loss",
                "eval_split",
                "mIoU",
                "mIoU_std",
                "precision",
                "recall",
                "TP",
                "FP",
                "FN",
                "num_images",
                "score_threshold",
                "nms_iou_threshold",
                "max_detections",
                "avg_pred_per_image",
                "miou_rule",
                "matching_rule",
            ],
        )
        writer_csv.writeheader()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for step, batch in enumerate(train_loader, start=1):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss detected at epoch={epoch}, step={step}: {loss.item()}"
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            train_losses.append(float(loss.item()))
            global_step += 1

            step_info = {
                "global_step": global_step,
                "epoch": epoch,
                "step_in_epoch": step,
                "loss": float(loss.item()),
                "lr_main": float(optimizer.param_groups[0]["lr"]),
                "lr_backbone": float(optimizer.param_groups[1]["lr"]),
            }
            with step_log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(step_info) + "\n")

            if writer is not None:
                writer.add_scalar("train/loss_step", step_info["loss"], global_step)
                writer.add_scalar("train/lr_main", step_info["lr_main"], global_step)
                writer.add_scalar("train/lr_backbone", step_info["lr_backbone"], global_step)

            if step % args.log_every_steps == 0:
                logger.info(
                    "Epoch %d Step %d/%d Loss %.4f LR(main/backbone)=%.2e/%.2e",
                    epoch,
                    step,
                    len(train_loader),
                    loss.item(),
                    optimizer.param_groups[0]["lr"],
                    optimizer.param_groups[1]["lr"],
                )

        avg_loss = float(np.mean(train_losses)) if train_losses else 0.0
        logger.info("Epoch %d/%d Train Loss: %.4f", epoch, args.epochs, avg_loss)

        eval_result = evaluate_model(
            model=model,
            image_processor=image_processor,
            dataloader=test_loader,
            device=device,
            score_threshold=args.score_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            max_detections=args.max_detections,
        )
        metrics = eval_result["metrics"]
        print_metric_table(metrics)
        logger.info(
            "Test epoch %d: mIoU=%.4f mIoU_std=%.4f precision=%.4f recall=%.4f avg_pred_per_image=%.3f",
            epoch,
            metrics["mIoU"],
            metrics["mIoU_std"],
            metrics["precision"],
            metrics["recall"],
            metrics.get("avg_pred_per_image", 0.0),
        )

        current_recall = float(metrics.get("recall", 0.0))
        current_miou = float(metrics.get("mIoU", 0.0))
        is_better = (
            current_recall > best_recall
            or (
                abs(current_recall - best_recall) <= 1e-12
                and current_miou > best_miou_tiebreak
            )
        )

        if is_better:
            best_recall = current_recall
            best_miou_tiebreak = current_miou
            epochs_without_improve = 0
            torch.save(model.state_dict(), best_ckpt_path)
            with (output_dir / "best_test_roi_pred.json").open("w", encoding="utf-8") as f:
                json.dump(eval_result["records"], f, indent=2)
            logger.info("Best checkpoint updated: %s", best_ckpt_path)
        else:
            improvement = current_recall - best_recall
            if improvement < args.early_stopping_min_delta:
                epochs_without_improve += 1
            else:
                epochs_without_improve = 0

        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "eval_split": "test",
                **metrics,
            }
        )
        with (output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        with latest_metrics_file.open("w", encoding="utf-8") as f:
            json.dump(history[-1], f, indent=2)
        with epoch_csv_file.open("a", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(
                f,
                extrasaction="ignore",
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "eval_split",
                    "mIoU",
                    "mIoU_std",
                    "precision",
                    "recall",
                    "TP",
                    "FP",
                    "FN",
                    "num_images",
                    "score_threshold",
                    "nms_iou_threshold",
                    "max_detections",
                    "avg_pred_per_image",
                    "miou_rule",
                    "matching_rule",
                ],
            )
            writer_csv.writerow(history[-1])

        if writer is not None:
            writer.add_scalar("train/loss_epoch", avg_loss, epoch)
            writer.add_scalar("val/mIoU", metrics["mIoU"], epoch)
            writer.add_scalar("val/mIoU_std", metrics["mIoU_std"], epoch)
            writer.add_scalar("val/precision", metrics["precision"], epoch)
            writer.add_scalar("val/recall", metrics["recall"], epoch)
            writer.add_scalar("train/epochs_without_improve", epochs_without_improve, epoch)

        if epochs_without_improve >= args.early_stopping_patience:
            logger.info(
                "Early stopping triggered at epoch %d (patience=%d, min_delta=%.6f)",
                epoch,
                args.early_stopping_patience,
                args.early_stopping_min_delta,
            )
            break

    if writer is not None:
        writer.close()

    # Final test evaluation with best checkpoint (test is not used for model selection).
    best_state = torch.load(best_ckpt_path, map_location="cpu")
    model.load_state_dict(best_state)
    model.to(device)
    final_test_result = evaluate_model(
        model=model,
        image_processor=image_processor,
        dataloader=test_loader,
        device=device,
        score_threshold=args.score_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        max_detections=args.max_detections,
    )
    with (output_dir / "best_test_roi_pred.json").open("w", encoding="utf-8") as f:
        json.dump(final_test_result["records"], f, indent=2)
    with (logs_dir / "best_test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_test_result["metrics"], f, indent=2)
    logger.info(
        "Final test (best checkpoint): mIoU=%.4f mIoU_std=%.4f precision=%.4f recall=%.4f",
        final_test_result["metrics"]["mIoU"],
        final_test_result["metrics"]["mIoU_std"],
        final_test_result["metrics"]["precision"],
        final_test_result["metrics"]["recall"],
    )

    try:
        save_end_of_train_visualizations(
            dataset_root=Path(args.dataset_root),
            output_dir=output_dir,
            num_per_class=args.num_vis_per_class,
            logger=logger,
        )
    except Exception as e:
        logger.warning("Failed to generate end-of-train visualizations: %s", e)

    try:
        save_run_records(
            args=args,
            output_dir=output_dir,
            best_ckpt_path=best_ckpt_path,
            history=history,
            logger=logger,
        )
    except Exception as e:
        logger.warning("Failed to save run summary/registry: %s", e)

    logger.info(
        "Training complete. Best selection rule recall->mIoU: best_recall=%.2f best_mIoU_tiebreak=%.2f",
        best_recall * 100.0,
        best_miou_tiebreak * 100.0,
    )
    logger.info("Best checkpoint: %s", best_ckpt_path)


if __name__ == "__main__":
    main()
