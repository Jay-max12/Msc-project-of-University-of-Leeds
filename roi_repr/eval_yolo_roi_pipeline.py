"""End-to-end YOLO detector + roi_repr classifier evaluation pipeline.

For each test image:
  1. YOLO predicts boxes (detector-only; YOLO cls ignored)
  2. Filter by confidence; optionally keep a single box (largest area / highest conf)
  3. Classify ROI crop(s) with a trained roi_repr model
  4. Fuse ROI predictions -> image-level prediction
  5. Compare with test.txt labels; compute Acc / Balanced Acc / malignant Sens & Spec / CM
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from roi_repr.config import MethodName, ReprConfig
from roi_repr.data.splits import build_cv_folds, build_roi_transform
from roi_repr.eval_multi_roi import (
    fuse_binary_any_positive,
    load_model_from_checkpoint,
    predict_crop,
)
from train_resnet50_roi_classifier import compute_malignant_metrics
from train_yolov11_onestep import _disable_ultralytics_neptune_callback
from vae_yolo_roi import BoxSelection, list_pred_boxes_for_image

NoBoxFallback = Literal["full_image", "non_malignant", "skip"]
FUSION_RULE_MULTI = "any_roi_malignant -> malignant; all_non_malignant -> non_malignant"
FUSION_RULE_SINGLE = "single_roi: image prediction equals the sole ROI prediction"
FUSION_RULE = FUSION_RULE_MULTI  # backward compat for gradcam / oracle ablation
METRIC_KEYS = ("accuracy", "balanced_accuracy", "malignant_sensitivity", "malignant_specificity", "select")


def fusion_rule_for(box_selection: BoxSelection) -> str:
    if box_selection == "all":
        return FUSION_RULE_MULTI
    return FUSION_RULE_SINGLE


def build_metrics_dict(y_true_np: np.ndarray, y_pred_np: np.ndarray, cfg: ReprConfig) -> Dict:
    """Acc / Balanced Acc / malignant Sens & Spec / select; no per-class F1."""
    acc = float((y_true_np == y_pred_np).mean()) if y_true_np.size else 0.0
    bal = float(balanced_accuracy_score(y_true_np, y_pred_np)) if y_true_np.size else 0.0
    mal_sens, mal_spec = compute_malignant_metrics(
        y_pred_np,
        y_true_np,
        malignant_id=cfg.malignant_id,
    )
    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(cfg.num_classes)))
    class_names = [cfg.class_tag(i) for i in range(cfg.num_classes)]

    return {
        "num_images": int(y_true_np.size),
        "malignant_id": int(cfg.malignant_id),
        "accuracy": acc,
        "balanced_accuracy": bal,
        "malignant_sensitivity": mal_sens,
        "malignant_specificity": mal_spec,
        "select": 0.5 * (mal_sens + mal_spec),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": class_names,
    }


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def save_visualization(
    image: Image.Image,
    record: Dict,
    out_path: Path,
    cfg: ReprConfig,
) -> None:
    """Draw YOLO boxes, per-ROI class labels, and GT vs fused prediction."""
    vis = image.copy().convert("RGB")
    draw = ImageDraw.Draw(vis)
    w, h = vis.size

    fallback_used = bool(record.get("fallback_used", False))
    roi_boxes = record.get("roi_boxes", [])
    roi_preds = record.get("roi_preds", [])
    gt_class = int(record.get("gt_class", 0))
    pred_class = int(record.get("pred_class", 0))

    for idx, box_entry in enumerate(roi_boxes):
        box = box_entry["box_xyxy"]
        conf = float(box_entry.get("conf", 0.0))
        source = str(box_entry.get("source", "yolo"))
        x0, y0, x1, y1 = [float(v) for v in box]
        is_fallback = source == "full_image_fallback"
        color = (0, 200, 0) if is_fallback else (255, 0, 0)
        width = 3 if not is_fallback else 2

        if is_fallback:
            for offset in (0, 4):
                draw.rectangle(
                    [x0 + offset, y0 + offset, x1 - offset, y1 - offset],
                    outline=color,
                    width=1,
                )
        else:
            draw.rectangle([x0, y0, x1, y1], outline=color, width=width)

        pred_id = int(roi_preds[idx]) if idx < len(roi_preds) else -1
        pred_tag = cfg.class_tag(pred_id) if pred_id >= 0 else "?"
        label_y = max(y0 - 18, 2)
        if not is_fallback:
            draw.text((x0 + 2, label_y), f"{conf:.2f} {pred_tag}", fill=color)
        else:
            draw.text((x0 + 2, label_y), f"fallback {pred_tag}", fill=color)

    gt_tag = cfg.class_tag(gt_class)
    pred_tag = cfg.class_tag(pred_class)
    correct = gt_class == pred_class
    status = "CORRECT" if correct else "WRONG"
    title = f"GT={gt_tag} | Pred={pred_tag} | {status}"
    if fallback_used:
        title += " | full_image_fallback"
    draw.rectangle([0, 0, w, 42], fill=(0, 0, 0))
    draw.text((8, 6), title, fill=(0, 255, 0) if correct else (255, 80, 80))
    draw.text((8, 24), f"boxes={len(roi_boxes)} fusion={record.get('fusion_rule', '')}", fill=(255, 255, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(out_path, quality=95)


@torch.no_grad()
def evaluate_yolo_roi_split(
    cfg: ReprConfig,
    model: nn.Module,
    yolo_model,
    samples: List[Tuple[str, int]],
    imgs_dir: Path,
    device: torch.device,
    *,
    eval_score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    imgsz: int,
    no_box_fallback: NoBoxFallback,
    box_selection: BoxSelection = "all",
    vis_dir: Path | None = None,
) -> Dict:
    fusion_rule = fusion_rule_for(box_selection)
    transform = build_roi_transform(cfg)
    y_true: List[int] = []
    y_pred: List[int] = []
    records: Dict[str, Dict] = {}

    for image_name, raw_cls in samples:
        image_path = imgs_dir / image_name
        if not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        gt_label = int(cfg.map_label(raw_cls))

        box_items = list_pred_boxes_for_image(
            yolo_model,
            image_path,
            eval_score_threshold=eval_score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
            imgsz=imgsz,
            box_selection=box_selection,
        )
        fallback_used = False

        if not box_items:
            if no_box_fallback == "skip":
                continue
            if no_box_fallback == "full_image":
                box_items = [
                    {
                        "box_xyxy": [0.0, 0.0, float(w), float(h)],
                        "conf": 0.0,
                        "source": "full_image_fallback",
                    }
                ]
                fallback_used = True
            else:
                box_items = []

        roi_preds = [
            predict_crop(model, transform, image, entry["box_xyxy"], device)
            for entry in box_items
        ]

        if cfg.task_mode == "binary":
            if roi_preds:
                if box_selection == "all":
                    fused = fuse_binary_any_positive(roi_preds, malignant_id=cfg.malignant_id)
                else:
                    fused = int(roi_preds[0])
            elif no_box_fallback == "non_malignant":
                fused = 0
            else:
                fused = 0
        else:
            fused = int(max(roi_preds, key=lambda p: (p == cfg.malignant_id, p))) if roi_preds else 0

        y_true.append(gt_label)
        y_pred.append(fused)
        record = {
            "gt_class": gt_label,
            "pred_class": fused,
            "num_boxes": len(box_items),
            "roi_boxes": box_items,
            "roi_preds": [int(v) for v in roi_preds],
            "fusion_rule": fusion_rule,
            "box_selection": box_selection,
            "fallback_used": fallback_used,
        }
        records[image_name] = record

        if vis_dir is not None:
            save_visualization(image, record, vis_dir / image_name, cfg)

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    metrics = build_metrics_dict(y_true_np, y_pred_np, cfg)

    return {
        **metrics,
        "records": records,
    }


def load_yolo_model(yolo_ckpt: Path):
    _disable_ultralytics_neptune_callback()
    from ultralytics import YOLO

    return YOLO(str(yolo_ckpt))


def resolve_yolo_checkpoint(
    *,
    fold: int,
    yolo_ckpt: Path,
    yolo_cv_root: Path | None,
) -> Path:
    """Resolve one detector checkpoint per fold for fair fixed-CV evaluation."""
    if yolo_cv_root is None:
        return yolo_ckpt
    return yolo_cv_root / f"fold{fold}" / "weights" / "best.pt"


def evaluate_fold(
    cfg: ReprConfig,
    fold: int,
    ckpt_path: Path,
    yolo_model,
    device: torch.device,
    *,
    split: str = "test",
    eval_score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    imgsz: int,
    no_box_fallback: NoBoxFallback,
    box_selection: BoxSelection,
    output_dir: Path,
    save_vis: bool,
    yolo_ckpt: Path,
) -> Dict:
    folds = build_cv_folds(cfg)
    fold_splits = folds[int(fold)]
    samples = fold_splits[split] if split != "all_train" else fold_splits["train"]
    imgs_dir = cfg.dataset_root / "imgs"
    model = load_model_from_checkpoint(cfg, ckpt_path, device)

    vis_dir = output_dir / f"fold{fold}" / "vis" if save_vis else None
    metrics = evaluate_yolo_roi_split(
        cfg,
        model,
        yolo_model,
        samples,
        imgs_dir,
        device,
        eval_score_threshold=eval_score_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
        imgsz=imgsz,
        no_box_fallback=no_box_fallback,
        box_selection=box_selection,
        vis_dir=vis_dir,
    )

    return {
        "method": cfg.method,
        "task_mode": cfg.task_mode,
        "fold": int(fold),
        "split": split,
        "checkpoint": str(ckpt_path.resolve()),
        "yolo_ckpt": str(yolo_ckpt.resolve()),
        "eval_score_threshold": float(eval_score_threshold),
        "nms_iou_threshold": float(nms_iou_threshold),
        "max_detections": int(max_detections),
        "imgsz": int(imgsz),
        "no_box_fallback": no_box_fallback,
        "box_selection": box_selection,
        "fusion_rule": fusion_rule_for(box_selection),
        "metrics": {k: v for k, v in metrics.items() if k != "records"},
        "records": metrics["records"],
    }


def print_metrics_block(title: str, m: Dict) -> None:
    print(f"\n== {title} ==")
    print(
        f"N={m['num_images']} acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} "
        f"malignant_sens={m['malignant_sensitivity']:.4f} "
        f"malignant_spec={m['malignant_specificity']:.4f} select={m['select']:.4f}"
    )
    labels = m.get("confusion_matrix_labels", [])
    print("Confusion matrix (rows=true, cols=pred):" + (f" {labels}" if labels else ""))
    print(np.array(m["confusion_matrix"]))


def summarize_method_cv(fold_results: List[Dict]) -> Dict:
    keys = list(METRIC_KEYS)
    out: Dict = {"n_folds": len(fold_results), "folds": []}
    for fr in fold_results:
        out["folds"].append({"fold": fr["fold"], "metrics": {k: fr["metrics"][k] for k in keys}})
    out["metrics_mean_std"] = {}
    for k in keys:
        m, s = _mean_std([fr["metrics"][k] for fr in fold_results])
        out["metrics_mean_std"][k] = {"mean": m, "std": s}
    return out


def run_method_cv(
    method: MethodName,
    *,
    task_mode: str,
    dataset_root: Path | None,
    output_dir: Path,
    split: str,
    device: torch.device,
    yolo_ckpt: Path,
    eval_score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    imgsz: int,
    no_box_fallback: NoBoxFallback,
    box_selection: BoxSelection,
    save_vis: bool,
) -> Dict:
    cfg = ReprConfig(method=method, task_mode=task_mode)  # type: ignore[arg-type]
    if dataset_root is not None:
        cfg.dataset_root = dataset_root
    cfg.__post_init__()

    method_dir = output_dir / f"{method}_cv{cfg.n_cv_folds}"
    method_dir.mkdir(parents=True, exist_ok=True)
    yolo_model = load_yolo_model(yolo_ckpt)
    fold_results: List[Dict] = []

    for fold in range(cfg.n_cv_folds):
        ckpt = cfg.output_dir / f"fold{fold}" / "checkpoints" / "best_model.pth"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        print(f"\n{'=' * 60}")
        print(f"fold={fold} method={cfg.method} task={cfg.task_mode} split={split}")
        print(f"classifier_ckpt={ckpt}")
        print(f"yolo_ckpt={yolo_ckpt}")
        print(f"box_selection={box_selection}")
        print(f"{'=' * 60}")

        result = evaluate_fold(
            cfg,
            fold,
            ckpt,
            yolo_model,
            device,
            split=split,
            eval_score_threshold=eval_score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
            imgsz=imgsz,
            no_box_fallback=no_box_fallback,
            box_selection=box_selection,
            output_dir=method_dir,
            save_vis=save_vis,
            yolo_ckpt=yolo_ckpt,
        )
        fold_results.append(result)
        print_metrics_block(f"YOLO+{cfg.method.upper()} fold{fold}", result["metrics"])

        fold_path = method_dir / f"fold{fold}_{split}.json"
        with fold_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {fold_path}")

    cv_summary = summarize_method_cv(fold_results)
    summary = {
        "method": method,
        "task_mode": task_mode,
        "split": split,
        "yolo_ckpt": str(yolo_ckpt.resolve()),
        "eval_score_threshold": float(eval_score_threshold),
        "nms_iou_threshold": float(nms_iou_threshold),
        "max_detections": int(max_detections),
        "imgsz": int(imgsz),
        "no_box_fallback": no_box_fallback,
        "box_selection": box_selection,
        "fusion_rule": fusion_rule_for(box_selection),
        "cv_summary": cv_summary,
        "fold_results": [
            {"fold": fr["fold"], "metrics": fr["metrics"]}
            for fr in fold_results
        ],
    }
    summary_path = method_dir / f"summary_{split}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved method summary: {summary_path}")
    return summary


def run_all_methods(
    *,
    task_mode: str,
    dataset_root: Path | None,
    output_dir: Path,
    split: str,
    device: torch.device,
    yolo_ckpt: Path,
    eval_score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    imgsz: int,
    no_box_fallback: NoBoxFallback,
    box_selection: BoxSelection,
    save_vis: bool,
    methods: Sequence[MethodName] | None = None,
) -> Dict:
    methods = list(methods or ["cls", "ae", "vae", "siamese", "triplet"])
    output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: Dict[str, Dict] = {}
    for method in methods:
        all_summaries[method] = run_method_cv(
            method,
            task_mode=task_mode,
            dataset_root=dataset_root,
            output_dir=output_dir,
            split=split,
            device=device,
            yolo_ckpt=yolo_ckpt,
            eval_score_threshold=eval_score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
            imgsz=imgsz,
            no_box_fallback=no_box_fallback,
            box_selection=box_selection,
            save_vis=save_vis,
        )

    leaderboard = []
    for method, summary in all_summaries.items():
        ms = summary["cv_summary"]["metrics_mean_std"]
        leaderboard.append(
            {
                "method": method,
                "acc": ms["accuracy"]["mean"],
                "bal_acc": ms["balanced_accuracy"]["mean"],
                "malignant_sensitivity": ms["malignant_sensitivity"]["mean"],
                "malignant_specificity": ms["malignant_specificity"]["mean"],
                "select": ms["select"]["mean"],
            }
        )
    leaderboard.sort(key=lambda x: x["select"], reverse=True)

    combined = {
        "task_mode": task_mode,
        "split": split,
        "yolo_ckpt": str(yolo_ckpt.resolve()),
        "eval_score_threshold": float(eval_score_threshold),
        "nms_iou_threshold": float(nms_iou_threshold),
        "max_detections": int(max_detections),
        "imgsz": int(imgsz),
        "no_box_fallback": no_box_fallback,
        "box_selection": box_selection,
        "fusion_rule": fusion_rule_for(box_selection),
        "methods": all_summaries,
        "leaderboard": leaderboard,
    }
    combined_path = output_dir / f"all_methods_summary_{split}.json"
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"\nSaved combined summary: {combined_path}")

    print("\n" + "=" * 72)
    print("YOLO + ROI CLASSIFIER PIPELINE LEADERBOARD (3-fold test mean)")
    print("=" * 72)
    print(f"{'method':<10} {'acc':>8} {'bal_acc':>8} {'mal_sens':>8} {'mal_spec':>8} {'select':>8}")
    print("(mal_sens / mal_spec: malignant class sensitivity & specificity)")
    for row in leaderboard:
        print(
            f"{row['method']:<10} "
            f"{row['acc']:>8.4f} {row['bal_acc']:>8.4f} "
            f"{row['malignant_sensitivity']:>8.4f} {row['malignant_specificity']:>8.4f} "
            f"{row['select']:>8.4f}"
        )
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("YOLO detector + roi_repr classifier end-to-end evaluation")
    parser.add_argument(
        "--method",
        type=str,
        default="ae",
        choices=["cls", "ae", "vae", "siamese", "triplet", "all"],
    )
    parser.add_argument("--task_mode", type=str, default="binary", choices=["binary", "multiclass"])
    parser.add_argument(
        "--classifier_head",
        type=str,
        default="mlp",
        choices=["linear", "mlp"],
        help="Must match the classifier checkpoint architecture.",
    )
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_model.pth")
    parser.add_argument(
        "--classifier_root",
        type=str,
        default=None,
        help="Root containing fold{N}/checkpoints/best_model.pth",
    )
    parser.add_argument("--yolo_ckpt", type=str, default="outputs/yolov11_onestep_run1/weights/best.pt")
    parser.add_argument(
        "--yolo_cv_root",
        type=str,
        default=None,
        help="Root containing fold{N}/weights/best.pt; overrides --yolo_ckpt per fold",
    )
    parser.add_argument("--cv_split_file", type=str, default=None)
    parser.add_argument("--n_cv_folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all_folds", action="store_true")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train", "all_train"])
    parser.add_argument("--output_dir", type=str, default="outputs/yolo_roi_repr_pipeline")
    parser.add_argument("--eval_score_threshold", type=float, default=0.5)
    parser.add_argument("--nms_iou_threshold", type=float, default=0.5)
    parser.add_argument("--max_detections", type=int, default=5, help="Candidate pool size before single-box selection")
    parser.add_argument(
        "--box_selection",
        type=str,
        default="largest_area",
        choices=["all", "highest_conf", "largest_area"],
        help="all=multi-ROI fusion; largest_area/highest_conf=single ROI only",
    )
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument(
        "--no_box_fallback",
        type=str,
        default="full_image",
        choices=["full_image", "non_malignant", "skip"],
    )
    parser.add_argument("--save_vis", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ReprConfig(
        method=args.method,
        task_mode=args.task_mode,
        classifier_head=args.classifier_head,
        n_cv_folds=args.n_cv_folds,
        seed=args.seed,
    )
    if args.dataset_root:
        cfg.dataset_root = Path(args.dataset_root)
    if args.classifier_root:
        cfg.output_dir = Path(args.classifier_root)
    if args.cv_split_file:
        cfg.cv_split_file = Path(args.cv_split_file)
    cfg.__post_init__()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    yolo_ckpt = Path(args.yolo_ckpt)
    yolo_cv_root = Path(args.yolo_cv_root) if args.yolo_cv_root else None
    no_box_fallback: NoBoxFallback = args.no_box_fallback  # type: ignore[assignment]
    box_selection: BoxSelection = args.box_selection  # type: ignore[assignment]

    if args.method == "all":
        if yolo_cv_root is not None or args.classifier_root or args.cv_split_file:
            raise ValueError(
                "Fixed-CV roots are supported for one method at a time; "
                "run --method ae or --method cls explicitly."
            )
        run_all_methods(
            task_mode=args.task_mode,
            dataset_root=Path(args.dataset_root) if args.dataset_root else None,
            output_dir=out_dir,
            split=args.split,
            device=device,
            yolo_ckpt=yolo_ckpt,
            eval_score_threshold=args.eval_score_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            max_detections=args.max_detections,
            imgsz=args.imgsz,
            no_box_fallback=no_box_fallback,
            box_selection=box_selection,
            save_vis=args.save_vis,
        )
        return

    folds = list(range(cfg.n_cv_folds)) if args.all_folds else [int(args.fold)]
    method_dir = out_dir / f"{cfg.method}_cv{cfg.n_cv_folds}"
    method_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for fold in folds:
        fold_yolo_ckpt = resolve_yolo_checkpoint(
            fold=fold,
            yolo_ckpt=yolo_ckpt,
            yolo_cv_root=yolo_cv_root,
        )
        if not fold_yolo_ckpt.is_file():
            raise FileNotFoundError(f"YOLO checkpoint not found: {fold_yolo_ckpt}")
        yolo_model = load_yolo_model(fold_yolo_ckpt)
        if args.checkpoint:
            ckpt = Path(args.checkpoint)
        else:
            ckpt = cfg.output_dir / f"fold{fold}" / "checkpoints" / "best_model.pth"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        print(f"\n{'=' * 60}")
        print(f"fold={fold} method={cfg.method} task={cfg.task_mode} split={args.split}")
        print(f"classifier_ckpt={ckpt}")
        print(f"yolo_ckpt={fold_yolo_ckpt}")
        print(f"cv_split_file={cfg.cv_split_file}")
        print(f"box_selection={box_selection}")
        print(f"{'=' * 60}")

        result = evaluate_fold(
            cfg,
            fold,
            ckpt,
            yolo_model,
            device,
            split=args.split,
            eval_score_threshold=args.eval_score_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            max_detections=args.max_detections,
            imgsz=args.imgsz,
            no_box_fallback=no_box_fallback,
            box_selection=box_selection,
            output_dir=method_dir,
            save_vis=args.save_vis,
            yolo_ckpt=fold_yolo_ckpt,
        )
        all_results.append(result)
        print_metrics_block(f"YOLO+{cfg.method.upper()} fold{fold}", result["metrics"])

        fold_path = method_dir / f"fold{fold}_{args.split}.json"
        with fold_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {fold_path}")

    if len(all_results) > 1:
        cv_summary = summarize_method_cv(all_results)
        summary_path = method_dir / f"summary_{args.split}.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "method": cfg.method,
                    "task_mode": cfg.task_mode,
                    "split": args.split,
                    "cv_split_file": str(cfg.cv_split_file.resolve()) if cfg.cv_split_file else None,
                    "classifier_root": str(cfg.output_dir.resolve()),
                    "yolo_cv_root": str(yolo_cv_root.resolve()) if yolo_cv_root else None,
                    "yolo_checkpoints": [fr["yolo_ckpt"] for fr in all_results],
                    "box_selection": box_selection,
                    "fusion_rule": fusion_rule_for(box_selection),
                    "cv_summary": cv_summary,
                    "fold_results": [
                        {"fold": fr["fold"], "metrics": fr["metrics"]}
                        for fr in all_results
                    ],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
