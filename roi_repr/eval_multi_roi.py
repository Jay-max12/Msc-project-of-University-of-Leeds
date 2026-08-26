"""Multi-ROI fusion evaluation for roi_repr classifiers (AE etc.).

Uses clinical boxes only: `nml` (normal), `bmt` (benign), `malg` (malignant).
Excludes `abn` (abnormal ROI shell), `stn` (stone), etc.
Fuses per-ROI predictions with:
  any ROI malignant -> image malignant; all non-malignant -> non_malignant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

from roi_repr.config import MethodName, ReprConfig
from roi_repr.data.splits import build_cv_folds, build_roi_transform, list_valid_samples
from roi_repr.models.repr_model import RoiReprModel
from train_resnet50_roi_classifier import (
    choose_roi_box,
    compute_malignant_metrics,
    load_bbox_annotations,
    sanitize_box_xyxy,
)

BoxItem = Tuple[str, List[float]]
# GBCU bbox_annot.json: nml (normal ROI), bmt (benign mural thickening), malg (malignancy).
INCLUDED_EVAL_BOX_LABELS = frozenset({"nml", "bmt", "malg"})
BOX_LABEL_ALIASES = {
    "mal": "malg",
    "ben": "bmt",
    "nml": "nml",
    "normal": "nml",
    "malignant": "malg",
    "benign": "bmt",
}


def normalize_box_label(label: str) -> str:
    key = str(label).lower().strip()
    return BOX_LABEL_ALIASES.get(key, key)


def list_eval_boxes(raw_bbs: Sequence[Sequence]) -> List[BoxItem]:
    """GT boxes limited to nml (normal) / bmt (benign) / malg (malignant)."""
    out: List[BoxItem] = []
    for item in raw_bbs:
        if len(item) != 2:
            continue
        label, box = item
        norm = normalize_box_label(str(label))
        if norm not in INCLUDED_EVAL_BOX_LABELS:
            continue
        if not isinstance(box, list) or len(box) != 4:
            continue
        out.append((norm, [float(v) for v in box]))
    return out


def fuse_binary_any_positive(roi_preds: Sequence[int], malignant_id: int = 1) -> int:
    """阳全阳、全阴才阴: any malignant ROI -> malignant."""
    return int(malignant_id) if int(malignant_id) in roi_preds else 0


def load_model_from_checkpoint(cfg: ReprConfig, ckpt_path: Path, device: torch.device) -> RoiReprModel:
    model = RoiReprModel(cfg).to(device)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_crop(
    model: nn.Module,
    transform,
    image: Image.Image,
    box: Sequence[float],
    device: torch.device,
) -> int:
    w, h = image.size
    x0, y0, x1, y1 = sanitize_box_xyxy(box, width=w, height=h)
    crop = image.crop((x0, y0, x1, y1))
    tensor = transform(crop).unsqueeze(0).to(device, non_blocking=True)
    logits = model(tensor)["logits"]
    return int(torch.argmax(logits, dim=1).item())


def evaluate_split(
    cfg: ReprConfig,
    model: nn.Module,
    samples: List[Tuple[str, int]],
    bbox_annot: Dict,
    imgs_dir: Path,
    device: torch.device,
    *,
    mode: str,
) -> Dict:
    transform = build_roi_transform(cfg)
    y_true: List[int] = []
    y_pred: List[int] = []
    records: Dict[str, Dict] = {}

    for image_name, raw_cls in samples:
        if image_name not in bbox_annot:
            continue
        image_path = imgs_dir / image_name
        if not image_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        gt_label = int(cfg.map_label(raw_cls))
        raw_bbs = bbox_annot[image_name]["bbs"]

        if mode == "single_roi":
            box = choose_roi_box(raw_bbs)
            roi_preds = [predict_crop(model, transform, image, box, device)]
            box_items = [("chosen", box)]
        elif mode == "multi_roi":
            box_items = list_eval_boxes(raw_bbs)
            roi_preds = [
                predict_crop(model, transform, image, box, device)
                for _, box in box_items
            ]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if cfg.task_mode == "binary":
            fused = fuse_binary_any_positive(roi_preds, malignant_id=cfg.malignant_id)
        else:
            fused = int(max(roi_preds, key=lambda p: (p == cfg.malignant_id, p)))

        y_true.append(gt_label)
        y_pred.append(fused)
        records[image_name] = {
            "gt_class": gt_label,
            "pred_class": fused,
            "roi_preds": [int(v) for v in roi_preds],
            "roi_boxes": [{"label": lb, "box": [float(v) for v in bx]} for lb, bx in box_items],
        }

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    acc = float((y_true_np == y_pred_np).mean()) if y_true_np.size else 0.0
    bal = float(balanced_accuracy_score(y_true_np, y_pred_np)) if y_true_np.size else 0.0
    mal_sens, mal_spec = compute_malignant_metrics(y_pred_np, y_true_np, malignant_id=cfg.malignant_id)
    target_names = [cfg.class_tag(i) for i in range(cfg.num_classes)]
    report = classification_report(
        y_true_np,
        y_pred_np,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(cfg.num_classes)))

    return {
        "mode": mode,
        "num_images": int(y_true_np.size),
        "accuracy": acc,
        "balanced_accuracy": bal,
        "mal_sens": mal_sens,
        "mal_spec": mal_spec,
        "select": 0.5 * (mal_sens + mal_spec),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "records": records,
    }


def evaluate_fold(
    cfg: ReprConfig,
    fold: int,
    ckpt_path: Path,
    device: torch.device,
    split: str = "test",
) -> Dict:
    folds = build_cv_folds(cfg)
    fold_splits = folds[int(fold)]
    samples = fold_splits[split] if split != "all_train" else list_valid_samples(cfg, "train")
    bbox_annot = load_bbox_annotations(cfg.dataset_root / "bbox_annot.json")
    imgs_dir = cfg.dataset_root / "imgs"
    model = load_model_from_checkpoint(cfg, ckpt_path, device)

    multi = evaluate_split(cfg, model, samples, bbox_annot, imgs_dir, device, mode="multi_roi")
    single = evaluate_split(cfg, model, samples, bbox_annot, imgs_dir, device, mode="single_roi")

    return {
        "method": cfg.method,
        "task_mode": cfg.task_mode,
        "fold": int(fold),
        "split": split,
        "checkpoint": str(ckpt_path.resolve()),
        "included_box_labels": sorted(INCLUDED_EVAL_BOX_LABELS),
        "box_label_aliases": dict(BOX_LABEL_ALIASES),
        "fusion_rule": "any_roi_malignant -> malignant; all_non_malignant -> non_malignant; no_box -> non_malignant",
        "multi_roi": {k: v for k, v in multi.items() if k != "records"},
        "single_roi_baseline": {k: v for k, v in single.items() if k != "records"},
        "records_multi_roi": multi["records"],
    }


def print_metrics_block(title: str, m: Dict) -> None:
    print(f"\n== {title} ==")
    print(
        f"N={m['num_images']} acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} "
        f"mal_sens={m['mal_sens']:.4f} mal_spec={m['mal_spec']:.4f} select={m['select']:.4f}"
    )
    print(m["classification_report"])
    print("Confusion matrix:")
    print(np.array(m["confusion_matrix"]))


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def summarize_method_cv(fold_results: List[Dict]) -> Dict:
    keys = ["accuracy", "balanced_accuracy", "mal_sens", "mal_spec", "select"]
    out: Dict = {"n_folds": len(fold_results), "folds": []}
    for fr in fold_results:
        out["folds"].append(
            {
                "fold": fr["fold"],
                "multi_roi": {k: fr["multi_roi"][k] for k in keys},
                "single_roi_baseline": {k: fr["single_roi_baseline"][k] for k in keys},
            }
        )
    for mode_key in ("multi_roi", "single_roi_baseline"):
        out[f"{mode_key}_mean_std"] = {}
        for k in keys:
            m, s = _mean_std([fr[mode_key][k] for fr in fold_results])
            out[f"{mode_key}_mean_std"][k] = {"mean": m, "std": s}
    return out


def run_method_cv(
    method: MethodName,
    *,
    task_mode: str,
    dataset_root: Path | None,
    output_dir: Path,
    split: str,
    device: torch.device,
) -> Dict:
    cfg = ReprConfig(method=method, task_mode=task_mode)  # type: ignore[arg-type]
    if dataset_root is not None:
        cfg.dataset_root = dataset_root
    cfg.__post_init__()

    method_dir = output_dir / f"{method}_cv{cfg.n_cv_folds}"
    method_dir.mkdir(parents=True, exist_ok=True)
    fold_results: List[Dict] = []

    for fold in range(cfg.n_cv_folds):
        ckpt = cfg.output_dir / f"fold{fold}" / "checkpoints" / "best_model.pth"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        print(f"\n{'=' * 60}")
        print(f"fold={fold} method={cfg.method} task={cfg.task_mode} split={split}")
        print(f"checkpoint={ckpt}")
        print(f"include boxes: {sorted(INCLUDED_EVAL_BOX_LABELS)} (nml=normal, bmt=benign, malg=malignant)")
        print(f"{'=' * 60}")

        result = evaluate_fold(cfg, fold, ckpt, device, split=split)
        fold_results.append(result)
        print_metrics_block("MULTI-ROI (fusion)", result["multi_roi"])
        print_metrics_block("SINGLE-ROI baseline (choose_roi_box)", result["single_roi_baseline"])

        fold_path = method_dir / f"fold{fold}_{split}.json"
        with fold_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {fold_path}")

    cv_summary = summarize_method_cv(fold_results)
    summary = {
        "method": method,
        "task_mode": task_mode,
        "split": split,
        "included_box_labels": sorted(INCLUDED_EVAL_BOX_LABELS),
        "box_label_aliases": dict(BOX_LABEL_ALIASES),
        "fusion_rule": "any_roi_malignant -> malignant; all_non_malignant -> non_malignant; no_box -> non_malignant",
        "cv_summary": cv_summary,
        "fold_results": [
            {
                "fold": fr["fold"],
                "multi_roi": fr["multi_roi"],
                "single_roi_baseline": fr["single_roi_baseline"],
            }
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
        )

    leaderboard = []
    for method, summary in all_summaries.items():
        ms = summary["cv_summary"]["multi_roi_mean_std"]
        bs = summary["cv_summary"]["single_roi_baseline_mean_std"]
        leaderboard.append(
            {
                "method": method,
                "multi_acc": ms["accuracy"]["mean"],
                "multi_sens": ms["mal_sens"]["mean"],
                "multi_spec": ms["mal_spec"]["mean"],
                "multi_select": ms["select"]["mean"],
                "single_acc": bs["accuracy"]["mean"],
                "single_sens": bs["mal_sens"]["mean"],
                "single_spec": bs["mal_spec"]["mean"],
                "single_select": bs["select"]["mean"],
                "delta_select": ms["select"]["mean"] - bs["select"]["mean"],
            }
        )
    leaderboard.sort(key=lambda x: x["multi_select"], reverse=True)

    combined = {
        "task_mode": task_mode,
        "split": split,
        "included_box_labels": sorted(INCLUDED_EVAL_BOX_LABELS),
        "box_label_aliases": dict(BOX_LABEL_ALIASES),
        "fusion_rule": "any_roi_malignant -> malignant; all_non_malignant -> non_malignant; no_box -> non_malignant",
        "methods": all_summaries,
        "leaderboard": leaderboard,
    }
    combined_path = output_dir / f"all_methods_summary_{split}.json"
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"\nSaved combined summary: {combined_path}")

    print("\n" + "=" * 72)
    print("MULTI-ROI FUSION LEADERBOARD (3-fold test mean)")
    print("=" * 72)
    print(f"{'method':<10} {'acc':>8} {'sens':>8} {'spec':>8} {'select':>8} {'d_sel':>8}")
    for row in leaderboard:
        print(
            f"{row['method']:<10} "
            f"{row['multi_acc']:>8.4f} {row['multi_sens']:>8.4f} "
            f"{row['multi_spec']:>8.4f} {row['multi_select']:>8.4f} "
            f"{row['delta_select']:>+8.4f}"
        )
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Multi-ROI fusion evaluation for roi_repr models")
    parser.add_argument(
        "--method",
        type=str,
        default="ae",
        choices=["cls", "ae", "vae", "siamese", "triplet", "all"],
    )
    parser.add_argument("--task_mode", type=str, default="binary", choices=["binary", "multiclass"])
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_model.pth")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all_folds", action="store_true")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train", "all_train"])
    parser.add_argument("--output_dir", type=str, default="outputs/roi_repr/binary/multi_roi_fusion_cv3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ReprConfig(method=args.method, task_mode=args.task_mode)
    if args.dataset_root:
        cfg.dataset_root = Path(args.dataset_root)
    cfg.__post_init__()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "all":
        run_all_methods(
            task_mode=args.task_mode,
            dataset_root=Path(args.dataset_root) if args.dataset_root else None,
            output_dir=out_dir,
            split=args.split,
            device=device,
        )
        return

    folds = list(range(cfg.n_cv_folds)) if args.all_folds else [int(args.fold)]
    all_results = []

    for fold in folds:
        if args.checkpoint:
            ckpt = Path(args.checkpoint)
        else:
            ckpt = cfg.output_dir / f"fold{fold}" / "checkpoints" / "best_model.pth"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        print(f"\n{'=' * 60}")
        print(f"fold={fold} method={cfg.method} task={cfg.task_mode} split={args.split}")
        print(f"checkpoint={ckpt}")
        print(f"include boxes: {sorted(INCLUDED_EVAL_BOX_LABELS)} (nml=normal, bmt=benign, malg=malignant)")
        print(f"{'=' * 60}")

        result = evaluate_fold(cfg, fold, ckpt, device, split=args.split)
        all_results.append(result)
        print_metrics_block("MULTI-ROI (fusion)", result["multi_roi"])
        print_metrics_block("SINGLE-ROI baseline (choose_roi_box)", result["single_roi_baseline"])

        fold_path = out_dir / f"fold{fold}_{args.split}.json"
        with fold_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {fold_path}")

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "method": cfg.method,
                "task_mode": cfg.task_mode,
                "split": args.split,
                "folds": all_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
