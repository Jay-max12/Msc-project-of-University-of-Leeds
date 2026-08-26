"""Oracle ROI ablation: compare GT-box vs YOLO-box ROI classification on the same images.

GT oracle uses a single main ROI via ``choose_roi_box`` (prefer nml/abn, largest area),
aligned with roi_repr classifier training — not multi clinical boxes (nml/bmt/malg).

Error attribution (binary, same fusion rule):
  - detection_degradation: GT oracle correct, YOLO wrong  -> blame detector/crop
  - classifier_failure:    GT oracle wrong                  -> blame classifier
  - both_correct / both_wrong_same / yolo_rescue / both_wrong_diff
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np
import torch

GT_BOX_SOURCE = "bbox_annot.json (choose_roi_box: nml/abn, max area)"

from roi_repr.config import MethodName, ReprConfig
from roi_repr.data.splits import build_cv_folds, list_valid_samples
from roi_repr.eval_multi_roi import evaluate_split, load_model_from_checkpoint
from roi_repr.eval_yolo_roi_pipeline import (
    FUSION_RULE,
    METRIC_KEYS,
    NoBoxFallback,
    _mean_std,
    build_metrics_dict,
    evaluate_yolo_roi_split,
    load_yolo_model,
)
from train_resnet50_roi_classifier import load_bbox_annotations

AblationCategory = Literal[
    "both_correct",
    "detection_degradation",
    "yolo_rescue",
    "classifier_failure",
    "both_wrong_different",
]


def categorize_image(gt_pred: int, yolo_pred: int, label: int) -> AblationCategory:
    gt_ok = int(gt_pred) == int(label)
    yolo_ok = int(yolo_pred) == int(label)
    if gt_ok and yolo_ok:
        return "both_correct"
    if gt_ok and not yolo_ok:
        return "detection_degradation"
    if not gt_ok and yolo_ok:
        return "yolo_rescue"
    if not gt_ok and not yolo_ok:
        if int(gt_pred) == int(yolo_pred):
            return "classifier_failure"
        return "both_wrong_different"
    return "both_wrong_different"


def compare_gt_yolo_records(
    gt_records: Dict[str, Dict],
    yolo_records: Dict[str, Dict],
) -> Dict:
    common = sorted(set(gt_records) & set(yolo_records))
    categories: Dict[str, List[str]] = {
        "both_correct": [],
        "detection_degradation": [],
        "yolo_rescue": [],
        "classifier_failure": [],
        "both_wrong_different": [],
    }
    per_image: Dict[str, Dict] = {}

    for name in common:
        gt_rec = gt_records[name]
        yolo_rec = yolo_records[name]
        label = int(gt_rec["gt_class"])
        gt_pred = int(gt_rec["pred_class"])
        yolo_pred = int(yolo_rec["pred_class"])
        cat = categorize_image(gt_pred, yolo_pred, label)
        categories[cat].append(name)
        per_image[name] = {
            "gt_class": label,
            "gt_box_pred": gt_pred,
            "yolo_box_pred": yolo_pred,
            "category": cat,
            "gt_num_boxes": int(gt_rec.get("num_boxes", len(gt_rec.get("roi_boxes", [])))),
            "yolo_num_boxes": int(yolo_rec.get("num_boxes", len(yolo_rec.get("roi_boxes", [])))),
            "yolo_fallback_used": bool(yolo_rec.get("fallback_used", False)),
        }

    n = len(common)
    counts = {k: len(v) for k, v in categories.items()}
    return {
        "num_images": n,
        "counts": counts,
        "rates": {k: (float(v) / n if n else 0.0) for k, v in counts.items()},
        "detection_error_rate": float(counts["detection_degradation"] / n) if n else 0.0,
        "classifier_failure_rate": float(
            (counts["classifier_failure"] + counts["both_wrong_different"]) / n
        )
        if n
        else 0.0,
        "oracle_accuracy": float(counts["both_correct"] + counts["detection_degradation"]) / n
        if n
        else 0.0,
        "yolo_accuracy": float(counts["both_correct"] + counts["yolo_rescue"]) / n if n else 0.0,
        "per_image": per_image,
        "image_lists": categories,
    }


def _records_to_arrays(records: Dict[str, Dict]) -> Tuple[np.ndarray, np.ndarray]:
    names = sorted(records.keys())
    y_true = np.array([int(records[n]["gt_class"]) for n in names], dtype=np.int64)
    y_pred = np.array([int(records[n]["pred_class"]) for n in names], dtype=np.int64)
    return y_true, y_pred


def print_ablation_block(title: str, result: Dict) -> None:
    gt_m = result["gt_oracle"]["metrics"]
    yolo_m = result["yolo"]["metrics"]
    ab = result["ablation"]
    print(f"\n== {title} ==")
    print(
        f"N={gt_m['num_images']} | GT oracle: acc={gt_m['accuracy']:.4f} "
        f"mal_sens={gt_m['malignant_sensitivity']:.4f} mal_spec={gt_m['malignant_specificity']:.4f} "
        f"select={gt_m['select']:.4f}"
    )
    print(
        f"N={yolo_m['num_images']} | YOLO box:  acc={yolo_m['accuracy']:.4f} "
        f"mal_sens={yolo_m['malignant_sensitivity']:.4f} mal_spec={yolo_m['malignant_specificity']:.4f} "
        f"select={yolo_m['select']:.4f}"
    )
    print(
        f"Delta select (YOLO - GT) = {yolo_m['select'] - gt_m['select']:+.4f} | "
        f"detection_degradation={ab['counts']['detection_degradation']} "
        f"classifier_failure={ab['counts']['classifier_failure']} "
        f"both_wrong_diff={ab['counts']['both_wrong_different']} "
        f"yolo_rescue={ab['counts']['yolo_rescue']}"
    )


def evaluate_oracle_ablation_fold(
    cfg: ReprConfig,
    fold: int,
    ckpt_path: Path,
    yolo_model,
    samples: List[Tuple[str, int]],
    device: torch.device,
    *,
    split: str,
    eval_score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    imgsz: int,
    no_box_fallback: NoBoxFallback,
    yolo_ckpt: Path,
) -> Dict:
    bbox_annot = load_bbox_annotations(cfg.dataset_root / "bbox_annot.json")
    imgs_dir = cfg.dataset_root / "imgs"
    model = load_model_from_checkpoint(cfg, ckpt_path, device)

    gt_out = evaluate_split(
        cfg,
        model,
        samples,
        bbox_annot,
        imgs_dir,
        device,
        mode="single_roi",
    )
    yolo_out = evaluate_yolo_roi_split(
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
        vis_dir=None,
    )

    gt_records = gt_out["records"]
    yolo_records = yolo_out["records"]
    ablation = compare_gt_yolo_records(gt_records, yolo_records)

    gt_y_true, gt_y_pred = _records_to_arrays(gt_records)
    yolo_y_true, yolo_y_pred = _records_to_arrays(yolo_records)

    gt_metrics = build_metrics_dict(gt_y_true, gt_y_pred, cfg)
    yolo_metrics = build_metrics_dict(yolo_y_true, yolo_y_pred, cfg)

    return {
        "method": cfg.method,
        "task_mode": cfg.task_mode,
        "fold": int(fold),
        "split": split,
        "num_samples_requested": len(samples),
        "checkpoint": str(ckpt_path.resolve()),
        "yolo_ckpt": str(yolo_ckpt.resolve()),
        "eval_score_threshold": float(eval_score_threshold),
        "nms_iou_threshold": float(nms_iou_threshold),
        "max_detections": int(max_detections),
        "imgsz": int(imgsz),
        "no_box_fallback": no_box_fallback,
        "fusion_rule": FUSION_RULE,
        "gt_box_source": GT_BOX_SOURCE,
        "gt_oracle_mode": "single_roi",
        "yolo_box_source": "yolo_detector",
        "gt_oracle": {
            "metrics": {k: gt_metrics[k] for k in (*METRIC_KEYS, "num_images", "malignant_id", "confusion_matrix", "confusion_matrix_labels")},
            "records": gt_records,
        },
        "yolo": {
            "metrics": {k: yolo_metrics[k] for k in (*METRIC_KEYS, "num_images", "malignant_id", "confusion_matrix", "confusion_matrix_labels")},
            "records": yolo_records,
        },
        "ablation": {
            k: v for k, v in ablation.items() if k != "per_image"
        },
        "ablation_per_image": ablation["per_image"],
        "delta_select": float(yolo_metrics["select"] - gt_metrics["select"]),
        "delta_accuracy": float(yolo_metrics["accuracy"] - gt_metrics["accuracy"]),
    }


def resolve_samples(cfg: ReprConfig, split: str, fold: int) -> List[Tuple[str, int]]:
    if split in {"train", "test"}:
        return list_valid_samples(cfg, split)
    folds = build_cv_folds(cfg)
    fold_splits = folds[int(fold)]
    if split not in fold_splits:
        raise ValueError(f"Unknown split: {split}")
    return fold_splits[split]


def summarize_ablation_cv(fold_results: List[Dict]) -> Dict:
    out: Dict = {"n_folds": len(fold_results), "folds": []}
    for fr in fold_results:
        out["folds"].append(
            {
                "fold": fr["fold"],
                "gt_select": fr["gt_oracle"]["metrics"]["select"],
                "yolo_select": fr["yolo"]["metrics"]["select"],
                "delta_select": fr["delta_select"],
                "detection_degradation": fr["ablation"]["counts"]["detection_degradation"],
                "classifier_failure": fr["ablation"]["counts"]["classifier_failure"],
            }
        )

    def _ms(key_path: str) -> Dict:
        if key_path == "gt":
            vals = [fr["gt_oracle"]["metrics"]["select"] for fr in fold_results]
        elif key_path == "yolo":
            vals = [fr["yolo"]["metrics"]["select"] for fr in fold_results]
        else:
            vals = [fr["delta_select"] for fr in fold_results]
        m, s = _mean_std(vals)
        return {"mean": m, "std": s}

    out["gt_select_mean_std"] = _ms("gt")
    out["yolo_select_mean_std"] = _ms("yolo")
    out["delta_select_mean_std"] = _ms("delta")

    for metric_key in METRIC_KEYS:
        gt_vals = [fr["gt_oracle"]["metrics"][metric_key] for fr in fold_results]
        yolo_vals = [fr["yolo"]["metrics"][metric_key] for fr in fold_results]
        gm, gs = _mean_std(gt_vals)
        ym, ys = _mean_std(yolo_vals)
        out[f"gt_{metric_key}_mean_std"] = {"mean": gm, "std": gs}
        out[f"yolo_{metric_key}_mean_std"] = {"mean": ym, "std": ys}

    det_vals = [fr["ablation"]["counts"]["detection_degradation"] for fr in fold_results]
    cls_vals = [fr["ablation"]["counts"]["classifier_failure"] for fr in fold_results]
    out["detection_degradation_count_mean_std"] = dict(zip(("mean", "std"), _mean_std(det_vals)))
    out["classifier_failure_count_mean_std"] = dict(zip(("mean", "std"), _mean_std(cls_vals)))
    return out


def run_method_ablation(
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
    all_folds: bool,
    fold: int,
) -> Dict:
    cfg = ReprConfig(method=method, task_mode=task_mode)  # type: ignore[arg-type]
    if dataset_root is not None:
        cfg.dataset_root = dataset_root
    cfg.__post_init__()

    method_dir = output_dir / f"{method}_cv{cfg.n_cv_folds}"
    method_dir.mkdir(parents=True, exist_ok=True)
    yolo_model = load_yolo_model(yolo_ckpt)
    folds = list(range(cfg.n_cv_folds)) if all_folds else [int(fold)]
    fold_results: List[Dict] = []

    for f in folds:
        ckpt = cfg.output_dir / f"fold{f}" / "checkpoints" / "best_model.pth"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        samples = resolve_samples(cfg, split, f)
        print(f"\n{'=' * 60}")
        print(f"fold={f} method={cfg.method} split={split} N={len(samples)}")
        print(f"classifier_ckpt={ckpt}")
        print(f"GT boxes: choose_roi_box (nml/abn single) | YOLO: {yolo_ckpt}")
        print(f"{'=' * 60}")

        result = evaluate_oracle_ablation_fold(
            cfg,
            f,
            ckpt,
            yolo_model,
            samples,
            device,
            split=split,
            eval_score_threshold=eval_score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
            imgsz=imgsz,
            no_box_fallback=no_box_fallback,
            yolo_ckpt=yolo_ckpt,
        )
        fold_results.append(result)
        print_ablation_block(f"{cfg.method.upper()} fold{f}", result)

        fold_path = method_dir / f"fold{f}_{split}_ablation.json"
        with fold_path.open("w", encoding="utf-8") as f_out:
            json.dump(result, f_out, indent=2, ensure_ascii=False)
        print(f"Saved: {fold_path}")

    summary = {
        "method": method,
        "task_mode": task_mode,
        "split": split,
        "fusion_rule": FUSION_RULE,
        "gt_oracle_mode": "single_roi",
        "gt_box_source": GT_BOX_SOURCE,
        "yolo_ckpt": str(yolo_ckpt.resolve()),
        "no_box_fallback": no_box_fallback,
        "cv_summary": summarize_ablation_cv(fold_results),
        "fold_results": [
            {
                "fold": fr["fold"],
                "gt_oracle": fr["gt_oracle"]["metrics"],
                "yolo": fr["yolo"]["metrics"],
                "delta_select": fr["delta_select"],
                "ablation_counts": fr["ablation"]["counts"],
            }
            for fr in fold_results
        ],
    }
    summary_path = method_dir / f"summary_{split}_ablation.json"
    with summary_path.open("w", encoding="utf-8") as f_out:
        json.dump(summary, f_out, indent=2, ensure_ascii=False)
    print(f"Saved: {summary_path}")
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
    all_folds: bool,
    fold: int,
    methods: Sequence[MethodName] | None = None,
) -> Dict:
    methods = list(methods or ["cls", "ae", "vae", "siamese", "triplet"])
    output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: Dict[str, Dict] = {}
    for method in methods:
        all_summaries[method] = run_method_ablation(
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
            all_folds=all_folds,
            fold=fold,
        )

    leaderboard = []
    for method, summary in all_summaries.items():
        cv = summary["cv_summary"]
        leaderboard.append(
            {
                "method": method,
                "gt_select": cv["gt_select_mean_std"]["mean"],
                "yolo_select": cv["yolo_select_mean_std"]["mean"],
                "delta_select": cv["delta_select_mean_std"]["mean"],
                "detection_degradation_mean": cv["detection_degradation_count_mean_std"]["mean"],
                "classifier_failure_mean": cv["classifier_failure_count_mean_std"]["mean"],
            }
        )
    leaderboard.sort(key=lambda x: x["delta_select"])

    combined = {
        "task_mode": task_mode,
        "split": split,
        "fusion_rule": FUSION_RULE,
        "gt_oracle_mode": "single_roi",
        "gt_box_source": GT_BOX_SOURCE,
        "methods": all_summaries,
        "leaderboard_by_delta_select": leaderboard,
    }
    combined_path = output_dir / f"all_methods_summary_{split}_ablation.json"
    with combined_path.open("w", encoding="utf-8") as f_out:
        json.dump(combined, f_out, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"ORACLE ROI ABLATION LEADERBOARD (split={split}, sorted by delta_select YOLO-GT)")
    print("=" * 80)
    print(
        f"{'method':<10} {'gt_sel':>8} {'yolo_sel':>8} {'d_sel':>8} "
        f"{'det_err':>8} {'cls_err':>8}"
    )
    for row in leaderboard:
        print(
            f"{row['method']:<10} "
            f"{row['gt_select']:>8.4f} {row['yolo_select']:>8.4f} {row['delta_select']:>+8.4f} "
            f"{row['detection_degradation_mean']:>8.1f} {row['classifier_failure_mean']:>8.1f}"
        )
    print("det_err = images where GT oracle correct but YOLO wrong (detection blame)")
    print("cls_err = images where GT oracle also wrong (classifier blame)")
    print(f"Saved: {combined_path}")
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("GT vs YOLO oracle ROI ablation for roi_repr classifiers")
    parser.add_argument("--method", type=str, default="ae", choices=["cls", "ae", "vae", "siamese", "triplet", "all"])
    parser.add_argument("--task_mode", type=str, default="binary", choices=["binary", "multiclass"])
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "val"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all_folds", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/roi_repr/binary/oracle_roi_ablation_single_roi",
    )
    parser.add_argument("--yolo_ckpt", type=str, default="outputs/yolov11_onestep_run1/weights/best.pt")
    parser.add_argument("--eval_score_threshold", type=float, default=0.5)
    parser.add_argument("--nms_iou_threshold", type=float, default=0.5)
    parser.add_argument("--max_detections", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument(
        "--no_box_fallback",
        type=str,
        default="full_image",
        choices=["full_image", "non_malignant", "skip"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    yolo_ckpt = Path(args.yolo_ckpt)
    no_box_fallback: NoBoxFallback = args.no_box_fallback  # type: ignore[assignment]

    common_kwargs = dict(
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
        all_folds=args.all_folds,
        fold=args.fold,
    )

    if args.method == "all":
        run_all_methods(**common_kwargs)
        return

    run_method_ablation(method=args.method, **common_kwargs)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
