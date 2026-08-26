from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

from roi_repr.data.cv_splits import load_cv_folds, stratify_for_task_mode

ROI_LABELS = {"nml", "abn"}
TaskMode = Literal["detect", "detect_cls"]

YOLO_TRAIN_AUGMENT_KWARGS = {
    # Align with existing two-stage setup: only horizontal flip.
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.0,
    "close_mosaic": 0,
}


def _disable_ultralytics_neptune_callback() -> None:
    """Neptune 回调在 SETTINGS['neptune']==True 时会 `import neptune`，可能拖入与 numpy ABI 不兼容的 pandas 并崩溃。

    通过 `dict.__setitem__` 只改进程内 SETTINGS，不写回 Ultralytics 全局 settings.json。
    """
    from ultralytics.utils import SETTINGS

    dict.__setitem__(SETTINGS, "neptune", False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def load_bbox_annotations(annotation_file: Path) -> Dict[str, Dict]:
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


def sanitize_box_xyxy(box: Sequence[float], width: int, height: int) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = [float(v) for v in box]
    x0 = max(0.0, min(x0, width - 1))
    y0 = max(0.0, min(y0, height - 1))
    x1 = max(0.0, min(x1, width))
    y1 = max(0.0, min(y1, height))
    if x1 <= x0:
        x1 = min(width, x0 + 1.0)
    if y1 <= y0:
        y1 = min(height, y0 + 1.0)
    return x0, y0, x1, y1


def xyxy_to_yolo(box: Sequence[float], width: int, height: int) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = sanitize_box_xyxy(box, width=width, height=height)
    cx = ((x0 + x1) / 2.0) / float(width)
    cy = ((y0 + y1) / 2.0) / float(height)
    bw = (x1 - x0) / float(width)
    bh = (y1 - y0) / float(height)
    return cx, cy, bw, bh


def iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def compute_malignant_metrics(preds: np.ndarray, labels: np.ndarray, malignant_id: int = 1) -> Tuple[float, float]:
    pred_pos = preds == malignant_id
    true_pos = labels == malignant_id
    tp = int(np.logical_and(pred_pos, true_pos).sum())
    fn = int(np.logical_and(~pred_pos, true_pos).sum())
    tn = int(np.logical_and(~pred_pos, ~true_pos).sum())
    fp = int(np.logical_and(pred_pos, ~true_pos).sum())
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return sensitivity, specificity


def map_to_binary_class(raw_class_id: int) -> int:
    # Original labels are {0,1,2}; treat 2 as malignant and others as non-malignant.
    return 1 if int(raw_class_id) == 2 else 0


def label_class_id(raw_class_id: int, task_mode: TaskMode) -> int:
    if task_mode == "detect":
        return 0
    return map_to_binary_class(raw_class_id)


def _yaml_class_names(task_mode: TaskMode) -> List[str]:
    if task_mode == "detect":
        return ["  0: roi"]
    return ["  0: non_malignant", "  1: malignant"]


Sample = Tuple[str, int]
FoldSplits = Dict[str, Union[int, List[Sample]]]


def list_valid_samples(dataset_root: Path, split: str) -> List[Sample]:
    bbox_annot = load_bbox_annotations(dataset_root / "bbox_annot.json")
    imgs_dir = dataset_root / "imgs"
    rows = read_split_file(dataset_root / f"{split}.txt")
    out: List[Sample] = []
    for name, cls_id in rows:
        if name not in bbox_annot:
            continue
        if not (imgs_dir / name).exists():
            continue
        out.append((name, int(cls_id)))
    return out


def build_cv_folds(
    dataset_root: Path,
    *,
    n_cv_folds: int = 3,
    seed: int = 42,
    task_mode: TaskMode = "detect_cls",
    cv_split_file: Path | str | None = None,
) -> List[FoldSplits]:
    """Load pre-defined CV folds shared with roi_repr."""
    stratify = stratify_for_task_mode(task_mode)
    folds, split_path = load_cv_folds(
        dataset_root,
        n_cv_folds=n_cv_folds,
        seed=seed,
        stratify=stratify,
        cv_split_file=cv_split_file,
    )
    print(f"[CV splits] loaded fixed manifest: {split_path}")
    return folds


def _raw_class_dist(rows: List[Sample]) -> str:
    cnt = np.bincount([r[1] for r in rows], minlength=3)
    return f"normal={int(cnt[0])}, benign={int(cnt[1])}, malignant={int(cnt[2])}"


def _binary_class_dist(rows: List[Sample]) -> str:
    labels = [map_to_binary_class(r[1]) for r in rows]
    cnt = np.bincount(labels, minlength=2)
    return f"non_malignant={int(cnt[0])}, malignant={int(cnt[1])}"


def print_cv_summary(folds: List[FoldSplits], *, n_cv_folds: int, task_mode: TaskMode) -> None:
    print(f"[CV] {n_cv_folds}-fold stratified on train ({len(folds[0]['train']) + len(folds[0]['val'])} samples)")
    for fd in folds:
        fi = int(fd["fold"])
        train_rows = fd["train"]
        val_rows = fd["val"]
        print(f"  fold {fi}: train ({len(train_rows):4d}) raw: {_raw_class_dist(train_rows)}")
        if task_mode == "detect_cls":
            print(f"           task: {_binary_class_dist(train_rows)}")
        print(f"           fold_val ({len(val_rows):4d}) raw: {_raw_class_dist(val_rows)}")
        if task_mode == "detect_cls":
            print(f"           task: {_binary_class_dist(val_rows)}")
    test_rows = folds[0]["test"]
    print(f"[test ] held-out ({len(test_rows):4d}) raw: {_raw_class_dist(test_rows)}")
    if task_mode == "detect_cls":
        print(f"         task: {_binary_class_dist(test_rows)}")


def prepare_yolo_dataset(
    dataset_root: Path,
    generated_dir: Path,
    task_mode: TaskMode = "detect_cls",
    train_items: Optional[List[Sample]] = None,
    val_items: Optional[List[Sample]] = None,
    test_items: Optional[List[Sample]] = None,
) -> Dict[str, Path]:
    imgs_dir = dataset_root / "imgs"
    bbox_annot = load_bbox_annotations(dataset_root / "bbox_annot.json")
    if train_items is None:
        train_items = read_split_file(dataset_root / "train.txt")
    if val_items is None:
        val_items = read_split_file(dataset_root / "test.txt")
    if test_items is None:
        test_items = read_split_file(dataset_root / "test.txt")

    images_train = generated_dir / "images" / "train"
    images_val = generated_dir / "images" / "val"
    images_test = generated_dir / "images" / "test"
    labels_train = generated_dir / "labels" / "train"
    labels_val = generated_dir / "labels" / "val"
    labels_test = generated_dir / "labels" / "test"
    for d in (images_train, images_val, images_test, labels_train, labels_val, labels_test):
        d.mkdir(parents=True, exist_ok=True)

    def _prepare_split(items: List[Tuple[str, int]], images_dir: Path, labels_dir: Path) -> None:
        for image_name, class_id in items:
            src_image_path = imgs_dir / image_name
            if not src_image_path.exists():
                continue
            if image_name not in bbox_annot:
                continue
            try:
                box = choose_roi_box(bbox_annot[image_name]["bbs"])
            except Exception:
                continue
            dst_image_path = images_dir / image_name
            dst_image_path.parent.mkdir(parents=True, exist_ok=True)
            if not dst_image_path.exists():
                dst_image_path.symlink_to(src_image_path.resolve())

            with Image.open(src_image_path) as img:
                w, h = img.size
            cx, cy, bw, bh = xyxy_to_yolo(box, width=w, height=h)
            yolo_cls = label_class_id(int(class_id), task_mode=task_mode)
            image_rel = Path(image_name)
            label_path = (labels_dir / image_rel).with_suffix(".txt")
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(
                f"{yolo_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n",
                encoding="utf-8",
            )

    _prepare_split(train_items, images_train, labels_train)
    _prepare_split(val_items, images_val, labels_val)
    _prepare_split(test_items, images_test, labels_test)

    yaml_name = "gbcu_onestep_detect.yaml" if task_mode == "detect" else "gbcu_onestep_yolov11.yaml"
    data_yaml = generated_dir / yaml_name
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {str(generated_dir.resolve())}",
                "train: images/train",
                "val: images/val",
                "names:",
                *_yaml_class_names(task_mode),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "data_yaml": data_yaml,
    }


def build_train_kwargs(
    *,
    data_yaml: Path,
    output_dir: Path,
    model: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
    imgsz: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    seed: int,
    eval_score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
) -> Dict:
    return {
        "data": str(data_yaml),
        "epochs": int(epochs),
        "batch": int(batch_size),
        "imgsz": int(imgsz),
        "workers": int(num_workers),
        "seed": int(seed),
        "optimizer": "SGD",
        "lr0": float(lr),
        "momentum": float(momentum),
        "weight_decay": float(weight_decay),
        "project": str(output_dir.parent.resolve()),
        "name": output_dir.name,
        "exist_ok": True,
        "pretrained": True,
        "val": True,
        "conf": float(eval_score_threshold),
        "iou": float(nms_iou_threshold),
        "max_det": int(max_detections),
        **YOLO_TRAIN_AUGMENT_KWARGS,
    }


def resolve_best_checkpoint(run_dir: Path) -> Path:
    best_ckpt = run_dir / "weights" / "best.pt"
    if not best_ckpt.exists():
        best_ckpt = run_dir / "weights" / "last.pt"
    return best_ckpt


def run_evaluation(
    eval_model,
    *,
    dataset_root: Path,
    task_mode: TaskMode,
    model_name: str,
    eval_score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    fallback_class: int,
    imgsz: int,
    eval_items: Optional[List[Sample]] = None,
) -> Tuple[Dict, Dict[str, Dict]]:
    test_items = eval_items if eval_items is not None else read_split_file(dataset_root / "test.txt")
    bbox_annot = load_bbox_annotations(dataset_root / "bbox_annot.json")
    image_dir = dataset_root / "imgs"

    det_tp, det_fp, det_fn = 0, 0, 0
    det_ious: List[float] = []
    cls_preds: List[int] = []
    cls_gts: List[int] = []
    records: Dict[str, Dict] = {}

    for image_name, class_id in test_items:
        image_path = image_dir / image_name
        if not image_path.exists() or image_name not in bbox_annot:
            continue
        try:
            gt_box = choose_roi_box(bbox_annot[image_name]["bbs"])
        except Exception:
            continue

        raw_res = eval_model.predict(
            source=str(image_path),
            conf=0.001,
            iou=float(nms_iou_threshold),
            max_det=max(int(max_detections), 1),
            verbose=False,
            imgsz=int(imgsz),
        )[0]
        boxes = raw_res.boxes
        all_xyxy = boxes.xyxy.cpu().numpy() if boxes is not None and boxes.xyxy is not None else np.zeros((0, 4))
        all_conf = boxes.conf.cpu().numpy() if boxes is not None and boxes.conf is not None else np.zeros((0,))
        all_cls = boxes.cls.cpu().numpy().astype(np.int32) if boxes is not None and boxes.cls is not None else np.zeros((0,), dtype=np.int32)

        keep = all_conf > float(eval_score_threshold)
        det_xyxy = all_xyxy[keep]
        det_cls = all_cls[keep]

        pred_box = None
        center_in_gt = False
        det_iou = 0.0

        if det_xyxy.shape[0] == 0:
            det_fn += 1
            det_ious.append(0.0)
        else:
            hit_count = 0
            miss_count = 0
            best_iou = 0.0
            best_box = None
            gx0, gy0, gx1, gy1 = [float(v) for v in gt_box]
            for box in det_xyxy.tolist():
                px0, py0, px1, py1 = [float(v) for v in box]
                cx = (px0 + px1) / 2.0
                cy = (py0 + py1) / 2.0
                hit = gx0 <= cx <= gx1 and gy0 <= cy <= gy1
                if hit:
                    hit_count += 1
                else:
                    miss_count += 1
                iou_now = iou_xyxy([px0, py0, px1, py1], gt_box)
                if iou_now >= best_iou:
                    best_iou = iou_now
                    best_box = [px0, py0, px1, py1]
                    center_in_gt = hit
            pred_box = best_box
            det_iou = best_iou
            det_ious.append(det_iou)
            if hit_count > 0:
                det_tp += 1
                det_fp += (hit_count - 1) + miss_count
            else:
                det_fn += 1
                det_fp += miss_count

        record: Dict[str, Union[int, float, bool, List, None]] = {
            "gt_box": [float(v) for v in gt_box],
            "pred_box": pred_box,
            "det_iou": float(det_iou),
            "center_in_gt": bool(center_in_gt),
        }

        if task_mode == "detect_cls":
            roi_preds: List[int]
            if det_cls.shape[0] > 0:
                roi_preds = [int(v) for v in det_cls.tolist()]
            elif all_cls.shape[0] > 0:
                roi_preds = [int(all_cls[0])]
            else:
                roi_preds = [int(fallback_class)]

            # Binary fusion rule: if any ROI is malignant -> malignant, else non-malignant.
            cls_pred = 1 if 1 in roi_preds else 0

            cls_preds.append(int(cls_pred))
            gt_binary = map_to_binary_class(int(class_id))
            cls_gts.append(int(gt_binary))
            record.update(
                {
                    "gt_class": int(gt_binary),
                    "pred_class": int(cls_pred),
                    "roi_preds": roi_preds,
                }
            )

        records[image_name] = record

    det_recall = float(det_tp / (det_tp + det_fn)) if (det_tp + det_fn) > 0 else 0.0
    det_precision = float(det_tp / (det_tp + det_fp)) if (det_tp + det_fp) > 0 else 0.0
    det_miou = float(np.mean(det_ious)) if det_ious else 0.0
    num_images = len(records)

    summary: Dict = {
        "task_mode": task_mode,
        "detector_metrics": {
            "model": str(model_name),
            "mIoU": det_miou,
            "precision": det_precision,
            "recall": det_recall,
            "TP": int(det_tp),
            "FP": int(det_fp),
            "FN": int(det_fn),
            "num_images": int(num_images),
            "eval_score_threshold": float(eval_score_threshold),
            "nms_iou_threshold": float(nms_iou_threshold),
            "max_detections": int(max_detections),
        },
    }

    if task_mode == "detect_cls":
        cls_preds_np = np.asarray(cls_preds, dtype=np.int32)
        cls_gts_np = np.asarray(cls_gts, dtype=np.int32)
        cls_acc = float((cls_preds_np == cls_gts_np).mean()) if cls_gts_np.size > 0 else 0.0
        cls_sens, cls_spec = compute_malignant_metrics(cls_preds_np, cls_gts_np, malignant_id=1)
        summary["classifier_metrics"] = {
            "accuracy": cls_acc,
            "malignant_sensitivity": cls_sens,
            "malignant_specificity": cls_spec,
            "num_images": int(num_images),
        }

    return summary, records


def summarize_cv_results(fold_results: List[Dict]) -> Dict:
    def _mean_std(key_path: Tuple[str, ...]) -> Tuple[float, float]:
        vals = []
        for r in fold_results:
            obj = r
            for k in key_path:
                obj = obj[k]
            vals.append(float(obj))
        arr = np.asarray(vals, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=0))

    det_keys = ["mIoU", "precision", "recall"]
    cls_keys = ["accuracy", "malignant_sensitivity", "malignant_specificity"]
    summary: Dict = {
        "n_folds": len(fold_results),
        "folds": [
            {
                "fold": r["fold"],
                "detector_metrics": r["detector_metrics"],
                "classifier_metrics": r.get("classifier_metrics"),
                "checkpoint": r.get("checkpoint"),
            }
            for r in fold_results
        ],
        "detector_metrics_mean_std": {
            k: {"mean": _mean_std(("detector_metrics", k))[0], "std": _mean_std(("detector_metrics", k))[1]}
            for k in det_keys
        },
    }
    if fold_results and "classifier_metrics" in fold_results[0]:
        summary["classifier_metrics_mean_std"] = {
            k: {"mean": _mean_std(("classifier_metrics", k))[0], "std": _mean_std(("classifier_metrics", k))[1]}
            for k in cls_keys
        }
    return summary


def print_cv_results_summary(cv_summary: Dict) -> None:
    print(f"\n== CV summary ({cv_summary['n_folds']} folds, test set) ==")
    det = cv_summary["detector_metrics_mean_std"]
    print(
        f"det mIoU={det['mIoU']['mean']:.4f}±{det['mIoU']['std']:.4f} "
        f"precision={det['precision']['mean']:.4f}±{det['precision']['std']:.4f} "
        f"recall={det['recall']['mean']:.4f}±{det['recall']['std']:.4f}"
    )
    if "classifier_metrics_mean_std" in cv_summary:
        cls = cv_summary["classifier_metrics_mean_std"]
        print(
            f"cls accuracy={cls['accuracy']['mean']:.4f}±{cls['accuracy']['std']:.4f} "
            f"mal_sens={cls['malignant_sensitivity']['mean']:.4f}±{cls['malignant_sensitivity']['std']:.4f} "
            f"mal_spec={cls['malignant_specificity']['mean']:.4f}±{cls['malignant_specificity']['std']:.4f}"
        )


def print_summary(summary: Dict) -> None:
    det = summary["detector_metrics"]
    print("== YOLOv11 one-step detector ==")
    print(
        f"mIoU={det['mIoU']:.4f} precision={det['precision']:.4f} recall={det['recall']:.4f} "
        f"(TP={det['TP']} FP={det['FP']} FN={det['FN']})"
    )
    if "classifier_metrics" in summary:
        cls = summary["classifier_metrics"]
        print("== YOLOv11 one-step classifier ==")
        print(
            f"accuracy={cls['accuracy']:.4f} "
            f"malignant_sensitivity={cls['malignant_sensitivity']:.4f} "
            f"malignant_specificity={cls['malignant_specificity']:.4f}"
        )


def run_single_fold(
    args: argparse.Namespace,
    fold_splits: FoldSplits,
    *,
    YOLO,
) -> Dict:
    fold = int(fold_splits["fold"])
    output_dir = Path(args.output_dir)
    fold_dir = output_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = fold_dir / "generated_yolo_data"
    generated_dir.mkdir(parents=True, exist_ok=True)

    task_mode: TaskMode = args.task_mode
    generated = prepare_yolo_dataset(
        dataset_root=Path(args.dataset_root),
        generated_dir=generated_dir,
        task_mode=task_mode,
        train_items=fold_splits["train"],
        val_items=fold_splits["val"],
        test_items=fold_splits["test"],
    )
    data_yaml = generated["data_yaml"]
    print(f"\n===== [YOLO] fold={fold} TASK_MODE={task_mode} =====")
    print(f"Prepared YOLO dataset yaml: {data_yaml}")

    model = YOLO(args.model)
    model.train(
        **build_train_kwargs(
            data_yaml=data_yaml,
            output_dir=fold_dir,
            model=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            imgsz=args.imgsz,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            seed=args.seed + fold,
            eval_score_threshold=args.eval_score_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            max_detections=args.max_detections,
        )
    )

    best_ckpt = resolve_best_checkpoint(fold_dir)
    print(f"Using checkpoint for evaluation: {best_ckpt}")
    eval_model = YOLO(str(best_ckpt))

    summary, records = run_evaluation(
        eval_model,
        dataset_root=Path(args.dataset_root),
        task_mode=task_mode,
        model_name=args.model,
        eval_score_threshold=args.eval_score_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        max_detections=args.max_detections,
        fallback_class=args.fallback_class,
        imgsz=args.imgsz,
        eval_items=fold_splits["test"],
    )

    fold_result = {
        "fold": fold,
        "checkpoint": str(best_ckpt),
        "detector_metrics": summary["detector_metrics"],
    }
    if "classifier_metrics" in summary:
        fold_result["classifier_metrics"] = summary["classifier_metrics"]

    with (fold_dir / "fold_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (fold_dir / "fold_records.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print_summary(summary)
    return fold_result


def run_cv_experiment(args: argparse.Namespace, *, YOLO) -> Dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_mode: TaskMode = args.task_mode
    folds = build_cv_folds(
        Path(args.dataset_root),
        n_cv_folds=int(args.n_cv_folds),
        seed=int(args.seed),
        task_mode=task_mode,
        cv_split_file=getattr(args, "cv_split_file", None),
    )
    print_cv_summary(folds, n_cv_folds=int(args.n_cv_folds), task_mode=task_mode)
    print("Augmentation alignment: keep horizontal flip only (fliplr=0.5), disable mosaic/mixup/scale/rotation.")

    fold_results: List[Dict] = []
    for fd in folds:
        fold_dir = output_dir / f"fold{fd['fold']}"
        if (fold_dir / "fold_summary.json").exists():
            print(f"Already complete, loading: {fold_dir / 'fold_summary.json'}")
            with (fold_dir / "fold_summary.json").open("r", encoding="utf-8") as f:
                summary = json.load(f)
            fold_results.append(
                {
                    "fold": int(fd["fold"]),
                    "checkpoint": str(resolve_best_checkpoint(fold_dir)),
                    "detector_metrics": summary["detector_metrics"],
                    **(
                        {"classifier_metrics": summary["classifier_metrics"]}
                        if "classifier_metrics" in summary
                        else {}
                    ),
                }
            )
            continue
        fold_results.append(run_single_fold(args, fd, YOLO=YOLO))

    cv_summary = summarize_cv_results(fold_results)
    out = {
        "task_mode": task_mode,
        "n_cv_folds": int(args.n_cv_folds),
        "cv_summary": cv_summary,
        "fold_results": fold_results,
    }
    with (output_dir / "cv_results.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print_cv_results_summary(cv_summary)
    print(f"Saved CV summary to: {output_dir / 'cv_results.json'}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train YOLOv11 one-step ROI+classification on GBCU")
    parser.add_argument("--dataset_root", type=str, default="GBCU")
    parser.add_argument("--output_dir", type=str, default="outputs/yolov11_onestep_run1")
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--eval_score_threshold", type=float, default=0.5)
    parser.add_argument("--nms_iou_threshold", type=float, default=0.5)
    parser.add_argument("--max_detections", type=int, default=5)
    parser.add_argument("--fallback_class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n_cv_folds",
        type=int,
        default=1,
        help="Stratified K-fold on train split; 1 keeps legacy train/val=test hold-out training.",
    )
    parser.add_argument(
        "--cv_split_file",
        type=str,
        default=None,
        help="Path to fixed CV split JSON (default: dataset_root/cv_splits/cv{n}_seed{seed}_{stratify}.json)",
    )
    parser.add_argument(
        "--task_mode",
        type=str,
        default="detect_cls",
        choices=["detect", "detect_cls"],
        help="detect: single-class ROI detection; detect_cls: ROI detection + binary classification.",
    )
    return parser.parse_args()


def run_legacy_experiment(args: argparse.Namespace, *, YOLO) -> Dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = output_dir / "generated_yolo_data"
    generated_dir.mkdir(parents=True, exist_ok=True)

    task_mode: TaskMode = args.task_mode
    generated = prepare_yolo_dataset(
        dataset_root=Path(args.dataset_root),
        generated_dir=generated_dir,
        task_mode=task_mode,
    )
    data_yaml = generated["data_yaml"]
    print(f"Prepared YOLO dataset yaml: {data_yaml}")
    print("Augmentation alignment: keep horizontal flip only (fliplr=0.5), disable mosaic/mixup/scale/rotation.")

    model = YOLO(args.model)
    model.train(
        **build_train_kwargs(
            data_yaml=data_yaml,
            output_dir=output_dir,
            model=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            imgsz=args.imgsz,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            seed=args.seed,
            eval_score_threshold=args.eval_score_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            max_detections=args.max_detections,
        )
    )

    best_ckpt = resolve_best_checkpoint(output_dir)
    print(f"Using checkpoint for evaluation: {best_ckpt}")
    eval_model = YOLO(str(best_ckpt))

    summary, records = run_evaluation(
        eval_model,
        dataset_root=Path(args.dataset_root),
        task_mode=task_mode,
        model_name=args.model,
        eval_score_threshold=args.eval_score_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        max_detections=args.max_detections,
        fallback_class=args.fallback_class,
        imgsz=args.imgsz,
    )

    with (output_dir / "pipeline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "pipeline_records.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print_summary(summary)
    print(f"Saved summary to: {output_dir / 'pipeline_summary.json'}")
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    try:
        _disable_ultralytics_neptune_callback()
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "Cannot import ultralytics. Please install it first: pip install ultralytics"
        ) from e

    if int(args.n_cv_folds) > 1:
        run_cv_experiment(args, YOLO=YOLO)
    else:
        run_legacy_experiment(args, YOLO=YOLO)


if __name__ == "__main__":
    main()
