import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.ops import nms
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.transforms import functional as F
from transformers import AutoImageProcessor, DetrForObjectDetection
from detr_roi.preprocess import PreprocessConfig, apply_geometric_preprocess, map_boxes_xyxy_to_original

ROI_LABELS = {"nml", "abn"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def choose_roi_box(raw_bbs: List[List]) -> List[float]:
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


def load_bbox_annotations(annotation_file: Path) -> Dict[str, Dict]:
    with annotation_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_box_xyxy(box: List[float], width: int, height: int) -> Tuple[int, int, int, int]:
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


def iou_xyxy(box_a: List[float], box_b: List[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


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


def build_fasterrcnn_detector() -> nn.Module:
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=0,
        box_score_thresh=0.5,
        box_nms_thresh=0.5,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model.transform = GeneralizedRCNNTransform(
        min_size=800,
        max_size=1333,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    return model


def build_detr_detector(model_name: str) -> nn.Module:
    model = DetrForObjectDetection.from_pretrained(
        model_name,
        num_labels=1,
        ignore_mismatched_sizes=True,
        id2label={0: "roi"},
        label2id={"roi": 0},
    )
    return model


def build_classifier() -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(int(model.fc.in_features), 3)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate end-to-end detector -> ResNet50 pipeline")
    parser.add_argument("--dataset_root", type=str, default="GBCU")
    parser.add_argument(
        "--detector_type",
        type=str,
        default="fasterrcnn",
        choices=["fasterrcnn", "detr"],
        help="Choose ROI detector backend.",
    )
    parser.add_argument(
        "--detector_ckpt",
        type=str,
        default="outputs/faster_rcnn_gb_detector_run1/best_model.pth",
    )
    parser.add_argument(
        "--detector_model_name",
        type=str,
        default="facebook/detr-resnet-50",
        help="HF model name used only when detector_type=detr.",
    )
    parser.add_argument(
        "--classifier_ckpt",
        type=str,
        default="outputs/resnet50_roi_cls_run1/checkpoints/best_resnet50.pth",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/pipeline_eval_run1")
    parser.add_argument(
        "--eval_score_threshold",
        type=float,
        default=0.5,
        help="Use only detections with score > this value for metrics and classification crops.",
    )
    parser.add_argument("--nms_iou_threshold", type=float, default=0.5)
    parser.add_argument("--max_detections", type=int, default=5)
    parser.add_argument(
        "--detr_use_paper_preprocess",
        action="store_true",
        help="For DETR, apply training-aligned geometric preprocess before inference.",
    )
    parser.add_argument("--prep_shortest_edge", type=int, default=800)
    parser.add_argument("--prep_longest_edge", type=int, default=1333)
    parser.add_argument("--prep_center_crop_h", type=int, default=800)
    parser.add_argument("--prep_center_crop_w", type=int, default=800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = None
    detr_preprocess_cfg = None
    if args.detector_type == "fasterrcnn":
        print(
            "Detector=fasterrcnn: box_score_thresh=0.5, box_nms_thresh=0.5; "
            f"pipeline uses score > {args.eval_score_threshold:.3f}, nms_iou={args.nms_iou_threshold:.2f}, max_det={args.max_detections}."
        )
        detector = build_fasterrcnn_detector()
    else:
        if args.detr_use_paper_preprocess:
            detr_preprocess_cfg = PreprocessConfig(
                use_geometric=True,
                shortest_edge=args.prep_shortest_edge,
                longest_edge=args.prep_longest_edge,
                center_crop_h=args.prep_center_crop_h,
                center_crop_w=args.prep_center_crop_w,
            )
        print(
            f"Detector=detr ({args.detector_model_name}); "
            f"pipeline uses score > {args.eval_score_threshold:.3f}, nms_iou={args.nms_iou_threshold:.2f}, max_det={args.max_detections}, "
            f"paper_preprocess={args.detr_use_paper_preprocess}."
        )
        image_processor = AutoImageProcessor.from_pretrained(args.detector_model_name)
        detector = build_detr_detector(args.detector_model_name)

    print(f"Using detector checkpoint: {args.detector_ckpt}")
    detector_state = torch.load(args.detector_ckpt, map_location="cpu")
    detector.to(device)
    detector.load_state_dict(detector_state)
    detector.eval()
    if isinstance(detector_state, dict):
        det_epoch = detector_state.get("epoch")
        det_recall = detector_state.get("recall")
        det_miou = detector_state.get("mIoU")
        if det_epoch is not None or det_recall is not None or det_miou is not None:
            print("== Detector checkpoint metadata ==")
            epoch_text = str(int(det_epoch)) if det_epoch is not None else "N/A"
            recall_text = f"{float(det_recall):.4f}" if det_recall is not None else "N/A"
            print(f"epoch={epoch_text} recall={recall_text}")
            if det_miou is not None:
                print(f"mIoU={float(det_miou):.4f}")

    classifier = build_classifier()
    print(f"Using classifier checkpoint: {args.classifier_ckpt}")
    cls_raw = torch.load(args.classifier_ckpt, map_location="cpu")
    cls_state = cls_raw["model_state_dict"] if isinstance(cls_raw, dict) and "model_state_dict" in cls_raw else cls_raw
    classifier.load_state_dict(cls_state)
    classifier.to(device)
    classifier.eval()
    if isinstance(cls_raw, dict):
        ckpt_acc = cls_raw.get("test_acc")
        ckpt_sens = cls_raw.get("malignant_sensitivity")
        ckpt_spec = cls_raw.get("malignant_specificity")
        ckpt_epoch = cls_raw.get("epoch")
        if ckpt_acc is not None and ckpt_sens is not None and ckpt_spec is not None:
            print("== Classifier checkpoint metrics (ROI training/eval) ==")
            print(
                f"epoch={int(ckpt_epoch) if ckpt_epoch is not None else 'N/A'} "
                f"accuracy={float(ckpt_acc):.4f} malignant_sensitivity={float(ckpt_sens):.4f} "
                f"malignant_specificity={float(ckpt_spec):.4f}"
            )

    cls_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    print(
        "Classifier preprocess (aligned with ROI classifier training): "
        "Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize(ImageNet mean/std)."
    )

    test_items = read_split_file(dataset_root / "test.txt")
    bbox_annot = load_bbox_annotations(dataset_root / "bbox_annot.json")
    image_dir = dataset_root / "imgs"

    det_tp = 0
    det_fp = 0
    det_fn = 0
    det_ious: List[float] = []
    cls_preds: List[int] = []
    cls_gts: List[int] = []
    records: Dict[str, Dict] = {}

    with torch.no_grad():
        for image_name, class_id in test_items:
            image_path = image_dir / image_name
            if not image_path.exists():
                continue

            image = Image.open(image_path).convert("RGB")
            w, h = image.size
            gt_box = choose_roi_box(bbox_annot[image_name]["bbs"])

            if args.detector_type == "fasterrcnn":
                image_tensor = F.to_tensor(image).to(device)
                output = detector([image_tensor])[0]
                pred_boxes = output["boxes"]
                pred_scores = output["scores"]
            else:
                assert image_processor is not None
                detr_image = image
                detr_meta = None
                if detr_preprocess_cfg is not None:
                    detr_image, _, detr_meta = apply_geometric_preprocess(
                        image=image,
                        box_xyxy=gt_box,
                        config=detr_preprocess_cfg,
                    )
                proc_extra = {}
                if detr_meta is not None:
                    proc_extra = {
                        "do_resize": False,
                        "do_normalize": True,
                        "do_rescale": True,
                        "do_pad": True,
                    }
                inputs = image_processor(images=detr_image, return_tensors="pt", **proc_extra)
                pixel_values = inputs["pixel_values"].to(device)
                pixel_mask = inputs.get("pixel_mask")
                if pixel_mask is not None:
                    pixel_mask = pixel_mask.to(device)
                raw = detector(pixel_values=pixel_values, pixel_mask=pixel_mask)
                if detr_meta is not None:
                    target_sizes = torch.tensor(
                        [[int(detr_meta["geom_h"]), int(detr_meta["geom_w"])]],
                        device=device,
                        dtype=torch.long,
                    )
                else:
                    target_sizes = torch.tensor([[h, w]], device=device, dtype=torch.long)
                processed = image_processor.post_process_object_detection(
                    raw, threshold=args.eval_score_threshold, target_sizes=target_sizes
                )[0]
                pred_boxes = processed["boxes"]
                pred_scores = processed["scores"]
                if detr_meta is not None and pred_boxes.numel() > 0:
                    pred_boxes = map_boxes_xyxy_to_original(pred_boxes, detr_meta)
                if pred_boxes.numel() > 0:
                    keep = nms(pred_boxes, pred_scores, args.nms_iou_threshold)
                    if args.max_detections > 0:
                        keep = keep[: args.max_detections]
                    pred_boxes = pred_boxes[keep]
                    pred_scores = pred_scores[keep]

            pred_box = None
            center_in_gt = False
            det_iou = 0.0
            if pred_boxes.numel() == 0 or pred_scores.numel() == 0:
                det_fn += 1
                det_ious.append(0.0)
            else:
                keep = pred_scores > args.eval_score_threshold
                if int(keep.sum().item()) == 0:
                    det_fn += 1
                    det_ious.append(0.0)
                    boxes_kept = []
                else:
                    boxes_kept = pred_boxes[keep].detach().cpu().tolist()
                if not boxes_kept:
                    pass  # FN already counted; IoU already appended when score filter empty
                else:
                    hit_count = 0
                    miss_count = 0
                    best_iou = 0.0
                    best_box = None
                    gx0, gy0, gx1, gy1 = [float(v) for v in gt_box]
                    for box in boxes_kept:
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

            # Paper-style classification-time ROI handling:
            # A) no detected box -> fallback to full image;
            # B) multiple detected boxes -> classify each selected ROI and fuse by rule.
            roi_preds: List[int] = []
            if pred_boxes.numel() == 0 or pred_scores.numel() == 0:
                full_tensor = cls_tf(image).unsqueeze(0).to(device)
                logits = classifier(full_tensor)
                roi_preds.append(int(torch.argmax(logits, dim=1).item()))
            else:
                keep = pred_scores > args.eval_score_threshold
                if int(keep.sum().item()) == 0:
                    full_tensor = cls_tf(image).unsqueeze(0).to(device)
                    logits = classifier(full_tensor)
                    roi_preds.append(int(torch.argmax(logits, dim=1).item()))
                else:
                    kept_boxes = pred_boxes[keep]
                    if args.max_detections > 0 and int(kept_boxes.shape[0]) > args.max_detections:
                        kept_scores = pred_scores[keep]
                        topk_idx = torch.argsort(kept_scores, descending=True)[: args.max_detections]
                        kept_boxes = kept_boxes[topk_idx]
                    for idx in range(len(kept_boxes)):
                        box_i = [float(v) for v in kept_boxes[idx].detach().cpu().tolist()]
                        x0, y0, x1, y1 = sanitize_box_xyxy(box_i, width=w, height=h)
                        crop = image.crop((x0, y0, x1, y1))
                        crop_tensor = cls_tf(crop).unsqueeze(0).to(device)
                        logits = classifier(crop_tensor)
                        roi_preds.append(int(torch.argmax(logits, dim=1).item()))

            if 2 in roi_preds:
                cls_pred = 2
            elif roi_preds and all(p == 0 for p in roi_preds):
                cls_pred = 0
            else:
                cls_pred = 1

            cls_preds.append(cls_pred)
            cls_gts.append(int(class_id))
            records[image_name] = {
                "gt_class": int(class_id),
                "pred_class": int(cls_pred),
                "roi_preds": roi_preds,
                "gt_box": [float(v) for v in gt_box],
                "pred_box": pred_box,
                "det_iou": float(det_iou),
                "center_in_gt": bool(center_in_gt),
            }

    det_recall = float(det_tp / (det_tp + det_fn)) if (det_tp + det_fn) > 0 else 0.0
    det_precision = float(det_tp / (det_tp + det_fp)) if (det_tp + det_fp) > 0 else 0.0
    det_miou = float(np.mean(det_ious)) if det_ious else 0.0

    cls_preds_np = np.asarray(cls_preds, dtype=np.int32)
    cls_gts_np = np.asarray(cls_gts, dtype=np.int32)
    cls_acc = float((cls_preds_np == cls_gts_np).mean()) if cls_gts_np.size > 0 else 0.0
    cls_sens, cls_spec = compute_malignant_metrics(cls_preds_np, cls_gts_np, malignant_id=2)
    print("== Classifier (3 metrics, pre-summary) ==")
    print(
        f"accuracy={cls_acc:.4f} malignant_sensitivity={cls_sens:.4f} "
        f"malignant_specificity={cls_spec:.4f} (N={int(len(cls_gts))})"
    )

    summary = {
        "detector_metrics": {
            "detector_type": args.detector_type,
            "mIoU": det_miou,
            "precision": det_precision,
            "recall": det_recall,
            "TP": int(det_tp),
            "FP": int(det_fp),
            "FN": int(det_fn),
            "num_images": int(len(cls_gts)),
            "eval_score_threshold": float(args.eval_score_threshold),
            "nms_iou_threshold": float(args.nms_iou_threshold),
            "max_detections": int(args.max_detections),
        },
        "classifier_metrics": {
            "accuracy": cls_acc,
            "malignant_sensitivity": cls_sens,
            "malignant_specificity": cls_spec,
            "num_images": int(len(cls_gts)),
        },
    }

    with (output_dir / "pipeline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "pipeline_records.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print("== Detector ==")
    print(
        f"mIoU={det_miou:.4f} precision={det_precision:.4f} recall={det_recall:.4f} "
        f"(TP={det_tp} FP={det_fp} FN={det_fn})"
    )
    print("== Classifier ==")
    print(
        f"accuracy={cls_acc:.4f} malignant_sensitivity={cls_sens:.4f} malignant_specificity={cls_spec:.4f}"
    )
    print(f"Saved summary to: {output_dir / 'pipeline_summary.json'}")


if __name__ == "__main__":
    main()
