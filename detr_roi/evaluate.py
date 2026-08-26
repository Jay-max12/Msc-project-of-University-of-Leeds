import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torchvision.ops import nms
from transformers import (
    AutoImageProcessor,
    DeformableDetrForObjectDetection,
    DetrForObjectDetection,
)

from detr_roi.dataset import build_dataloader
from detr_roi.metrics import iou_xyxy, summarize_detection
from detr_roi.preprocess import PreprocessConfig, map_boxes_xyxy_to_original


def _box_center_xy(box: List[float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = box
    return (float(x0 + x1) / 2.0, float(y0 + y1) / 2.0)


def _point_in_xyxy(x: float, y: float, box: List[float]) -> bool:
    x0, y0, x1, y1 = box
    return (x0 <= x <= x1) and (y0 <= y <= y1)


def match_predictions_to_gt(
    pred_boxes: List[List[float]],
    pred_scores: List[float],
    gt_boxes: List[List[float]],
) -> Tuple[int, int, int, List[Dict], float]:
    """Greedy one-to-one matching by descending prediction score.

    Precision/Recall TP-FP-FN follow the paper's localization rule:
    - TP: predicted box center lies inside a (not-yet-matched) GT box.
    - FP: otherwise.
    - FN: GT boxes without matched predictions (includes zero prediction case).
    mIoU statistics are still computed from IoU values.
    """
    if not gt_boxes and not pred_boxes:
        return 0, 0, 0, [], 0.0

    order = sorted(range(len(pred_boxes)), key=lambda idx: pred_scores[idx], reverse=True)
    matched_gt = set()
    matches: List[Dict] = []
    tp = 0
    fp = 0
    best_iou = 0.0

    for pred_idx in order:
        pred_box = pred_boxes[pred_idx]
        cx, cy = _box_center_xy(pred_box)
        best_gt_idx = -1
        best_local_iou = 0.0
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            _iou = iou_xyxy(pred_box, gt_box)
            if _iou > best_local_iou:
                best_local_iou = _iou
                best_gt_idx = gt_idx

        best_iou = max(best_iou, best_local_iou)
        if best_gt_idx >= 0 and _point_in_xyxy(cx, cy, gt_boxes[best_gt_idx]):
            matched_gt.add(best_gt_idx)
            tp += 1
            matches.append(
                {
                    "pred_idx": pred_idx,
                    "gt_idx": best_gt_idx,
                    "iou": best_local_iou,
                    "score": pred_scores[pred_idx],
                    "center_x": cx,
                    "center_y": cy,
                    "center_in_gt": True,
                }
            )
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    if gt_boxes and not pred_boxes:
        best_iou = 0.0
    return tp, fp, fn, matches, best_iou


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    image_processor,
    dataloader,
    device: torch.device,
    score_threshold: float = 0.1,
    nms_iou_threshold: float = 0.5,
    max_detections: int = 10,
) -> Dict:
    model.eval()
    iou_values: List[float] = []
    records: Dict[str, Dict] = {}
    tp = fp = fn = 0
    total_preds = 0

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        metas = batch.get("preprocess_metas")
        if metas is not None and metas[0] is not None:
            target_sizes = torch.tensor(
                [[m["geom_h"], m["geom_w"]] for m in metas],
                device=device,
                dtype=torch.long,
            )
        else:
            target_sizes = batch["orig_sizes"].to(device)
        results = image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=score_threshold,
            target_sizes=target_sizes,
        )

        for i, result in enumerate(results):
            image_name = batch["image_names"][i]
            gt_boxes = [batch["gt_boxes_xyxy"][i].tolist()]

            pred_boxes: List[List[float]] = []
            pred_scores: List[float] = []
            if len(result["scores"]) > 0:
                boxes = result["boxes"].detach().cpu()
                scores = result["scores"].detach().cpu()
                keep = nms(boxes, scores, nms_iou_threshold)
                if max_detections > 0:
                    keep = keep[:max_detections]
                boxes = boxes[keep]
                scores = scores[keep]
                if metas is not None and metas[i] is not None:
                    boxes = map_boxes_xyxy_to_original(boxes, metas[i])
                pred_boxes = boxes.tolist()
                pred_scores = scores.tolist()

            image_tp, image_fp, image_fn, matches, best_iou = match_predictions_to_gt(
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                gt_boxes=gt_boxes,
            )
            tp += image_tp
            fp += image_fp
            fn += image_fn
            total_preds += len(pred_boxes)
            iou_values.append(best_iou)

            records[image_name] = {
                "image_id": image_name,
                "Boxes": pred_boxes,
                "Scores": pred_scores,
                "Gold": gt_boxes,
                "best_iou": best_iou,
                "matches": matches,
                "num_pred": len(pred_boxes),
                "num_gt": len(gt_boxes),
            }

    metrics = summarize_detection(iou_values=iou_values, tp=tp, fp=fp, fn=fn)
    metrics["miou_rule"] = "best_single_pred_box_vs_gt_box"
    metrics["matching_rule"] = "center_point_inside_gt"
    metrics["score_threshold"] = score_threshold
    metrics["nms_iou_threshold"] = nms_iou_threshold
    metrics["max_detections"] = max_detections
    metrics["avg_pred_per_image"] = float(total_preds / len(iou_values)) if iou_values else 0.0
    return {"metrics": metrics, "records": records}


def print_metric_table(metrics: Dict) -> None:
    miou = metrics["mIoU"] * 100.0
    miou_std = metrics["mIoU_std"] * 100.0
    precision = metrics["precision"] * 100.0
    recall = metrics["recall"] * 100.0
    print("\nModel\t\tmIoU\t\tPrecision\tRecall")
    print(
        f"Detector\t{miou:.1f} +- {miou_std:.1f}\t"
        f"{precision:.1f}\t\t{recall:.1f}"
    )
    print(
        f"TP={metrics['TP']} FP={metrics['FP']} FN={metrics['FN']} "
        f"Images={metrics['num_images']}"
    )
    print(f"AvgPred/Image={metrics['avg_pred_per_image']:.2f}")


def parse_args():
    parser = argparse.ArgumentParser("Evaluate DETR ROI detector on GBCU")
    parser.add_argument("--dataset_root", type=str, default="GBCU")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="facebook/detr-resnet-50")
    parser.add_argument(
        "--model_type",
        type=str,
        default="detr",
        choices=["detr", "deformable_detr"],
        help="Backbone family to evaluate while keeping post-processing unchanged.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--nms_iou_threshold", type=float, default=0.5)
    parser.add_argument("--max_detections", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="outputs/detr_roi")
    parser.add_argument(
        "--use_paper_preprocess",
        action="store_true",
        help="Resize (shortest/longest edge), fixed center crop, then HF rescale+normalize.",
    )
    parser.add_argument("--prep_shortest_edge", type=int, default=256)
    parser.add_argument("--prep_longest_edge", type=int, default=1333)
    parser.add_argument("--prep_center_crop_h", type=int, default=224)
    parser.add_argument("--prep_center_crop_w", type=int, default=224)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = AutoImageProcessor.from_pretrained(args.model_name)
    model_cls = DetrForObjectDetection if args.model_type == "detr" else DeformableDetrForObjectDetection
    model = model_cls.from_pretrained(
        args.model_name,
        num_labels=1,
        ignore_mismatched_sizes=True,
        id2label={0: "roi"},
        label2id={"roi": 0},
    )
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)

    prep_cfg = None
    if args.use_paper_preprocess:
        prep_cfg = PreprocessConfig(
            use_geometric=True,
            shortest_edge=args.prep_shortest_edge,
            longest_edge=args.prep_longest_edge,
            center_crop_h=args.prep_center_crop_h,
            center_crop_w=args.prep_center_crop_w,
        )
    dataloader = build_dataloader(
        dataset_root=Path(args.dataset_root),
        split_name=args.split,
        image_processor=image_processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        preprocess_config=prep_cfg,
    )
    result = evaluate_model(
        model=model,
        image_processor=image_processor,
        dataloader=dataloader,
        device=device,
        score_threshold=args.score_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        max_detections=args.max_detections,
    )
    print_metric_table(result["metrics"])

    metrics_file = output_dir / f"metrics_{args.split}.json"
    records_file = output_dir / f"roi_pred_{args.split}.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, indent=2)
    with records_file.open("w", encoding="utf-8") as f:
        json.dump(result["records"], f, indent=2)
    print(f"Saved metrics to: {metrics_file}")
    print(f"Saved predictions to: {records_file}")


if __name__ == "__main__":
    main()
