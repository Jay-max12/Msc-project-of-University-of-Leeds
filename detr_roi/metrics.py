from typing import List

import numpy as np


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


def union_iou_xyxy(pred_boxes: List[List[float]], gt_box: List[float]) -> float:
    """IoU between GT box and union area of predicted boxes."""
    if not pred_boxes:
        return 0.0

    gx0, gy0, gx1, gy1 = [float(v) for v in gt_box]
    if gx1 <= gx0 or gy1 <= gy0:
        return 0.0
    gt_area = (gx1 - gx0) * (gy1 - gy0)

    rects = []
    for b in pred_boxes:
        x0, y0, x1, y1 = [float(v) for v in b]
        if x1 <= x0 or y1 <= y0:
            continue
        rects.append((x0, y0, x1, y1))
    if not rects:
        return 0.0

    xs = sorted({x for r in rects for x in (r[0], r[2])})
    if len(xs) < 2:
        return 0.0

    union_pred_area = 0.0
    inter_area = 0.0
    for i in range(len(xs) - 1):
        lx, rx = xs[i], xs[i + 1]
        if rx <= lx:
            continue
        seg_w = rx - lx
        intervals = []
        inter_intervals = []
        for x0, y0, x1, y1 in rects:
            if x0 < rx and x1 > lx:
                intervals.append((y0, y1))
                # intersection strip with GT
                iy0 = max(y0, gy0)
                iy1 = min(y1, gy1)
                if iy1 > iy0 and not (rx <= gx0 or lx >= gx1):
                    inter_intervals.append((iy0, iy1))

        if intervals:
            intervals.sort()
            cy0, cy1 = intervals[0]
            covered_h = 0.0
            for y0, y1 in intervals[1:]:
                if y0 <= cy1:
                    cy1 = max(cy1, y1)
                else:
                    covered_h += max(0.0, cy1 - cy0)
                    cy0, cy1 = y0, y1
            covered_h += max(0.0, cy1 - cy0)
            union_pred_area += seg_w * covered_h

        if inter_intervals:
            inter_intervals.sort()
            cy0, cy1 = inter_intervals[0]
            covered_h = 0.0
            for y0, y1 in inter_intervals[1:]:
                if y0 <= cy1:
                    cy1 = max(cy1, y1)
                else:
                    covered_h += max(0.0, cy1 - cy0)
                    cy0, cy1 = y0, y1
            covered_h += max(0.0, cy1 - cy0)
            inter_area += seg_w * covered_h

    denom = union_pred_area + gt_area - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def summarize_detection(iou_values: List[float], tp: int, fp: int, fn: int) -> dict:
    miou = float(np.mean(iou_values)) if iou_values else 0.0
    miou_std = float(np.std(iou_values)) if iou_values else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return {
        "mIoU": miou,
        "mIoU_std": miou_std,
        "precision": precision,
        "recall": recall,
        "num_images": len(iou_values),
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }
