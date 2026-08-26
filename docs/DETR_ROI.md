# DETR ROI Training on GBCU

This module trains a ROI detector with a pre-trained DETR backbone for the GBCU dataset.

## What is trained

- Model (standard): `facebook/detr-resnet-50` (pre-trained on COCO)
- Model (optional): Deformable DETR (via `--model_type deformable_detr`)
- Task: single-class ROI detection (`roi`)
- Ground-truth box selection from `bbox_annot.json`:
  - Prefer labels `nml` and `abn` (the large ROI box used in original work)
  - Fallback: largest box in image

## Metrics

The scripts report:

- `mIoU` (mean IoU across images)
- `Precision` (TP / (TP + FP))
- `Recall` (TP / (TP + FN))

Matching rule (current implementation):

- One GT ROI per image
- Multiple predictions may exist; NMS uses `--nms_iou_threshold`
- **Precision / Recall**: true positive if the **center of the predicted box** lies inside the GT box (paper-style localization)
- **mIoU**: mean per-image IoU between **union of predicted boxes** and the GT box (`miou_rule` in metrics logs)

## Files

- `detr_roi/train.py`: training + per-epoch evaluation
- `detr_roi/evaluate.py`: standalone evaluation
- `train_detr_roi.slurm`: cluster submission template
- `requirements.txt`: dependencies

## Quick start

From `GBCNET_Plus`:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m detr_roi.train \
  --dataset_root GBCU \
  --output_dir outputs/detr_roi_run1
```

Train with Deformable DETR (other logic unchanged):

```bash
python -m detr_roi.train \
  --dataset_root GBCU \
  --model_type deformable_detr \
  --model_name SenseTime/deformable-detr \
  --output_dir outputs/deformable_detr_roi_run1
```

Evaluate best checkpoint:

```bash
python -m detr_roi.evaluate \
  --dataset_root GBCU \
  --split test \
  --checkpoint outputs/detr_roi_run1/checkpoints/best_miou.pth \
  --output_dir outputs/detr_roi_run1
```

Evaluate a Deformable DETR checkpoint:

```bash
python -m detr_roi.evaluate \
  --dataset_root GBCU \
  --split test \
  --model_type deformable_detr \
  --model_name SenseTime/deformable-detr \
  --checkpoint outputs/deformable_detr_roi_run1/checkpoints/best_miou.pth \
  --output_dir outputs/deformable_detr_roi_run1
```

## Outputs

- `outputs/.../checkpoints/best_miou.pth`: best model by mIoU
- `outputs/.../history.json`: train/eval history by epoch
- `outputs/.../metrics_test.json`: final metrics
- `outputs/.../roi_pred_test.json`: per-image predictions and IoU
- `outputs/.../logs/train.log`: console + file training log
- `outputs/.../logs/train_steps.jsonl`: step-level loss/LR trace
- `outputs/.../logs/epoch_metrics.csv`: epoch-level metrics table
- `outputs/.../logs/latest_metrics.json`: latest epoch snapshot
- `outputs/.../tensorboard/`: TensorBoard events (if `--use_tensorboard`)

## Monitoring

Tail training log in real-time:

```bash
tail -f outputs/detr_roi_run1/logs/train.log
```

Launch TensorBoard on server (if enabled):

```bash
tensorboard --logdir outputs/detr_roi_run1/tensorboard --port 6006
```
