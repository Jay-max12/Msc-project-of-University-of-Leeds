"""Unified evaluation metrics."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

from roi_repr.config import ReprConfig
from roi_repr.losses.losses import compute_batch_loss


@torch.no_grad()
def eval_split(
    cfg: ReprConfig,
    model: nn.Module,
    loader: DataLoader,
    cls_crit: nn.Module,
    device: torch.device,
    lambda_recon: float | None = None,
) -> dict:
    model.eval()
    buckets = {"total": [], "cls": [], "recon": [], "kl": []}
    preds, labels = [], []
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        out = model(x)
        _, parts = compute_batch_loss(
            cfg, cfg.method, out, y, x, cls_crit, lambda_recon=lambda_recon
        )
        for k in buckets:
            buckets[k].append(parts.get(k, 0.0))
        preds.extend(out["logits"].argmax(1).cpu().numpy().tolist())
        labels.extend(y.cpu().numpy().tolist())

    y_true = np.array(labels, dtype=np.int64)
    y_pred = np.array(preds, dtype=np.int64)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    bal = float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0

    from train_resnet50_roi_classifier import compute_malignant_metrics

    mal_sens, mal_spec = compute_malignant_metrics(y_pred, y_true, malignant_id=cfg.malignant_id)
    select = 0.5 * (mal_sens + mal_spec)
    return {
        "loss_total": float(np.mean(buckets["total"])) if buckets["total"] else 0.0,
        "loss_cls": float(np.mean(buckets["cls"])) if buckets["cls"] else 0.0,
        "loss_recon": float(np.mean(buckets["recon"])) if buckets["recon"] else 0.0,
        "loss_kl": float(np.mean(buckets["kl"])) if buckets["kl"] else 0.0,
        "accuracy": acc,
        "balanced_accuracy": bal,
        "mal_sens": mal_sens,
        "mal_spec": mal_spec,
        "select": select,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def checkpoint_score(cfg: ReprConfig, val_metrics: dict) -> float:
    return float(val_metrics["select"])


def format_metrics_line(prefix: str, m: dict) -> str:
    if not m:
        return f"{prefix}(no metrics)"
    return (
        f"{prefix}acc={m.get('accuracy', 0.0):.4f} bal_acc={m.get('balanced_accuracy', 0.0):.4f} "
        f"mal_sens={m.get('mal_sens', 0.0):.4f} mal_spec={m.get('mal_spec', 0.0):.4f} "
        f"select={m.get('select', 0.0):.4f} loss_cls={m.get('loss_cls', 0.0):.4f}"
    )
