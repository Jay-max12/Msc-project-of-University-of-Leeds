"""Loss functions for all representation-learning methods."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from roi_repr.config import ReprConfig


def classification_loss(logits: torch.Tensor, labels: torch.Tensor, crit: nn.Module) -> torch.Tensor:
    return crit(logits, labels)


def reconstruction_loss(recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(recon, x, reduction="mean")


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def contrastive_loss(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
    pair_label: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    dist = F.pairwise_distance(emb1, emb2, p=2)
    pair_label = pair_label.view(-1).to(dist.dtype)
    loss_pos = pair_label * dist.pow(2)
    loss_neg = (1.0 - pair_label) * F.relu(margin - dist).pow(2)
    return 0.5 * (loss_pos + loss_neg).mean()


def compute_batch_loss(
    cfg: ReprConfig,
    method: str,
    out: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    x: torch.Tensor | None,
    cls_crit: nn.Module,
    lambda_recon: float | None = None,
    triplet_crit: nn.Module | None = None,
    extra: Dict[str, torch.Tensor] | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    parts: Dict[str, float] = {}
    l_cls = classification_loss(out["logits"], labels, cls_crit)
    parts["cls"] = float(l_cls.item())
    total = cfg.lambda_cls * l_cls

    if method in ("ae", "vae") and x is not None and "recon" in out:
        l_recon = reconstruction_loss(out["recon"], x)
        parts["recon"] = float(l_recon.item())
        recon_weight = cfg.lambda_recon if lambda_recon is None else lambda_recon
        total = total + recon_weight * l_recon
        if method == "vae" and "mu" in out and "logvar" in out:
            l_kl = kl_loss(out["mu"], out["logvar"])
            parts["kl"] = float(l_kl.item())
            total = total + cfg.beta_kl * l_kl
        else:
            parts["kl"] = 0.0

    if method == "siamese" and extra is not None:
        l_metric = contrastive_loss(
            extra["emb1"],
            extra["emb2"],
            extra["pair_label"],
            margin=cfg.margin,
        )
        parts["metric"] = float(l_metric.item())
        total = total + cfg.lambda_contrastive * l_metric

    if method == "triplet" and extra is not None and triplet_crit is not None:
        l_metric = triplet_crit(extra["emb_a"], extra["emb_p"], extra["emb_n"])
        parts["metric"] = float(l_metric.item())
        total = total + cfg.lambda_triplet * l_metric

    parts["total"] = float(total.item())
    return total, parts
