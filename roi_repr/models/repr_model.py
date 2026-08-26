"""Unified representation-learning models (cls / ae / vae / siamese / triplet)."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from roi_repr.config import ReprConfig
from roi_repr.models.backbone import build_resnet50_backbone
from roi_repr.models.decoder import RoiDecoder
from roi_repr.models.heads import EmbeddingHead, build_classifier_head


class RoiReprModel(nn.Module):
    """
    Shared path:
      x -> backbone (2048) -> embedding_head -> z (128) -> classifier -> logits

    AE/VAE add decoder on z (train from mu for VAE).
    """

    def __init__(self, cfg: ReprConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.method = cfg.method
        self.backbone, feat_dim = build_resnet50_backbone(cfg.pretrained)
        self.embedding_head = EmbeddingHead(feat_dim, cfg.embedding_dim)
        self.classifier = build_classifier_head(
            cfg.classifier_head,
            cfg.embedding_dim,
            cfg.num_classes,
            hidden_dim=cfg.classifier_hidden_dim,
            dropout=cfg.classifier_dropout,
            norm=cfg.classifier_norm,
            activation=cfg.classifier_activation,
        )

        if cfg.method in ("vae",):
            self.fc_logvar = nn.Linear(feat_dim, cfg.embedding_dim)
        else:
            self.fc_logvar = None

        if cfg.method in ("ae", "vae"):
            self.decoder = RoiDecoder(cfg.embedding_dim, cfg.roi_size)
        else:
            self.decoder = None

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def embed(self, feat: torch.Tensor) -> torch.Tensor:
        return self.embedding_head(feat)

    def encode_vae(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.encode_features(x)
        mu = self.embed(feat)
        assert self.fc_logvar is not None
        logvar = self.fc_logvar(feat)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        assert self.decoder is not None
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.method == "vae":
            mu, logvar = self.encode_vae(x)
            z = self.reparameterize(mu, logvar) if self.training else mu
            logits = self.classifier(mu)
            out: Dict[str, torch.Tensor] = {
                "z": z,
                "mu": mu,
                "logvar": logvar,
                "logits": logits,
                "recon": self.decode(z),
            }
            return out

        feat = self.encode_features(x)
        z = self.embed(feat)
        logits = self.classifier(z)
        out = {"z": z, "logits": logits}
        if self.decoder is not None:
            out["recon"] = self.decode(z)
        return out

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "vae":
            mu, _ = self.encode_vae(x)
            return mu
        feat = self.encode_features(x)
        return self.embed(feat)
