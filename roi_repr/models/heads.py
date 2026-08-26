"""Shared embedding and classifier heads."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

NormType = Literal["batchnorm", "layernorm"]
ActivationType = Literal["relu", "gelu"]
ClassifierHeadType = Literal["linear", "mlp"]


def _build_norm(norm: NormType, dim: int) -> nn.Module:
    if norm == "batchnorm":
        return nn.BatchNorm1d(dim)
    return nn.LayerNorm(dim)


def _build_activation(activation: ActivationType) -> nn.Module:
    if activation == "gelu":
        return nn.GELU()
    return nn.ReLU(inplace=True)


class EmbeddingHead(nn.Module):
    def __init__(self, in_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, embedding_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.proj(feat)


class ClassifierHead(nn.Module):
    """Shallow MLP head: embedding_dim -> hidden_dim -> num_classes."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        norm: NormType = "batchnorm",
        activation: ActivationType = "relu",
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            _build_norm(norm, hidden_dim),
            _build_activation(activation),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def build_classifier_head(
    head_type: ClassifierHeadType,
    embedding_dim: int,
    num_classes: int,
    *,
    hidden_dim: int = 64,
    dropout: float = 0.3,
    norm: NormType = "batchnorm",
    activation: ActivationType = "relu",
) -> nn.Module:
    if head_type == "linear":
        return nn.Linear(embedding_dim, num_classes)
    return ClassifierHead(
        embedding_dim,
        num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        norm=norm,
        activation=activation,
    )
