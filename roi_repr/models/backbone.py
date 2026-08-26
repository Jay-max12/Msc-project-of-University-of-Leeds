"""Shared ResNet50 encoder."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50_backbone(pretrained: bool = True) -> tuple[nn.Module, int]:
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    m = resnet50(weights=weights)
    feat_dim = int(m.fc.in_features)
    backbone = nn.Sequential(
        m.conv1,
        m.bn1,
        m.relu,
        m.maxpool,
        m.layer1,
        m.layer2,
        m.layer3,
        m.layer4,
        m.avgpool,
        nn.Flatten(),
    )
    return backbone, feat_dim
