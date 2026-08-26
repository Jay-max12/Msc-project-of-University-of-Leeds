"""Lightweight decoder shared by AE and VAE (z -> 224x224 RGB)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoiDecoder(nn.Module):
    """z (B, latent_dim) -> recon (B, 3, out_size, out_size)."""

    def __init__(self, latent_dim: int = 128, out_size: int = 224) -> None:
        super().__init__()
        self.out_size = out_size
        self.base = 7
        channels = 256
        self.fc = nn.Linear(latent_dim, channels * self.base * self.base)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), 256, self.base, self.base)
        h = F.interpolate(h, size=self.out_size, mode="bilinear", align_corners=False)
        return self.conv(h)
