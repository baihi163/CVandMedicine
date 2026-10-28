"""Lightweight U-Net baseline (2D / 3D) for medical segmentation."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, spatial_dims: int = 2, dropout: float = 0.0):
        super().__init__()
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        norm = nn.BatchNorm2d if spatial_dims == 2 else nn.BatchNorm3d
        layers: list[nn.Module] = [
            conv(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm(out_ch),
            nn.ReLU(inplace=True),
            conv(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            norm(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Classic U-Net encoder–decoder with skip connections."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 64, 128, 256, 512),
        spatial_dims: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 or 3")

        self.spatial_dims = spatial_dims
        pool = nn.MaxPool2d if spatial_dims == 2 else nn.MaxPool3d
        up = nn.ConvTranspose2d if spatial_dims == 2 else nn.ConvTranspose3d
        head = nn.Conv2d if spatial_dims == 2 else nn.Conv3d

        feats = list(features)
        self.encoder = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for f in feats:
            self.encoder.append(ConvBlock(prev, f, spatial_dims, dropout))
            self.pools.append(pool(2))
            prev = f

        self.bottleneck = ConvBlock(feats[-1], feats[-1] * 2, spatial_dims, dropout)

        self.ups = nn.ModuleList()
        self.decoder = nn.ModuleList()
        prev = feats[-1] * 2
        for f in reversed(feats):
            self.ups.append(up(prev, f, kernel_size=2, stride=2))
            self.decoder.append(ConvBlock(f * 2, f, spatial_dims, dropout))
            prev = f

        self.head = head(feats[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for enc, pool in zip(self.encoder, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.ups, self.decoder, reversed(skips)):
            x = up(x)
            # handle odd shapes from pooling / transpose
            if x.shape[2:] != skip.shape[2:]:
                x = _center_crop_or_pad(x, skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.head(x)


def _center_crop_or_pad(x: torch.Tensor, target_size: torch.Size) -> torch.Tensor:
    """Match spatial size of skip connection (crop or zero-pad)."""
    for dim, (cur, tgt) in enumerate(zip(x.shape[2:], target_size), start=2):
        if cur == tgt:
            continue
        if cur > tgt:
            start = (cur - tgt) // 2
            slices = [slice(None)] * x.ndim
            slices[dim] = slice(start, start + tgt)
            x = x[tuple(slices)]
        else:
            pad_total = tgt - cur
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            # F.pad pads last dims first
            pads = [0, 0] * (x.ndim - 2)
            idx = (x.ndim - 1 - dim) * 2
            pads[idx] = pad_before
            pads[idx + 1] = pad_after
            x = torch.nn.functional.pad(x, pads)
    return x
