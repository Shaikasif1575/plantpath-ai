"""Compact U-Net used for severity segmentation (lesion and leaf masks).

Single source of truth for the architecture: the training script
(``scripts/train_segmentation.py``) imports this class so the checkpoint that
is trained is byte-for-byte the same architecture the backend loads. Binary
segmentation: logits shape (N, 1, H, W); apply sigmoid + threshold at 0.5.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3x3 -> ReLU -> Conv3x3 -> ReLU) with batch norm."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.block(x)


class UNet(nn.Module):
    """Classic encoder/decoder U-Net with transposed-convolution upsampling."""

    def __init__(self, n_channels: int = 3, base: int = 16, depth: int = 4, n_classes: int = 1) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        chans = [base * (2**i) for i in range(depth + 1)]  # e.g. 16,32,64,128,256

        # Encoder
        self.enc = nn.ModuleList()
        prev = n_channels
        for c in chans[:-1]:
            self.enc.append(DoubleConv(prev, c))
            prev = c
        self.bottleneck = DoubleConv(chans[-2], chans[-1])

        # Decoder
        self.upconv = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.upconv.append(nn.ConvTranspose2d(chans[i], chans[i - 1], kernel_size=2, stride=2))
            self.dec.append(DoubleConv(chans[i], chans[i - 1]))

        self.head = nn.Conv2d(chans[0], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        skips = []
        for layer in self.enc:
            x = layer(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconv, self.dec, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:  # odd input sizes
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(x)


def build_unet(arch: dict) -> UNet:
    """Instantiate a UNet from a stored arch dict (n_channels/base/depth/n_classes)."""
    return UNet(
        n_channels=int(arch.get("n_channels", 3)),
        base=int(arch.get("base", 16)),
        depth=int(arch.get("depth", 4)),
        n_classes=int(arch.get("n_classes", 1)),
    )
