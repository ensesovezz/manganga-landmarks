"""U-Net en PyTorch (entrada 1xHxW, salida 1xHxW con logits).

Reimplementación estándar: la arquitectura original del upstream no fue
publicada (solo los pesos ``.h5``, que tampoco están), así que se entrena
desde cero con los pares ``train/`` + ``masks/`` del repo.

El target son discos de radio 4 px en los 19 landmarks (ver ``cria_mascara.py``),
por eso la salida es un mapa de un solo canal y los centros se recuperan con
``manganga.landmarks.detect_dots``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """U-Net con skip connections y upsampling aprendido (ConvTranspose2d)."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth
        self.base = base

        channels = [base * (2**i) for i in range(depth)]
        self.downs = nn.ModuleList()
        prev = in_ch
        for ch in channels:
            self.downs.append(DoubleConv(prev, ch))
            prev = ch

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(channels[-1], channels[-1] * 2)

        self.ups = nn.ModuleList()
        for ch in reversed(channels):
            self.ups.append(nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(ch * 2, ch))

        self.head = nn.Conv2d(channels[0], out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rellena a múltiplo de 2**depth y recorta al final.
        factor = 2**self.depth
        _, _, h, w = x.shape
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for step in range(0, len(self.ups), 2):
            up, conv = self.ups[step], self.ups[step + 1]
            skip = skips[-(step // 2 + 1)]
            x = up(x)
            # Por si el tamaño no coincide exactamente (padding impar).
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = conv(x)

        x = self.head(x)
        if pad_h or pad_w:
            x = x[..., :h, :w]
        return x


def build_model(base: int = 32, depth: int = 4, in_ch: int = 1, out_ch: int = 1) -> UNet:
    return UNet(in_ch=in_ch, out_ch=out_ch, base=base, depth=depth)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
