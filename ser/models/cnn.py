"""A compact 2-D CNN over log-mel spectrograms.

Input : [B, 1, n_mels, T]
Output: [B, num_classes] logits

Four conv blocks (Conv-BN-ReLU x2 + MaxPool) followed by global average pooling
and a linear classifier. Global pooling makes the head invariant to the time
dimension, so variable clip lengths still work.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class MelCNN(nn.Module):
    def __init__(self, num_classes: int, in_ch: int = 1,
                 channels: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        blocks = []
        c_in = in_ch
        for c_out in channels:
            blocks.append(ConvBlock(c_in, c_out))
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
