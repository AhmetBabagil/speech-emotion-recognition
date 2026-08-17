'''From-scratch PyTorch models for both final methods.

No pretrained weights or ready-made architectures are loaded anywhere; both
classifiers are built purely from torch.nn primitives.
'''

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class OptimSettings:
    '''Optimization hyperparameters shared by both methods.'''

    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 8

    def validate(self) -> None:
        if self.batch_size <= 0 or self.patience <= 0:
            raise ValueError(f'Invalid optimization settings: {self}.')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError(f'Invalid learning rate: {self.learning_rate}.')
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError(f'Invalid weight decay: {self.weight_decay}.')


# --- Method 1: CNN over log-mel images ------------------------------------------


@dataclass(frozen=True)
class CNNConfig:
    '''One CNN hyperparameter candidate.'''

    channels: tuple[int, ...] = (32, 64, 128)
    dropout: float = 0.3
    optim: OptimSettings = field(default_factory=OptimSettings)

    def validate(self) -> None:
        self.optim.validate()
        if not self.channels or any(c <= 0 for c in self.channels):
            raise ValueError(f'Invalid channel widths: {self.channels}.')
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f'Invalid dropout: {self.dropout}.')

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result['channels'] = list(self.channels)
        return result


class ConvBlock(nn.Module):
    '''Conv-BN-ReLU x2 followed by 2x2 max pooling.'''

    def __init__(self, in_ch: int, out_ch: int) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MelCNN(nn.Module):
    '''Stacked conv blocks + global average pooling + linear classifier.

    Input [B, n_mels, T] mel images are unsqueezed to a single channel.
    '''

    def __init__(self, num_classes: int, config: CNNConfig) -> None:
        super().__init__()
        config.validate()
        blocks: list[nn.Module] = []
        in_ch = 1
        for out_ch in config.channels:
            blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.channels[-1], num_classes),
        )
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f'Expected [B, n_mels, T] input, got {tuple(x.shape)}.')
        x = self.features(x.unsqueeze(1))
        return self.classifier(self.pool(x).flatten(1))


# --- Method 2: LSTM/GRU over interval feature series ----------------------------


@dataclass(frozen=True)
class RNNConfig:
    '''One recurrent-model hyperparameter candidate.'''

    rnn_type: str = 'gru'
    hidden_size: int = 192
    num_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.3
    pooling: str = 'mean'
    optim: OptimSettings = field(default_factory=lambda: OptimSettings(batch_size=64, learning_rate=1e-3))

    def validate(self) -> None:
        self.optim.validate()
        if self.rnn_type not in {'lstm', 'gru'}:
            raise ValueError(f'rnn_type must be lstm or gru, got {self.rnn_type!r}.')
        if self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError(f'Invalid RNN size: {self}.')
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f'Invalid dropout: {self.dropout}.')
        if self.pooling not in {'last', 'mean', 'max', 'attn'}:
            raise ValueError(
                f'pooling must be last/mean/max/attn, got {self.pooling!r}.'
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SeqRNN(nn.Module):
    '''LSTM/GRU classifier over a fixed-length feature series [B, T, D].'''

    def __init__(self, input_dim: int, num_classes: int, config: RNNConfig) -> None:
        super().__init__()
        config.validate()
        if input_dim <= 0 or num_classes <= 1:
            raise ValueError(f'Invalid dimensions: {input_dim}, {num_classes}.')
        rnn_cls = nn.LSTM if config.rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        directions = 2 if config.bidirectional else 1
        rnn_out = config.hidden_size * directions
        # Learned attention over time steps (Mirsamadi et al., 2017 style);
        # built from a single Linear layer, so still fully from scratch.
        self.attention = nn.Linear(rnn_out, 1) if config.pooling == 'attn' else None
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(rnn_out, num_classes),
        )
        self.input_dim = input_dim
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[2] != self.input_dim:
            raise ValueError(
                f'Expected [B, T, {self.input_dim}] input, got {tuple(x.shape)}.'
            )
        outputs, _ = self.rnn(x)
        if self.config.pooling == 'last':
            pooled = outputs[:, -1]
        elif self.config.pooling == 'mean':
            pooled = outputs.mean(dim=1)
        elif self.config.pooling == 'attn':
            weights = torch.softmax(self.attention(outputs), dim=1)
            pooled = (weights * outputs).sum(dim=1)
        else:
            pooled = outputs.max(dim=1).values
        return self.head(pooled)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
