'''Configurable MLP implemented from scratch with PyTorch layers.'''

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class MLPConfig:
    '''A single hyperparameter candidate.'''

    batch_size: int = 64
    learning_rate: float = 3e-4
    patience: int = 8
    hidden_dims: tuple[int, ...] = (512, 256)
    activation: str = 'relu'
    batch_norm: bool = True
    dropout: float = 0.3
    weight_decay: float = 1e-4

    def validate(self) -> None:
        if self.batch_size <= 0 or self.patience <= 0:
            raise ValueError(f'Invalid optimization settings: {self}.')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError(f'Invalid learning rate: {self.learning_rate}.')
        if not self.hidden_dims or any(size <= 0 for size in self.hidden_dims):
            raise ValueError(f'Invalid hidden dimensions: {self.hidden_dims}.')
        if self.activation not in {'relu', 'gelu', 'tanh', 'leaky_relu'}:
            raise ValueError(f'Unsupported activation: {self.activation}.')
        if (
            not math.isfinite(self.dropout)
            or not 0.0 <= self.dropout < 1.0
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0.0
        ):
            raise ValueError(f'Invalid regularization settings: {self}.')

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result['hidden_dims'] = list(self.hidden_dims)
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> 'MLPConfig':
        values = dict(values)
        values['hidden_dims'] = tuple(int(v) for v in values['hidden_dims'])
        config = cls(**values)
        config.validate()
        return config


def _activation(name: str) -> nn.Module:
    if name == 'relu':
        return nn.ReLU()
    if name == 'gelu':
        return nn.GELU()
    if name == 'tanh':
        return nn.Tanh()
    if name == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(f'Unsupported activation: {name}.')


class MLP(nn.Module):
    '''Fully connected classifier; no pretrained or ready-made model is used.'''

    def __init__(self, input_dim: int, num_classes: int, config: MLPConfig) -> None:
        super().__init__()
        config.validate()
        if input_dim <= 0 or num_classes <= 1:
            raise ValueError(f'Invalid MLP dimensions: {input_dim}, {num_classes}.')

        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in config.hidden_dims:
            layers.append(nn.Linear(previous, hidden))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(_activation(config.activation))
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            previous = hidden
        layers.append(nn.Linear(previous, num_classes))
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.config = config
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.config.activation in {'relu', 'leaky_relu'}:
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f'Expected input shape (batch, {self.input_dim}), got '
                f'{tuple(features.shape)}.'
            )
        return self.network(features)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
