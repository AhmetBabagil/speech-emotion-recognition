'''Forward-pass shape checks for the final models and augmentations.'''

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from final.augment import FeatureNoise, SpecMask  # noqa: E402
from final.models import (  # noqa: E402
    CNNConfig,
    MelCNN,
    OptimSettings,
    RNNConfig,
    SeqRNN,
)


def test_cnn_forward_shape() -> None:
    model = MelCNN(6, CNNConfig(channels=(8, 16)))
    logits = model(torch.randn(4, 64, 128))
    assert logits.shape == (4, 6)


@pytest.mark.parametrize('pooling', ['last', 'mean', 'max', 'attn'])
def test_rnn_forward_shape_all_poolings(pooling: str) -> None:
    config = RNNConfig(hidden_size=32, num_layers=1, pooling=pooling,
                       optim=OptimSettings())
    model = SeqRNN(44, 6, config)
    logits = model(torch.randn(4, 24, 44))
    assert logits.shape == (4, 6)


def test_rnn_rejects_bad_pooling() -> None:
    with pytest.raises(ValueError):
        RNNConfig(pooling='sum').validate()


def test_spec_mask_zeroes_but_keeps_shape() -> None:
    batch = torch.ones(3, 64, 128)
    masked = SpecMask()(batch)
    assert masked.shape == batch.shape
    assert (batch == 1.0).all()  # input is not modified in place
    assert (masked == 0.0).sum() > 0


def test_feature_noise_shape() -> None:
    torch.manual_seed(0)
    batch = torch.zeros(3, 24, 44)
    noisy = FeatureNoise(std=0.1)(batch)
    assert noisy.shape == batch.shape
    assert noisy.std() > 0.05
