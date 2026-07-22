'''Tests for the configurable from-scratch PyTorch MLP.'''

import pytest
import torch
from torch import nn

from odev3.model import MLP, MLPConfig, count_parameters


def test_mlp_forward_shape_and_parameter_count() -> None:
    config = MLPConfig(
        hidden_dims=(8, 4),
        batch_norm=False,
        dropout=0.0,
    )
    model = MLP(input_dim=16, num_classes=6, config=config)

    logits = model(torch.zeros(3, 16))

    assert logits.shape == (3, 6)
    expected_parameters = (16 * 8 + 8) + (8 * 4 + 4) + (4 * 6 + 6)
    assert count_parameters(model) == expected_parameters


def test_model_config_round_trip_preserves_hyperparameters() -> None:
    original = MLPConfig(
        batch_size=32,
        learning_rate=1e-3,
        patience=12,
        hidden_dims=(256, 128, 64),
        activation='gelu',
        batch_norm=False,
        dropout=0.5,
        weight_decay=0.0,
    )

    restored = MLPConfig.from_dict(original.to_dict())

    assert restored == original


@pytest.mark.parametrize('activation', ['relu', 'gelu', 'tanh', 'leaky_relu'])
def test_supported_activations_produce_finite_logits(activation: str) -> None:
    config = MLPConfig(
        hidden_dims=(12,),
        activation=activation,
        batch_norm=False,
        dropout=0.0,
    )
    model = MLP(input_dim=10, num_classes=6, config=config)

    logits = model(torch.randn(4, 10))

    assert torch.isfinite(logits).all()


def test_batch_normalization_and_dropout_switches_change_architecture() -> None:
    regularized = MLP(
        10,
        6,
        MLPConfig(hidden_dims=(12,), batch_norm=True, dropout=0.4),
    )
    plain = MLP(
        10,
        6,
        MLPConfig(hidden_dims=(12,), batch_norm=False, dropout=0.0),
    )

    assert any(isinstance(layer, nn.BatchNorm1d) for layer in regularized.modules())
    assert any(isinstance(layer, nn.Dropout) for layer in regularized.modules())
    assert not any(isinstance(layer, nn.BatchNorm1d) for layer in plain.modules())
    assert not any(isinstance(layer, nn.Dropout) for layer in plain.modules())


@pytest.mark.parametrize(
    'config',
    [
        MLPConfig(learning_rate=float('nan')),
        MLPConfig(learning_rate=float('inf')),
        MLPConfig(weight_decay=float('nan')),
    ],
)
def test_config_rejects_non_finite_optimization_values(config: MLPConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()


@pytest.mark.parametrize(
    'features',
    [torch.zeros(16), torch.zeros(2, 15)],
)
def test_forward_rejects_wrong_input_shape(features: torch.Tensor) -> None:
    model = MLP(
        input_dim=16,
        num_classes=6,
        config=MLPConfig(hidden_dims=(8,), batch_norm=False),
    )

    with pytest.raises(ValueError, match='Expected input shape'):
        model(features)
