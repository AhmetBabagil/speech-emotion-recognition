'''Weighted PyTorch training with validation-based early stopping.'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from odev3.dataset import ArrayDataset
from odev3.model import MLP, MLPConfig
from ser.evaluate import compute_metrics
from ser.utils import set_seed


def inverse_frequency_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    '''Return n / (K * n_k), exactly inverse to each training class count.'''

    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes)
    if len(labels) == 0 or len(counts) != num_classes or np.any(counts == 0):
        raise ValueError(
            f'Every class must occur in training data; counts={counts.tolist()}.'
        )
    weights = len(labels) / (num_classes * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)


def _loader(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
    num_workers: int,
    drop_last: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ArrayDataset(features, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, Any], np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    true_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            true_parts.append(labels.cpu().numpy())
            predicted_parts.append(probabilities.argmax(dim=1).cpu().numpy())
            probability_parts.append(probabilities.cpu().numpy())

    if total_examples == 0:
        raise ValueError('Evaluation loader produced no examples.')
    y_true = np.concatenate(true_parts)
    y_pred = np.concatenate(predicted_parts)
    probabilities = np.concatenate(probability_parts)
    return total_loss / total_examples, compute_metrics(y_true, y_pred), probabilities


@dataclass
class TrainingOutcome:
    model: MLP
    history: list[dict[str, Any]]
    best_epoch: int
    epochs_trained: int
    stopped_early: bool
    validation_loss: float
    validation_metrics: dict[str, Any]


def train_with_early_stopping(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    config: MLPConfig,
    *,
    input_dim: int,
    num_classes: int,
    device: torch.device,
    max_epochs: int = 60,
    seed: int = 42,
    num_workers: int = 0,
    amp: bool = True,
    min_delta: float = 1e-4,
) -> TrainingOutcome:
    '''Fit one trial and restore the epoch with best validation macro-F1.'''

    config.validate()
    if max_epochs <= 0:
        raise ValueError(f'max_epochs must be positive, got {max_epochs}.')
    set_seed(seed)

    model = MLP(input_dim, num_classes, config).to(device)
    class_weights = inverse_frequency_weights(train_labels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # BatchNorm requires at least two examples in each training batch.
    drop_last = (
        config.batch_norm
        and len(train_labels) > config.batch_size
        and len(train_labels) % config.batch_size == 1
    )
    train_loader = _loader(
        train_features,
        train_labels,
        batch_size=config.batch_size,
        shuffle=True,
        seed=seed,
        device=device,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    validation_loader = _loader(
        validation_features,
        validation_labels,
        batch_size=max(config.batch_size, 128),
        shuffle=False,
        seed=seed,
        device=device,
        num_workers=num_workers,
    )

    use_amp = bool(amp and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = -float('inf')
    best_loss = float('inf')
    best_metrics: dict[str, Any] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        examples_seen = 0
        train_true_parts: list[np.ndarray] = []
        train_predicted_parts: list[np.ndarray] = []
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(features)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            batch_size = labels.shape[0]
            running_loss += float(loss.item()) * batch_size
            examples_seen += batch_size
            train_true_parts.append(labels.detach().cpu().numpy())
            train_predicted_parts.append(logits.detach().argmax(dim=1).cpu().numpy())

        train_loss = running_loss / max(examples_seen, 1)
        train_metrics = compute_metrics(
            np.concatenate(train_true_parts),
            np.concatenate(train_predicted_parts),
        )
        val_loss, val_metrics, _ = _evaluate_loader(
            model, validation_loader, criterion, device
        )
        score = float(val_metrics['macro_f1'])
        improved = score > best_score + min_delta
        # Deterministic tie break: lower validation loss wins when macro-F1 is
        # numerically tied, without resetting patience for negligible changes.
        tied_but_lower_loss = abs(score - best_score) <= min_delta and val_loss < best_loss
        if improved or tied_but_lower_loss:
            best_score = score
            best_loss = val_loss
            best_epoch = epoch
            best_metrics = val_metrics
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append(
            {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_metrics['accuracy'],
                'train_balanced_accuracy': train_metrics['balanced_accuracy'],
                'train_macro_f1': train_metrics['macro_f1'],
                'train_weighted_f1': train_metrics['weighted_f1'],
                'val_loss': val_loss,
                'val_accuracy': val_metrics['accuracy'],
                'val_balanced_accuracy': val_metrics['balanced_accuracy'],
                'val_macro_f1': val_metrics['macro_f1'],
                'val_weighted_f1': val_metrics['weighted_f1'],
                'improved': bool(improved or tied_but_lower_loss),
            }
        )
        if epochs_without_improvement >= config.patience:
            break

    if best_state is None or best_metrics is None:
        raise RuntimeError('Training ended without a valid validation checkpoint.')
    model.load_state_dict(best_state)
    return TrainingOutcome(
        model=model,
        history=history,
        best_epoch=best_epoch,
        epochs_trained=len(history),
        stopped_early=len(history) < max_epochs,
        validation_loss=best_loss,
        validation_metrics=best_metrics,
    )


def evaluate_arrays(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    class_weights: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 0,
) -> tuple[float, dict[str, Any], np.ndarray]:
    '''Evaluate a fixed model without shuffling or gradient updates.'''

    loader = _loader(
        features,
        labels,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        device=device,
        num_workers=num_workers,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    return _evaluate_loader(model, loader, criterion, device)
