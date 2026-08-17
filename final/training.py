'''Modelden bağımsız, ağırlıklı loss + geçerleme temelli early stopping'li eğitim.

Aynı döngü hem Yöntem 1'in CNN'ini hem Yöntem 2'nin LSTM/GRU'sunu eğitir:
tek varsayımı, modelin bir öznitelik yığınını sınıf logit'lerine çevirmesidir.
Model seçimi geçerleme macro-F1 üzerinden yapılır ve eğitim bittiğinde en iyi
epoch'un ağırlıkları geri yüklenir.

Yönergenin iki açık şartı bu dosyada karşılanır:
1. "Sınıf veri sayıları farklıysa hata fonksiyonunu veri sayısıyla ters
   orantılı ağırlıklı tanımlayın"  -> inverse_frequency_weights
2. "PyTorch ile erken durdurma yapın" -> train_with_early_stopping
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from final.dataset import ArrayDataset
from final.models import OptimSettings
from ser.evaluate import compute_metrics
from ser.utils import set_seed


def inverse_frequency_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    '''Sınıf ağırlıkları: n / (K * n_k) — eğitimdeki sınıf sayısıyla ters orantılı.

    Örnek: bir sınıfın az kaydı varsa ağırlığı büyük olur; böylece loss,
    çoğunluk sınıfını ezberleyip azınlığı görmezden gelmeyi cezalandırır.
    '''

    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes)
    if len(labels) == 0 or len(counts) != num_classes or np.any(counts == 0):
        raise ValueError(
            f'Her sınıf eğitim verisinde bulunmalı; sayımlar={counts.tolist()}.'
        )
    weights = len(labels) / (num_classes * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    '''Tekrarlanabilir (tohumlu) bir DataLoader kurar.'''

    # Karıştırma sırası bu üretece bağlı; sabit tohum = aynı sıra = aynı sonuç.
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ArrayDataset(features, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',   # GPU'ya kopyalamayı hızlandırır
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, Any], np.ndarray]:
    '''Modeli gradyansız değerlendirir: (ortalama loss, metrikler, olasılıklar).'''

    model.eval()   # BatchNorm/Dropout'u değerlendirme kipine al
    total_loss = 0.0
    total_examples = 0
    true_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []

    with torch.no_grad():   # değerlendirmede geri yayılım yok -> bellek/hız kazancı
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            batch = labels.shape[0]
            total_loss += float(loss.item()) * batch
            total_examples += batch
            true_parts.append(labels.cpu().numpy())
            predicted_parts.append(probabilities.argmax(dim=1).cpu().numpy())
            probability_parts.append(probabilities.cpu().numpy())

    if total_examples == 0:
        raise ValueError('Değerlendirme yükleyicisi hiç örnek üretmedi.')
    y_true = np.concatenate(true_parts)
    y_pred = np.concatenate(predicted_parts)
    return (
        total_loss / total_examples,
        compute_metrics(y_true, y_pred),
        np.concatenate(probability_parts),
    )


@dataclass
class TrainingOutcome:
    '''Bir eğitim koşusunun tüm sonucu: model + geçmiş + en iyi epoch bilgileri.'''

    model: nn.Module
    history: list[dict[str, Any]]           # epoch başına metrikler (öğrenme eğrisi)
    best_epoch: int                         # geçerlemede en iyi olan epoch
    epochs_trained: int                     # fiilen koşulan epoch sayısı
    stopped_early: bool                     # erken mi durdu?
    validation_loss: float                  # en iyi epoch'taki geçerleme loss'u
    validation_metrics: dict[str, Any]      # en iyi epoch'taki geçerleme metrikleri


def train_with_early_stopping(
    model: nn.Module,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    optim: OptimSettings,
    *,
    num_classes: int,
    device: torch.device,
    max_epochs: int = 60,
    seed: int = 42,
    num_workers: int = 0,
    amp: bool = True,
    min_delta: float = 1e-4,
    train_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> TrainingOutcome:
    '''Bir adayı eğitir ve geçerleme macro-F1'i en iyi olan epoch'u geri yükler.

    Early stopping mantığı: her epoch sonunda geçerleme macro-F1 ölçülür;
    `patience` epoch boyunca anlamlı iyileşme (min_delta'dan büyük) olmazsa
    eğitim durur. Böylece model, geçerlemede bozulmaya başladığı (aşırı
    öğrenme) noktadan sonrasını "unutur".

    ``train_transform`` (ör. SpecAugment maskeleme) SADECE eğitim
    yığınlarına uygulanır; geçerleme dokunulmadan kalır ki model seçimi
    yansız (tarafsız) olsun.
    '''

    optim.validate()
    if max_epochs <= 0:
        raise ValueError(f'max_epochs pozitif olmalı, gelen {max_epochs}.')
    set_seed(seed)   # python/numpy/torch tohumları -> tekrarlanabilirlik

    model = model.to(device)
    # Yönerge şartı: sınıf sayısıyla ters orantılı ağırlıklı cross-entropy.
    class_weights = inverse_frequency_weights(train_labels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim.learning_rate,
        weight_decay=optim.weight_decay,
    )

    # BatchNorm istatistik hesaplayabilmek için her eğitim yığınında en az
    # 2 örnek ister; son yığın tek örnekli kalacaksa onu düşür.
    drop_last = (
        len(train_labels) > optim.batch_size
        and len(train_labels) % optim.batch_size == 1
    )
    train_loader = make_loader(
        train_features,
        train_labels,
        batch_size=optim.batch_size,
        shuffle=True,            # eğitimde sıra her epoch karışır
        seed=seed,
        device=device,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    validation_loader = make_loader(
        validation_features,
        validation_labels,
        batch_size=max(optim.batch_size, 128),
        shuffle=False,           # değerlendirmede karıştırmaya gerek yok
        seed=seed,
        device=device,
        num_workers=num_workers,
    )

    # AMP (karışık hassasiyet): GPU'da float16 ile daha hızlı eğitim;
    # GradScaler sayısal taşmaları dengeler. CPU'da otomatik kapalı.
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
        # ---- 1) Eğitim geçişi -------------------------------------------------
        model.train()
        running_loss = 0.0
        examples_seen = 0
        train_true: list[np.ndarray] = []
        train_pred: list[np.ndarray] = []
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if train_transform is not None:
                # Veri artırma (varsa) yalnızca burada, eğitim yığınında.
                features = train_transform(features)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(features)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()      # geri yayılım
            scaler.unscale_(optimizer)
            # Gradyan kırpma: tek bir kötü yığının eğitimi raydan çıkarmasını önler.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)             # ağırlık güncelle
            scaler.update()
            batch = labels.shape[0]
            running_loss += float(loss.item()) * batch
            examples_seen += batch
            train_true.append(labels.detach().cpu().numpy())
            train_pred.append(logits.detach().argmax(dim=1).cpu().numpy())

        # ---- 2) Geçerleme geçişi ---------------------------------------------
        train_loss = running_loss / max(examples_seen, 1)
        train_metrics = compute_metrics(np.concatenate(train_true), np.concatenate(train_pred))
        val_loss, val_metrics, _ = evaluate_loader(model, validation_loader, criterion, device)

        # ---- 3) En iyiyi güncelle + erken durdurma sayacı --------------------
        score = float(val_metrics['macro_f1'])
        improved = score > best_score + min_delta
        # Deterministik beraberlik bozma: macro-F1 sayısal olarak eşitse
        # geçerleme loss'u düşük olan kazanır; ama minicik değişimler
        # sabır sayacını sıfırlamaz (yoksa eğitim gereksiz uzar).
        tied_but_lower_loss = abs(score - best_score) <= min_delta and val_loss < best_loss
        if improved or tied_but_lower_loss:
            best_score = score
            best_loss = val_loss
            best_epoch = epoch
            best_metrics = val_metrics
            # En iyi anın ağırlıklarının kopyasını CPU'da sakla.
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Öğrenme eğrisi için epoch kaydı.
        history.append(
            {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_metrics['accuracy'],
                'train_macro_f1': train_metrics['macro_f1'],
                'val_loss': val_loss,
                'val_accuracy': val_metrics['accuracy'],
                'val_balanced_accuracy': val_metrics['balanced_accuracy'],
                'val_macro_f1': val_metrics['macro_f1'],
                'val_weighted_f1': val_metrics['weighted_f1'],
                'improved': bool(improved or tied_but_lower_loss),
            }
        )
        # ERKEN DURDURMA: sabır dolduysa döngüden çık.
        if epochs_without_improvement >= optim.patience:
            break

    if best_state is None or best_metrics is None:
        raise RuntimeError('Eğitim, geçerli bir geçerleme kontrol noktası üretmeden bitti.')
    # En iyi epoch'un ağırlıklarını geri yükle: dönen model, eğitimin
    # sonundaki değil geçerlemede EN İYİ olan modeldir.
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
    '''Sabit bir modeli karıştırmadan ve gradyansız değerlendirir (test için).'''

    loader = make_loader(
        features,
        labels,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        device=device,
        num_workers=num_workers,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    return evaluate_loader(model, loader, criterion, device)
