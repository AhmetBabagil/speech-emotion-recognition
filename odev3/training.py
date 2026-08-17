'''Sınıf ağırlıklı PyTorch eğitimi ve doğrulamaya dayalı erken durdurma.

Bu modül tek bir "deneme"nin (trial) eğitim döngüsünü içerir. Öne çıkan
tasarım kararları:

- **Sınıf ağırlıklı kayıp**: Veri kümelerinde duygu sınıfları dengesizdir
  (örn. "neutral" çok, "fear" az). CrossEntropyLoss'a ters-frekans ağırlığı
  vererek azınlık sınıfların hatalarını daha pahalı yaparız; aksi halde model
  çoğunluk sınıfını ezberleyerek "iyi" görünebilirdi.
- **Erken durdurma (early stopping)**: Her epoch sonunda doğrulama macro-F1
  ölçülür; ``patience`` epoch boyunca iyileşme olmazsa eğitim kesilir ve EN
  İYİ epoch'un ağırlıkları geri yüklenir. Bu, aşırı öğrenmeye karşı en ucuz
  ve en etkili savunmalardan biridir.
- **Determinizm**: Sabit tohum + tohumlu DataLoader üreteci sayesinde aynı
  konfigürasyon aynı sonucu verir; deneyler tekrarlanabilir olur.
- **AMP (otomatik karışık hassasiyet)**: CUDA varsa float16 ile hızlanma
  sağlanır; GradScaler sayısal taşmaları güvenle yönetir.
'''

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
    '''n / (K * n_k) döndürür — her eğitim sınıfı sayısının tam tersi.

    Formülün mantığı: n toplam örnek, K sınıf sayısı, n_k ise k sınıfının
    örnek sayısı olsun. Ağırlık w_k = n / (K * n_k) seçilirse:
    - Dengeli veri kümesinde (her n_k = n/K) tüm ağırlıklar 1.0 olur, yani
      ağırlıklandırma etkisiz kalır — güzel bir "sağlamlık" özelliği.
    - Nadir sınıfın ağırlığı 1'den büyük, sık sınıfınki 1'den küçük olur;
      kayıp fonksiyonu azınlık hatalarını daha çok cezalandırır.
    '''

    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes)
    # Sıfır sayımlı sınıf varsa bölme tanımsız olur; ayrıca eğitim verisinde
    # hiç görülmeyen bir sınıfı öğrenmek zaten imkansızdır — erken hata ver.
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
    # Ortak DataLoader kurulumunu tek yerde toplayan yardımcı.
    # Tohumlu Generator: shuffle sırası her çalıştırmada aynı olsun
    # (tekrarlanabilirlik). pin_memory yalnızca CUDA'da anlamlı — CPU->GPU
    # kopyalarını hızlandırır. persistent_workers, worker süreçlerinin her
    # epoch'ta yeniden başlatılma maliyetini ortadan kaldırır.
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
    # Modeli değerlendirme modunda (dropout kapalı, BatchNorm çalışma
    # istatistikleri sabit) tüm loader üzerinde çalıştırır; ortalama kaybı,
    # metrikleri ve sınıf olasılıklarını döndürür. Olasılıklar kalibrasyon
    # analizinde ayrıca kullanılacağı için burada toplanır.
    model.eval()
    total_loss = 0.0
    total_examples = 0
    true_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []

    # no_grad: gradyan hesabı kapatılır — bellek ve zaman tasarrufu; zaten
    # değerlendirme sırasında geriye yayılım yapılmaz.
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            loss = criterion(logits, labels)
            # softmax logitleri olasılığa çevirir; argmax en olası sınıfı verir.
            probabilities = torch.softmax(logits, dim=1)
            batch_size = labels.shape[0]
            # Kayıp batch ortalaması döner; örnek sayısıyla çarpıp toplayarak
            # sonda GERÇEK ortalamayı elde ederiz (son batch küçük olabilir).
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
    # Bir eğitim denemesinin tüm çıktısını taşıyan basit veri sınıfı:
    # - model: en iyi epoch ağırlıkları geri yüklenmiş halde
    # - history: her epoch'un kayıp/metrik kayıtları (öğrenme eğrileri için)
    # - best_epoch / epochs_trained / stopped_early: erken durdurma özeti
    # - validation_loss / validation_metrics: en iyi epoch'un doğrulama skoru
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
    '''Tek bir denemeyi eğitir ve doğrulama macro-F1'i en iyi olan epoch'u geri yükler.

    Seçim ölçütü olarak accuracy değil macro-F1 kullanılır: macro-F1 her
    sınıfa eşit ağırlık verir, dolayısıyla dengesiz veri kümelerinde azınlık
    sınıfları da hesaba katar. ``min_delta`` küçük bir eşik olup "gürültü
    kadar iyileşmeleri" gerçek ilerleme saymamayı sağlar.
    '''

    config.validate()
    if max_epochs <= 0:
        raise ValueError(f'max_epochs must be positive, got {max_epochs}.')
    # Tohum sabitleme: ağırlık başlangıcı, shuffle sırası, dropout maskeleri
    # hepsi bu tohumdan türediği için deneme birebir tekrarlanabilir.
    set_seed(seed)

    # ------------------------------------------------------------------
    # Kurulum: model, sınıf ağırlıklı kayıp ve AdamW optimizasyonu.
    # AdamW = Adam + "decoupled" weight decay; L2 cezasını gradyandan ayrı
    # uyguladığı için Adam+L2'den daha doğru bir düzenlileştirme sağlar.
    # ------------------------------------------------------------------
    model = MLP(input_dim, num_classes, config).to(device)
    class_weights = inverse_frequency_weights(train_labels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # BatchNorm her eğitim batch'inde en az iki örnek ister (tek örnekten
    # varyans hesaplanamaz). Veri boyutu batch_size'a bölündüğünde son batch
    # tam 1 örneklik kalıyorsa o son batch'i atarız; başka durumda atmayız
    # (veri kaybetmemek için koşullar bilinçli olarak bu kadar dar).
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
    # Doğrulama loader'ı: shuffle yok (sıra önemsiz, determinizm önemli);
    # batch en az 128 — değerlendirme gradyansız olduğundan büyük batch
    # bellek sorunu çıkarmadan hız kazandırır.
    validation_loader = _loader(
        validation_features,
        validation_labels,
        batch_size=max(config.batch_size, 128),
        shuffle=False,
        seed=seed,
        device=device,
        num_workers=num_workers,
    )

    # AMP yalnızca CUDA'da etkin: CPU'da float16 kazanç sağlamaz. GradScaler,
    # float16'da küçük gradyanların sıfıra yuvarlanmasını (underflow) önlemek
    # için kaybı ölçekler; enabled=False iken tamamen şeffaf çalışır.
    use_amp = bool(amp and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = -float('inf')
    best_loss = float('inf')
    best_metrics: dict[str, Any] | None = None
    epochs_without_improvement = 0

    # ------------------------------------------------------------------
    # Ana eğitim döngüsü: her epoch = tüm eğitim verisinden bir geçiş +
    # doğrulama ölçümü + "en iyi"yi güncelleme + erken durdurma kontrolü.
    # ------------------------------------------------------------------
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        examples_seen = 0
        train_true_parts: list[np.ndarray] = []
        train_predicted_parts: list[np.ndarray] = []
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # set_to_none=True: gradyanları sıfır tensörü yerine None yapar;
            # bir tık daha hızlı ve bellek dostudur.
            optimizer.zero_grad(set_to_none=True)
            # autocast: ileri geçişte uygun işlemleri float16'da çalıştırır.
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(features)
                loss = criterion(logits, labels)
            # AMP akışı: kaybı ölçekle -> geriye yayıl -> ölçeği geri al ->
            # gradyanları kırp -> adım at -> ölçeği güncelle.
            scaler.scale(loss).backward()
            # unscale: kırpmadan ÖNCE gradyanları gerçek ölçeğine döndürmek
            # şart; yoksa kırpma eşiği yanlış ölçekte uygulanırdı.
            scaler.unscale_(optimizer)
            # Gradyan kırpma (max_norm=5): nadir de olsa patlayan gradyanların
            # eğitimi raydan çıkarmasını önleyen bir emniyet kemeri.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            batch_size = labels.shape[0]
            running_loss += float(loss.item()) * batch_size
            examples_seen += batch_size
            # Eğitim tahminlerini de topluyoruz: eğitim/doğrulama eğrilerini
            # yan yana çizmek aşırı öğrenmeyi görünür kılar.
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
        # Deterministik beraberlik bozma: macro-F1 sayısal olarak eşitken daha
        # düşük doğrulama kaybı kazanır; ama önemsiz değişimler için patience
        # sıfırlanmaz. (Yani checkpoint güncellenir, sayaç işlemez — böylece
        # "yerinde sayan" model eğitimi sonsuza dek uzatamaz.)
        tied_but_lower_loss = abs(score - best_score) <= min_delta and val_loss < best_loss
        if improved or tied_but_lower_loss:
            best_score = score
            best_loss = val_loss
            best_epoch = epoch
            best_metrics = val_metrics
            # Ağırlıkların CPU kopyasını saklıyoruz: GPU belleğini işgal
            # etmez ve sonraki epoch'ların güncellemelerinden etkilenmez
            # (clone olmasa state_dict aynı tensörlere referans verirdi).
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Öğrenme eğrileri ve rapor için her epoch'un tam kaydı.
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
        # Erken durdurma: patience epoch boyunca gerçek iyileşme yoksa dur.
        if epochs_without_improvement >= config.patience:
            break

    if best_state is None or best_metrics is None:
        raise RuntimeError('Training ended without a valid validation checkpoint.')
    # En kritik adım: son epoch'un değil, EN İYİ epoch'un ağırlıklarını
    # geri yükle. Döndürülen model her zaman doğrulamada zirve yapan haldir.
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
    '''Sabit bir modeli, karıştırma ve gradyan güncellemesi olmadan değerlendirir.

    Test aşamasında kullanılır: seed=0 ve shuffle=False sabittir çünkü
    değerlendirmede rastgelelik istemeyiz. Kayıp, eğitimle karşılaştırılabilir
    olsun diye yine sınıf ağırlıklı hesaplanır.
    '''

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
