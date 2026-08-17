'''İki final yöntemi için sıfırdan tanımlanmış PyTorch modelleri.

Hiçbir yerde önceden eğitilmiş ağırlık ya da hazır mimari yüklenmez; her iki
sınıflandırıcı da yalnızca torch.nn'in temel yapı taşlarıyla (Conv2d, Linear,
GRU/LSTM, BatchNorm, Dropout) kurulur. Ödevin "hazır model yasak" kuralının
sağlandığı yer burasıdır.
'''

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class OptimSettings:
    '''İki yöntemin ortak eğitim (optimizasyon) hiperparametreleri.'''

    batch_size: int = 32          # bir adımda işlenen örnek sayısı
    learning_rate: float = 3e-4   # AdamW öğrenme oranı
    weight_decay: float = 1e-4    # L2 düzenlileştirme katsayısı
    patience: int = 8             # early stopping sabrı (epoch)

    def validate(self) -> None:
        if self.batch_size <= 0 or self.patience <= 0:
            raise ValueError(f'Geçersiz optimizasyon ayarı: {self}.')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError(f'Geçersiz öğrenme oranı: {self.learning_rate}.')
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError(f'Geçersiz weight decay: {self.weight_decay}.')


# --- Yöntem 1: log-mel görüntüsü üzerinde CNN ------------------------------------


@dataclass(frozen=True)
class CNNConfig:
    '''Tek bir CNN hiperparametre adayı.'''

    channels: tuple[int, ...] = (32, 64, 128)  # blok başına kanal (filtre) sayısı
    dropout: float = 0.3                       # sınıflandırıcı öncesi dropout
    optim: OptimSettings = field(default_factory=OptimSettings)

    def validate(self) -> None:
        self.optim.validate()
        if not self.channels or any(c <= 0 for c in self.channels):
            raise ValueError(f'Geçersiz kanal genişlikleri: {self.channels}.')
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f'Geçersiz dropout: {self.dropout}.')

    def to_dict(self) -> dict[str, Any]:
        '''JSON'a yazılabilir sözlük (tuple -> list çevirisiyle).'''

        result = asdict(self)
        result['channels'] = list(self.channels)
        return result


class ConvBlock(nn.Module):
    '''CNN'in temel yapı taşı: (Conv-BN-ReLU) x2 + 2x2 max pooling.

    - Conv2d(3x3): yerel zaman-frekans desenlerini yakalar.
    - BatchNorm: eğitimi kararlı ve hızlı yapar.
    - ReLU: doğrusal olmayan aktivasyon.
    - MaxPool(2): haritayı yarıya indirir; ağ derinleştikçe daha geniş
      bağlama bakılmasını sağlar.
    '''

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
    '''Üst üste ConvBlock'lar + global average pooling + doğrusal sınıflandırıcı.

    Girdi [B, n_mels, T] mel görüntüleridir; forward içinde tek kanala
    (unsqueeze) açılır. Global average pooling, son özellik haritasının
    uzamsal ortalamasını aldığı için ağın kafası zaman uzunluğuna duyarsızdır.
    '''

    def __init__(self, num_classes: int, config: CNNConfig) -> None:
        super().__init__()
        config.validate()
        blocks: list[nn.Module] = []
        in_ch = 1  # log-mel tek kanallı bir "gri görüntü" gibi işlenir
        for out_ch in config.channels:
            blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)     # her kanalı tek sayıya indir
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),          # aşırı öğrenmeye karşı
            nn.Linear(config.channels[-1], num_classes),
        )
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f'Beklenen girdi [B, n_mels, T], gelen {tuple(x.shape)}.')
        x = self.features(x.unsqueeze(1))        # [B, 1, mels, T] -> özellik haritaları
        return self.classifier(self.pool(x).flatten(1))


# --- Yöntem 2: aralık serisi üzerinde LSTM/GRU -----------------------------------


@dataclass(frozen=True)
class RNNConfig:
    '''Tek bir tekrarlayan-model hiperparametre adayı.'''

    rnn_type: str = 'gru'          # 'gru' veya 'lstm'
    hidden_size: int = 192         # gizli durum boyutu
    num_layers: int = 2            # üst üste RNN katmanı sayısı
    bidirectional: bool = True     # seriyi iki yönden de oku
    dropout: float = 0.3           # katmanlar arası + kafa öncesi dropout
    pooling: str = 'mean'          # zaman özetleme: last/mean/max/attn
    optim: OptimSettings = field(default_factory=lambda: OptimSettings(batch_size=64, learning_rate=1e-3))

    def validate(self) -> None:
        self.optim.validate()
        if self.rnn_type not in {'lstm', 'gru'}:
            raise ValueError(f'rnn_type lstm veya gru olmalı, gelen {self.rnn_type!r}.')
        if self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError(f'Geçersiz RNN boyutu: {self}.')
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f'Geçersiz dropout: {self.dropout}.')
        if self.pooling not in {'last', 'mean', 'max', 'attn'}:
            raise ValueError(
                f'pooling last/mean/max/attn olmalı, gelen {self.pooling!r}.'
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SeqRNN(nn.Module):
    '''Sabit uzunluktaki öznitelik serisi [B, T, D] üzerinde LSTM/GRU sınıflandırıcı.

    Akış: seri -> RNN (her zaman adımına bir gizli durum) -> zaman havuzlama
    (tüm adımları tek vektöre özetle) -> doğrusal katman -> sınıf logit'leri.
    '''

    def __init__(self, input_dim: int, num_classes: int, config: RNNConfig) -> None:
        super().__init__()
        config.validate()
        if input_dim <= 0 or num_classes <= 1:
            raise ValueError(f'Geçersiz boyutlar: {input_dim}, {num_classes}.')
        rnn_cls = nn.LSTM if config.rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,                    # girdi düzeni [B, T, D]
            bidirectional=config.bidirectional,
            # PyTorch kuralı: tek katmanlı RNN'de katman-arası dropout olmaz.
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        directions = 2 if config.bidirectional else 1
        rnn_out = config.hidden_size * directions
        # Zaman adımları üzerinde öğrenilmiş dikkat (Mirsamadi vd., 2017 tarzı);
        # tek bir Linear katmandan kurulduğu için yine tamamen sıfırdandır.
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
                f'Beklenen girdi [B, T, {self.input_dim}], gelen {tuple(x.shape)}.'
            )
        outputs, _ = self.rnn(x)                 # [B, T, hidden * yön]
        if self.config.pooling == 'last':
            pooled = outputs[:, -1]              # yalnız son zaman adımı
        elif self.config.pooling == 'mean':
            pooled = outputs.mean(dim=1)         # tüm adımların ortalaması
        elif self.config.pooling == 'attn':
            # Her adıma bir önem puanı ver, softmax'la ağırlığa çevir,
            # adımların ağırlıklı ortalamasını al.
            weights = torch.softmax(self.attention(outputs), dim=1)
            pooled = (weights * outputs).sum(dim=1)
        else:
            pooled = outputs.max(dim=1).values   # her boyutta en güçlü adım
        return self.head(pooled)


def count_parameters(model: nn.Module) -> int:
    '''Modelin öğrenilebilir parametre sayısı (rapor/sunumda kullanılır).'''

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
