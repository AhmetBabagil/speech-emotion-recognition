# İki final yöntemi için sıfırdan tanımlanmış PyTorch modelleri.
#
# Hiçbir yerde önceden eğitilmiş ağırlık ya da hazır mimari yüklenmez; her iki sınıflandırıcı da yalnızca torch.nn'in temel yapı taşlarıyla (Conv2d, Linear, GRU/LSTM, BatchNorm, Dropout) kurulur. Ödevin "hazır model yasak" kuralının sağlandığı yer burasıdır.

from __future__ import annotations  # tip ipuçlarını esnek yazmak için (teknik satır, kodu etkilemez)

from dataclasses import asdict, dataclass, field  # ayar sınıflarını kolay yazmak + sözlüğe çevirmek için
import math  # sayı sonlu mu (isfinite) diye kontrol etmek için
from typing import Any  # tip ipucu: "herhangi bir tip" anlamına gelir

import torch  # PyTorch: derin öğrenme kütüphanesi (tensörler, otomatik türev)
from torch import nn  # nn = sinir ağı katmanları (Conv2d, Linear, GRU, Dropout...)


@dataclass(frozen=True)  # değiştirilemez (kilitli) ayar sınıfı: aynı ayar hep aynı sonuç
class OptimSettings:  # İki yöntemin ortak eğitim (optimizasyon) hiperparametreleri.

    batch_size: int = 32          # bir adımda işlenen örnek sayısı
    learning_rate: float = 3e-4   # AdamW öğrenme oranı (adım büyüklüğü)
    weight_decay: float = 1e-4    # L2 düzenlileştirme katsayısı (ağırlıkları küçük tutar)
    patience: int = 8             # early stopping sabrı (kaç epoch iyileşme beklenir)

    def validate(self) -> None:  # ayarların anlamlı (pozitif, sonlu) olduğunu kontrol eder
        if self.batch_size <= 0 or self.patience <= 0:  # yığın boyu ve sabır pozitif olmalı
            raise ValueError(f'Geçersiz optimizasyon ayarı: {self}.')  # değilse açık hata ver
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:  # lr sonlu ve pozitif mi
            raise ValueError(f'Geçersiz öğrenme oranı: {self.learning_rate}.')  # değilse hata
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:  # weight decay sonlu ve >=0 mı
            raise ValueError(f'Geçersiz weight decay: {self.weight_decay}.')  # değilse hata


# --- Yöntem 1: log-mel görüntüsü üzerinde CNN ------------------------------------


@dataclass(frozen=True)  # kilitli ayar sınıfı
class CNNConfig:  # Tek bir CNN hiperparametre adayı.

    channels: tuple[int, ...] = (32, 64, 128)  # blok başına kanal (filtre) sayısı; 3 blok
    dropout: float = 0.3                       # sınıflandırıcı öncesi dropout oranı
    optim: OptimSettings = field(default_factory=OptimSettings)  # eğitim ayarları (yukarıdaki sınıf)

    def validate(self) -> None:  # CNN ayarları geçerli mi kontrol eder
        self.optim.validate()  # önce eğitim ayarlarını doğrula
        if not self.channels or any(c <= 0 for c in self.channels):  # kanal listesi boş/negatif olmamalı
            raise ValueError(f'Geçersiz kanal genişlikleri: {self.channels}.')  # hata
        if not 0.0 <= self.dropout < 1.0:  # dropout 0 ile 1 arasında olmalı
            raise ValueError(f'Geçersiz dropout: {self.dropout}.')  # hata

    def to_dict(self) -> dict[str, Any]:  # JSON'a yazılabilir sözlük (tuple -> list çevirisiyle).

        result = asdict(self)  # dataclass'ı sözlüğe çevir
        result['channels'] = list(self.channels)  # tuple'ı JSON'un anlayacağı list'e çevir
        return result  # sözlüğü döndür


class ConvBlock(nn.Module):  # nn.Module'den türeyen bir sinir ağı bloğu
    # CNN'in temel yapı taşı: (Conv-BN-ReLU) x2 + 2x2 max pooling.
    #
    # Conv2d(3x3): yerel zaman-frekans desenlerini yakalar. BatchNorm: eğitimi kararlı ve hızlı yapar. ReLU: doğrusal olmayan aktivasyon. MaxPool(2): haritayı yarıya indirir; ağ derinleştikçe daha geniş bağlama bakılmasını sağlar.

    def __init__(self, in_ch: int, out_ch: int) -> None:  # giriş kanalı ve çıkış kanalı sayısını alır
        super().__init__()  # nn.Module'ün kurucusunu çağır (zorunlu)
        self.block = nn.Sequential(  # katmanları sırayla dizen kapsayıcı
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),  # 3x3 evrişim; padding=1 boyutu korur
            nn.BatchNorm2d(out_ch),  # toplu normalizasyon: eğitimi kararlı yapar
            nn.ReLU(inplace=True),  # negatifleri sıfırlayan aktivasyon (doğrusal olmayan)
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),  # ikinci 3x3 evrişim
            nn.BatchNorm2d(out_ch),  # ikinci normalizasyon
            nn.ReLU(inplace=True),  # ikinci aktivasyon
            nn.MaxPool2d(2),  # 2x2 havuzlama: haritayı yarı boyuta indirir
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ileri geçiş: girdiyi bloktan geçirir
        return self.block(x)  # katman dizisini uygula ve sonucu döndür


class MelCNN(nn.Module):  # Yöntem 1'in modeli: mel görüntüsü sınıflandıran CNN
    # Üst üste ConvBlock'lar + global average pooling + doğrusal sınıflandırıcı.
    #
    # Girdi [B, n_mels, T] mel görüntüleridir; forward içinde tek kanala (unsqueeze) açılır. Global average pooling, son özellik haritasının uzamsal ortalamasını aldığı için ağın kafası zaman uzunluğuna duyarsızdır.

    def __init__(self, num_classes: int, config: CNNConfig) -> None:  # sınıf sayısı + CNN ayarını alır
        super().__init__()  # nn.Module kurucusu
        config.validate()  # ayarların geçerliliğini baştan kontrol et
        blocks: list[nn.Module] = []  # blokları biriktireceğimiz boş liste
        in_ch = 1  # log-mel tek kanallı bir "gri görüntü" gibi işlenir
        for out_ch in config.channels:  # her kanal boyutu için bir blok kur (32 -> 64 -> 128)
            blocks.append(ConvBlock(in_ch, out_ch))  # yeni bloğu listeye ekle
            in_ch = out_ch  # bir sonraki bloğun girişi bu bloğun çıkışı olur
        self.features = nn.Sequential(*blocks)  # blokları tek bir sıralı ağa dönüştür
        self.pool = nn.AdaptiveAvgPool2d(1)     # her kanalı tek sayıya indir (global average pooling)
        self.classifier = nn.Sequential(  # sınıflandırıcı kafa
            nn.Dropout(config.dropout),          # aşırı öğrenmeye karşı rastgele nöron kapatma
            nn.Linear(config.channels[-1], num_classes),  # son kanal sayısından 6 sınıfa doğrusal katman
        )
        self.config = config  # ayarı sakla (sonradan lazım olabilir)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ileri geçiş: mel görüntüsü -> 6 sınıf puanı
        if x.ndim != 3:  # girdi 3 boyutlu [B, mel, zaman] olmalı
            raise ValueError(f'Beklenen girdi [B, n_mels, T], gelen {tuple(x.shape)}.')  # değilse hata
        x = self.features(x.unsqueeze(1))        # tek kanal ekle [B,1,mel,T] -> özellik haritaları
        return self.classifier(self.pool(x).flatten(1))  # havuzla, düzleştir, sınıflandır -> logit'ler


# --- Yöntem 2: aralık serisi üzerinde LSTM/GRU -----------------------------------


@dataclass(frozen=True)  # kilitli ayar sınıfı
class RNNConfig:  # Tek bir tekrarlayan-model hiperparametre adayı.

    rnn_type: str = 'gru'          # 'gru' veya 'lstm' (biz gru kullandık)
    hidden_size: int = 192         # gizli durum boyutu (modelin hafıza genişliği)
    num_layers: int = 2            # üst üste RNN katmanı sayısı
    bidirectional: bool = True     # seriyi iki yönden de oku (baştan sona + sondan başa)
    dropout: float = 0.3           # katmanlar arası + kafa öncesi dropout
    pooling: str = 'mean'          # zaman özetleme yöntemi: last/mean/max/attn
    optim: OptimSettings = field(default_factory=lambda: OptimSettings(batch_size=64, learning_rate=1e-3))  # RNN'e özel eğitim ayarı

    def validate(self) -> None:  # RNN ayarları geçerli mi kontrol eder
        self.optim.validate()  # önce eğitim ayarlarını doğrula
        if self.rnn_type not in {'lstm', 'gru'}:  # tip yalnız lstm ya da gru olabilir
            raise ValueError(f'rnn_type lstm veya gru olmalı, gelen {self.rnn_type!r}.')  # hata
        if self.hidden_size <= 0 or self.num_layers <= 0:  # boyut ve katman pozitif olmalı
            raise ValueError(f'Geçersiz RNN boyutu: {self}.')  # hata
        if not 0.0 <= self.dropout < 1.0:  # dropout 0-1 arasında
            raise ValueError(f'Geçersiz dropout: {self.dropout}.')  # hata
        if self.pooling not in {'last', 'mean', 'max', 'attn'}:  # havuzlama bu dördünden biri olmalı
            raise ValueError(
                f'pooling last/mean/max/attn olmalı, gelen {self.pooling!r}.'  # hata
            )

    def to_dict(self) -> dict[str, Any]:  # ayarı JSON'a yazılabilir sözlüğe çevirir
        return asdict(self)  # dataclass -> sözlük


class SeqRNN(nn.Module):  # Yöntem 2'nin modeli: aralık serisini sınıflandıran BiGRU
    # Sabit uzunluktaki öznitelik serisi [B, T, D] üzerinde LSTM/GRU sınıflandırıcı.
    #
    # Akış: seri -> RNN (her zaman adımına bir gizli durum) -> zaman havuzlama (tüm adımları tek vektöre özetle) -> doğrusal katman -> sınıf logit'leri.

    def __init__(self, input_dim: int, num_classes: int, config: RNNConfig) -> None:  # öznitelik boyutu + sınıf sayısı + ayar
        super().__init__()  # nn.Module kurucusu
        config.validate()  # ayarları doğrula
        if input_dim <= 0 or num_classes <= 1:  # öznitelik boyutu ve sınıf sayısı mantıklı mı
            raise ValueError(f'Geçersiz boyutlar: {input_dim}, {num_classes}.')  # hata
        rnn_cls = nn.LSTM if config.rnn_type == 'lstm' else nn.GRU  # tipe göre LSTM ya da GRU sınıfını seç
        self.rnn = rnn_cls(  # tekrarlayan katmanı kur
            input_size=input_dim,  # her adımda kaç öznitelik girecek (53)
            hidden_size=config.hidden_size,  # gizli durum genişliği (192)
            num_layers=config.num_layers,  # kaç katman (2)
            batch_first=True,                    # girdi düzeni [B, T, D] (yığın önce)
            bidirectional=config.bidirectional,  # çift yönlü mü (evet)
            # PyTorch kuralı: tek katmanlı RNN'de katman-arası dropout olmaz.
            dropout=config.dropout if config.num_layers > 1 else 0.0,  # 2+ katmanda dropout uygula
        )
        directions = 2 if config.bidirectional else 1  # çift yönlüyse çıkış 2 kat genişler
        rnn_out = config.hidden_size * directions  # RNN çıkış boyutu (192 * 2 = 384)
        # Zaman adımları üzerinde öğrenilmiş dikkat (Mirsamadi vd., 2017 tarzı);
        # tek bir Linear katmandan kurulduğu için yine tamamen sıfırdandır.
        self.attention = nn.Linear(rnn_out, 1) if config.pooling == 'attn' else None  # attn seçiliyse dikkat katmanı
        self.head = nn.Sequential(  # sınıflandırıcı kafa
            nn.Dropout(config.dropout),  # dropout
            nn.Linear(rnn_out, num_classes),  # RNN çıkışından 6 sınıfa doğrusal katman
        )
        self.input_dim = input_dim  # beklenen öznitelik boyutunu sakla (kontrol için)
        self.config = config  # ayarı sakla

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ileri geçiş: seri -> 6 sınıf puanı
        if x.ndim != 3 or x.shape[2] != self.input_dim:  # girdi [B, T, 53] olmalı
            raise ValueError(
                f'Beklenen girdi [B, T, {self.input_dim}], gelen {tuple(x.shape)}.'  # değilse hata
            )
        outputs, _ = self.rnn(x)                 # seriyi RNN'den geçir -> her adım için çıktı [B, T, 384]
        if self.config.pooling == 'last':  # havuzlama = son adım
            pooled = outputs[:, -1]              # yalnız son zaman adımını al
        elif self.config.pooling == 'mean':  # havuzlama = ortalama (bizim seçimimiz)
            pooled = outputs.mean(dim=1)         # tüm zaman adımlarının ortalaması
        elif self.config.pooling == 'attn':  # havuzlama = dikkat
            # Her adıma bir önem puanı ver, softmax'la ağırlığa çevir,
            # adımların ağırlıklı ortalamasını al.
            weights = torch.softmax(self.attention(outputs), dim=1)  # önem ağırlıkları (toplamı 1)
            pooled = (weights * outputs).sum(dim=1)  # ağırlıklı ortalama
        else:  # havuzlama = max
            pooled = outputs.max(dim=1).values   # her boyutta en güçlü adımı al
        return self.head(pooled)  # özet vektörü sınıflandır -> logit'ler


def count_parameters(model: nn.Module) -> int:  # Modelin öğrenilebilir parametre sayısı (rapor/sunumda kullanılır).

    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # tüm eğitilebilir ağırlıkları say
