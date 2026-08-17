'''PyTorch katmanlarıyla sıfırdan kurulmuş, yapılandırılabilir MLP.

Bu modülde iki ana parça var:

1. ``MLPConfig``: Bir hiperparametre adayını (katman boyutları, aktivasyon,
   dropout oranı vb.) tek bir değişmez (frozen) veri sınıfında toplar.
   Böylece "hangi ayarlarla eğitildi?" sorusunun cevabı her zaman tek bir
   nesnede durur ve JSON'a çevrilip diske kaydedilebilir.
2. ``MLP``: Bu ayarlara göre katmanları dinamik olarak dizen, tamamen
   bağlantılı (fully connected) sınıflandırıcı. Ödev şartı gereği hazır ya da
   önceden eğitilmiş hiçbir model kullanılmaz; her şey ``nn.Linear`` gibi
   temel yapı taşlarından kurulur.
'''

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class MLPConfig:
    '''Tek bir hiperparametre adayı.

    ``frozen=True`` sayesinde nesne oluşturulduktan sonra alanları
    değiştirilemez; bu, aynı konfigürasyonun deney boyunca yanlışlıkla
    mutasyona uğramasını engeller ve sözlük anahtarı olarak kullanılabilmesini
    sağlar. Varsayılan değerler, aramaya başlamadan önce "makul bir orta
    nokta" olarak seçilmiştir.
    '''

    batch_size: int = 64
    learning_rate: float = 3e-4
    patience: int = 8
    hidden_dims: tuple[int, ...] = (512, 256)
    activation: str = 'relu'
    batch_norm: bool = True
    dropout: float = 0.3
    weight_decay: float = 1e-4

    def validate(self) -> None:
        # Neden ayrı bir validate metodu? Dataclass'lar alan tiplerini zorunlu
        # kılmaz; hatalı bir değer (örn. negatif öğrenme oranı) eğitim sırasında
        # sessizce saçma sonuçlar üretebilir. Burada hatayı EN ERKEN noktada,
        # anlaşılır bir mesajla yakalıyoruz ("fail fast" ilkesi).
        if self.batch_size <= 0 or self.patience <= 0:
            raise ValueError(f'Invalid optimization settings: {self}.')
        # math.isfinite kontrolü NaN ve sonsuz değerleri de eler; sadece
        # "> 0" yazsaydık NaN karşılaştırmaları sessizce False dönerdi.
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError(f'Invalid learning rate: {self.learning_rate}.')
        if not self.hidden_dims or any(size <= 0 for size in self.hidden_dims):
            raise ValueError(f'Invalid hidden dimensions: {self.hidden_dims}.')
        if self.activation not in {'relu', 'gelu', 'tanh', 'leaky_relu'}:
            raise ValueError(f'Unsupported activation: {self.activation}.')
        # Dropout bir olasılıktır: [0, 1) aralığında olmalı. 1.0'a izin
        # vermiyoruz çünkü tüm nöronları kapatmak öğrenmeyi imkansız kılar.
        # weight_decay (L2 cezası) ise negatif olamaz.
        if (
            not math.isfinite(self.dropout)
            or not 0.0 <= self.dropout < 1.0
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0.0
        ):
            raise ValueError(f'Invalid regularization settings: {self}.')

    def to_dict(self) -> dict[str, Any]:
        # JSON'a kaydederken tuple desteklenmediği için hidden_dims'i listeye
        # çeviriyoruz; geri kalan alanları asdict olduğu gibi kopyalar.
        result = asdict(self)
        result['hidden_dims'] = list(self.hidden_dims)
        return result

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> 'MLPConfig':
        # Diskten (JSON) okunan sözlüğü tekrar MLPConfig'e çevirir.
        # Önce kopyalıyoruz ki çağıranın sözlüğünü yerinde değiştirmeyelim;
        # ardından listeyi tuple'a döndürüp doğrulamadan geçiriyoruz.
        values = dict(values)
        values['hidden_dims'] = tuple(int(v) for v in values['hidden_dims'])
        config = cls(**values)
        config.validate()
        return config


def _activation(name: str) -> nn.Module:
    # İsimden aktivasyon katmanı üreten küçük bir "fabrika" fonksiyonu.
    # Her çağrıda YENİ bir modül döndürmek önemli: aynı nesneyi paylaşmak
    # PyTorch'ta genelde sorun çıkarmasa da katman listesini temiz tutar.
    if name == 'relu':
        return nn.ReLU()
    if name == 'gelu':
        return nn.GELU()
    if name == 'tanh':
        return nn.Tanh()
    if name == 'leaky_relu':
        # 0.01'lik negatif eğim, "ölü ReLU" problemini hafifletmek için
        # kullanılan standart değerdir.
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(f'Unsupported activation: {name}.')


class MLP(nn.Module):
    '''Tamamen bağlantılı sınıflandırıcı; hazır/önceden eğitilmiş model yok.

    Katman dizilimi her gizli katman için şöyledir:
    ``Linear -> (BatchNorm1d) -> Aktivasyon -> (Dropout)``
    ve en sonda sınıf sayısı kadar çıkış üreten bir ``Linear`` bulunur.
    Çıkışta softmax YOKTUR; çünkü eğitimde kullanılan ``CrossEntropyLoss``
    ham logit bekler (softmax'ı içeride kendisi uygular).
    '''

    def __init__(self, input_dim: int, num_classes: int, config: MLPConfig) -> None:
        super().__init__()
        # Girdi/çıktı boyutlarını ve konfigürasyonu en başta doğruluyoruz;
        # hatalı bir boyutla katman kurmak sonradan anlaşılması zor
        # şekil (shape) hatalarına yol açar.
        config.validate()
        if input_dim <= 0 or num_classes <= 1:
            raise ValueError(f'Invalid MLP dimensions: {input_dim}, {num_classes}.')

        # Katmanları bir listede biriktirip en sonda nn.Sequential'a veriyoruz.
        # Bu sayede gizli katman sayısı ve genişliği tamamen config'den gelir;
        # kodda sabit bir mimari yoktur.
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in config.hidden_dims:
            layers.append(nn.Linear(previous, hidden))
            if config.batch_norm:
                # BatchNorm aktivasyondan ÖNCE uygulanır: katman çıktısını
                # normalize ederek eğitimi kararlı hale getirir ve daha yüksek
                # öğrenme oranlarına izin verir.
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(_activation(config.activation))
            if config.dropout > 0.0:
                # Dropout aktivasyondan SONRA: eğitim sırasında nöronların bir
                # kısmını rastgele kapatarak aşırı öğrenmeyi (overfitting)
                # azaltır. dropout=0 ise katmanı hiç eklemeyiz.
                layers.append(nn.Dropout(config.dropout))
            previous = hidden
        # Çıkış katmanı: son gizli boyuttan sınıf sayısına projeksiyon.
        layers.append(nn.Linear(previous, num_classes))
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.config = config
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Ağırlık başlangıcı (initialization) rastgele ama BİLİNÇLİ seçilir:
        # - ReLU ailesi için Kaiming (He) init: ReLU negatifleri sıfırladığı
        #   için varyansı koruyacak şekilde ölçeklenmiş başlangıç gerekir.
        # - tanh/GELU gibi simetrik aktivasyonlar için Xavier (Glorot) init
        #   daha uygundur.
        # Bias'ları sıfırdan başlatmak yaygın ve güvenli bir tercihtir.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.config.activation in {'relu', 'leaky_relu'}:
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Şekil kontrolü: (batch, input_dim) beklenir. Yanlış şekilli girdi
        # PyTorch'un derinliklerinde kafa karıştırıcı hatalar üretebileceği
        # için burada açık bir mesajla erken durduruyoruz.
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f'Expected input shape (batch, {self.input_dim}), got '
                f'{tuple(features.shape)}.'
            )
        return self.network(features)


def count_parameters(model: nn.Module) -> int:
    # Sadece eğitilebilir (requires_grad=True) parametreleri sayar; raporda
    # model boyutunu belirtmek ve adaylar arasında karşılaştırma yapmak için
    # kullanılır.
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
