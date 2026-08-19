# Log-mel spektrogramlar üzerinde çalışan kompakt bir 2 boyutlu CNN.
#
# Girdi : [B, 1, n_mels, T]   (B=batch, 1=kanal, n_mels=frekans, T=zaman) Çıktı : [B, num_classes] logit'ler (softmax UYGULANMAZ — CrossEntropyLoss
# logit bekler; olasılık gerektiğinde değerlendirme kodu softmax uygular)
#
# Mimari: Dört konvolüsyon bloğu (2 x [Conv-BN-ReLU] + MaxPool) ve ardından global ortalama havuzlama + doğrusal sınıflandırıcı. Spektrogram bir "görüntü" gibi işlenir: alçak katmanlar yerel zaman-frekans desenlerini (enerji kenarları, harmonik çizgiler), üst katmanlar daha soyut birleşimleri öğrenir.
#
# Global havuzlama önemli bir tasarım kararıdır: sınıflandırıcı başlığını zaman boyutundan bağımsız kılar — [B, C, H', W'] hangi genişlikte olursa olsun havuz sonrası [B, C] elde edilir. Böylece klip uzunluğu (T) değişse bile model çalışmaya devam eder ve Flatten'lı tasarımlara göre çok daha az parametre gerekir.

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    # Temel yapı taşı: (Conv-BN-ReLU) x2 + MaxPool.
    #
    # * Conv2d(3x3, padding=1): boyutu korur; bias=False çünkü hemen ardından
    # gelen BatchNorm kendi öğrenilebilir kaydırmasını (beta) ekler — ayrıca
    # bias tutmak gereksiz parametre olurdu.
    # * BatchNorm: aktivasyon dağılımlarını dengeler; eğitimi hızlandırır ve
    # hafif bir düzenlileştirme etkisi yapar.
    # * ReLU(inplace=True): aktivasyonu yerinde yazarak bellekten tasarruf eder. * İki konvolüsyonu ARKA ARKAYA koymak (VGG deseni), tek büyük çekirdeğe göre
    # daha az parametreyle daha geniş bir görüş alanı (receptive field) sağlar.
    # * MaxPool(2): hem frekans hem zaman eksenini yarıya indirir — kademeli
    # soyutlama ve hesap tasarrufu.

    def __init__(self, in_ch: int, out_ch: int):
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

    def forward(self, x):
        return self.block(x)


class MelCNN(nn.Module):  # Konvolüsyonel gövde + global ortalama havuz + dropout'lu doğrusal başlık.

    def __init__(self, num_classes: int, in_ch: int = 1,
                 channels: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        # Blokları zincirle: kanal sayısı her blokta artar (32->64->128->256),
        # uzamsal boyut her blokta yarılanır. Config'ten farklı bir tuple
        # verilerek ağın derinliği/genişliği kod değişmeden ayarlanabilir.
        blocks = []
        c_in = in_ch
        for c_out in channels:
            blocks.append(ConvBlock(c_in, c_out))
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        # AdaptiveAvgPool2d(1): girdinin boyutu ne olursa olsun her kanalı tek
        # sayıya (o kanalın tüm harita üzerindeki ortalamasına) indirger.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),                       # başlıkta overfitting freni
            nn.Linear(channels[-1], num_classes),      # 256 -> 6 logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # [B, 1, n_mels, T] -> [B, C, H', W']
        x = self.pool(x).flatten(1)   # -> [B, C, 1, 1] -> [B, C]
        return self.classifier(x)     # -> [B, num_classes] (ham logit)
