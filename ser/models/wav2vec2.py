"""Transfer öğrenme: önceden eğitilmiş wav2vec2 üzerine sınıflandırma başlığı.

Girdi : [B, num_samples] ham dalga formu (16 kHz; normalizasyonu dataset yapar)
Çıktı : [B, num_classes] logit'ler

Transfer öğrenmenin fikri: wav2vec2, ETİKETSİZ binlerce saat konuşmayla
öz-denetimli (self-supervised) olarak eğitilmiştir ve genel amaçlı, zengin ses
temsilleri öğrenmiştir. Bizim gibi görece küçük SER veri kümeleri, bu hazır
temsillerin üstüne yalnızca küçük bir sınıflandırma başlığı öğrenerek sıfırdan
eğitimin ulaşamayacağı sonuçlara ulaşabilir.

Modelin iki parçası vardır: dalga formunu kaba temsile çeviren KONVOLÜSYONEL
öznitelik kodlayıcı ve bağlamı işleyen TRANSFORMER katmanları. Küçük veri
kümelerinde kodlayıcıyı DONDURMAK (freeze) önerilir: en genel, en alt seviye
özellikleri bozmadan yalnızca transformer + başlık ince ayarlanır (fine-tune).

``transformers`` paketi gerekir (``pip install -e .[transfer]``). Bu yol
hesap yükü nedeniyle GPU'ya yöneliktir.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Wav2Vec2Classifier(nn.Module):
    """Önceden eğitilmiş wav2vec2 gövdesi + ortalama havuz + 2 katmanlı MLP başlık."""

    def __init__(self, num_classes: int, pretrained_name: str = "facebook/wav2vec2-base",
                 freeze_feature_encoder: bool = True, dropout: float = 0.3):
        super().__init__()
        try:
            from transformers import Wav2Vec2Model
        except ImportError as e:  # pragma: no cover
            # transformers ağır ve isteğe bağlı bir bağımlılıktır; eksikse
            # kurulum komutunu da söyleyen anlaşılır bir hata ver.
            raise ImportError(
                "wav2vec2 model requires `transformers`. Install with "
                "`pip install -e .[transfer]` or `pip install transformers`."
            ) from e

        # Önceden eğitilmiş ağırlıklar Hugging Face Hub'dan indirilir
        # (ilk çağrıda; sonrası yerel önbellekten gelir).
        self.backbone = Wav2Vec2Model.from_pretrained(pretrained_name)
        if freeze_feature_encoder:
            # Konvolüsyonel kodlayıcının gradyanlarını kapat: bu parametreler
            # artık güncellenmez. Küçük veri kümesinde alt katmanları oynatmak
            # genel temsilleri bozup overfitting'i büyütür; ayrıca eğitim
            # belleği ve süresi de azalır.
            self.backbone.feature_extractor._freeze_parameters()
        # Başlığın giriş boyutu modelden okunur (base: 768) — model adı
        # değişse bile (örn. large: 1024) başlık otomatik uyum sağlar.
        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),   # gizli ara katman: temsili göreve uyarla
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        # Gövde, dalga formunu zaman adımı başına bir gizli vektöre çevirir.
        out = self.backbone(input_values).last_hidden_state  # [B, T', H]
        # Zaman üzerinden ortalama havuzlama: değişken uzunluklu diziyi tek
        # vektöre indirger — "kaydın tamamının ortalama temsili". Duygu klibin
        # geneline yayılan bir özellik olduğu için ortalama makul bir özettir.
        pooled = out.mean(dim=1)  # [B, H]
        return self.head(pooled)  # [B, num_classes] logit
