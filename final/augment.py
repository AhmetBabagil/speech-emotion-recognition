# Geliştirme aşamasının, yalnız eğitim yığınlarına uygulanan veri artırmaları.
#
# Her iki dönüşüm de: - normalizasyondan geçmiş öznitelik yığınları üzerinde çalışır, - SADECE eğitim yığınlarına uygulanır (geçerleme/test asla değişmez), - önceden eğitilmiş model gerektirmez (ödev kuralına uygun).
#
# SpecAugment tarzı maskeleme (Park vd., 2019) mel-CNN için, toplamsal Gauss gürültüsü aralık serileri için kullanılır. Amaç: modele her epoch'ta aynı örneğin biraz "bozulmuş" hâlini göstererek ezberlemeyi zorlaştırmak.

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpecMask:
    # Mel görüntülerinde [B, mels, T] rastgele frekans/zaman şeritlerini sıfırlar.
    #
    # Normalizasyon sonrası ortalama 0 olduğundan, maskelenen bölgeye 0 yazmak "ortalama değerle örtme" anlamına gelir. Model, kaydın bir kısmı görünmezken bile doğru sınıfı bulmayı öğrenmek zorunda kalır.

    freq_masks: int = 2     # kaç frekans şeridi maskelenecek
    freq_width: int = 8     # bir frekans şeridinin en fazla genişliği (mel bandı)
    time_masks: int = 2     # kaç zaman şeridi maskelenecek
    time_width: int = 16    # bir zaman şeridinin en fazla genişliği (kare)

    def validate(self) -> None:
        if min(self.freq_masks, self.freq_width, self.time_masks, self.time_width) < 0:
            raise ValueError(f'Maske ayarları negatif olamaz: {self}.')

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 3:
            raise ValueError(f'Beklenen yığın [B, mels, T], gelen {tuple(batch.shape)}.')
        # clone(): orijinal tensörü bozmamak için kopya üzerinde çalış.
        batch = batch.clone()
        n, n_mels, n_frames = batch.shape
        # Her örnek için ayrı rastgele genişlik ve başlangıç seçilir; böylece
        # yığındaki her kayıt farklı bir maskeleme görür.
        for _ in range(self.freq_masks):
            widths = torch.randint(0, self.freq_width + 1, (n,), device=batch.device)
            starts = (torch.rand(n, device=batch.device)
                      * (n_mels - widths).clamp(min=0)).long()
            for i in range(n):
                batch[i, starts[i] : starts[i] + widths[i], :] = 0.0  # yatay şerit
        for _ in range(self.time_masks):
            widths = torch.randint(0, self.time_width + 1, (n,), device=batch.device)
            starts = (torch.rand(n, device=batch.device)
                      * (n_frames - widths).clamp(min=0)).long()
            for i in range(n):
                batch[i, :, starts[i] : starts[i] + widths[i]] = 0.0  # dikey şerit
        return batch


@dataclass(frozen=True)
class FeatureNoise:
    # Normalize edilmiş aralık serilerine [B, T, D] toplamsal Gauss gürültüsü.
    #
    # Öznitelikler z-skorlandığı için std=0.1, "her özniteliğe kendi doğal değişkenliğinin %10'u kadar rastgele sarsıntı" demektir.

    std: float = 0.1

    def validate(self) -> None:
        if self.std < 0.0:
            raise ValueError(f'Gürültü std negatif olamaz: {self.std}.')

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 3:
            raise ValueError(f'Beklenen yığın [B, T, D], gelen {tuple(batch.shape)}.')
        return batch + torch.randn_like(batch) * self.std
