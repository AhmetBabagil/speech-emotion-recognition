# Geliştirme aşamasının, yalnız eğitim yığınlarına uygulanan veri artırmaları.
#
# Her iki dönüşüm de: normalizasyondan geçmiş öznitelik yığınları üzerinde çalışır, SADECE eğitim yığınlarına uygulanır (geçerleme/test asla değişmez), önceden eğitilmiş model gerektirmez (ödev kuralına uygun).
#
# SpecAugment tarzı maskeleme (Park vd., 2019) mel-CNN için, toplamsal Gauss gürültüsü aralık serileri için kullanılır. Amaç: modele her epoch'ta aynı örneğin biraz "bozulmuş" hâlini göstererek ezberlemeyi zorlaştırmak.

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

from dataclasses import dataclass  # ayar sınıflarını kolay yazmak için

import torch  # tensör işlemleri


@dataclass(frozen=True)  # kilitli ayar sınıfı
class SpecMask:
    # Mel görüntülerinde [B, mels, T] rastgele frekans/zaman şeritlerini sıfırlar.
    #
    # Normalizasyon sonrası ortalama 0 olduğundan, maskelenen bölgeye 0 yazmak "ortalama değerle örtme" anlamına gelir. Model, kaydın bir kısmı görünmezken bile doğru sınıfı bulmayı öğrenmek zorunda kalır.

    freq_masks: int = 2     # kaç frekans şeridi maskelenecek
    freq_width: int = 8     # bir frekans şeridinin en fazla genişliği (mel bandı)
    time_masks: int = 2     # kaç zaman şeridi maskelenecek
    time_width: int = 16    # bir zaman şeridinin en fazla genişliği (kare)

    def validate(self) -> None:  # ayarların geçerli (negatif olmayan) olduğunu kontrol eder
        if min(self.freq_masks, self.freq_width, self.time_masks, self.time_width) < 0:  # negatif var mı
            raise ValueError(f'Maske ayarları negatif olamaz: {self}.')  # varsa hata

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:  # bir yığına maskeleme uygular
        if batch.ndim != 3:  # yığın [B, mels, T] olmalı
            raise ValueError(f'Beklenen yığın [B, mels, T], gelen {tuple(batch.shape)}.')  # değilse hata
        # clone(): orijinal tensörü bozmamak için kopya üzerinde çalış.
        batch = batch.clone()  # kopya al (orijinali değiştirme)
        n, n_mels, n_frames = batch.shape  # yığın boyu, mel sayısı, kare sayısı
        # Her örnek için ayrı rastgele genişlik ve başlangıç seçilir; böylece
        # yığındaki her kayıt farklı bir maskeleme görür.
        for _ in range(self.freq_masks):  # her frekans maskesi için
            widths = torch.randint(0, self.freq_width + 1, (n,), device=batch.device)  # rastgele şerit genişlikleri
            starts = (torch.rand(n, device=batch.device)  # rastgele başlangıçlar
                      * (n_mels - widths).clamp(min=0)).long()
            for i in range(n):  # her örnek için
                batch[i, starts[i] : starts[i] + widths[i], :] = 0.0  # yatay şerit
        for _ in range(self.time_masks):  # her zaman maskesi için
            widths = torch.randint(0, self.time_width + 1, (n,), device=batch.device)  # rastgele genişlikler
            starts = (torch.rand(n, device=batch.device)  # rastgele başlangıçlar
                      * (n_frames - widths).clamp(min=0)).long()
            for i in range(n):  # her örnek için
                batch[i, :, starts[i] : starts[i] + widths[i]] = 0.0  # dikey şerit
        return batch  # maskelenmiş yığın


@dataclass(frozen=True)  # kilitli ayar sınıfı
class FeatureNoise:
    # Normalize edilmiş aralık serilerine [B, T, D] toplamsal Gauss gürültüsü.
    #
    # Öznitelikler z-skorlandığı için std=0.1, "her özniteliğe kendi doğal değişkenliğinin %10'u kadar rastgele sarsıntı" demektir.

    std: float = 0.1  # gürültü şiddeti (standart sapma)

    def validate(self) -> None:  # ayarı doğrula
        if self.std < 0.0:  # std negatif olamaz
            raise ValueError(f'Gürültü std negatif olamaz: {self.std}.')  # hata

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:  # yığına gürültü ekler
        if batch.ndim != 3:  # yığın [B, T, D] olmalı
            raise ValueError(f'Beklenen yığın [B, T, D], gelen {tuple(batch.shape)}.')  # değilse hata
        return batch + torch.randn_like(batch) * self.std  # her değere Gauss gürültüsü ekle
