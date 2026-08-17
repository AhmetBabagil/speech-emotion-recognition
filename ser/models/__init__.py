"""Modeller: klasik MFCC taban modeli, log-mel CNN, wav2vec2 transfer öğrenmesi.

Üç model ailesi bilinçli olarak "basitten karmaşığa" bir merdiven oluşturur:
  1. baseline  — sklearn (MFCC istatistikleri + SVM/logreg/RF): derin olmayan referans.
  2. cnn       — log-mel spektrogram üzerinde kompakt 2B CNN: ana derin model.
  3. wav2vec2  — önceden eğitilmiş öz-denetimli ses modelinin üstüne sınıflandırma
                 başlığı: transfer öğrenme deneyi.
"""

from .cnn import MelCNN
from .baseline import build_baseline


def build_model(cfg, num_classes: int):
    """Torch modelleri için fabrika fonksiyonu (sklearn taban modeli ayrı kurulur).

    "Fabrika" deseni burada işe yarar: eğitim kodu hangi sınıfın kurulacağını
    bilmek zorunda kalmaz; config'teki ``model.name`` dizgesi yeter. Yeni bir
    model eklemek = buraya bir dal eklemek.
    """
    name = cfg.model.name.lower()
    if name == "cnn":
        return MelCNN(
            num_classes=num_classes,
            in_ch=1,                                  # tek "görüntü kanalı": spektrogram
            channels=tuple(cfg.model.cnn_channels),   # her bloğun kanal sayıları
            dropout=cfg.model.dropout,
        )
    if name == "wav2vec2":
        # İçeride import: transformers paketi yalnızca wav2vec2 gerçekten
        # istendiğinde yüklenir; CNN kullanıcısı bu ağır bağımlılığı hiç ödemez.
        from .wav2vec2 import Wav2Vec2Classifier

        return Wav2Vec2Classifier(
            num_classes=num_classes,
            pretrained_name=cfg.model.pretrained_name,
            freeze_feature_encoder=cfg.model.freeze_feature_encoder,
            dropout=cfg.model.dropout,
        )
    # Tanınmayan isim: sessizce varsayılana düşmek yerine açık hata —
    # config'teki bir yazım hatası en erken burada yakalanır.
    raise ValueError(f"Unknown model.name={cfg.model.name!r} (expected 'cnn' or 'wav2vec2')")


__all__ = ["MelCNN", "build_baseline", "build_model"]
