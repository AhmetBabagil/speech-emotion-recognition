"""Konuşmadan Duygu Tanıma (Speech Emotion Recognition, SER) paketi.

Bu paket, projenin tamamının çekirdeğidir: CREMA-D ve MELD veri kümeleri
üzerinde hem korpus-içi (within-corpus) hem korpuslar-arası (cross-corpus)
duygu tanıma deneylerini yürütür.

İki veri kümesinin ortak kesişimi olan altı duygu sınıfı kullanılır:
angry (kızgın), disgust (iğrenme), fear (korku), happy (mutlu),
neutral (nötr), sad (üzgün). Böylece iki korpusun etiketleri aynı
indeks uzayında buluşur ve sonuçlar doğrudan karşılaştırılabilir olur.
"""

# Paketin sürüm numarası; pip/paketleme araçları ve loglar buradan okur.
__version__ = "0.1.0"
