"""Ödev 2 / Project_Assignment_2 — klasik makine öğrenmesi modelleri aşaması.

Bu paket, projenin ikinci ödevine ait kodu bir arada tutar. Ödev 1'de kurulan
altyapının (Wav2Vec2 öznitelik önbelleği, konuşmacı-bağımsız bölmeler, ortak
metrikler) üzerine üç ağaç tabanlı sınıflandırıcı eklenir:

  * Karar Ağacı (Decision Tree),
  * Rastgele Orman (Random Forest),
  * Gradient Boosting (HistGradientBoostingClassifier).

Öznitelikler yeniden çıkarılmaz; Ödev 1'in `odev1/cache/w2v` altındaki .npy
vektörleri doğrudan okunur. Böylece iki ödevin sonuçları aynı girdiler ve aynı
bölmeler üzerinden adil biçimde karşılaştırılabilir (KNN dahil karşılaştırma
tablosu da üretilir).

Dosyaların görev dağılımı:
  * ``model_pipeline.py`` — model tanımları, hiperparametre taraması, test ölçümü,
  * ``run_experiment.py`` — deneyleri komut satırından başlatan giriş noktası,
  * ``build_report.py``   — çıktılardan markdown rapor tabloları üretir,
  * ``build_docx.py``     — markdown raporu Word (.docx) belgesine çevirir.
"""
