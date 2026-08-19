# Final Projesi — Konuşmadan Duygu Tanıma

YAP 470 final ödevi: CREMA-D üzerinde, önceden eğitilmiş model kullanmadan
iki yöntem.

- **Yöntem 1:** log-mel spectrogram + sıfırdan CNN
- **Yöntem 2:** aralık başına akustik öznitelik serisi + sıfırdan LSTM/GRU
  (aralık sayısı ve genişliği hiperparametre)

Ortak protokol: konuşmacı-bağımsız eğitim/geçerleme/test bölmesi,
sınıf-ağırlıklı cross-entropy, geçerleme macro-F1 üzerinde erken durdurma,
geçerleme temelli hiperparametre araması + yerel iyileştirme turu. Test
kümesine yalnızca seçilen modeller dokunur.

**Nihai sonuç:** her iki yöntemin son hâli 5'er modelli topluluktur (softmax
ortalaması). Yöntem 1 (CNN) test doğruluğu %62,3; Yöntem 2 (BiGRU,
jitter+kontrast öznitelikleri) %68,6 — sıfırdan yayınlanmış en iyi bantla
(~%68 [kaynaklar.md, 4]) aynı seviye. Öznitelik seçimi ablasyonla yapıldı.

## Çalıştırma sırası

```bash
# 1. Ana deney: arama + iyileştirme turu + test değerlendirmesi
python final/run_experiment.py --grid-mode report --feature-workers 8

# 2. Geliştirme aşaması: SpecAugment / dikkat havuzlama / gürültü varyantları
python final/improve.py

# 3. Öznitelik ablasyonu: hangi grup ne kadar katıyor + nihai set (jitter+kontrast)
python final/ablasyon.py
python final/ablasyon_birak.py

# 4. Nihai topluluk modelleri (5'er model, softmax ortalaması) — resmi sonuçlar
python final/cnn_topluluk.py    # Yöntem 1 -> %62,3
python final/rnn_topluluk.py    # Yöntem 2 -> %68,6

# Hızlı sağlamlık kontrolü (küçük veri, dakikalar içinde):
python final/run_experiment.py --grid-mode quick --limit-per-split 60

# Demo (yönerge 8. bölüm): rastgele veya adıyla verilen örnekleri iki yöntemden geçir
python final/demo.py --rastgele 3
python final/demo.py final/demo_ornekleri/1013_TIE_DIS_XX.wav

# Tek dosya tahmini:
python final/predict.py ses.wav --model final/outputs/cremad/cnn/winner_model.pt
```

Geçerleme/test/demo kodlarının çalıştırılmış hali: `SONUCLAR_VE_DEMO.ipynb`
(yönerge 7. bölüm; test sonuçlarının birebir yeniden üretildiğini de gösterir).

## Dosyalar

| Dosya | İçerik |
|---|---|
| `features.py` | Mel görüntüsü (Yöntem 1) ve aralık öznitelik serisi (Yöntem 2) çıkarımı |
| `dataset.py` | Öznitelik önbelleği, yalnız-eğitim z-skor normalizasyonu, tensör veri kümeleri |
| `models.py` | Sıfırdan CNN ve LSTM/GRU (last/mean/max/attention havuzlama) |
| `training.py` | Ağırlıklı loss + erken durdurmalı ortak eğitim döngüsü |
| `search_space.py` | Deterministik hiperparametre adayları ve iyileştirme uzayları |
| `pipeline.py` | Bölme → öznitelik → arama → iyileştirme → test akışı |
| `augment.py` | SpecAugment maskeleme ve öznitelik gürültüsü (yalnız eğitim yığınlarına) |
| `improve.py` | Geliştirme aşaması koşucusu |
| `ablasyon.py`, `ablasyon_birak.py` | Öznitelik ablasyonu (kümülatif + bırak-birini); nihai set seçimi |
| `kaydet_resmi.py` | Nihai öznitelik setiyle tek modeli 5 koşu eğitip kaydeder |
| `cnn_topluluk.py`, `rnn_topluluk.py` | Nihai 5-model topluluklar (softmax ortalaması) |
| `cnn_gelistir.py`, `ses_artirma_dene.py` | Geliştirme denemeleri (mixup/label-smoothing; ses-uzayı artırma) |
| `demo.py` | Demo: iyi/kötü örnek dizininden rastgele veya adıyla girdi → iki yöntemin sonuçları |
| `demo_ornekleri/` | Test kümesinden seçilmiş 6 iyi + 4 kötü örnek (demo girdileri) |
| `SONUCLAR_VE_DEMO.ipynb` | Geçerleme + test (yeniden üretim kanıtıyla) + demo, çalıştırılmış çıktılarıyla |
| `predict.py` | Kaydedilen modelle WAV tahmini |
| `kaynaklar.md` | Literatür taraması ve kaynakça notları |
| `tests/` | Öznitelik/model birim testleri (`python -m pytest final/tests -q`) |

Çıktılar `final/outputs/<corpus>/` altına yazılır: `search_log.csv`,
`winner.json`, öğrenme eğrisi, test metrikleri + karışıklık matrisi,
`improvements.csv`, `method_comparison.csv`, model ağırlıkları.
