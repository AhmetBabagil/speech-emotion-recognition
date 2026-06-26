# Ödev 1 — KNN aşaması (Project_Assignment_1)

Konuşmadan duygu tanıma (CREMA-D + MELD) için **K-En Yakın Komşu (KNN)** aşaması.
Akış: **Wav2Vec2 (donmuş öznitelik çıkarıcı) → StandardScaler → PCA → KNN**.
İncelenen hiperparametreler: **F** (öznitelik vektör boyutu), **P** (PCA boyutu),
**K** (komşu sayısı).

> **Kütüphane kuralı:** ML iş akışında yalnızca `numpy / pandas / scikit-learn`.
> `torch` / `transformers` SADECE Wav2Vec2 öznitelik çıkarımında kullanılır.

## Çalıştırma sırası

```bash
# 0) (bir kez) veri + manifest — proje kökünden:
python scripts/download_data.py --datasets cremad meld
python scripts/build_manifest.py

# 1) Wav2Vec2 öznitelikleri çıkar ve cache'le (yavaş, bir kez; resumable):
python odev1/extract.py                                  # tam veri
python odev1/extract.py --manifest odev1/manifest_subset.csv   # altküme (hızlı)

# 2) KNN + PCA hiperparametre gridini koş (iki veri seti):
python odev1/run_experiment.py --manifest odev1/manifest_subset.csv

# 3) Rapor tablolarını üret (Doc'a yapıştırmaya hazır markdown):
python odev1/build_report.py
```

## Dosyalar

| Dosya | Görev |
|-------|-------|
| `features_w2v.py` | Wav2Vec2 donmuş embedding çıkarımı + cache; mean/std/max havuzlama → 3·H. `load_pooled()` ile F=768/1536/2304 dilimlenir. |
| `extract.py` | Manifest'teki tüm klipler için öznitelik çıkarır (CLI). |
| `knn_pipeline.py` | Veri seti başına F×P×K ızgara araması, en iyi seçimi, test ve karmaşıklık matrisi (numpy/pandas/sklearn). |
| `run_experiment.py` | Deneyi iki veri setinde koşar (CLI). |
| `evaluation.py` | Metrikler (sklearn) + karmaşıklık matrisi çizimi (matplotlib). |
| `build_report.py` | Çıktılardan rapor tablolarını markdown üretir. |
| `RAPOR.md` | İlerleme raporu (şablona göre). |

## Çıktılar

`odev1/outputs/<veri seti>/`: `validation_grid.csv` (120 kombinasyon),
`result.json` (en iyi config + test), `confusion_matrix.png`.
`odev1/outputs/`: `test_comparison.csv`, `overall_best_confusion.png`, `summary.json`.

## Değerlendirme protokolü

Denek-bağımsız bölme (konuşmacı tek bir kümede): %70 eğitim / %15 geçerleme / %15 test.
Geçerleme hiperparametre seçer; test yalnız nihai ölçüm. Metrikler: doğruluk, dengeli
doğruluk, makro-F1, ağırlıklı-F1, karmaşıklık matrisi.
