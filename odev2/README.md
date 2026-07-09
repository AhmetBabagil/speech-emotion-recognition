# Odev 2 - ML modellerinin gelistirilmesi ve karsilastirilmasi

Bu klasor Project_Assignment_2 icindir ve 1. odev klasorunden ayridir.

Konu: konusmadan duygu tanima. Veri setleri: CREMA-D ve MELD. Ses girdileri,
1. asamada kullanilan donmus Wav2Vec2 oznitelik cache'i ile vektore cevrilir;
bu asamada ML is akisi yalnizca `numpy`, `pandas` ve `scikit-learn` kullanir.

## Akis

1. Oznitelikler hazir degilse once 1. asamadaki extraction calistirilir.

```bash
python odev1/extract.py --manifest odev1/manifest_subset.csv
```

2. Karar agaci, rastgele orman ve gradient boosting icin validasyon aramasi
   calistirilir.

```bash
python odev2/run_experiment.py --manifest odev1/manifest_subset.csv --grid-mode report
```

Hizli kod kontrolu icin (`report` yerine kucuk grid):

```bash
python odev2/run_experiment.py --manifest odev1/manifest_subset.csv --quick --corpora cremad
```

3. Google Doc'a yapistirilacak tablolar uretilir.

```bash
python odev2/build_report.py
```

## Dosyalar

| Dosya | Gorev |
|---|---|
| `model_pipeline.py` | Her veri seti ve model icin feature boyutu, PCA boyutu ve model hiperparametrelerini validation setinde arar; en iyiyi train+val ile tekrar egitir ve test setinde olcer. |
| `run_experiment.py` | Deneyleri CLI ile calistirir. |
| `build_report.py` | Deney ciktilarindan rapora yapistirilabilir markdown tablolar uretir. |

## Optimize edilen hiperparametreler

| Model | Hiperparametreler |
|---|---|
| Karar Agaci | `criterion`, `max_depth`, `min_samples_split` |
| Rastgele Orman | `n_estimators`, `max_depth`, `max_features` |
| Gradient Boosting (`HistGradientBoostingClassifier`) | `learning_rate`, `max_depth` |

Her model icin ayrica Wav2Vec2 havuzlama/vektor boyutu (`768`, `1536`, `2304`)
ve PCA cikti boyutu validasyonla secilir. Rapor/teslim modunda PCA icin
`none`, `64`, `128`, `256`; tam grid modunda ek olarak `32` ve `512` denenir. Sinif dengesizligi icin karar agaci ve rastgele ormanda
`class_weight="balanced"`, gradient boosting'de `sample_weight` kullanilir.

## Ciktilar

`odev2/outputs/<veri_seti>/` altinda model bazli validasyon gridleri, test
sonuclari ve karmasiklik matrisleri bulunur. `odev2/outputs/model_comparison.csv`
yalnizca 2. asamadaki modelleri, `odev2/outputs/test_comparison_with_knn.csv`
ise 1. asamadaki KNN sonucunu da iceren genel test karsilastirmasini verir.

## Teslim notu

Rapor, validation gridleri, test karsilastirma tablolari ve karmasiklik matrisleri odev2/outputs altinda teslim icin saklanir. Drive teslim paketinde ayni ciktinin DOCX/PDF ve zip kopyalari da bulunur.
