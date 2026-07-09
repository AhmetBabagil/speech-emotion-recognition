# Project Assignment 2 - Rapor tablolari

Not: Asagidaki validasyon tablolarinda her veri seti-model ikilisi icin en yuksek validasyon makro-F1 degerine sahip ilk kombinasyonlar verilmistir. Calistirilan grid CSV dosyalari `odev2/outputs/<veri_seti>/` altindadir.

Deney bilgisi: manifest `odev1\manifest_subset.csv`, feature cache `odev1/cache/w2v`, mod `rapor/teslim gridi`.

## Deney kapsami

| Veri seti | Model | Kombinasyon | F secenekleri | PCA secenekleri | Hiperparametre kombinasyonu |
|---|---|---|---|---|---|
| CREMA-D | Karar Agaci | 216 | 768, 1536, 2304 | none, 256, 64, 128 | 18 |
| CREMA-D | Gradient Boosting | 24 | 768, 1536, 2304 | none, 64 | 4 |
| CREMA-D | Rastgele Orman | 72 | 768, 1536, 2304 | none, 128, 64, 256 | 6 |
| MELD | Karar Agaci | 216 | 768, 1536, 2304 | 128, 64, none, 256 | 18 |
| MELD | Gradient Boosting | 24 | 768, 1536, 2304 | 64, none | 4 |
| MELD | Rastgele Orman | 72 | 768, 1536, 2304 | 64, 128, 256, none | 6 |

## CREMA-D - Karar Agaci validasyon

| Sira | F | PCA | Hiperparametreler | Val dogr. | Val dengeli dogr. | Val makro-F1 |
|---|---|---|---|---|---|---|
| 1 | 1536 | none | criterion=entropy, max_depth=8, min_samples_split=10 | 0.4049 | 0.4056 | 0.4031 |
| 2 | 1536 | none | criterion=log_loss, max_depth=8, min_samples_split=10 | 0.4049 | 0.4056 | 0.4031 |
| 3 | 1536 | none | criterion=entropy, max_depth=16, min_samples_split=2 | 0.3902 | 0.3881 | 0.3859 |
| 4 | 1536 | none | criterion=log_loss, max_depth=16, min_samples_split=2 | 0.3902 | 0.3881 | 0.3859 |
| 5 | 1536 | none | criterion=entropy, max_depth=8, min_samples_split=2 | 0.3902 | 0.3917 | 0.3851 |
| 6 | 1536 | none | criterion=log_loss, max_depth=8, min_samples_split=2 | 0.3902 | 0.3917 | 0.3851 |
| 7 | 1536 | none | criterion=entropy, max_depth=None, min_samples_split=10 | 0.3829 | 0.3806 | 0.3804 |
| 8 | 1536 | none | criterion=entropy, max_depth=16, min_samples_split=10 | 0.3829 | 0.3806 | 0.3804 |
| 9 | 1536 | none | criterion=log_loss, max_depth=None, min_samples_split=10 | 0.3829 | 0.3806 | 0.3804 |
| 10 | 1536 | none | criterion=log_loss, max_depth=16, min_samples_split=10 | 0.3829 | 0.3806 | 0.3804 |
| 11 | 2304 | none | criterion=entropy, max_depth=8, min_samples_split=10 | 0.3829 | 0.3849 | 0.378 |
| 12 | 2304 | none | criterion=log_loss, max_depth=8, min_samples_split=10 | 0.3829 | 0.3849 | 0.378 |

## CREMA-D - Gradient Boosting validasyon

| Sira | F | PCA | Hiperparametreler | Val dogr. | Val dengeli dogr. | Val makro-F1 |
|---|---|---|---|---|---|---|
| 1 | 1536 | none | learning_rate=0.1, max_depth=3 | 0.5244 | 0.5226 | 0.5238 |
| 2 | 1536 | none | learning_rate=0.05, max_depth=3 | 0.522 | 0.5206 | 0.5151 |
| 3 | 768 | none | learning_rate=0.1, max_depth=3 | 0.5073 | 0.5052 | 0.5043 |
| 4 | 2304 | none | learning_rate=0.1, max_depth=3 | 0.5098 | 0.5056 | 0.5016 |
| 5 | 768 | none | learning_rate=0.05, max_depth=3 | 0.5024 | 0.5016 | 0.4966 |
| 6 | 2304 | none | learning_rate=0.05, max_depth=3 | 0.5024 | 0.5008 | 0.4949 |
| 7 | 2304 | none | learning_rate=0.1, max_depth=1 | 0.4976 | 0.496 | 0.4858 |
| 8 | 1536 | none | learning_rate=0.1, max_depth=1 | 0.4927 | 0.4913 | 0.4808 |
| 9 | 768 | none | learning_rate=0.1, max_depth=1 | 0.4854 | 0.4825 | 0.478 |
| 10 | 768 | 64 | learning_rate=0.05, max_depth=3 | 0.478 | 0.4786 | 0.4683 |
| 11 | 768 | 64 | learning_rate=0.1, max_depth=3 | 0.4756 | 0.475 | 0.4664 |
| 12 | 2304 | 64 | learning_rate=0.1, max_depth=3 | 0.4732 | 0.475 | 0.4629 |

## CREMA-D - Rastgele Orman validasyon

| Sira | F | PCA | Hiperparametreler | Val dogr. | Val dengeli dogr. | Val makro-F1 |
|---|---|---|---|---|---|---|
| 1 | 1536 | none | max_depth=16, max_features=log2, n_estimators=100 | 0.5049 | 0.5048 | 0.4964 |
| 2 | 2304 | none | max_depth=24, max_features=sqrt, n_estimators=100 | 0.5049 | 0.5028 | 0.4941 |
| 3 | 768 | none | max_depth=None, max_features=log2, n_estimators=100 | 0.5 | 0.4984 | 0.4924 |
| 4 | 768 | none | max_depth=24, max_features=log2, n_estimators=100 | 0.5 | 0.498 | 0.4923 |
| 5 | 2304 | none | max_depth=None, max_features=sqrt, n_estimators=100 | 0.5024 | 0.5004 | 0.4904 |
| 6 | 2304 | 128 | max_depth=16, max_features=log2, n_estimators=100 | 0.5 | 0.5004 | 0.487 |
| 7 | 1536 | none | max_depth=24, max_features=log2, n_estimators=100 | 0.5 | 0.4972 | 0.4863 |
| 8 | 1536 | none | max_depth=None, max_features=log2, n_estimators=100 | 0.5 | 0.4976 | 0.4859 |
| 9 | 768 | none | max_depth=16, max_features=log2, n_estimators=100 | 0.4951 | 0.494 | 0.4841 |
| 10 | 2304 | 64 | max_depth=24, max_features=log2, n_estimators=100 | 0.4927 | 0.4909 | 0.4821 |
| 11 | 2304 | none | max_depth=16, max_features=log2, n_estimators=100 | 0.4927 | 0.4909 | 0.4807 |
| 12 | 1536 | none | max_depth=None, max_features=sqrt, n_estimators=100 | 0.4927 | 0.4913 | 0.4804 |

## MELD - Karar Agaci validasyon

| Sira | F | PCA | Hiperparametreler | Val dogr. | Val dengeli dogr. | Val makro-F1 |
|---|---|---|---|---|---|---|
| 1 | 768 | 128 | criterion=gini, max_depth=8, min_samples_split=10 | 0.2512 | 0.2579 | 0.2492 |
| 2 | 768 | 128 | criterion=gini, max_depth=8, min_samples_split=2 | 0.2488 | 0.2549 | 0.2461 |
| 3 | 768 | 64 | criterion=gini, max_depth=None, min_samples_split=2 | 0.2365 | 0.2313 | 0.2299 |
| 4 | 768 | 64 | criterion=gini, max_depth=8, min_samples_split=2 | 0.2291 | 0.2349 | 0.2285 |
| 5 | 768 | 64 | criterion=gini, max_depth=8, min_samples_split=10 | 0.2291 | 0.2348 | 0.2283 |
| 6 | 768 | 64 | criterion=entropy, max_depth=16, min_samples_split=2 | 0.2266 | 0.2221 | 0.2223 |
| 7 | 768 | 64 | criterion=log_loss, max_depth=16, min_samples_split=2 | 0.2266 | 0.2221 | 0.2223 |
| 8 | 768 | 64 | criterion=gini, max_depth=16, min_samples_split=2 | 0.2266 | 0.2218 | 0.2203 |
| 9 | 768 | 64 | criterion=entropy, max_depth=None, min_samples_split=10 | 0.2266 | 0.2177 | 0.217 |
| 10 | 768 | 64 | criterion=entropy, max_depth=16, min_samples_split=10 | 0.2266 | 0.2177 | 0.217 |
| 11 | 768 | 64 | criterion=log_loss, max_depth=None, min_samples_split=10 | 0.2266 | 0.2177 | 0.217 |
| 12 | 768 | 64 | criterion=log_loss, max_depth=16, min_samples_split=10 | 0.2266 | 0.2177 | 0.217 |

## MELD - Gradient Boosting validasyon

| Sira | F | PCA | Hiperparametreler | Val dogr. | Val dengeli dogr. | Val makro-F1 |
|---|---|---|---|---|---|---|
| 1 | 1536 | 64 | learning_rate=0.1, max_depth=3 | 0.2365 | 0.2312 | 0.2268 |
| 2 | 2304 | none | learning_rate=0.05, max_depth=3 | 0.234 | 0.2251 | 0.2234 |
| 3 | 768 | none | learning_rate=0.1, max_depth=3 | 0.2291 | 0.2193 | 0.2204 |
| 4 | 1536 | none | learning_rate=0.1, max_depth=3 | 0.2241 | 0.2172 | 0.2185 |
| 5 | 768 | 64 | learning_rate=0.1, max_depth=1 | 0.2365 | 0.2232 | 0.215 |
| 6 | 1536 | 64 | learning_rate=0.05, max_depth=3 | 0.2192 | 0.2172 | 0.2138 |
| 7 | 768 | 64 | learning_rate=0.05, max_depth=3 | 0.2241 | 0.2157 | 0.212 |
| 8 | 2304 | none | learning_rate=0.1, max_depth=1 | 0.2291 | 0.2216 | 0.2112 |
| 9 | 1536 | none | learning_rate=0.05, max_depth=3 | 0.234 | 0.2157 | 0.2082 |
| 10 | 2304 | 64 | learning_rate=0.1, max_depth=1 | 0.2241 | 0.2135 | 0.2046 |
| 11 | 2304 | 64 | learning_rate=0.05, max_depth=3 | 0.2143 | 0.2061 | 0.202 |
| 12 | 768 | 64 | learning_rate=0.05, max_depth=1 | 0.2241 | 0.2109 | 0.201 |

## MELD - Rastgele Orman validasyon

| Sira | F | PCA | Hiperparametreler | Val dogr. | Val dengeli dogr. | Val makro-F1 |
|---|---|---|---|---|---|---|
| 1 | 2304 | 64 | max_depth=16, max_features=sqrt, n_estimators=100 | 0.2586 | 0.2379 | 0.2349 |
| 2 | 768 | 128 | max_depth=None, max_features=log2, n_estimators=100 | 0.2389 | 0.2323 | 0.234 |
| 3 | 1536 | 64 | max_depth=None, max_features=log2, n_estimators=100 | 0.2537 | 0.2355 | 0.2333 |
| 4 | 768 | 64 | max_depth=24, max_features=sqrt, n_estimators=100 | 0.2537 | 0.2342 | 0.2295 |
| 5 | 2304 | 256 | max_depth=None, max_features=log2, n_estimators=100 | 0.2414 | 0.2277 | 0.2274 |
| 6 | 2304 | 128 | max_depth=None, max_features=sqrt, n_estimators=100 | 0.2611 | 0.2341 | 0.2246 |
| 7 | 1536 | 128 | max_depth=24, max_features=sqrt, n_estimators=100 | 0.2365 | 0.2211 | 0.2228 |
| 8 | 1536 | 128 | max_depth=None, max_features=sqrt, n_estimators=100 | 0.2488 | 0.2264 | 0.2224 |
| 9 | 2304 | 128 | max_depth=16, max_features=log2, n_estimators=100 | 0.2389 | 0.2213 | 0.2202 |
| 10 | 1536 | 64 | max_depth=16, max_features=sqrt, n_estimators=100 | 0.234 | 0.2213 | 0.2198 |
| 11 | 768 | 128 | max_depth=None, max_features=sqrt, n_estimators=100 | 0.234 | 0.22 | 0.2186 |
| 12 | 1536 | 128 | max_depth=16, max_features=log2, n_estimators=100 | 0.234 | 0.2187 | 0.2185 |

## 2. asama modelleri test karsilastirmasi

| Veri seti | Model | F | PCA | Hiperparametreler | Test dogr. | Test dengeli dogr. | Test makro-F1 | Test agirlikli-F1 |
|---|---|---|---|---|---|---|---|---|
| CREMA-D | Gradient Boosting | 1536 | none | learning_rate=0.1, max_depth=3 | 0.5202 | 0.5179 | 0.5144 | 0.5153 |
| CREMA-D | Rastgele Orman | 1536 | none | max_depth=16, max_features=log2, n_estimators=100 | 0.486 | 0.484 | 0.4721 | 0.4717 |
| CREMA-D | Karar Agaci | 1536 | none | criterion=entropy, max_depth=8, min_samples_split=10 | 0.3614 | 0.361 | 0.3496 | 0.3494 |
| MELD | Gradient Boosting | 1536 | 64 | learning_rate=0.1, max_depth=3 | 0.2093 | 0.2072 | 0.2027 | 0.2068 |
| MELD | Rastgele Orman | 2304 | 64 | max_depth=16, max_features=sqrt, n_estimators=100 | 0.2116 | 0.2146 | 0.2006 | 0.2004 |
| MELD | Karar Agaci | 768 | 128 | criterion=gini, max_depth=8, min_samples_split=10 | 0.1465 | 0.1475 | 0.143 | 0.1463 |

## KNN dahil genel test karsilastirmasi

| Veri seti | Model | F | PCA | Hiperparametreler | Test dogr. | Test dengeli dogr. | Test makro-F1 | Test agirlikli-F1 |
|---|---|---|---|---|---|---|---|---|
| CREMA-D | Gradient Boosting | 1536 | none | learning_rate=0.1, max_depth=3 | 0.5202 | 0.5179 | 0.5144 | 0.5153 |
| CREMA-D | Rastgele Orman | 1536 | none | max_depth=16, max_features=log2, n_estimators=100 | 0.486 | 0.484 | 0.4721 | 0.4717 |
| CREMA-D | KNN (Odev 1) | 2304 | 256 | K=15 | 0.4798 | 0.4803 | 0.4657 | 0.4641 |
| CREMA-D | Karar Agaci | 1536 | none | criterion=entropy, max_depth=8, min_samples_split=10 | 0.3614 | 0.361 | 0.3496 | 0.3494 |
| MELD | Gradient Boosting | 1536 | 64 | learning_rate=0.1, max_depth=3 | 0.2093 | 0.2072 | 0.2027 | 0.2068 |
| MELD | Rastgele Orman | 2304 | 64 | max_depth=16, max_features=sqrt, n_estimators=100 | 0.2116 | 0.2146 | 0.2006 | 0.2004 |
| MELD | KNN (Odev 1) | 768 | 128 | K=5 | 0.2047 | 0.2063 | 0.1941 | 0.1938 |
| MELD | Karar Agaci | 768 | 128 | criterion=gini, max_depth=8, min_samples_split=10 | 0.1465 | 0.1475 | 0.143 | 0.1463 |

## Otomatik bulgular

- CREMA-D icin en iyi model Gradient Boosting (test makro-F1=0.5144).
- MELD icin en iyi model Gradient Boosting (test makro-F1=0.2027).
- Genel en iyi sonuc CREMA-D veri setinde Gradient Boosting ile elde edildi (test makro-F1=0.5144).