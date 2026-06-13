# Konuşmadan Duygu Tanıma — Proje Raporu (Şablon)

> **YAP 470 / BİL 570 — Grup 23 — Ahmet Babagil (211101067)**
> Bu şablon, kod çalıştırıldıkça üretilen çıktılarla (tablolar, karışıklık
> matrisleri) doldurulmak üzere hazırlanmıştır. Yer tutucular `〔...〕` ile
> işaretlenmiştir. İlgili görseller `outputs/` altında otomatik üretilir.

## 1. Problem ve Amaç
Kısa bir konuşma kaydından, **yalnızca sesin akustik/prozodik özelliklerine**
(ton, perde, tempo, enerji) dayanarak ifade edilen baskın duygunun otomatik
tanınması. Metin/ASR veya dil modeli **kullanılmaz**. Problem çok-sınıflı,
tek-etiketli bir ses sınıflandırma problemidir.

Ortak 6 duygu: **angry, disgust, fear, happy, neutral, sad**.

## 2. Veri Setleri
| Veri Seti | Tür | Koşul | Kayıt (ortak 6) | Konuşmacı |
|-----------|-----|-------|------------------|-----------|
| CREMA-D | Oyunculu | Stüdyo (kontrollü) | ~7.442 | 91 oyuncu |
| MELD | Diyalog (TV) | Gerçek-ortam | ~10.000 (`surprise` çıkarıldı) | Çok sayıda |

- **Etiket eşlemesi:** CREMA-D `ANG/DIS/FEA/HAP/NEU/SAD` → kanonik 6 sınıf;
  MELD `anger/disgust/fear/joy/neutral/sadness` → kanonik 6 sınıf
  (`joy→happy`, `sadness→sad`), `surprise` ortak sınıflar dışında olduğundan
  atılır.
- **Birleştirilmiş manifest:** `data/processed/manifest.csv`
  (`path, corpus, speaker, split, emotion, label_idx`).
- **CREMA-D sınıf dağılımı (gerçek):** angry/disgust/fear/happy/sad = 1271,
  neutral = 1087 (neutral'de yoğunluk seviyesi olmadığından daha az).
- MELD sınıf dağılımı dengesizdir (neutral baskın) → makro-F1 ve dengeli
  doğruluk raporlanır.

## 3. Yöntem
**Öznitelikler.** (a) Temel model için MFCC özet istatistikleri
(MFCC + Δ + ΔΔ'nin zaman üzerinden ortalama/std → 240 boyutlu vektör);
(b) derin modeller için log-mel spektrogram (`n_mels=64`, 16 kHz).

**Modeller.**
1. **Temel (baseline):** MFCC istatistikleri → StandardScaler → SVM (RBF)
   / LogReg / RandomForest. (sklearn)
2. **CNN:** log-mel spektrogram üzerinde 4 evrişim bloğu + global ortalama
   havuzlama + doğrusal sınıflandırıcı. (PyTorch)
3. **Transfer öğrenme:** ön-eğitimli `wav2vec2-base` üzerine sınıflandırma
   başlığı (ince ayar). (HuggingFace transformers)

**Değerlendirme protokolü.**
- **Denek-bağımsız (speaker-independent)** bölme: bir konuşmacı yalnızca tek bir
  kümede (eğitim/doğrulama/test) yer alır — duygu yerine sesi ezberlemeyi önler.
- **Çapraz-veri-seti (cross-corpus):** A üzerinde eğit, B'nin tamamında test et
  (ve tersi) — alan kayması (domain shift) altında genelleme ölçülür.
- Sınıf dengesizliği için **dengeli sınıf ağırlıkları**; metrikler:
  doğruluk, dengeli doğruluk, **makro-F1**, ağırlıklı-F1, sınıf-bazlı
  kesinlik/duyarlılık/F1 ve **karışıklık matrisi**.

## 4. Deney Düzeni
- Donanım: eğitim RTX 5080 (16 GB, CUDA, AMP); geliştirme CPU.
- Tekrarlanabilirlik: sabit tohum (`seed=42`), kaydedilen `config.yaml`.
- Komutlar: bkz. `README.md`. Tüm matris: `python scripts/run_all.py`.

## 5. Sonuçlar
> `python scripts/aggregate_results.py` → `outputs/results.csv` / `results.md`.

### 5.1 Veri-seti-içi (within-corpus)
| Deney | Doğruluk | Dengeli Doğr. | Makro-F1 |
|-------|----------|----------------|----------|
| baseline_cremad (SVM) | **0.523** | **0.520** | **0.520** |
| cnn_cremad | 〔...〕 | 〔...〕 | 〔...〕 |
| cnn_meld | 〔...〕 | 〔...〕 | 〔...〕 |
| wav2vec2_cremad | 〔...〕 | 〔...〕 | 〔...〕 |

> baseline_cremad gerçek sonuç (denek-bağımsız bölme, yalnızca eğitim kümesinde
> uyarlanmış StandardScaler+SVM, 6 sınıf, şans = %16.7): doğruluk %52.3,
> makro-F1 0.520. Klasik MFCC+SVM için CREMA-D'de tipik aralıktadır.
> Karışıklık matrisi: `outputs/baseline_cremad/test_confusion_matrix.png`.

Karışıklık matrisleri: `outputs/<deney>/test_confusion_matrix.png`.

### 5.2 Çapraz-veri-seti (cross-corpus)
> `outputs/<exp>_crosscorpus/macro_f1_matrix.png` ve `summary.csv`.

| Eğitim → Test | Makro-F1 |
|---------------|----------|
| CREMA-D → CREMA-D | 〔...〕 |
| MELD → MELD | 〔...〕 |
| CREMA-D → MELD | 〔...〕 |
| MELD → CREMA-D | 〔...〕 |

**Beklenti:** Köşegen (veri-seti-içi) değerleri, köşegen-dışı (çapraz) değerlerden
belirgin biçimde yüksektir; bu düşüş, alan kayması probleminin somut kanıtıdır ve
projenin temel bulgusudur.

## 6. Tartışma
- Hangi sınıflar karışıyor? (örn. CREMA-D'de fear↔sad, MELD'de neutral baskınlığı)
- Stüdyo↔gerçek-ortam farkının başarımı nasıl etkilediği.
- Temel model vs CNN vs transfer öğrenme karşılaştırması.
- Çapraz-veri-seti düşüşünün nedenleri (kayıt koşulları, konuşmacı/dil dağılımı,
  etiketleme öznelliği).

## 7. Sonuç
Her iki veri setinde de çalışan, denek-bağımsız ve çapraz-veri-seti değerlendirme
yapan bir konuşmadan duygu tanıma sistemi geliştirildi. 〔Özet bulgular〕.

## 8. Tekrarlanabilirlik
Kod: `〔GitHub bağlantısı〕`. Kurulum ve çalıştırma: `README.md`.
Her deney klasörü `config.yaml`, `history.json`, `test_metrics.json` ve karışıklık
matrisini içerir.
