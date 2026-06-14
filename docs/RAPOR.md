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
| baseline_cremad (MFCC+SVM) | 0.523 | 0.520 | 0.520 |
| **cnn_cremad (log-mel CNN)** | **0.555** | **0.554** | **0.557** |
| cnn_meld (log-mel CNN, denek-bağımsız) | 0.304 | 0.253 | 0.206 |
| wav2vec2_cremad | 〔GPU'da çalıştırılacak〕 | | |

> **Temel model vs CNN (CREMA-D, denek-bağımsız, şans = %16.7):** Klasik MFCC+SVM
> temel modeli makro-F1 **0.520** verirken, log-mel spektrogram üzerinde eğitilen
> CNN makro-F1 **0.557**'ye çıkar (~+3.7 puan) — derin modelin beklenen üstünlüğü.
> CNN en kolay sınıf `angry` (F1 0.65), en zor `fear` (F1 0.47); 22 epoch, en iyi
> epoch 14 (erken durdurma). cnn_meld ve wav2vec2 derin eğitimleri RTX 5080
> makinesinde çalıştırılmak üzere hazırdır (bu geliştirme makinesi yalnızca CPU).

CREMA-D temel model (solda) ve CNN (sağda) karışıklık matrisleri:

![CREMA-D temel model](figures/baseline_cremad_confusion.png)
![CREMA-D CNN](figures/cnn_cremad_confusion.png)

### 5.2 Çapraz-veri-seti (cross-corpus) — GERÇEK SONUÇLAR
İki model için de denek-bağımsız çapraz-veri-seti matrisi (makro-F1):

| Eğitim → Test | MFCC+LogReg | log-mel CNN |
|---------------|-------------|-------------|
| CREMA-D → CREMA-D (içi) | 0.502 | **0.557** |
| MELD → MELD (içi) | 0.188 | **0.206** |
| **CREMA-D → MELD (çapraz)** | 0.083 | 0.100 |
| **MELD → CREMA-D (çapraz)** | 0.191 | 0.105 |

> Görseller: `figures/crosscorpus_macro_f1.png` (LogReg),
> `figures/cnn_crosscorpus_macro_f1.png` (CNN); ham veriler
> `outputs/*_crosscorpus/summary.csv`.

**Bulgu 1 — alan kayması (projenin temel katkısı):** Her iki modelde de köşegen
(veri-seti-içi) değerleri köşegen-dışı (çapraz) değerlerden belirgin biçimde
yüksektir. CNN, CREMA-D'de makro-F1 **0.557** elde ederken aynı model MELD'de
**0.100**'e çöker (~5.6 kat düşüş). Bu, stüdyo→gerçek-ortam genellemesinin somut,
ölçülmüş kanıtıdır ve önerideki "literatürde kolayca yükselmeyen zorlu problem"
tezini doğrular.

**Bulgu 2 — daha güçlü model alan kaymasını çözmez:** CNN, veri-seti-içi başarımı
LogReg'e göre belirgin artırır (CREMA-D 0.502→0.557, MELD 0.188→0.206); ancak
çapraz-veri-seti başarımı düşük kalır, hatta MELD→CREMA-D yönünde CNN (0.105)
LogReg'den (0.191) **daha kötü** genelleştirir. Yani daha yüksek kapasite, alana
özgü ipuçlarını daha çok ezberleyip çapraz-alan genellemesini iyileştirmeyebilir —
bu, alan kaymasının kapasiteyle çözülmediğini gösteren öğretici bir sonuçtur.

![CNN çapraz-veri-seti makro-F1 matrisi](figures/cnn_crosscorpus_macro_f1.png)

**Bulgu 3 — metrik seçimi:** MELD kendi içinde de zordur (gerçek-ortam koşulları +
neutral baskınlığı). Bu nedenle doğruluk yerine **makro-F1** ve **dengeli doğruluk**
raporlanması kritiktir (her şeye "neutral" demek yüksek doğruluk ama düşük makro-F1
verir).

## 6. Tartışma
- Hangi sınıflar karışıyor? (örn. CREMA-D'de fear↔sad, MELD'de neutral baskınlığı)
- Stüdyo↔gerçek-ortam farkının başarımı nasıl etkilediği.
- Temel model vs CNN vs transfer öğrenme karşılaştırması.
- Çapraz-veri-seti düşüşünün nedenleri (kayıt koşulları, konuşmacı/dil dağılımı,
  etiketleme öznelliği).

## 7. Sonuç
Her iki veri setinde (CREMA-D + MELD, ~19.500 kayıt, ortak 6 duygu) çalışan,
**denek-bağımsız** ve **çapraz-veri-seti** değerlendirme yapan bir konuşmadan duygu
tanıma sistemi geliştirildi. Başlıca bulgular:

1. **Model ilerlemesi işe yarıyor:** CREMA-D'de MFCC+SVM temel modeli (makro-F1
   0.520) → log-mel CNN (makro-F1 0.557). Derin model klasik temeli geçti.
2. **Çapraz-veri-seti genelleme zordur (projenin temel bulgusu):** CREMA-D üzerinde
   eğitilen model kendi test kümesinde makro-F1 0.502 elde ederken MELD üzerinde
   0.083'e çöker — stüdyo→gerçek-ortam alan kaymasının somut, ölçülmüş kanıtı.
3. **Metrik seçimi kritiktir:** MELD'in sınıf dengesizliği nedeniyle doğruluk
   yanıltıcıdır; makro-F1 ve dengeli doğruluk gerçek başarımı gösterir.

Derin MELD-içi ve wav2vec2 transfer-öğrenme deneyleri (GPU gerektirir) RTX 5080
makinesinde aynı kod ve konfigürasyonlarla çalıştırılmaya hazırdır.

## 8. Tekrarlanabilirlik ve doğrulama
Kod: <https://github.com/AhmetBabagil/speech-emotion-recognition>.
Kurulum ve çalıştırma: `README.md`. Sabit tohum (seed=42), kaydedilen `config.yaml`,
tohumlanmış veri artırma ve DataLoader üreteci ile sonuçlar tekrarlanabilir. Her
deney klasörü `config.yaml`, `history.json`, `test_metrics.json` ve karışıklık
matrisini içerir; tüm sonuçlar `scripts/aggregate_results.py` ile toplanır.

**Doğrulama notu:** Kod, çok-ajanlı düşmanca denetimden geçirildi. Bu denetimlerde,
öznitelik önbelleğinin (cache) yalnızca dosya adıyla anahtarlanması nedeniyle MELD'in
bölme başına yeniden başlayan `dia{D}_utt{U}` kimliklerinin çakışabileceği bir hata
bulundu ve düzeltildi (anahtar artık bölme klasörünü içerir). Tüm MELD/çapraz-veri-seti
deneyleri düzeltilmiş önbellekle yeniden koşuldu; sonuçlar **değişmedi** (MELD başarımı
zaten taban seviyesine yakın olduğundan öznitelik gürültüsü makro-F1'i etkilemedi),
böylece raporlanan sayılar doğrulanmış oldu. (CREMA-D dosya adları benzersiz olduğundan
hiç etkilenmemişti.)
