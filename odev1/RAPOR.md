# YAP 470 / BİL 570 – Proje İlerleme Raporu 1

**Grup 23 — Ahmet Babagil (211101067)** · Konu: Konuşmadan Duygu Tanıma
Bu aşamada model: **K-En Yakın Komşu (KNN)**. Akış: Wav2Vec2 (donmuş öznitelik
çıkarıcı) → StandardScaler → PCA → KNN. İncelenen hiperparametreler: **F** (öznitelik
vektör boyutu), **P** (PCA çıktı boyutu), **K** (komşu sayısı).

> Not: ML iş akışında yalnızca **numpy / pandas / scikit-learn** kullanılmıştır.
> torch/transformers SADECE Wav2Vec2 öznitelik çıkarımı aşamasında kullanılır.

---

## Erişim Bilgileri

**Orijinal Veri Seti Adı:** CREMA-D ve MELD (iki ayrı veri seti).

**Orijinal Veri Seti Bağlantısı:**
- CREMA-D: <https://github.com/CheyneyComputerScience/CREMA-D>
  (ses, Hugging Face aynası `AbstractTTS/CREMA-D` ile indirildi — orijinal dosya
  adları korunur)
- MELD: <https://github.com/declare-lab/MELD>
  (ham veri: `http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz`)

**Proje Çalışmasının Yüklendiği Google Drive Bağlantısı:** 〔YÜKLENECEK — link buraya〕
Kod + git geçmişi: <https://github.com/AhmetBabagil/speech-emotion-recognition>
(Link izin istemeden erişilebilir olacak şekilde public yapılacaktır.)

> **Veri seti boyutu notu:** MELD ham verisi (~11 GB) Google Drive'a yüklenemeyecek
> kadar büyük olabilir. Bu durumda veri seti **USB ile ertesi gün** teslim
> edilecektir. (CREMA-D ~0.6 GB; MELD'ten çıkarılan 16 kHz ses ~ birkaç GB.)

**Çalıştırılacak Dosyalar ve Ne Yaptıkları:**

| Dosya | Ne yapar |
|-------|----------|
| `scripts/download_data.py` | CREMA-D + MELD'i indirir; MELD videolarından sesi (ffmpeg) çıkarır |
| `scripts/build_manifest.py` | İki veri setini ortak 6 duyguda birleşik manifest'e (`data/processed/manifest.csv`) yazar |
| `odev1/extract.py` | **Wav2Vec2** ile her klipten öznitelik çıkarır ve diske cache'ler (yavaş, bir kez) |
| `odev1/run_experiment.py` | **KNN+PCA hiperparametre gridini** iki veri setinde koşar; tablolar + karmaşıklık matrisi üretir |

Sıra: `download_data.py` → `build_manifest.py` → `odev1/extract.py` → `odev1/run_experiment.py`.

---

## Veri Seti ve Değerlendirme Metrikleri

**Ortak sınıflar.** İki veri setinin kesişimi olan 6 duygu kullanılır:
`angry, disgust, fear, happy, neutral, sad`. (MELD'deki `surprise` ortak olmadığından
atılır; CREMA-D'de zaten yoktur.) Toplam ~19.512 kayıt (CREMA-D 7.442 + MELD 12.070).

**Eğitim / Geçerleme / Test oluşturma — denek-bağımsız (speaker-independent).**
Her veri seti **ayrı ayrı** bölünür ve bir **konuşmacı yalnızca tek bir kümede** yer
alır (CREMA-D'de oyuncu kimliği, MELD'de konuşmacı adı). Bu, modelin duygu yerine
sesi/konuşmacıyı ezberlemesini engeller. Konuşmacılar sabit tohumla (seed=42)
karıştırılıp bölünür: **%70 eğitim / %15 geçerleme / %15 test** (konuşmacı bazında).
Geçerleme kümesi hiperparametre (F, P, K) seçimi için; test kümesi yalnızca en iyi
yapılandırmanın nihai ölçümü için kullanılır (test hiçbir seçimde kullanılmaz).

**Değerlendirme metrikleri.** Sınıflandırma problemi olduğundan: **doğruluk**, **dengeli
doğruluk**, **makro-F1** (sınıf dengesizliği için), **ağırlıklı-F1** ve **karmaşıklık
matrisi**. MELD sınıf dağılımı dengesiz (neutral baskın) olduğundan makro-F1 ve dengeli
doğruluk öne çıkarılır. (Şablondaki MSE ve X=yıllık kazanç sütunları sırasıyla
regresyon ve finans projeleri içindir; bu sınıflandırma görevine uygulanmaz.)

---

## Yöntem

### Öznitelik çıkarımı (PCA öncesi — ayrıntılı)
1. **Ses okuma.** Her kayıt 16 kHz, tek kanal (mono) olarak okunur (librosa). Çok uzun
   klipler en fazla **6 sn** olacak şekilde merkezden kırpılır; çok kısa olanlar
   Wav2Vec2'nin minimum girdi uzunluğuna ped'lenir.
2. **Normalizasyon.** Her kayıt kendi içinde sıfır-ortalama / birim-varyans'a
   normalize edilir (Wav2Vec2'nin beklediği biçim).
3. **Wav2Vec2 (donmuş).** Ön-eğitimli `facebook/wav2vec2-base` modeli **ince ayar
   YAPILMADAN**, yalnızca öznitelik çıkarıcı olarak kullanılır. Her klip için son gizli
   katman dizisi `[T, 768]` elde edilir.
4. **Zaman havuzlama.** Bu dizi zaman ekseninde üç istatistikle havuzlanır:
   ortalama (mean), standart sapma (std) ve maksimum (max). Birleştirilerek
   `[mean | std | max] = 3·768 = 2304` boyutlu bir vektör diske kaydedilir (cache).
5. **Öznitelik boyutu F (hiperparametre-1).** Bu cache dilimlenerek üç farklı boyut
   elde edilir:
   - **F = 768** : yalnız mean
   - **F = 1536** : mean + std
   - **F = 2304** : mean + std + max

### PCA ve sınıflandırma
6. **Ölçekleme.** `StandardScaler` **yalnızca eğitim** kümesine uydurulur, geçerleme/test
   ona göre dönüştürülür (test istatistiği sızıntısı yok).
7. **PCA (hiperparametre-2, P).** Eğitim kümesine uydurulur; çıktı boyutu P taranır:
   **PCA yok / 32 / 64 / 128 / 256**. (Önce PCA'sız, sonra PCA ile.)
8. **KNN (hiperparametre-3, K).** `KNeighborsClassifier`, K ∈ {1, 3, 5, 7, 11, 15, 21,
   31}. Geçerleme makro-F1'ine göre en iyi (F, P, K) seçilir; en iyi yapılandırma
   eğitim+geçerleme üzerine yeniden uydurulup test edilir.

Toplam ızgara: 3 (F) × 5 (P) × 8 (K) = **120 kombinasyon / veri seti**.

---

## Deney Sonuçları Tablosu

Tam ızgara (3 F × 5 P × 8 K = 120 kombinasyon / veri seti)
`odev1/outputs/<veri seti>/validation_grid.csv` içindedir. Aşağıda her **(F, P)**
ikilisi için **en iyi K** satırı (geçerleme makro-F1'ine göre) özetlenmiştir.
P = `none` PCA kullanılmadığı durumdur.

**CREMA-D — geçerleme (her F×P için en iyi K):**

| Deney | F | P | K | Doğruluk | Makro-F1 |
|-------|---|---|---|----------|----------|
| 1 | 768 | 128 | 15 | 0.476 | 0.463 |
| 2 | 768 | 256 | 15 | 0.466 | 0.454 |
| 3 | 768 | 64 | 15 | 0.466 | 0.453 |
| 4 | 768 | none | 15 | 0.461 | 0.450 |
| 5 | 768 | 32 | 31 | 0.454 | 0.438 |
| 6 | 1536 | none | 31 | 0.461 | 0.445 |
| 7 | 1536 | 256 | 31 | 0.459 | 0.441 |
| 8 | 1536 | 32 | 31 | 0.454 | 0.438 |
| 9 | 1536 | 128 | 15 | 0.446 | 0.428 |
| 10 | 1536 | 64 | 15 | 0.437 | 0.421 |
| 11 | **2304** | **256** | **15** | **0.481** | **0.468** |
| 12 | 2304 | none | 15 | 0.478 | 0.463 |
| 13 | 2304 | 128 | 15 | 0.473 | 0.459 |
| 14 | 2304 | 64 | 21 | 0.456 | 0.443 |
| 15 | 2304 | 32 | 31 | 0.451 | 0.434 |

**MELD — geçerleme (her F×P için en iyi K):**

| Deney | F | P | K | Doğruluk | Makro-F1 |
|-------|---|---|---|----------|----------|
| 1 | **768** | **128** | **5** | **0.256** | **0.246** |
| 2 | 768 | 256 | 5 | 0.259 | 0.243 |
| 3 | 768 | none | 5 | 0.249 | 0.233 |
| 4 | 768 | 32 | 5 | 0.244 | 0.229 |
| 5 | 768 | 64 | 5 | 0.239 | 0.225 |
| 6 | 1536 | none | 21 | 0.246 | 0.223 |
| 7 | 1536 | 32 | 21 | 0.239 | 0.218 |
| 8 | 1536 | 256 | 15 | 0.232 | 0.217 |
| 9 | 1536 | 128 | 15 | 0.229 | 0.213 |
| 10 | 1536 | 64 | 15 | 0.224 | 0.208 |
| 11 | 2304 | 256 | 15 | 0.239 | 0.226 |
| 12 | 2304 | none | 21 | 0.237 | 0.219 |
| 13 | 2304 | 64 | 21 | 0.237 | 0.217 |
| 14 | 2304 | 128 | 15 | 0.232 | 0.217 |
| 15 | 2304 | 32 | 5 | 0.217 | 0.210 |

---

## Model Geçerleme

Hiperparametrelerin geçerleme başarımına etkisi (yukarıdaki tablolardan):

- **Öznitelik boyutu F.** CREMA-D'de en yüksek başarım **F = 2304** (mean+std+max)
  ile elde edildi; daha zengin istatistikler kontrollü/temiz kayıtlarda yardımcı
  oluyor. MELD'de ise tersine **F = 768** (yalnız mean) en iyi; gerçek-ortam
  gürültüsü altında ek istatistikler (std, max) gürültü taşıyıp KNN'i bozuyor
  (yüksek boyutta "boyut laneti" + sınırlı örnek).
- **PCA boyutu P.** Her iki veri setinde de **orta düzey PCA (128–256)** en iyi ya
  da PCA'sıza çok yakın; **agresif indirgeme (P = 32)** başarımı düşürüyor (bilgi
  kaybı). PCA'nın asıl faydası boyut/hız: 2304 → 256 ile ~%89 boyut azalması, başarım
  neredeyse korunarak (hatta hafif artarak) sağlanıyor.
- **Komşu sayısı K.** CREMA-D **K = 15** (orta), MELD **K = 5** (küçük) tercih ediyor.
  K = 1 her ikisinde de gürültüye duyarlı (aşırı uyum); çok büyük K (31) sınırları
  bulanıklaştırıyor. En iyi K, veri/ayrışabilirliğe göre orta bir değerde.

**Test karşılaştırma tablosu (her veri setinin en iyisi):**

| Veri seti | F | P | K | Doğruluk | Dengeli doğr. | Makro-F1 |
|-----------|---|---|---|----------|----------------|----------|
| CREMA-D | 2304 | 256 | 15 | **0.480** | 0.480 | **0.466** |
| MELD | 768 | 128 | 5 | 0.205 | 0.206 | 0.194 |

**Genel en iyi sonuç — karmaşıklık matrisi:** CREMA-D (makro-F1 0.466).
`odev1/figures/overall_best_confusion.png` (ayrıca her veri setinin kendi matrisi
`odev1/figures/<veri seti>_confusion.png`).

---

## Sonuç ve Değerlendirme

Bu aşamada, Wav2Vec2 (donmuş) öznitelikleri üzerinde **KNN** ile konuşmadan duygu
tanıma kuruldu ve üç hiperparametre (F, P, K) iki veri setinde ayrı ayrı tarandı.

- **KNN başarımı (denek-bağımsız, şans = %16.7).** CREMA-D'de en iyi yapılandırma
  test makro-F1 **0.466** (doğruluk %48.0) verdi — Wav2Vec2 özniteliklerinin
  duygu bilgisini taşıdığını, basit bir KNN ile bile şansın ~3 katı başarıya
  ulaşıldığını gösteriyor. MELD'de en iyi makro-F1 **0.194** — gerçek-ortam
  koşulları (arka plan gürültüsü, değişken kayıt) ve güçlü sınıf dengesizliği
  (neutral baskın) nedeniyle belirgin biçimde daha zor.
- **Veri setleri arası fark.** CREMA-D (kontrollü/stüdyo) ile MELD (gerçek-ortam)
  arasındaki büyük başarım farkı, kayıt koşullarının duygu tanımayı doğrudan
  etkilediğini ortaya koyuyor; bu, bir sonraki aşamalarda incelenecek alan-kayması
  (domain shift) konusunun ön göstergesidir.
- **Hiperparametre çıkarımları.** (i) Öznitelik boyutu veri setine göre farklı
  optimum veriyor: temiz veride zengin (2304), gürültülü veride sade (768) öznitelik
  daha iyi. (ii) Orta düzey PCA boyut/başarım dengesini iyi kuruyor; agresif indirgeme
  zararlı. (iii) Orta bir K (5–15) en iyi; aşırı küçük/büyük K başarımı düşürüyor.
- **Metrik notu.** MELD dengesiz olduğundan doğruluk yanıltıcıdır; **makro-F1** ve
  **dengeli doğruluk** gerçek başarımı yansıtır (karmaşıklık matrisinde tahminlerin
  baskın sınıfa kayması görülür).

> **Kapsam notu (ilerleme raporu).** Bu ilk aşamada KNN hiperparametre çalışması,
> hesaplama süresini sınırlamak için her veri setinden **dengeli bir altküme**
> (CREMA-D 2.480 + MELD 2.719 = 5.199 kayıt) üzerinde, denek-bağımsız bölmeyle
> yürütülmüştür. Wav2Vec2 öznitelik çıkarımı tüm veri için yeniden çalıştırılabilir
> (resumable cache) ve sonraki aşamalarda tam veriye genişletilecektir.
