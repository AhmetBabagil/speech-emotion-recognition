# Final raporu için kaynakça notları

İnternet taramasıyla derlenen, iki yöntemimizle doğrudan ilgili çalışmalar ve
rapora aktarılacak beklenti aralıkları. (Derleme tarihi: 2026-08-17)

## Kaynaklar

1. Cao, Cooper, Keutmann, Gur, Nenkova, Verma (2014). "CREMA-D: Crowd-sourced
   Emotional Multimodal Actors Dataset." IEEE Trans. Affective Computing
   5(4):377-390. doi:10.1109/TAFFC.2014.2336244.
   Veri seti makalesi — 7.442 kayıt, 91 aktör, 6 duygu. İnsanların yalnız sesten
   tanıma başarısı %40,9.
2. Satt, Rozenberg, Hoory (2017). "Efficient Emotion Recognition from Speech
   Using Deep Learning on Spectrograms." Interspeech 2017, 1089-1093.
   Mel spectrogram üzerinde sıfırdan CNN / CNN-LSTM; IEMOCAP 4 sınıf,
   konuşmacı-bağımsız: %66 (CNN), %68 (CNN-LSTM).
3. Etienne ve ark. (2018). "CNN+LSTM Architecture for Speech Emotion
   Recognition with Data Augmentation." arXiv:1802.05630.
   Spectrogram + CNN-BLSTM, IEMOCAP: %64,5 WA / %61,7 UA; veri artırma ve sınıf
   dengeleme tartışması.
4. Ristea, Ionescu (2021). "Self-Paced Ensemble Learning for Speech and Audio
   Classification." Interspeech 2021. arXiv:2103.11988.
   Sıfırdan eğitilen 5'li ResNet-18 topluluğu: CREMA-D 6 sınıf %68,12 —
   önceden eğitimsiz modeller için yayınlanmış en iyi sonuçlardan.
5. "Evaluating raw waveforms with deep learning frameworks for speech emotion
   recognition" (2023). arXiv:2307.02820.
   CREMA-D 6 sınıf, 80/20 bölme: CNN %69,72, LSTM %61,59, CNN-LSTM %63,72 —
   CNN ile LSTM'in aynı veri üzerinde doğrudan karşılaştırması.
6. Mirsamadi, Barsoum, Zhang (2017). "Automatic speech emotion recognition
   using recurrent neural networks with local attention." ICASSP 2017.
   Çerçeve-düzeyi öznitelikler (F0, enerji, ZCR, MFCC+delta) üzerinde BLSTM +
   dikkat havuzlama; IEMOCAP konuşmacı-bağımsız %58,8 UA. Yöntem 2'nin
   literatürdeki birebir karşılığı.
7. Lee, Tashev (2015). "High-level feature representation using recurrent
   neural network for speech emotion recognition." Interspeech 2015, 1537-1540.
   Çerçeve-düzeyi öznitelikler üzerinde BLSTM; DNN tabana karşı %12'ye varan
   WA artışı.
8. Kesim, Helli, Cavsak (2023). "A Comparison of Time-based Models for
   Multimodal Emotion Recognition." arXiv:2306.13076 (Türkçe).
   CREMA-D ses için CNN öznitelik + GRU/LSTM/Transformer karşılaştırması;
   en iyi F1 = 0,640 (GRU).
9. Eyben ve ark. (2016). "The Geneva Minimalistic Acoustic Parameter Set
   (GeMAPS) for Voice Research and Affective Computing." IEEE TAFFC 7(2).
   Segment başına istatistiksel fonksiyonellerle standart akustik öznitelik
   kümesi — Yöntem 2'deki aralık-başına ortalama/std yaklaşımının dayanağı.
10. Park ve ark. (2019). "SpecAugment." Interspeech 2019. arXiv:1904.08779.
    Log-mel üzerinde zaman/frekans maskeleme — önceden eğitilmiş model
    gerektirmeyen veri artırma (olası geliştirme adımı).
11. Ma ve ark. (2024). "EmoBox: Multilingual Multi-corpus Speech Emotion
    Recognition Toolkit and Benchmark." Interspeech 2024. arXiv:2406.07162.
    Standart konuşmacı-bağımsız katmanlar; CREMA-D'de büyük önceden eğitilmiş
    modeller bile 58-77 UA bandında (wav2vec2-base 62,0; HuBERT-large 73,8).
12. "Speech emotion recognition with light weight deep neural ensemble model
    using hand crafted features." Scientific Reports (2025). PMC11977261.
    Rastgele 80/10/10 bölmeyle %98,66 iddiası — konuşmacı-bağımsız sonuçlarla
    KARŞILAŞTIRILAMAZ; şişirilmiş bölme rejimine örnek olarak anılmalı.

## Tasarım kararlarının literatür karşılıkları

- Log-mel ayarları (25 ms pencere / ~10-16 ms atlama, 64-128 mel, log genlik,
  sabit uzunlukta kırpma/dolgu): [2, 3] ile uyumlu.
- Küçük sıfırdan CNN yeterli; CREMA-D'nin 7,4k kaydında büyük ağlar aşırı
  öğreniyor: [2, 4].
- Yöntem 2 öznitelik kümesi (MFCC+delta, RMS enerji, ZCR, spektral
  centroid/rolloff; aralık başına ortalama/std): [6, 9, 12] standardı.
- RNN boyutu: 1-2 katman BiLSTM/GRU, 128-256 birim; zaman üzerinde ortalama
  veya dikkat havuzlama (dikkat, son-adım gizli durumdan iyi): [6, 7].
- Sınıf-ağırlıklı cross-entropy + geçerleme macro-F1/UA üzerinde erken durdurma:
  [3, 11]. CREMA-D hafif dengesiz (nötr 1.087'ye karşı 1.271/duygu).
- Konuşmacı-bağımsız (aktör kimliğine göre) bölme şart; kayıt-bazlı rastgele
  bölme sonuçları şişirir: [11, 12].
- Önceden eğitim gerektirmeyen veri artırma seçenekleri: SpecAugment maskeleme
  [10], Gauss gürültüsü / perde kaydırma [5, 12], VTLP [3]. Yalnız eğitim
  katmanına, bölmeden SONRA uygulanmalı.

## Beklenen sonuç aralıkları (CREMA-D, 6 sınıf, konuşmacı-bağımsız, sıfırdan)

- Taban çizgileri: şans ~%17; insan (yalnız ses) %40,9 [1].
- Yöntem 1 (mel + CNN): %55-65 doğruluk / macro-F1 0,50-0,62 sağlam sonuç;
  %68 civarı yayınlanmış tavana yakın (toplulukla alındı) [4].
- Yöntem 2 (aralık öznitelikleri + LSTM/GRU): tipik olarak CNN'in 3-8 puan
  altı — %45-58 beklenir [5, 6, 8].
- Bağlam: önceden eğitilmiş dev modeller bile konuşmacı-bağımsız CREMA-D'de
  62-77 UA [11]; sıfırdan %55 üstü gerçekten iyi. %85+ iddiaları rastgele
  bölme/sızıntı ürünü [12].
