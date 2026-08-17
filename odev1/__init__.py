"""Ödev 1 / Project_Assignment_1 — K-En Yakın Komşu (KNN) aşaması.

Bu paket, konuşmadan duygu tanıma (Speech Emotion Recognition) projesinin ilk
ödevine ait tüm kodu bir arada tutar. Kullanılan veri setleri CREMA-D ve MELD,
bu aşamada kullanılan sınıflandırıcı ise **KNN**'dir.

Boru hattının (pipeline) genel akışı şöyledir:

    Wav2Vec2 (donmuş, yalnızca öznitelik çıkarıcı)  →  StandardScaler  →  PCA  →  KNN

Yani ham ses dosyaları önce Wav2Vec2 ile sabit boyutlu sayısal vektörlere
çevrilir; bu vektörler ölçeklenir, istenirse PCA ile boyutu indirgenir ve en
sonunda KNN ile duygu sınıfı tahmin edilir.

İncelenen üç hiperparametre:
  1. Öznitelik vektör boyutu (F)  — Wav2Vec2 çıktısının nasıl havuzlandığına bağlı,
  2. PCA çıktı boyutu (P)         — boyut indirgeme uygulanıp uygulanmayacağı ve kaça,
  3. Komşu sayısı (K)             — KNN'de oy kullanan en yakın örnek sayısı.

Önemli kural: Makine öğrenimi iş akışında yalnızca numpy / pandas / scikit-learn
kullanılır. torch / transformers SADECE öznitelik çıkarımı aşamasında kullanılır
(features_w2v.py içinde, ödev kurallarına göre izinli olan tek yer).

Paketteki dosyaların görev dağılımı:
  * ``extract.py``       — komut satırından öznitelik çıkarımını başlatır,
  * ``features_w2v.py``  — Wav2Vec2 ile vektör üretimi ve disk önbelleği (cache),
  * ``knn_pipeline.py``  — hiperparametre taraması, model seçimi ve test ölçümü,
  * ``evaluation.py``    — metrik hesapları ve karmaşıklık matrisi çizimi,
  * ``run_experiment.py``— tüm deneyi tek komutla çalıştıran giriş noktası,
  * ``build_report.py``  — çıktılardan rapora yapıştırılacak tabloları üretir.
"""
