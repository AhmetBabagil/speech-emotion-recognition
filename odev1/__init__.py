"""Ödev 1 / Project_Assignment_1 — K-En Yakın Komşu (KNN) aşaması.

Konuşmadan duygu tanıma (CREMA-D + MELD) için, bu aşamada model **KNN**'dir.
Akış: Wav2Vec2 (donmuş, yalnızca öznitelik çıkarıcı) → StandardScaler → PCA → KNN.
İncelenen hiperparametreler: (1) öznitelik vektör boyutu, (2) PCA boyutu, (3) K.

Kural: Makine öğrenimi iş akışında yalnızca numpy / pandas / scikit-learn
kullanılır. torch / transformers SADECE öznitelik çıkarımı aşamasında kullanılır
(features_w2v.py içinde, izinli).
"""
