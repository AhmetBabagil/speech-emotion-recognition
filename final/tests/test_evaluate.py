# evaluate.py metrik hesabı için birim testleri.
#
# En önemlisi: dengesiz veride "hep çoğunluk sınıfını söyleyen" bir modelin yüksek DOĞRULUK ama düşük MACRO-F1 aldığını kanıtlar — yani projede neden model seçimini macro-F1 ile yaptığımızı somut olarak gösterir.

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

from pathlib import Path  # dosya yolları
import sys  # import yolu

import numpy as np  # diziler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # proje kökünü import yoluna ekle

from ser.constants import CANONICAL_EMOTIONS, NUM_CLASSES  # noqa: E402  # duygu adları + sınıf sayısı
from ser.evaluate import compute_metrics  # noqa: E402  # metrik hesabı


def test_perfect_predictions() -> None:  # Tüm tahminler doğruysa: doğruluk ve macro-F1 = 1.
    y = np.array([0, 1, 2, 3, 4, 5, 0, 1])  # gerçek etiketler (6 sınıf da var)
    m = compute_metrics(y, y)  # tahmin = gerçek
    assert m['accuracy'] == 1.0  # doğruluk tam
    assert m['macro_f1'] == 1.0  # macro-F1 tam
    assert m['balanced_accuracy'] == 1.0  # dengeli doğruluk tam


def test_majority_baseline_is_high_accuracy_low_macro_f1() -> None:  # "%90 tuzağı": hep çoğunluğu söylemek yüksek doğruluk ama düşük macro-F1 verir.
    y_true = np.array([0] * 90 + [1] * 10)  # dengesiz: 90 sınıf-0, 10 sınıf-1
    y_pred = np.zeros(100, dtype=int)  # model hep "0" der
    m = compute_metrics(y_true, y_pred)  # metrikler
    assert m['accuracy'] > 0.85  # doğruluk yüksek (yanıltıcı)
    assert m['macro_f1'] < 0.30  # macro-F1 düşük (gerçeği söyler)
    assert m['balanced_accuracy'] < 0.60  # dengeli doğruluk da düşük
    # DERS: doğruluk kanar, macro-F1 kanmaz -> model seçimi macro-F1 ile yapılır.


def test_confusion_matrix_always_six_by_six() -> None:  # Karışıklık matrisi, bazı sınıflar eksik olsa bile hep 6x6 olmalı.
    y_true = np.array([0, 0, 1, 1])  # yalnız 2 sınıf mevcut
    y_pred = np.array([0, 1, 1, 0])  # yine 2 sınıf
    m = compute_metrics(y_true, y_pred)  # metrikler
    cm = np.asarray(m['confusion_matrix'])  # matrisi al
    assert cm.shape == (NUM_CLASSES, NUM_CLASSES)  # 6x6 (indeksler duygularla hizalı kalır)


def test_per_class_has_all_emotions() -> None:  # Sınıf-bazı sözlük altı duygunun hepsini içermeli.
    y = np.array([0, 1, 2, 3, 4, 5])  # her sınıftan bir örnek
    m = compute_metrics(y, y)  # metrikler
    assert set(m['per_class'].keys()) == set(CANONICAL_EMOTIONS)  # 6 duygu da var
    for emotion in CANONICAL_EMOTIONS:  # her duygu için
        assert 'f1' in m['per_class'][emotion]  # F1 alanı olmalı
