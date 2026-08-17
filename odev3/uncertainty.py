'''Ayrılmış (held-out) sınıflandırma metrikleri için belirsizlik tahminleri.

Neden belirsizlik? Test kümesi sonlu olduğu için "accuracy = 0.61" gibi tek
bir sayı, gerçek performansın yalnızca bir TAHMİNİDİR. Test kümesi biraz
farklı örneklerden oluşsaydı sayı da biraz farklı çıkardı. Bu modül,
bootstrap (yeniden örnekleme) yöntemiyle bu "örneklem şansı"nın metrikleri
ne kadar oynatabileceğini ölçer ve her metrik için bir güven aralığı üretir.

Temel fikir: elimizdeki (gerçek etiket, tahmin) çiftlerini iadeli olarak
tekrar tekrar örnekleyip her seferinde metriği yeniden hesaplarsak, ortaya
çıkan dağılımın yüzdelikleri bize güven aralığını verir.
'''

from __future__ import annotations

from typing import Any

import numpy as np

from ser.constants import CANONICAL_EMOTIONS, NUM_CLASSES
from ser.evaluate import compute_metrics


# Güven aralığı raporlanan çekirdek metrikler. compute_metrics daha fazlasını
# döndürse de raporda odaklandığımız dört ana skor bunlardır.
CORE_METRICS = (
    'accuracy',
    'balanced_accuracy',
    'macro_f1',
    'weighted_f1',
)


def percentile_bounds(
    samples: np.ndarray,
    confidence: float,
) -> tuple[float, float]:
    '''Bootstrap örnekleri için eşit kuyruklu yüzdelik sınırlarını döndürür.

    "Eşit kuyruklu" (equal-tailed) şu demek: %95 güven için dağılımın alt
    %2.5'ini ve üst %2.5'ini dışarıda bırakıp aradaki aralığı alırız.
    Bu, percentile bootstrap yönteminin standart aralık tanımıdır.
    '''

    sample_array = np.asarray(samples, dtype=np.float64)
    if sample_array.ndim != 1 or len(sample_array) == 0:
        raise ValueError('samples must be a non-empty one-dimensional array.')
    if not 0.0 < confidence < 1.0:
        raise ValueError('confidence must be strictly between 0 and 1.')
    # confidence=0.95 ise tail=0.025 olur; quantile çağrısı tek seferde hem
    # alt hem üst sınırı hesaplar.
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(sample_array, [tail, 1.0 - tail])
    return float(lower), float(upper)


def validate_label_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    '''Hizalanmış, tek boyutlu tamsayı etiket dizileri döndürür.

    Bootstrap'e girmeden önce girdileri sıkıca doğruluyoruz: uzunluklar eşit
    mi, etiketler tamsayı mı, geçerli sınıf aralığında mı? Bu kontroller
    olmadan hatalı bir girdi (örn. olasılık vektörü ya da negatif etiket)
    sessizce yanlış güven aralıkları üretebilirdi.
    '''

    true_array = np.asarray(y_true)
    predicted_array = np.asarray(y_pred)
    if true_array.ndim != 1 or predicted_array.ndim != 1:
        raise ValueError('y_true and y_pred must be one-dimensional.')
    if len(true_array) == 0 or len(true_array) != len(predicted_array):
        raise ValueError(
            'y_true and y_pred must be non-empty and have equal lengths.'
        )
    if not np.issubdtype(true_array.dtype, np.integer):
        raise ValueError('y_true must contain integer class labels.')
    if not np.issubdtype(predicted_array.dtype, np.integer):
        raise ValueError('y_pred must contain integer class labels.')
    if np.any(true_array < 0) or np.any(predicted_array < 0):
        raise ValueError('Class labels cannot be negative.')
    if np.any(true_array >= NUM_CLASSES) or np.any(predicted_array >= NUM_CLASSES):
        raise ValueError(f'Class labels must be smaller than {NUM_CLASSES}.')
    # copy=False: dtype zaten int64 ise gereksiz kopya oluşturma.
    return (
        true_array.astype(np.int64, copy=False),
        predicted_array.astype(np.int64, copy=False),
    )


def stratified_resample_indices(
    labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    '''Gözlenen her sınıfın İÇİNDE iadeli örnekleme yapar.

    Neden tabakalı (stratified)? Düz bootstrap'te şans eseri bir sınıftan hiç
    örnek çekilmeyebilir; o zaman macro-F1 gibi sınıf başına ortalanan
    metrikler tanımsız/çarpık olur. Her sınıftan kendi boyutunda iadeli
    örnek çekerek sınıf dağılımını her bootstrap kopyasında sabit tutuyoruz;
    böylece yalnızca "sınıf içi" örneklem belirsizliğini ölçmüş oluyoruz.
    '''

    label_array = np.asarray(labels)
    if label_array.ndim != 1 or len(label_array) == 0:
        raise ValueError('labels must be a non-empty one-dimensional array.')

    sampled_parts: list[np.ndarray] = []
    for label in np.unique(label_array):
        # O sınıfa ait konumları bul, aynı sayıda konumu iadeli olarak çek.
        class_indices = np.flatnonzero(label_array == label)
        sampled_parts.append(
            rng.choice(class_indices, size=len(class_indices), replace=True)
        )
    # Parçaları tek dizide birleştir; sıralama önemsiz çünkü metrikler
    # örneklerin sırasından bağımsızdır.
    return np.concatenate(sampled_parts).astype(np.int64, copy=False)


def bootstrap_metric_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    '''Tabakalı bootstrap ile held-out metriklerin belirsizliğini tahmin eder.

    Akış:
    1. Girdileri doğrula, her sınıfın test kümesinde bulunduğundan emin ol.
    2. Gözlenen (orijinal) metrikleri bir kez hesapla.
    3. ``iterations`` kez: tabakalı yeniden örnekle, metrikleri hesapla,
       dağılım dizilerine yaz.
    4. Her metrik için yüzdelik sınırlarından güven aralığı çıkar.

    Sonuç sözlüğü, raporda "yöntem + parametreler + aralıklar" şeklinde
    eksiksiz aktarılabilsin diye tüm ayarları da (seed dahil) içerir; bu,
    deneyin tekrarlanabilirliği için önemlidir.
    '''

    true_array, predicted_array = validate_label_arrays(y_true, y_pred)
    # Çok az iterasyonla yüzdelikler güvenilmez olur; 100 alt sınırı kaba
    # bir emniyet eşiğidir (varsayılan 2000 çok daha sağlıklıdır).
    if iterations < 100:
        raise ValueError('iterations must be at least 100.')
    if not 0.0 < confidence < 1.0:
        raise ValueError('confidence must be strictly between 0 and 1.')

    # Her kanonik sınıf test kümesinde en az bir kez görülmeli; aksi halde
    # tabakalı örnekleme o sınıfı hiç üretemez ve metrikler eksik kalır.
    class_counts = np.bincount(true_array, minlength=NUM_CLASSES)
    if np.any(class_counts == 0):
        raise ValueError(
            'Every canonical class must occur in y_true; '
            f'counts={class_counts.tolist()}.'
        )

    # Gözlenen (nokta) tahmin: aralığın merkezine koyacağımız gerçek skor.
    observed = compute_metrics(true_array, predicted_array)
    # Her metrik için iterations uzunluğunda boş dizi ayır; döngüde doldurulur.
    distributions = {
        metric: np.empty(iterations, dtype=np.float64)
        for metric in CORE_METRICS
    }
    # Sabit tohumlu üreteç: aynı girdiyle her çalıştırmada aynı aralıklar
    # çıkar (tekrarlanabilirlik).
    rng = np.random.default_rng(seed)
    for iteration in range(iterations):
        indices = stratified_resample_indices(true_array, rng)
        # Aynı indeks dizisini hem gerçek hem tahmin etiketlerine uyguluyoruz;
        # böylece (y_true, y_pred) çiftleri bozulmadan yeniden örneklenir.
        sampled = compute_metrics(
            true_array[indices],
            predicted_array[indices],
        )
        for metric in CORE_METRICS:
            distributions[metric][iteration] = float(sampled[metric])

    # Dağılımlardan güven aralıklarını çıkar.
    intervals: dict[str, dict[str, float]] = {}
    for metric in CORE_METRICS:
        lower, upper = percentile_bounds(distributions[metric], confidence)
        intervals[metric] = {
            'estimate': float(observed[metric]),
            'lower': lower,
            'upper': upper,
        }

    return {
        'method': 'class-stratified percentile bootstrap',
        'iterations': int(iterations),
        'confidence': float(confidence),
        'seed': int(seed),
        'sample_size': int(len(true_array)),
        'class_counts': {
            emotion: int(class_counts[index])
            for index, emotion in enumerate(CANONICAL_EMOTIONS)
        },
        'metrics': intervals,
    }
