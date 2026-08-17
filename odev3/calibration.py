'''Ayrılmış (held-out) çok sınıflı tahminler için olasılık kalibrasyonu metrikleri.

Kalibrasyon nedir? Bir model "%80 eminim" dediğinde gerçekten yüzde 80
oranında haklı çıkıyorsa o model "kalibre"dir. Sinir ağları çoğu zaman
AŞIRI ÖZGÜVENLİDİR: doğruluk oranlarından daha yüksek güven skorları
üretirler. Bu modül iki iş yapar:

1. **Ölçme**: NLL, Brier skoru, ECE (beklenen kalibrasyon hatası) ve güven
   dilimi (bin) istatistikleriyle kalibrasyon kalitesini sayısallaştırır.
2. **Düzeltme**: Sıcaklık ölçekleme (temperature scaling) — logitleri tek
   bir T sayısına bölerek güveni yumuşatan/keskinleştiren, tahmin edilen
   sınıfı ASLA değiştirmeyen en basit kalibrasyon yöntemi. T doğrulama
   kümesinde NLL'i en aza indirerek seçilir.

Dosyadaki tüm fonksiyonlar saf numpy ile yazılmıştır ve girdilerini çok
sıkı doğrular; çünkü kalibrasyon metrikleri geçersiz olasılıklarla sessizce
anlamsız sayılar üretebilir.
'''

from __future__ import annotations

from typing import Any

import numpy as np

from ser.constants import NUM_CLASSES


# Satır toplamlarının 1'den sapmasına izin verilen tolerans: float
# aritmetiğinde toplamlar tam 1.0 çıkmayabilir, ufak sapma normaldir.
ROW_SUM_TOLERANCE = 1e-6
# log(0) = -sonsuz felaketini önlemek için olasılıklara uygulanan taban.
PROBABILITY_FLOOR = 1e-12
# Sıcaklık aramasının makul sınırları: 0.25 (çok keskinleştir) ile
# 10 (çok yumuşat) dışındaki değerler pratikte anlamsızdır.
MIN_TEMPERATURE = 0.25
MAX_TEMPERATURE = 10.0


def validate_probability_matrix(probabilities: np.ndarray) -> np.ndarray:
    '''Sonlu, normalize edilmiş float64 çok sınıflı olasılık matrisi döndürür.

    Kontrol listesi: 2 boyutlu mu, sütun sayısı sınıf sayısına eşit mi, boş
    değil mi, sayısal mı, tüm değerler [0,1] aralığında ve sonlu mu, her
    satır 1'e toplanıyor mu? Bu kontroller "olasılık gibi görünen ama
    olmayan" girdileri (örn. ham logitler) en baştan reddeder.
    '''

    probability_array = np.asarray(probabilities)
    if probability_array.ndim != 2:
        raise ValueError('probabilities must be a two-dimensional matrix.')
    if probability_array.shape[1] != NUM_CLASSES:
        raise ValueError(
            f'probabilities must contain exactly {NUM_CLASSES} class columns.'
        )
    if probability_array.shape[0] == 0:
        raise ValueError('probabilities and labels cannot be empty.')
    if not np.issubdtype(probability_array.dtype, np.number):
        raise ValueError('probabilities must contain numeric values.')

    # float64'e yükselt: kalibrasyon hesapları küçük farklara duyarlıdır,
    # float32 hassasiyeti yetersiz kalabilir.
    probability_array = probability_array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(probability_array)):
        raise ValueError('probabilities must contain only finite values.')
    if np.any(probability_array < 0.0) or np.any(probability_array > 1.0):
        raise ValueError('probabilities must remain between 0 and 1.')

    row_sums = probability_array.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=ROW_SUM_TOLERANCE, rtol=0.0):
        raise ValueError('each probability row must sum to 1.')
    return probability_array


def validate_probability_inputs(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    '''Hizalanmış float64 olasılıklar ve tamsayı sınıf etiketleri döndürür.

    Olasılık matrisinin doğrulamasına ek olarak etiketlerin de matrisle aynı
    uzunlukta, tamsayı ve geçerli sınıf aralığında olduğunu garanti eder.
    '''

    probability_array = validate_probability_matrix(probabilities)
    label_array = np.asarray(labels)
    if label_array.ndim != 1:
        raise ValueError('labels must be one-dimensional.')
    if len(label_array) != len(probability_array):
        raise ValueError('probabilities and labels must have equal sample counts.')
    if not np.issubdtype(label_array.dtype, np.integer):
        raise ValueError('labels must contain integer class indices.')

    label_array = label_array.astype(np.int64, copy=False)
    if np.any(label_array < 0) or np.any(label_array >= NUM_CLASSES):
        raise ValueError(f'labels must remain between 0 and {NUM_CLASSES - 1}.')
    return probability_array, label_array


def temperature_scale_probabilities(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    '''Tahmin edilen sınıfı koruyarak sınıf güvenini yeniden ölçekler.

    Matematik: p -> softmax(log(p) / T). T > 1 dağılımı yumuşatır (güven
    azalır), T < 1 keskinleştirir (güven artar). log monoton olduğu için
    argmax DEĞİŞMEZ — yani accuracy/F1 gibi metrikler aynı kalır; yalnızca
    olasılıkların "ne kadar iddialı" olduğu değişir. Kalibrasyonun bu kadar
    güvenli bir işlem olmasının sırrı budur.
    '''

    probability_array = validate_probability_matrix(probabilities)
    # bool, Python'da int'in alt sınıfıdır; float(True)=1.0 sessizce geçerdi.
    # Bunu bilinçli olarak reddediyoruz — True bir sıcaklık değeri değildir.
    if isinstance(temperature, (bool, np.bool_)):
        raise ValueError('temperature must be a positive finite number.')
    try:
        temperature_value = float(temperature)
    except (TypeError, ValueError) as error:
        raise ValueError(
            'temperature must be a positive finite number.'
        ) from error
    if not np.isfinite(temperature_value) or temperature_value <= 0.0:
        raise ValueError('temperature must be a positive finite number.')
    # T=1 kimlik dönüşümüdür; kopya döndürerek çağıranın matrisini koruruz.
    if temperature_value == 1.0:
        return probability_array.copy()

    # Olasılıklardan "logit benzeri" değerlere dönüş: log(p). Softmax kayma
    # değişmezi (shift-invariant) olduğundan log(p), gerçek logitlerin yerine
    # güvenle geçer. clip, log(0)'ı engeller.
    log_probabilities = np.log(
        np.clip(probability_array, PROBABILITY_FLOOR, 1.0)
    )
    scaled_logits = log_probabilities / temperature_value
    # Sayısal kararlılık numarası: her satırdan maksimumu çıkarmak exp'in
    # taşmasını (overflow) önler; softmax sonucu matematiksel olarak aynıdır.
    scaled_logits -= scaled_logits.max(axis=1, keepdims=True)
    exponentials = np.exp(scaled_logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _temperature_nll(
    log_probabilities: np.ndarray,
    labels: np.ndarray,
    temperature: float,
) -> float:
    # Verilen sıcaklık için ortalama negatif log-olabilirliği (NLL) hesaplar.
    # NLL = log-sum-exp(z) - z_dogru_sinif ortalaması. Burada "log-sum-exp
    # numarası" kullanılır: önce satır maksimumu çıkarılır ki exp taşmasın,
    # sonra maksimum geri eklenir. Böylece uç sıcaklıklarda bile sayısal
    # olarak güvenli bir NLL elde edilir.
    scaled_logits = log_probabilities / temperature
    row_maxima = scaled_logits.max(axis=1, keepdims=True)
    log_normalizers = row_maxima[:, 0] + np.log(
        np.exp(scaled_logits - row_maxima).sum(axis=1)
    )
    # Fancy indexing: her satırdan doğru sınıfın logit'ini seç.
    true_logits = scaled_logits[np.arange(len(labels)), labels]
    return float(np.mean(log_normalizers - true_logits))


def _best_temperature_index(
    temperatures: np.ndarray,
    losses: np.ndarray,
) -> int:
    # En düşük NLL'i veren sıcaklığın indeksini bulur. Beraberlik durumunda
    # |log T| en küçük olanı, yani 1.0'a EN YAKIN sıcaklığı seçer: eşit
    # kayıpta "en az müdahale" ilkesi uygulanır (Occam'ın usturası gibi).
    # log ölçeğinde uzaklık kullanılır çünkü 0.5 ve 2.0, 1'e eşit uzaklıktadır.
    minimum_loss = float(losses.min())
    tied = np.flatnonzero(
        np.isclose(losses, minimum_loss, rtol=1e-12, atol=1e-14)
    )
    tie_distances = np.abs(np.log(temperatures[tied]))
    return int(tied[int(tie_distances.argmin())])


def fit_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    minimum: float = MIN_TEMPERATURE,
    maximum: float = MAX_TEMPERATURE,
    coarse_steps: int = 161,
    fine_steps: int = 161,
) -> dict[str, Any]:
    '''Doğrulama NLL'ini en aza indirerek tek bir skaler sıcaklık öğrenir.

    Yöntem iki aşamalı logaritmik ızgara araması:
    1. **Kaba aşama**: [minimum, maximum] aralığını log ölçekte eşit aralıklı
       noktalara böl (geomspace), her noktada NLL hesapla, en iyisini bul.
    2. **İnce aşama**: Kaba kazananın hemen komşuları arasında daha sık bir
       ızgara kurup aramayı tekrarla.

    Neden gradyan tabanlı optimizasyon değil? Tek boyutlu, düzgün bir problem
    için ızgara araması hem yeterince hassas hem de tamamen deterministik ve
    bağımlılıksızdır. 1.0 her iki aşamada da ızgaraya bilerek eklenir; böylece
    "hiç ölçekleme yapma" seçeneği her zaman adaylar arasındadır ve NLL
    asla T=1'den kötüye gidemez.
    '''

    probability_array, label_array = validate_probability_inputs(
        probabilities,
        labels,
    )
    # bool'ların sayı yerine geçmesini burada da engelle.
    if isinstance(minimum, (bool, np.bool_)) or isinstance(
        maximum, (bool, np.bool_)
    ):
        raise ValueError('temperature range must contain 1 with positive bounds.')
    minimum_value = float(minimum)
    maximum_value = float(maximum)
    # Aralık 1.0'ı içermek zorunda: 0 < min < 1 < max. Aksi halde "değişiklik
    # yapma" seçeneği aramanın dışında kalırdı.
    if (
        not np.isfinite(minimum_value)
        or not np.isfinite(maximum_value)
        or not 0.0 < minimum_value < 1.0 < maximum_value
    ):
        raise ValueError('temperature range must contain 1 with positive bounds.')
    # Adım sayıları en az 21 olmalı; daha azı ızgarayı anlamsızca seyrek yapar.
    for name, steps in (
        ('coarse_steps', coarse_steps),
        ('fine_steps', fine_steps),
    ):
        if (
            isinstance(steps, (bool, np.bool_))
            or not isinstance(steps, (int, np.integer))
            or steps < 21
        ):
            raise ValueError(f'{name} must be an integer of at least 21.')

    # log(p) bir kez hesaplanır ve tüm sıcaklık denemelerinde yeniden kullanılır.
    log_probabilities = np.log(
        np.clip(probability_array, PROBABILITY_FLOOR, 1.0)
    )
    # Kaba ızgara: log ölçekte eşit aralıklı + 1.0 garanti içeride.
    # np.unique hem sıralar hem olası kopyayı temizler.
    coarse_temperatures = np.unique(
        np.append(
            np.geomspace(minimum_value, maximum_value, int(coarse_steps)),
            1.0,
        )
    )
    coarse_losses = np.array(
        [
            _temperature_nll(log_probabilities, label_array, temperature)
            for temperature in coarse_temperatures
        ]
    )
    coarse_index = _best_temperature_index(
        coarse_temperatures,
        coarse_losses,
    )
    # İnce aşamanın aralığı: kaba kazananın bir sol ve bir sağ komşusu.
    # min/max korumaları, kazanan ızgaranın ucundaysa taşmayı önler.
    lower_index = max(0, coarse_index - 1)
    upper_index = min(len(coarse_temperatures) - 1, coarse_index + 1)
    fine_temperatures = np.geomspace(
        coarse_temperatures[lower_index],
        coarse_temperatures[upper_index],
        int(fine_steps),
    )
    temperatures = np.unique(
        np.concatenate((fine_temperatures, np.array([1.0])))
    )
    losses = np.array(
        [
            _temperature_nll(log_probabilities, label_array, temperature)
            for temperature in temperatures
        ]
    )
    best_index = _best_temperature_index(temperatures, losses)
    temperature = float(temperatures[best_index])
    # Rapor için: kalibrasyon öncesi (T=1) ve sonrası NLL — iyileşme miktarı
    # sıcaklık ölçeklemenin gerçekten işe yarayıp yaramadığını gösterir.
    nll_before = _temperature_nll(log_probabilities, label_array, 1.0)
    nll_after = float(losses[best_index])

    return {
        'method': 'two-stage logarithmic grid search',
        'fitted_on': 'validation probabilities',
        'temperature': temperature,
        'minimum': minimum_value,
        'maximum': maximum_value,
        'coarse_steps': int(coarse_steps),
        'fine_steps': int(fine_steps),
        'validation_nll_before': nll_before,
        'validation_nll_after': nll_after,
        'validation_nll_improvement': nll_before - nll_after,
    }


def multiclass_negative_log_likelihood(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    '''Doğru sınıfa atanan olasılığın ortalama negatif logaritmasını döndürür.

    NLL, kalibrasyonun "her şey dahil" ölçütüdür: model doğru sınıfa yüksek
    olasılık verdikçe düşer; emin olup yanılınca sert cezalanır. Cross-entropy
    kaybının değerlendirme halidir.
    '''

    probability_array, label_array = validate_probability_inputs(
        probabilities,
        labels,
    )
    row_indices = np.arange(len(label_array))
    # Her satırdan doğru sınıfın olasılığını çek.
    true_probabilities = probability_array[row_indices, label_array]
    # Taban kırpması: model doğru sınıfa ~0 olasılık verdiyse log patlamasın.
    safe_probabilities = np.clip(
        true_probabilities,
        PROBABILITY_FLOOR,
        1.0,
    )
    return float(-np.log(safe_probabilities).mean())


def multiclass_brier_score(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    '''One-hot sınıf hedeflerine karşı ortalama toplam kare hatayı döndürür.

    Brier skoru = ||p - onehot(y)||^2 ortalaması. NLL'e alternatif, sınırlı
    (0 ile 2 arası) ve uç olasılıklara karşı daha az hassas bir kalibrasyon
    ölçütüdür. Düşük değer daha iyidir.
    '''

    probability_array, label_array = validate_probability_inputs(
        probabilities,
        labels,
    )
    # np.eye ile birim matris kur, etiketlerle satır seç: hızlı one-hot üretimi.
    one_hot_targets = np.eye(NUM_CLASSES, dtype=np.float64)[label_array]
    squared_errors = np.square(probability_array - one_hot_targets)
    return float(squared_errors.sum(axis=1).mean())


def confidence_bin_statistics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 10,
) -> list[dict[str, Any]]:
    '''Eşit genişlikte dilimler içinde güven ve doğruluğu özetler.

    Güvenilirlik diyagramının (reliability diagram) veri tarafı: [0,1]
    aralığı ``bins`` eşit parçaya bölünür; her örnek, en yüksek sınıf
    olasılığına (güvenine) göre bir dilime düşer. Dilim başına ortalama
    güven ile gerçek doğruluk karşılaştırılır — mükemmel kalibre bir modelde
    bu ikisi eşit olurdu, aradaki fark (absolute_gap) kalibrasyon hatasıdır.
    '''

    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)):
        raise ValueError('bins must be an integer.')
    if bins < 2:
        raise ValueError('bins must be at least 2.')

    probability_array, label_array = validate_probability_inputs(
        probabilities,
        labels,
    )
    predictions = probability_array.argmax(axis=1)
    confidences = probability_array.max(axis=1)
    correct = predictions == label_array
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    # searchsorted ile dilim ataması: iç kenarlara (ilk ve son hariç) göre
    # her güven değerinin hangi dilime düştüğü bulunur. side='right' sınır
    # değerlerinin tutarlı (sol dilime dahil) işlenmesini sağlar.
    assignments = np.searchsorted(
        edges[1:-1],
        confidences,
        side='right',
    )

    statistics: list[dict[str, Any]] = []
    sample_size = len(label_array)
    for bin_index in range(int(bins)):
        mask = assignments == bin_index
        count = int(mask.sum())
        # Boş dilimlerde ortalama güven/doğruluk tanımsızdır; None bırakılır
        # ki 0.0 ile karışmasın (0.0 "kötü", None "veri yok" demektir).
        summary: dict[str, Any] = {
            'bin': bin_index + 1,
            'lower': float(edges[bin_index]),
            'upper': float(edges[bin_index + 1]),
            'count': count,
            'fraction': float(count / sample_size),
            'mean_confidence': None,
            'accuracy': None,
            'absolute_gap': None,
        }
        if count:
            mean_confidence = float(confidences[mask].mean())
            accuracy = float(correct[mask].mean())
            summary.update(
                {
                    'mean_confidence': mean_confidence,
                    'accuracy': accuracy,
                    'absolute_gap': abs(accuracy - mean_confidence),
                }
            )
        statistics.append(summary)
    return statistics


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    '''Güven dilimlerine göre ağırlıklı mutlak kalibrasyon hatasını döndürür.

    ECE = her dilimin |doğruluk - ortalama güven| farkının, dilimdeki örnek
    oranıyla ağırlıklandırılmış toplamı. Tek sayıda "model kendine ne kadar
    haksız güveniyor?" sorusunun cevabıdır; 0 mükemmel kalibrasyondur.
    Boş dilimler (count=0) toplamdan doğal olarak dışarıda kalır.
    '''

    statistics = confidence_bin_statistics(
        probabilities,
        labels,
        bins=bins,
    )
    weighted_gaps = (
        entry['fraction'] * entry['absolute_gap']
        for entry in statistics
        if entry['count'] > 0
    )
    return float(sum(weighted_gaps))


def classification_calibration_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, Any]:
    '''JSON'a hazır, ayrılmış (held-out) çok sınıflı kalibrasyon özeti döndürür.

    Tüm kalibrasyon metriklerini tek bir sözlükte toplayan çatı fonksiyon:
    NLL, Brier, ECE, MCE (en kötü dilimin farkı) ve
    ``confidence_minus_accuracy`` (pozitifse model aşırı özgüvenli demektir).
    Sonuç doğrudan json.dumps ile diske yazılabilir.
    '''

    probability_array, label_array = validate_probability_inputs(
        probabilities,
        labels,
    )
    statistics = confidence_bin_statistics(
        probability_array,
        label_array,
        bins=bins,
    )
    nonempty_gaps = [
        float(entry['absolute_gap'])
        for entry in statistics
        if entry['count'] > 0
    ]
    predictions = probability_array.argmax(axis=1)
    confidences = probability_array.max(axis=1)
    accuracy = float((predictions == label_array).mean())
    mean_confidence = float(confidences.mean())

    return {
        'method': 'top-label equal-width confidence calibration',
        'sample_size': int(len(label_array)),
        'num_classes': NUM_CLASSES,
        'num_bins': int(bins),
        'negative_log_likelihood': multiclass_negative_log_likelihood(
            probability_array,
            label_array,
        ),
        'multiclass_brier_score': multiclass_brier_score(
            probability_array,
            label_array,
        ),
        'expected_calibration_error': expected_calibration_error(
            probability_array,
            label_array,
            bins=bins,
        ),
        # MCE: en kötü (en büyük farklı) dolu dilimin kalibrasyon hatası.
        'maximum_calibration_error': max(nonempty_gaps),
        'accuracy': accuracy,
        'mean_confidence': mean_confidence,
        'confidence_minus_accuracy': mean_confidence - accuracy,
        'bins': statistics,
    }
