'''Hızlı, rapor ve tam çalıştırmalar için deterministik hiperparametre adayları.

Rastgele arama yerine SABİT (deterministik) aday listeleri kullanıyoruz.
Bunun iki nedeni var:
1. Tekrarlanabilirlik — aynı komut her çalıştırmada aynı adayları dener.
2. Yorumlanabilirlik — 'report' modunda adaylar "tek seferde tek faktör"
   mantığıyla kurulduğu için her hiperparametrenin etkisi raporda temiz
   biçimde izole edilebilir.

Üç mod vardır:
- ``quick``  : 2 küçük aday; kodun uçtan uca çalıştığını dakikalar içinde
  doğrulamak için (duman testi).
- ``report`` : Güçlü bir taban ayarın etrafında tek-faktör varyasyonları;
  ödev raporundaki hiperparametre etkisi analizinin kaynağı.
- ``full``   : Kartesyen çarpımla daha geniş bir tarama; en iyi skoru
  kovalamak için.
'''

from __future__ import annotations

from dataclasses import replace
from itertools import product

from odev3.model import MLPConfig


GRID_MODES = ('quick', 'report', 'full')


def _unique(configs: list[MLPConfig]) -> list[MLPConfig]:
    # Aday listesinden kopyaları ayıklar ama SIRAYI korur (set'e atmak sırayı
    # bozardı; sıra, deney kayıtlarının okunabilirliği için önemli).
    # Her adayı ayrıca validate'ten geçirerek geçersiz kombinasyonların
    # aramaya sızmasını engelleriz.
    result: list[MLPConfig] = []
    seen: set[tuple] = set()
    for config in configs:
        config.validate()
        key = (
            config.batch_size,
            config.learning_rate,
            config.patience,
            config.hidden_dims,
            config.activation,
            config.batch_norm,
            config.dropout,
            config.weight_decay,
        )
        if key not in seen:
            seen.add(key)
            result.append(config)
    return result


def _report_candidates() -> list[MLPConfig]:
    '''Yorumlanabilir etkiler için güçlü bir taban etrafında tek-faktör kontrolleri.

    Deney tasarımı mantığı: önce makul bir "base" konfigürasyon sabitlenir,
    sonra her adayda SADECE BİR alan değiştirilir (``dataclasses.replace``
    tam da bunu yapar: verilen alan dışında her şeyi kopyalar). Böylece
    "dropout 0.3'ten 0.5'e çıkınca ne oldu?" sorusuna, diğer her şey sabitken
    cevap verilebilir — klasik kontrollü deney yaklaşımı.
    '''

    base = MLPConfig()
    return _unique(
        [
            base,
            # Batch boyutu: küçük (gürültülü ama sık güncelleme) vs büyük.
            replace(base, batch_size=32),
            replace(base, batch_size=128),
            # Öğrenme oranı: bir kademe yukarı ve aşağı.
            replace(base, learning_rate=1e-3),
            replace(base, learning_rate=1e-4),
            # Erken durdurma sabrı: agresif vs toleranslı.
            replace(base, patience=5),
            replace(base, patience=12),
            # Mimari: daha sığ/dar ve daha derin/geniş varyantlar.
            replace(base, hidden_dims=(256,)),
            replace(base, hidden_dims=(512, 256, 128)),
            replace(base, hidden_dims=(1024, 512, 256)),
            # Aktivasyon alternatifleri.
            replace(base, activation='gelu'),
            replace(base, activation='tanh'),
            # Düzenlileştirme bileşenlerini tek tek aç/kapat.
            replace(base, batch_norm=False),
            replace(base, dropout=0.0),
            replace(base, dropout=0.5),
            replace(base, weight_decay=0.0),
        ]
    )


def _full_candidates() -> list[MLPConfig]:
    # 'full' mod: seçilmiş değer kümelerinin kartesyen çarpımı.
    # 2*2*2*2*2*2*2 = 128 aday üretir; weight_decay sabit tutularak
    # kombinasyon patlaması sınırlanır. itertools.product, iç içe 7 döngü
    # yazmadan tüm kombinasyonları sırayla verir.
    configs = []
    for values in product(
        (32, 64),            # batch_size
        (1e-3, 3e-4),        # learning_rate
        (6, 10),             # patience
        ((256,), (512, 256)),  # hidden_dims
        ('relu', 'gelu'),    # activation
        (False, True),       # batch_norm
        (0.2, 0.5),          # dropout
    ):
        configs.append(
            MLPConfig(
                batch_size=values[0],
                learning_rate=values[1],
                patience=values[2],
                hidden_dims=values[3],
                activation=values[4],
                batch_norm=values[5],
                dropout=values[6],
                weight_decay=1e-4,
            )
        )
    return _unique(configs)


def _scaled_hidden_dims(hidden_dims: tuple[int, ...], factor: float) -> tuple[int, ...]:
    '''Genişlikleri ölçeklerken pratik 32'nin katlarında tutar.

    Örnek: (512, 256) * 1.5 -> (768, 384). Önce çarpar, sonra en yakın 32
    katına yuvarlar; 32 tabanı hem donanım dostu hem de "temiz" katman
    boyutları üretir. max(32, ...) alt sınırı, küçültme sırasında katmanın
    yok olacak kadar daralmasını engeller.
    '''

    return tuple(
        max(32, int(round((size * factor) / 32.0)) * 32)
        for size in hidden_dims
    )


def refinement_space(
    winner: MLPConfig,
    *,
    exclude: list[MLPConfig] | tuple[MLPConfig, ...] = (),
) -> list[MLPConfig]:
    '''Doğrulama şampiyonunun etrafında yerel bir ikinci aşama araması kurar.

    Strateji "kaba arama + ince ayar" (coarse-to-fine): ilk ızgara en iyi
    bölgeyi bulur, bu fonksiyon da o kazananın KOMŞULARINI üretir — her
    hiperparametreyi bir kademe aşağı/yukarı oynatır. min/max sınırları
    (örn. öğrenme oranı için [1e-5, 5e-3]) komşuların saçma değerlere
    taşmasını engeller.

    ``exclude`` parametresi, ilk aşamada ZATEN denenmiş adayların ikinci
    aşamada boşuna yeniden eğitilmesini önler.
    '''

    winner.validate()
    # Kazananın kullanmadığı iki aktivasyonu da komşu olarak deneriz.
    alternative_activations = [
        name for name in ('relu', 'gelu', 'tanh') if name != winner.activation
    ]
    candidates = _unique(
        [
            # Öğrenme oranını yarıla/iki katına çıkar (sınırlar içinde).
            replace(winner, learning_rate=max(1e-5, winner.learning_rate / 2.0)),
            replace(winner, learning_rate=min(5e-3, winner.learning_rate * 2.0)),
            # Batch boyutunu yarıla/iki katına çıkar.
            replace(winner, batch_size=max(16, winner.batch_size // 2)),
            replace(winner, batch_size=min(256, winner.batch_size * 2)),
            # Erken durdurma sabrını +-2 oynat.
            replace(winner, patience=max(3, winner.patience - 2)),
            replace(winner, patience=min(20, winner.patience + 2)),
            # Dropout'u +-0.1 oynat; round(.., 2) yüzen nokta artıklarını
            # (0.30000000004 gibi) temizler.
            replace(winner, dropout=round(max(0.0, winner.dropout - 0.1), 2)),
            replace(winner, dropout=round(min(0.7, winner.dropout + 0.1), 2)),
            # Ağı yarı/1.5 kat genişlikte dene.
            replace(
                winner,
                hidden_dims=_scaled_hidden_dims(winner.hidden_dims, 0.5),
            ),
            replace(
                winner,
                hidden_dims=_scaled_hidden_dims(winner.hidden_dims, 1.5),
            ),
            # BatchNorm'u tersine çevir; iki alternatif aktivasyonu dene.
            replace(winner, batch_norm=not winner.batch_norm),
            replace(winner, activation=alternative_activations[0]),
            replace(winner, activation=alternative_activations[1]),
            # weight_decay'i aç/kapat (hangisi kapalıysa diğerini dene).
            replace(
                winner,
                weight_decay=0.0 if winner.weight_decay > 0.0 else 1e-4,
            ),
        ]
    )
    # Kazananın kendisini ve daha önce denenmiş adayları listeden çıkar.
    excluded = set(exclude)
    return [config for config in candidates if config != winner and config not in excluded]


def search_space(mode: str) -> list[MLPConfig]:
    '''Doğrulama veya test verisine DOKUNMADAN aday listesini döndürür.

    Bu fonksiyon yalnızca konfigürasyon nesneleri üretir; hangi adayın iyi
    olduğuna karar vermek eğitim/doğrulama döngüsünün işidir. Aday üretimini
    veriden tamamen ayırmak, arama uzayının veri sızıntısından etkilenmesini
    yapısal olarak imkansız kılar.
    '''

    if mode == 'quick':
        # Duman testi adayları: küçük ağlar, düşük sabır — amaç skor değil,
        # boru hattının uçtan uca çalıştığını hızla görmek.
        return [
            MLPConfig(
                batch_size=64,
                learning_rate=1e-3,
                patience=2,
                hidden_dims=(64,),
                activation='relu',
                batch_norm=False,
                dropout=0.0,
                weight_decay=0.0,
            ),
            MLPConfig(
                batch_size=64,
                learning_rate=3e-4,
                patience=3,
                hidden_dims=(128, 64),
                activation='gelu',
                batch_norm=True,
                dropout=0.3,
                weight_decay=1e-4,
            ),
        ]
    if mode == 'report':
        return _report_candidates()
    if mode == 'full':
        return _full_candidates()
    raise ValueError(f'Unknown grid mode {mode!r}; expected one of {GRID_MODES}.')
