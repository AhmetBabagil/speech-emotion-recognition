'''İki final yöntemi için öznitelik çıkarma.

Yöntem 1: Her ses kaydı, 2 boyutlu bir log-mel spectrogram "görüntüsüne"
çevrilir (CNN'in girdisi). Spectrogram, sesin zaman-frekans haritasıdır:
yatay eksen zaman, dikey eksen frekans, renk/değer o andaki enerji.

Yöntem 2: Ses kaydı, sayısı ve genişliği HİPERPARAMETRE olan eşit aralıklı
pencerelere bölünür; her aralıktan klasik akustik istatistikler (MFCC,
enerji, spektral özellikler) çıkarılır. Sonuç [T, D] boyutlu bir öznitelik
serisidir ve LSTM/GRU'ya girdi olur.

librosa yalnızca sinyal düzeyi öznitelik çıkarımı için kullanılır — ödev
yönergesi buna açıkça izin verir; hiçbir yerde önceden eğitilmiş model yok.
'''

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np

# Tüm kayıtlar bu örnekleme hızına indirgenir (konuşma için standart değer).
SAMPLE_RATE = 16_000


def _fingerprint(payload: dict) -> str:
    '''Ayarlardan kısa ve kararlı bir kimlik üretir.

    Önbellek (cache) klasörleri bu kimlikle adlandırılır; böylece farklı
    ayarlarla çıkarılmış öznitelikler asla birbirine karışmaz.
    '''

    text = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]


def _load_audio(audio_path: str | Path, sample_rate: int) -> np.ndarray:
    '''Bir ses dosyasını mono ve hedef örnekleme hızında yükler, doğrular.'''

    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f'Ses dosyası bulunamadı: {audio_path}')
    # sr=... vermek librosa'ya "gerekirse yeniden örnekle" der; mono=True
    # stereo kayıtları tek kanala indirir.
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f'Ses dosyası boş: {audio_path}')
    if not np.isfinite(audio).all():
        raise ValueError(f'Ses dosyasında geçersiz (NaN/inf) örnek var: {audio_path}')
    return audio


# --- Yöntem 1: log-mel spectrogram görüntüsü -------------------------------------


@dataclass(frozen=True)
class MelImageConfig:
    '''CNN'e girecek 2 boyutlu log-mel görüntüsünün tüm ayarları.

    frozen=True: değerler sonradan değiştirilemez; böylece aynı config
    her zaman aynı özniteliği üretir (önbellek güvenliği için önemli).
    '''

    sample_rate: int = SAMPLE_RATE
    n_mels: int = 64          # frekans ekseni çözünürlüğü (mel bandı sayısı)
    n_frames: int = 128       # zaman ekseni uzunluğu (sabitlenmiş kare sayısı)
    n_fft: int = 1024         # FFT pencere genişliği (64 ms @ 16 kHz)
    hop_length: int = 256     # pencereler arası atlama (16 ms @ 16 kHz)
    fmin: float = 20.0        # insan işitmesinin alt sınırı civarı
    fmax: float = 8_000.0     # konuşma enerjisinin çoğu bu bandın altında
    top_db: float = 80.0      # dB ölçeğinde kırpma aralığı

    def validate(self) -> None:
        '''Anlamsız ayarları en baştan reddet (sessiz hata yerine açık hata).'''

        if min(self.sample_rate, self.n_mels, self.n_frames, self.n_fft, self.hop_length) <= 0:
            raise ValueError(f'Mel parametreleri pozitif olmalı: {self}.')
        if self.fmin < 0.0 or self.fmax <= self.fmin or self.top_db <= 0.0:
            raise ValueError(f'Geçersiz frekans/dB ayarı: {self}.')

    @property
    def shape(self) -> tuple[int, int]:
        '''Üretilecek görüntünün boyutu: (mel bandı, zaman karesi).'''

        return (self.n_mels, self.n_frames)

    @property
    def fingerprint(self) -> str:
        '''Bu ayarlara özel önbellek kimliği (ör. "mel_a1b2c3...").'''

        self.validate()
        return 'mel_' + _fingerprint(asdict(self))


def fix_frames(mel_db: np.ndarray, n_frames: int) -> np.ndarray:
    '''Mel matrisinin zaman eksenini tam olarak n_frames sütuna sabitler.

    Kayıtlar farklı uzunlukta olduğundan CNN'e vermeden önce hepsini aynı
    boyuta getirmek gerekir: uzun kayıt ortadan kırpılır, kısa kayıt en
    düşük enerji değeriyle (sessizlik gibi) sağdan doldurulur.
    '''

    current = mel_db.shape[1]
    if current == 0:
        # Hiç kare üretilemediyse tamamen "sessizlik" döndür.
        return np.full((mel_db.shape[0], n_frames), -80.0, dtype=np.float32)
    if current < n_frames:
        # Kısa kayıt: sağa dolgu. Dolgu değeri olarak matristeki en düşük
        # değeri kullanıyoruz ki dolgu, gerçek sessizlikten ayrışmasın.
        floor = float(np.min(mel_db))
        return np.pad(
            mel_db,
            ((0, 0), (0, n_frames - current)),
            mode='constant',
            constant_values=floor,
        ).astype(np.float32, copy=False)
    if current > n_frames:
        # Uzun kayıt: ortadan kırp (duygu genellikle kaydın ortasında en belirgindir).
        start = (current - n_frames) // 2
        return mel_db[:, start : start + n_frames]
    return mel_db.astype(np.float32, copy=False)


def extract_mel_image(
    audio_path: str | Path,
    config: MelImageConfig,
) -> np.ndarray:
    '''Bir kaydı [n_mels, n_frames] boyutlu float32 log-mel matrisine çevirir.

    Adımlar: ses yükle -> mel spectrogram (güç) -> dB (log) ölçeğine çevir
    -> zaman eksenini sabitle. CNN bu matrisi tek kanallı görüntü gibi işler.
    '''

    config.validate()
    audio = _load_audio(audio_path, config.sample_rate)
    mel_power = librosa.feature.melspectrogram(
        y=audio,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=min(config.fmax, config.sample_rate / 2),  # Nyquist sınırını aşma
        power=2.0,
    )
    # dB'ye çevirirken referans olarak matristeki en büyük değeri alıyoruz.
    # Pozitif alt sınır, tamamen sessiz kayıtta sıfıra bölmeyi engeller.
    reference = max(float(np.max(mel_power)), float(np.finfo(np.float32).tiny))
    mel_db = librosa.power_to_db(mel_power, ref=reference, top_db=config.top_db)
    image = fix_frames(mel_db, config.n_frames)
    if image.shape != config.shape or not np.isfinite(image).all():
        raise ValueError(f'Geçersiz mel görüntüsü ({audio_path}): shape={image.shape}.')
    return image


# --- Yöntem 2: aralık öznitelik serisi -------------------------------------------


@dataclass(frozen=True)
class IntervalConfig:
    '''Aralık düzeni ve aralık-içi ayarlar (RNN öznitelik serisi için).

    n_intervals ve interval_ms, ödevin hiperparametre olarak aranmasını
    istediği iki değerdir. Aralık başlangıçları kayda eşit yayılır: kısa
    kayıtta aralıklar örtüşür, uzun kayıtta aralarında boşluk kalır —
    her iki durumda da kayıt başına tam n_intervals satır üretilir, yani
    RNN her zaman sabit uzunlukta bir seri görür.
    '''

    sample_rate: int = SAMPLE_RATE
    n_intervals: int = 24     # kayıt kaç aralığa bölünecek (hiperparametre)
    interval_ms: int = 300    # her aralığın genişliği, milisaniye (hiperparametre)
    n_mfcc: int = 13          # aralık başına MFCC katsayısı sayısı
    n_fft: int = 512          # aralık içi kısa FFT penceresi (32 ms)
    hop_length: int = 160     # aralık içi atlama (10 ms)
    # --- Ablasyon bayrakları: her mantıklı öznitelik grubu ayrı açılıp kapanır.
    # İki ablasyonun (kümülatif + bırak-birini, her biri 5 koşu) sonucu:
    # en iyi kombinasyon jitter/shimmer + spektral kontrast (53 boyut, %66,9).
    # Pitch, kontrastla birlikteyken fazlalık kaldı (bırak-birini: çıkarınca
    # doğruluk arttı) -> KAPALI. delta2 ve bant zaten düşürdüğü için KAPALI.
    # Bu, kanıta dayalı öznitelik seçiminin ürünü.
    use_pitch: bool = False     # kontrastla birlikte fazlalık — bırak-birini kanıtı
    use_jitter: bool = True     # +2: jitter + shimmer titremesi
    use_contrast: bool = True   # +7: spektral kontrast — en güçlü tekil grup
    use_delta2: bool = False    # +13: delta-delta MFCC — ablasyonda düşürdü
    use_bandwidth: bool = False # +2: bant genişliği + düzlük — ablasyonda düşürdü
    feature_version: int = 3   # öznitelik seti sürümü (fingerprint'i ayırır)

    def validate(self) -> None:
        if min(self.sample_rate, self.n_intervals, self.interval_ms,
               self.n_mfcc, self.n_fft, self.hop_length) <= 0:
            raise ValueError(f'Aralık parametreleri pozitif olmalı: {self}.')

    @property
    def interval_samples(self) -> int:
        '''Bir aralığın örnek (sample) cinsinden uzunluğu.'''

        return int(round(self.sample_rate * self.interval_ms / 1000.0))

    @property
    def feature_dim(self) -> int:
        '''Aralık başına öznitelik sayısı — açık bayraklara göre hesaplanır.

        Taban: 13 MFCC ort + 13 MFCC std + 13 delta-MFCC ort + 5 skaler
        (enerji, enerji std, ZCR, centroid, rolloff) = 44.
        Bayraklar: +3 pitch, +2 jitter/shimmer, +7 kontrast, +13 delta-delta,
        +2 bant genişliği/düzlük.
        '''

        boyut = 3 * self.n_mfcc + 5           # taban 44
        if self.use_pitch:
            boyut += 3
        if self.use_jitter:
            boyut += 2
        if self.use_contrast:
            boyut += 7
        if self.use_delta2:
            boyut += self.n_mfcc               # delta-delta ortalaması (13)
        if self.use_bandwidth:
            boyut += 2
        return boyut

    @property
    def shape(self) -> tuple[int, int]:
        '''Üretilecek serinin boyutu: (aralık sayısı, öznitelik sayısı).'''

        return (self.n_intervals, self.feature_dim)

    @property
    def fingerprint(self) -> str:
        '''Bu ayarlara özel önbellek kimliği (ör. "seq_9a8b7c...").'''

        self.validate()
        return 'seq_' + _fingerprint(asdict(self))


def interval_starts(total_samples: int, config: IntervalConfig) -> np.ndarray:
    '''n_intervals adet pencerenin başlangıç indekslerini eşit aralıkla yerleştirir.'''

    # Son pencerenin kayıt dışına taşmaması için kullanılabilir açıklık:
    span = max(total_samples - config.interval_samples, 0)
    if config.n_intervals == 1:
        return np.array([span // 2], dtype=np.int64)
    # 0'dan span'a eşit aralıklı n_intervals başlangıç noktası.
    return np.linspace(0, span, config.n_intervals).round().astype(np.int64)


def _interval_features(segment: np.ndarray, config: IntervalConfig) -> np.ndarray:
    '''Tek bir aralığı sabit uzunlukta bir istatistik vektörüne özetler.

    Aralık içinde kısa pencerelerle (32 ms) kare-düzeyi öznitelikler
    hesaplanır, sonra bu karelerin ortalama/std'si alınır. Böylece aralık
    ne kadar uzun olursa olsun çıktı hep aynı boyutta kalır.
    '''

    # Aralık FFT penceresinden bile kısaysa sıfırla doldur.
    if len(segment) < config.n_fft:
        segment = np.pad(segment, (0, config.n_fft - len(segment)))
    # MFCC: konuşma tınısını (spektral zarfı) özetleyen klasik katsayılar.
    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=config.sample_rate,
        n_mfcc=config.n_mfcc,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )
    # Delta-MFCC: katsayıların zaman içindeki değişim hızı (dinamik bilgi).
    if mfcc.shape[1] >= 2:
        delta = librosa.feature.delta(mfcc, width=min(9, mfcc.shape[1] // 2 * 2 + 1))
    else:
        delta = np.zeros_like(mfcc)
    # Delta-delta MFCC (ivme): değişimin değişimi — hızlanan/yavaşlayan tını akışı.
    if config.use_delta2:
        if mfcc.shape[1] >= 2:
            delta2 = librosa.feature.delta(mfcc, order=2,
                                           width=min(9, mfcc.shape[1] // 2 * 2 + 1))
        else:
            delta2 = np.zeros_like(mfcc)
    # STFT'yi bir kez hesaplayıp enerji ve spektral özniteliklerde paylaşıyoruz.
    stft = np.abs(
        librosa.stft(segment, n_fft=config.n_fft, hop_length=config.hop_length)
    )
    rms = librosa.feature.rms(S=stft, frame_length=config.n_fft)[0]      # enerji
    zcr = librosa.feature.zero_crossing_rate(                            # sıfır geçişi
        segment, frame_length=config.n_fft, hop_length=config.hop_length
    )[0]
    centroid = librosa.feature.spectral_centroid(S=stft, sr=config.sample_rate)[0]
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=config.sample_rate)[0]

    skalerler = [
        float(np.log1p(rms.mean())),                            # log enerji
        float(rms.std()),                                       # enerji dalgalanması
        float(zcr.mean()),                                      # gürültülülük göstergesi
        float(centroid.mean() / (config.sample_rate / 2)),      # spektral ağırlık merkezi (0-1)
        float(rolloff.mean() / (config.sample_rate / 2)),       # enerji %85 sınırı (0-1)
    ]
    if config.use_pitch or config.use_jitter:
        # Pitch (temel frekans F0): tonlamanın taşıyıcısı; duygunun en güçlü ipucu.
        # pyin her çerçeve için F0 + "tonlu mu" bilgisi verir. Aralık içi kısa
        # sinyalde daha seyrek çerçeve (2x hop) yeterli ve pyin'i belirgin hızlandırır.
        f0, voiced_flag, _ = librosa.pyin(
            segment, sr=config.sample_rate,
            fmin=65, fmax=400,                  # insan konuşma perdesi bandı (Hz)
            frame_length=config.n_fft,
            hop_length=config.hop_length * 2,   # daha az çerçeve -> daha hızlı
        )
        f0_tonlu = f0[voiced_flag]              # yalnızca tonlu çerçevelerin perdesi
    if config.use_pitch:
        if len(f0_tonlu) > 0:
            pitch_ort = float(np.log1p(np.mean(f0_tonlu)))   # log-perde ortalaması
            pitch_std = float(np.std(f0_tonlu))              # perde dalgalanması
        else:
            pitch_ort = 0.0                     # hiç tonlu çerçeve yoksa (fısıltı/sessizlik)
            pitch_std = 0.0
        tonlu_oran = float(np.mean(voiced_flag))             # aralığın ne kadarı tonlu
        skalerler += [pitch_ort, pitch_std, tonlu_oran]
    if config.use_jitter:
        # Jitter: ardışık perde değerleri arasındaki ortalama görece değişim.
        # Yüksek jitter = "titrek/gergin ses" -> duygusal uyarılma göstergesi.
        if len(f0_tonlu) >= 2:
            jitter = float(np.mean(np.abs(np.diff(f0_tonlu))) / (np.mean(f0_tonlu) + 1e-6))
        else:
            jitter = 0.0
        # Shimmer: ardışık genlik (enerji) değerleri arasındaki ortalama görece değişim.
        if len(rms) >= 2:
            shimmer = float(np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-6))
        else:
            shimmer = 0.0
        skalerler += [jitter, shimmer]
    if config.use_contrast:
        # Spektral kontrast: her frekans bandında tepe-vadi enerji farkı. Sesin
        # "tokluğu/boğukluğu" -> MFCC'nin yakalamadığı tamamlayıcı doku bilgisi.
        kontrast = librosa.feature.spectral_contrast(S=stft, sr=config.sample_rate)
        skalerler += [float(v) for v in kontrast.mean(axis=1)]   # 7 bant ortalaması
    if config.use_bandwidth:
        # Spektral bant genişliği: enerjinin merkez etrafındaki yayılımı.
        # Spektral düzlük (flatness): sesin ne kadar "gürültü gibi" (1) veya
        # "tonlu" (0) olduğu — konuşma/gürültü ayrımına duyarlı.
        bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=config.sample_rate)[0]
        flatness = librosa.feature.spectral_flatness(S=stft)[0]
        skalerler += [
            float(bandwidth.mean() / (config.sample_rate / 2)),  # 0-1 ölçekli
            float(flatness.mean()),
        ]

    parts = [
        mfcc.mean(axis=1),      # 13 değer: ortalama tını
        mfcc.std(axis=1),       # 13 değer: tınının aralık içi değişkenliği
        delta.mean(axis=1),     # 13 değer: ortalama değişim hızı
    ]
    if config.use_delta2:
        parts.append(delta2.mean(axis=1))   # 13 değer: ortalama ivme
    parts.append(skalerler)
    vector = np.concatenate([np.asarray(p, dtype=np.float32).ravel() for p in parts])
    if vector.shape != (config.feature_dim,):
        raise ValueError(f'Beklenmeyen aralık vektörü boyutu: {vector.shape}.')
    return vector


def extract_interval_series(
    audio_path: str | Path,
    config: IntervalConfig,
    audio: np.ndarray | None = None,
) -> np.ndarray:
    '''Bir kaydı [n_intervals, feature_dim] boyutlu float32 seriye çevirir.

    RNN bu seriyi satır satır (aralık aralık) okuyarak duygunun zaman
    içindeki seyrini modeller.

    ``audio`` verilirse dosyadan yükleme atlanır ve doğrudan bu dalga formu
    kullanılır (ör. veri artırma için pitch-shift edilmiş bir kayıt). None
    (varsayılan) ise davranış değişmez: kayıt ``audio_path``'ten yüklenir.
    '''

    config.validate()
    if audio is None:
        audio = _load_audio(audio_path, config.sample_rate)
    window = config.interval_samples
    rows = []
    for start in interval_starts(len(audio), config):
        segment = audio[start : start + window]
        # Kaydın sonuna denk gelen pencere kısa kalırsa sıfırla tamamla.
        if len(segment) < window:
            segment = np.pad(segment, (0, window - len(segment)))
        rows.append(_interval_features(segment, config))
    series = np.stack(rows).astype(np.float32, copy=False)
    if series.shape != config.shape or not np.isfinite(series).all():
        raise ValueError(
            f'Geçersiz aralık serisi ({audio_path}): shape={series.shape}.'
        )
    return series
