'''PyTorch MLP için sabit boyutlu log-mel öznitelikleri.

MLP'nin girişi sabit uzunlukta bir vektör olmak zorundadır; oysa ses
kayıtları farklı sürelerdedir. Bu modül her ses dosyasını şu adımlarla
sabit boyutlu tek bir vektöre dönüştürür:

1. Sesi tek kanala (mono) indirip sabit örnekleme hızına yeniden örnekle.
2. Mel spektrogram hesapla (insan kulağının frekans algısına yakın bir
   frekans ekseni kullanan bir tür zaman-frekans temsili).
3. Güç değerlerini desibele (log ölçek) çevir — küçük enerji farkları da
   görünür olsun diye.
4. Zaman eksenini tam olarak ``n_frames`` sütuna sabitle (kırp/doldur ya da
   yeniden boyutlandır).
5. Matrisi düzleştirip (n_mels * n_frames) uzunluğunda float32 vektör yap.
'''

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np


@dataclass(frozen=True)
class MelSpecConfig:
    '''Önbelleğe alınan öznitelik vektörünü etkileyen TÜM parametreler.

    Bu sınıfın var olma sebebi önbellek (cache) tutarlılığıdır: bu
    parametrelerden herhangi biri değişirse üretilen vektör de değişir.
    Hepsini tek bir değişmez nesnede toplayıp aşağıdaki ``fingerprint``
    özelliğiyle kısa bir kimliğe çevirince, farklı ayarlarla üretilmiş
    önbellekler birbirine karışamaz.
    '''

    sample_rate: int = 16_000
    n_mels: int = 64
    n_frames: int = 64
    n_fft: int = 1024
    hop_length: int = 512
    fmin: float = 20.0
    fmax: float = 8_000.0
    top_db: float = 80.0
    frame_strategy: str = 'crop_pad'

    def validate(self) -> None:
        '''Anlamlı bir mel vektörü üretemeyecek ayarları reddeder.'''

        # Tüm tamsayı alanlar pozitif olmalı; hangisinin hatalı olduğunu
        # tek tek listeleyerek hata mesajını bilgilendirici yapıyoruz.
        integer_fields = {
            'sample_rate': self.sample_rate,
            'n_mels': self.n_mels,
            'n_frames': self.n_frames,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
        }
        invalid = {name: value for name, value in integer_fields.items() if value <= 0}
        if invalid:
            raise ValueError(f'Mel parameters must be positive; invalid={invalid}.')
        # Frekans bandı mantıklı olmalı: 0 <= fmin < fmax.
        if self.fmin < 0.0 or self.fmax <= self.fmin:
            raise ValueError(
                f'Expected 0 <= fmin < fmax, got fmin={self.fmin}, '
                f'fmax={self.fmax}.'
            )
        if self.top_db <= 0.0:
            raise ValueError(f'top_db must be positive, got {self.top_db}.')
        # Zaman eksenini sabitlemek için desteklenen iki strateji var;
        # yazım hatalarını burada yakalıyoruz.
        if self.frame_strategy not in {'crop_pad', 'resize'}:
            raise ValueError(
                'frame_strategy must be crop_pad or resize, got '
                f'{self.frame_strategy!r}.'
            )

    @property
    def vector_size(self) -> int:
        # Düzleştirilmiş vektörün uzunluğu = mel bandı sayısı x kare sayısı.
        # MLP'nin giriş katmanı bu boyuta göre kurulur.
        return self.n_mels * self.n_frames

    @property
    def fingerprint(self) -> str:
        '''Uyumsuz önbellekleri ayrı tutmak için kısa ve kararlı bir kimlik.

        Config'i deterministik JSON'a çevirip (anahtarlar sıralı, boşluksuz)
        SHA-1 özetinin ilk 12 karakterini alırız. Aynı ayarlar her zaman aynı
        kimliği üretir; tek bir parametre bile değişse kimlik değişir ve yeni
        bir önbellek dosyası oluşturulur.
        '''

        self.validate()
        payload = json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))
        return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]


DEFAULT_CONFIG = MelSpecConfig()

# Geriye dönük uyumluluk için sabitler: not defterleri (notebook) ve testler
# bu isimleri doğrudan kullanıyor. Tek doğruluk kaynağı DEFAULT_CONFIG'tir;
# buradaki sabitler yalnızca ona birer takma addır.
SAMPLE_RATE = DEFAULT_CONFIG.sample_rate
N_MELS = DEFAULT_CONFIG.n_mels
N_FRAMES = DEFAULT_CONFIG.n_frames
N_FFT = DEFAULT_CONFIG.n_fft
HOP_LENGTH = DEFAULT_CONFIG.hop_length
VECTOR_SIZE = DEFAULT_CONFIG.vector_size


def fix_frames(
    mel_db: np.ndarray,
    n_frames: int = N_FRAMES,
    *,
    strategy: str = 'crop_pad',
) -> np.ndarray:
    '''Bir log-mel matrisini tam olarak n_frames zaman sütununa dönüştürür.

    Ses kayıtları farklı uzunlukta olduğundan spektrogramın sütun sayısı da
    değişkendir; MLP ise sabit boyut ister. İki strateji sunuyoruz:

    - ``crop_pad``: Uzunsa ortadan kırp (konuşmanın önemli kısmı genelde
      ortadadır), kısaysa sağdan sessizlik değeriyle doldur. Zaman ölçeğini
      korur; hız/tempo bilgisi bozulmaz.
    - ``resize``: Zaman eksenini doğrusal interpolasyonla n_frames'e esnet
      ya da sıkıştır. Kaydın tamamını kullanır ama tempoyu deforme eder.

    Hangisinin daha iyi çalıştığı deneysel bir sorudur; bu yüzden ikisi de
    hiperparametre olarak aranabilir durumda.
    '''

    mel_db = np.asarray(mel_db, dtype=np.float32)
    if mel_db.ndim != 2 or mel_db.shape[0] == 0:
        raise ValueError(f'Expected a non-empty 2-D mel matrix, got {mel_db.shape}.')
    if n_frames <= 0:
        raise ValueError(f'n_frames must be positive, got {n_frames}.')
    if strategy not in {'crop_pad', 'resize'}:
        raise ValueError(f'Unknown frame strategy: {strategy!r}.')

    current_frames = mel_db.shape[1]
    if current_frames == 0:
        # Hiç sütun yoksa (aşırı kısa ses) tümüyle "sessizlik" (-80 dB,
        # yani top_db tabanı) döndürürüz; boru hattı çökmesin diye.
        return np.full((mel_db.shape[0], n_frames), -80.0, dtype=np.float32)
    if strategy == 'resize' and current_frames != n_frames:
        if current_frames == 1:
            # Tek sütun varsa interpolasyon tanımsızdır; sütunu tekrarlayarak
            # hedef genişliğe ulaşırız.
            return np.repeat(mel_db, n_frames, axis=1).astype(
                np.float32, copy=False
            )
        # Her mel bandını (satırı) [0,1] aralığına yerleştirilmiş konumlar
        # üzerinden 1-D doğrusal interpolasyonla yeni genişliğe taşırız.
        source_positions = np.linspace(0.0, 1.0, current_frames)
        target_positions = np.linspace(0.0, 1.0, n_frames)
        resized = np.empty((mel_db.shape[0], n_frames), dtype=np.float32)
        for mel_index, row in enumerate(mel_db):
            resized[mel_index] = np.interp(target_positions, source_positions, row)
        return resized
    if current_frames < n_frames:
        # crop_pad + kısa kayıt: sağa, matrisin en düşük değeriyle (pratikte
        # sessizlik tabanı) dolgu yaparız. Sabit bir sayı yerine min kullanmak,
        # dolgunun gerçek sessizlik seviyesiyle tutarlı olmasını sağlar.
        floor = float(np.min(mel_db))
        return np.pad(
            mel_db,
            ((0, 0), (0, n_frames - current_frames)),
            mode='constant',
            constant_values=floor,
        ).astype(np.float32, copy=False)
    if current_frames > n_frames:
        # crop_pad + uzun kayıt: ortadan n_frames genişliğinde pencere kes.
        # Baş/son yerine ORTA tercih edilir çünkü kayıtların başı ve sonu
        # çoğu zaman sessizlik içerir.
        start = (current_frames - n_frames) // 2
        return mel_db[:, start : start + n_frames]
    # Zaten doğru genişlikteyse dokunmadan geri ver.
    return mel_db


def extract_melspec(
    audio_path: str | Path,
    config: MelSpecConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    '''Tek bir ses dosyasını sonlu değerli float32 log-mel vektörüne çevirir.

    Fonksiyonun sonunda katı bir doğrulama vardır: şekil doğru mu, tüm
    değerler sonlu mu? Bozuk bir dosyanın NaN üretip eğitimi sessizce
    zehirlemesindense burada gürültülü şekilde hata vermesini isteriz.
    '''

    config.validate()
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f'Audio file does not exist: {audio_path}')

    # librosa.load: dosyayı okur, mono'ya indirir ve config'teki örnekleme
    # hızına yeniden örnekler; böylece tüm kayıtlar aynı zaman çözünürlüğüne
    # sahip olur.
    audio, _ = librosa.load(audio_path, sr=config.sample_rate, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f'Audio file is empty: {audio_path}')
    if not np.isfinite(audio).all():
        raise ValueError(f'Audio file contains non-finite samples: {audio_path}')

    # power=2.0: genlik yerine güç spektrogramı (genliğin karesi).
    # fmax'i Nyquist frekansıyla (sample_rate/2) sınırlıyoruz; üstü fiziksel
    # olarak temsil edilemez ve librosa uyarı üretirdi.
    mel_power = librosa.feature.melspectrogram(
        y=audio,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=min(config.fmax, config.sample_rate / 2),
        power=2.0,
    )
    # Pozitif bir referans, sessiz kayıtlarda sıfıra bölmeyi engeller.
    # (Tamamen sessiz bir dosyada max=0 olurdu; tiny en küçük pozitif float.)
    reference = max(float(np.max(mel_power)), float(np.finfo(np.float32).tiny))
    # Güç -> desibel: log ölçek, insan işitmesine daha yakın ve MLP için
    # daha dengeli bir değer aralığı üretir. top_db en düşük seviyeyi
    # (ref - top_db) ile sınırlar, yani -80 dB altı kırpılır.
    mel_db = librosa.power_to_db(mel_power, ref=reference, top_db=config.top_db)
    mel_fixed = fix_frames(
        mel_db,
        config.n_frames,
        strategy=config.frame_strategy,
    )
    # 2-D matrisi tek boyuta düzleştir: MLP'nin beklediği vektör biçimi.
    vector = mel_fixed.reshape(config.vector_size).astype(np.float32, copy=False)

    # Son emniyet kontrolü: yukarıdaki adımlardan herhangi biri beklenmedik
    # bir çıktı ürettiyse (yanlış şekil, NaN/inf) hemen ve açık hata ver.
    if vector.shape != (config.vector_size,) or not np.isfinite(vector).all():
        raise ValueError(
            f'Invalid mel vector for {audio_path}: shape={vector.shape}, '
            f'finite={bool(np.isfinite(vector).all())}'
        )
    return vector
