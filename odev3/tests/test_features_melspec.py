'''MLP'nin kullandığı mel-spektrogram öznitelik katmanının testleri.

Öznitelik katmanı boru hattının ilk halkasıdır; buradaki bir hata her şeyi
zehirler. Testler dört alanı kapsar: ödevin boyut şartı, önbellek parmak
izinin parametre duyarlılığı, zaman-sabitleme (fix_frames) stratejilerinin
matematiği ve uçtan uca çıkarımın (gerçek bir WAV dosyasıyla) sağlamlığı.
'''

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from odev3.features_melspec import (
    DEFAULT_CONFIG,
    MelSpecConfig,
    extract_melspec,
    fix_frames,
)


def _write_tone(path: Path, *, frequency: float = 220.0) -> None:
    # Test için 1 saniyelik saf sinüs tonu üretip WAV olarak yazar.
    # Gerçek konuşma kaydına ihtiyaç yok: çıkarımın şekil/sonluluk
    # garantilerini sınamak için deterministik bir sinyal yeterli.
    sample_rate = DEFAULT_CONFIG.sample_rate
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = 0.25 * np.sin(2.0 * np.pi * frequency * time)
    sf.write(path, audio, sample_rate)


def test_default_vector_meets_assignment_minimum() -> None:
    '''Varsayılan konfig 64x64=4096 boyut üretmeli ve ödevin 4000 alt sınırını sağlamalı.'''

    assert DEFAULT_CONFIG.vector_size == 4096
    assert DEFAULT_CONFIG.vector_size >= 4000


def test_config_fingerprint_changes_with_feature_parameters() -> None:
    '''Parmak izi parametre değişiminde değişmeli, aynı ayarlarda aynı kalmalı.

    Bu, önbellek güvenliğinin özüdür: farklı ayarların ürettiği vektörler
    farklı klasörlere gider (fingerprint farklı), aynı ayarlar ise önbelleği
    yeniden kullanabilir (fingerprint aynı).
    '''

    changed = MelSpecConfig(hop_length=256)
    resized = MelSpecConfig(frame_strategy='resize')

    assert DEFAULT_CONFIG.fingerprint != changed.fingerprint
    assert DEFAULT_CONFIG.fingerprint != resized.fingerprint
    assert DEFAULT_CONFIG.fingerprint == MelSpecConfig().fingerprint


@pytest.mark.parametrize(
    'config',
    [
        MelSpecConfig(sample_rate=0),
        MelSpecConfig(n_frames=-1),
        MelSpecConfig(fmin=100.0, fmax=50.0),
        MelSpecConfig(top_db=0.0),
        MelSpecConfig(frame_strategy='unknown'),
    ],
)
def test_invalid_configs_are_rejected(config: MelSpecConfig) -> None:
    '''Fiziksel/mantıksal olarak anlamsız ayarlar (sıfır örnekleme hızı, ters frekans bandı vb.) reddedilmeli.'''

    with pytest.raises(ValueError):
        config.validate()


def test_fix_frames_right_pads_short_input_with_floor() -> None:
    '''crop_pad: kısa matris sağdan, matrisin EN DÜŞÜK değeriyle doldurulmalı.

    Buradaki min -4.0; dolgu sütunlarının tamamı -4.0 olmalı ki eklenen
    kısım gerçek sessizlik tabanıyla tutarlı olsun.
    '''

    mel = np.array([[2.0, 3.0], [-4.0, 5.0]], dtype=np.float32)

    fixed = fix_frames(mel, n_frames=4)

    assert fixed.shape == (2, 4)
    np.testing.assert_array_equal(fixed[:, :2], mel)
    np.testing.assert_array_equal(fixed[:, 2:], np.full((2, 2), -4.0))


def test_fix_frames_centre_crops_long_input() -> None:
    '''crop_pad: uzun matris ORTADAN kırpılmalı (9 sütundan 5 istenince 2:7 dilimi).'''

    mel = np.arange(18, dtype=np.float32).reshape(2, 9)

    fixed = fix_frames(mel, n_frames=5)

    np.testing.assert_array_equal(fixed, mel[:, 2:7])


def test_fix_frames_resize_uses_the_complete_time_axis() -> None:
    '''resize: 5 sütun 3'e inerken uçlar korunmalı, ara değer doğrusal interpolasyonla gelmeli.

    Beklenen sütunlar kaynak eksenin 0, 0.5 ve 1.0 konumlarına denk gelir;
    yani ilk ve son sütun aynen kalır, ortadaki tam orta değer olur.
    '''

    mel = np.array(
        [[0.0, 10.0, 20.0, 30.0, 40.0], [5.0, 15.0, 25.0, 35.0, 45.0]],
        dtype=np.float32,
    )

    fixed = fix_frames(mel, n_frames=3, strategy='resize')

    expected = np.array([[0.0, 20.0, 40.0], [5.0, 25.0, 45.0]], dtype=np.float32)
    np.testing.assert_allclose(fixed, expected)


def test_fix_frames_resize_repeats_a_single_frame() -> None:
    '''resize: tek sütunluk girişte interpolasyon tanımsızdır; sütun tekrarlanmalı.'''

    mel = np.array([[2.0], [-4.0]], dtype=np.float32)

    fixed = fix_frames(mel, n_frames=4, strategy='resize')

    np.testing.assert_array_equal(
        fixed,
        np.array([[2.0, 2.0, 2.0, 2.0], [-4.0, -4.0, -4.0, -4.0]]),
    )


def test_extract_melspec_returns_finite_float32_vector(tmp_path: Path) -> None:
    '''Uçtan uca çıkarım: gerçek WAV -> 4096'lık, sonlu, float32 vektör.

    dB ölçeği referansa göre hesaplandığından değerler yaklaşık
    [-top_db, 0] aralığında kalmalıdır; 1e-4 payı float yuvarlama içindir.
    '''

    audio_path = tmp_path / 'tone.wav'
    _write_tone(audio_path)

    vector = extract_melspec(audio_path)

    assert vector.shape == (4096,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()
    assert float(vector.max()) <= 1e-4
    assert float(vector.min()) >= -DEFAULT_CONFIG.top_db - 1e-4


def test_extract_melspec_rejects_missing_file(tmp_path: Path) -> None:
    '''Var olmayan ses dosyası, belirsiz bir librosa hatası yerine FileNotFoundError vermeli.'''

    with pytest.raises(FileNotFoundError):
        extract_melspec(tmp_path / 'missing.wav')
