'''Öznitelik önbelleği ve yalnızca-eğitim normalizasyonunun testleri.

dataset.py'nin üç kritik davranışı sınanır:
1. Önbellek anahtarı kaynak dosya değişince değişiyor mu; geçerli önbellek
   yeniden kullanılıyor, bozuk önbellek onarılıyor mu?
2. Standardizer istatistikleri doğru mu; sabit sütunlar ve NaN/inf girişler
   doğru ele alınıyor mu?
3. Paralel öznitelik yükleme, satır sırasını (etiket hizasını) koruyor mu?

Testlerde gerçek ses çıkarımı yapılmaz: monkeypatch ile extract_melspec /
load_or_extract sahte (fake) fonksiyonlarla değiştirilir. Böylece testler
hızlı ve librosa'dan bağımsız kalır; sadece dataset.py'nin mantığı ölçülür.
'''

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from odev3 import dataset as dataset_module
from odev3.dataset import (
    ArrayDataset,
    FeatureStandardizer,
    feature_cache_path,
    load_feature_matrix,
    load_or_extract,
)
from odev3.features_melspec import DEFAULT_CONFIG


def test_feature_cache_path_changes_when_source_file_changes(tmp_path: Path) -> None:
    '''Kaynak dosyanın içeriği değişince önbellek yolu da değişmeli.

    Yol; dosyanın boyutu ve değişiklik zamanından türetilen özeti içerir.
    Dosya güncellenirse eski önbellek otomatik olarak "kayıp" olur ve
    yeni içerik için yeni vektör üretilir — bayat önbellek riski sıfırlanır.
    '''

    audio_path = tmp_path / 'clip.wav'
    audio_path.write_bytes(b'first version')
    first_path = feature_cache_path(audio_path, tmp_path / 'cache')

    audio_path.write_bytes(b'a different second version')
    second_path = feature_cache_path(audio_path, tmp_path / 'cache')

    assert first_path != second_path


def test_load_or_extract_reuses_valid_cache(tmp_path: Path, monkeypatch) -> None:
    '''Geçerli bir önbellek varken çıkarım İKİNCİ kez çağrılmamalı.

    Sahte extract_melspec bir sayaç tutar: iki load_or_extract çağrısına
    rağmen sayaç 1'de kalmalı (ikinci çağrı diskten okumalı) ve iki sonuç
    da beklenen vektörle aynı olmalı.
    '''

    audio_path = tmp_path / 'clip.wav'
    audio_path.write_bytes(b'placeholder audio')
    expected = np.arange(DEFAULT_CONFIG.vector_size, dtype=np.float32)
    calls = 0

    def fake_extract(path, config):
        nonlocal calls
        calls += 1
        assert Path(path) == audio_path
        return expected.copy()

    monkeypatch.setattr(dataset_module, 'extract_melspec', fake_extract)

    first = load_or_extract(audio_path, tmp_path / 'cache')
    second = load_or_extract(audio_path, tmp_path / 'cache')

    assert calls == 1
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)


def test_load_or_extract_replaces_invalid_cache(tmp_path: Path, monkeypatch) -> None:
    '''Yanlış boyutlu (bozuk) bir önbellek girdisi sessizce yenisiyle değiştirilmeli.

    Önbelleğe kasten 2 elemanlık (geçersiz) bir dizi yazılır; load_or_extract
    bunu reddetmeli, çıkarımı çalıştırmalı ve diskteki dosyayı doğru vektörle
    güncellemeli.
    '''

    audio_path = tmp_path / 'clip.wav'
    audio_path.write_bytes(b'placeholder audio')
    cache_path = feature_cache_path(audio_path, tmp_path / 'cache')
    cache_path.parent.mkdir(parents=True)
    np.save(cache_path, np.array([1.0, 2.0], dtype=np.float32))
    expected = np.full(DEFAULT_CONFIG.vector_size, 7.0, dtype=np.float32)

    monkeypatch.setattr(
        dataset_module,
        'extract_melspec',
        lambda path, config: expected.copy(),
    )

    actual = load_or_extract(audio_path, tmp_path / 'cache')

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.load(cache_path), expected)


def test_standardizer_uses_training_statistics_and_handles_constant_column() -> None:
    '''fit+transform sonrası ortalama ~0, std ~1 olmalı; sabit sütun sıfıra dönüşmeli.

    Üçüncü sütun tamamen 5.0 (sabit): std=0 olduğu için ölçek 1.0'a
    zorlanmalı ve dönüşüm sonucu sıfır olmalı — sıfıra bölme çökmesi yok.
    '''

    train = np.array(
        [
            [1.0, 10.0, 5.0],
            [3.0, 20.0, 5.0],
            [5.0, 30.0, 5.0],
        ],
        dtype=np.float32,
    )

    standardizer = FeatureStandardizer.fit(train)
    transformed = standardizer.transform(train)

    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(transformed[:, :2].std(axis=0), 1.0, atol=1e-6)
    np.testing.assert_array_equal(transformed[:, 2], np.zeros(3, dtype=np.float32))
    assert standardizer.scale[2] == 1.0


def test_standardizer_rejects_non_finite_training_values() -> None:
    '''Eğitim matrisinde NaN varsa fit açık bir hatayla durmalı (sessiz NaN yayılımı yasak).'''

    train = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)

    with pytest.raises(ValueError, match='finite'):
        FeatureStandardizer.fit(train)


def test_standardizer_rejects_non_finite_transform_values() -> None:
    '''Dönüştürülecek veride sonsuz değer varsa transform da reddedilmeli.'''

    standardizer = FeatureStandardizer.fit(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    )

    with pytest.raises(ValueError, match='finite'):
        standardizer.transform(np.array([[np.inf, 2.0]], dtype=np.float32))


def test_array_dataset_converts_numpy_types_for_pytorch() -> None:
    '''ArrayDataset, numpy tiplerini PyTorch'un beklediği tiplere çevirmeli.

    float64 öznitelikler float32'ye, int32 etiketler long'a (int64) inmeli;
    CrossEntropyLoss etiketlerde long ister, aksi çalışma zamanı hatasıdır.
    '''

    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    labels = np.array([2, 1], dtype=np.int32)

    dataset = ArrayDataset(features, labels)
    first_features, first_label = dataset[0]

    assert len(dataset) == 2
    assert first_features.dtype == torch.float32
    assert first_label.dtype == torch.long
    torch.testing.assert_close(first_features, torch.tensor([1.0, 2.0]))
    assert first_label.item() == 2


def test_parallel_feature_loading_preserves_record_order(tmp_path: Path, monkeypatch) -> None:
    '''workers=2 ile paralel yükleme, satır sırasını ve etiket hizasını bozmamalı.

    Bu, veri bütünlüğünün en sinsi hatasına karşı bir sigortadır: paralel
    yükleme sırası karıştırsaydı öznitelikler yanlış etiketlerle eşleşir,
    model "çalışır" ama saçma öğrenirdi. Her sahte dosyaya farklı sabit değer
    verilerek çıktı sırasının girdi sırasıyla aynı olduğu doğrulanır.
    '''

    records = pd.DataFrame(
        {
            'path': ['first.wav', 'second.wav', 'third.wav'],
            'label_idx': [2, 0, 4],
        }
    )
    values = {'first.wav': 1.0, 'second.wav': 2.0, 'third.wav': 3.0}

    def fake_load(path, cache_dir, config):
        return np.full(config.vector_size, values[path], dtype=np.float32)

    monkeypatch.setattr(dataset_module, 'load_or_extract', fake_load)

    features, labels = load_feature_matrix(
        records,
        tmp_path / 'cache',
        workers=2,
        show_progress=False,
    )

    np.testing.assert_array_equal(features[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(labels, [2, 0, 4])
