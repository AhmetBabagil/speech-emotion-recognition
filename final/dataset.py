'''Öznitelik önbelleği, yalnız-eğitimle normalizasyon ve bellek-içi veri kümeleri.

İki yöntem de kayıt başına sabit boyutlu bir float32 dizi ürettiği için tek
bir önbellek ve tek bir yükleyici, hem CNN görüntülerini hem RNN serilerini
karşılar. Öznitelik çıkarımı pahalı olduğundan her sonuç diske yazılır ve
sonraki denemelerde saniyeler içinde geri okunur.
'''

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import threading
from typing import Callable, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


def feature_cache_path(
    audio_path: str | Path,
    cache_dir: str | Path,
    fingerprint: str,
) -> Path:
    '''Kaynak dosyaya ve öznitelik ayarlarına özel, kararlı önbellek yolu üretir.

    Yol iki şeye bağlıdır:
    1. fingerprint: öznitelik ayarlarının kimliği (klasör adı) — farklı
       ayarların çıktıları karışamaz.
    2. Kaynak dosyanın yolu + boyutu + değişiklik zamanı — ses dosyası
       değişirse eski önbellek otomatik geçersizleşir.
    '''

    audio_path = Path(audio_path)
    stat = audio_path.stat()
    identity = (
        f'{os.path.normcase(str(audio_path.resolve()))}\0{stat.st_size}\0{stat.st_mtime_ns}'
    )
    digest = hashlib.sha1(identity.encode('utf-8')).hexdigest()[:16]
    return Path(cache_dir) / fingerprint / f'{audio_path.stem[:48]}_{digest}.npy'


def load_or_extract(
    audio_path: str | Path,
    cache_dir: str | Path,
    fingerprint: str,
    extract: Callable[[str | Path], np.ndarray],
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    '''Geçerli bir önbellek kaydı varsa onu döndürür; yoksa hesaplayıp saklar.'''

    cache_path = feature_cache_path(audio_path, cache_dir, fingerprint)
    if cache_path.is_file():
        try:
            cached = np.asarray(np.load(cache_path, allow_pickle=False), dtype=np.float32)
            # Boyut ve sonluluk kontrolü: bozuk kayıt varsa aşağıda yeniden üretilir.
            if cached.shape == expected_shape and np.isfinite(cached).all():
                return cached
        except (OSError, ValueError):
            pass  # yarım kalmış/bozuk önbellek dosyası aşağıda değiştirilir

    array = extract(audio_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Önce geçici ada yazıp sonra atomik os.replace: iki işlem aynı anda
    # yazsa bile dosya asla yarım hâlde okunmaz.
    temp_path = cache_path.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.tmp.npy')
    np.save(temp_path, array, allow_pickle=False)
    os.replace(temp_path, cache_path)
    return array


def _progress(iterator: Iterable, total: int, description: str, enabled: bool):
    '''tqdm kuruluysa ilerleme çubuğu göster; değilse sessizce devam et.'''

    if not enabled:
        return iterator
    try:
        from tqdm import tqdm

        return tqdm(iterator, total=total, desc=description, unit='ses')
    except ImportError:
        return iterator


def load_feature_tensor(
    records,
    cache_dir: str | Path,
    fingerprint: str,
    extract: Callable[[str | Path], np.ndarray],
    expected_shape: tuple[int, ...],
    *,
    workers: int = 1,
    show_progress: bool = True,
    description: str = 'Öznitelikler',
) -> tuple[np.ndarray, np.ndarray]:
    '''Bir katmanın (fold) tüm kayıtlarını tek bir [N, ...] tensöre toplar.

    workers > 1 verilirse dosyalar iş parçacıklarıyla paralel işlenir
    (öznitelik çıkarımı G/Ç + librosa ağırlıklı olduğu için thread yeterli).
    '''

    records = records.reset_index(drop=True)
    paths = records['path'].astype(str).tolist()
    labels = records['label_idx'].to_numpy(dtype=np.int64, copy=True)
    if not paths:
        raise ValueError('Boş katmandan öznitelik üretilemez.')

    def load_one(path: str) -> np.ndarray:
        return load_or_extract(path, cache_dir, fingerprint, extract, expected_shape)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            arrays = list(
                _progress(executor.map(load_one, paths), len(paths), description, show_progress)
            )
    else:
        arrays = list(_progress(map(load_one, paths), len(paths), description, show_progress))

    features = np.stack(arrays).astype(np.float32, copy=False)
    if features.shape != (len(records), *expected_shape):
        raise ValueError(f'Beklenmeyen öznitelik tensörü boyutu: {features.shape}.')
    return features, labels


@dataclass(frozen=True)
class Standardizer:
    '''YALNIZCA eğitim katmanından öğrenilen z-skor parametreleri.

    Neden önemli: ortalama/std'yi tüm veriden öğrenmek, test bilgisinin
    eğitime sızması demektir (data leakage). Burada fit() sadece eğitim
    verisiyle çağrılır; geçerleme ve test aynı parametrelerle dönüştürülür.

    Eksen mantığı: [N, mels, T] mel görüntülerinde istatistik mel bandı
    başına (feature_axis=1), [N, T, D] serilerde öznitelik boyutu başına
    (feature_axis=2) tutulur; kalan eksenler (örnek + zaman) üzerinden
    ortalama alınır.
    '''

    mean: np.ndarray
    scale: np.ndarray
    feature_axis: int

    @classmethod
    def fit(cls, features: np.ndarray, feature_axis: int, epsilon: float = 1e-6) -> 'Standardizer':
        features = np.asarray(features)
        if features.ndim != 3 or len(features) == 0:
            raise ValueError(f'Boş olmayan 3 boyutlu tensör bekleniyor: {features.shape}.')
        if not np.isfinite(features).all():
            raise ValueError('Eğitim öznitelikleri sonlu olmalı.')
        axes = tuple(axis for axis in range(features.ndim) if axis != feature_axis)
        mean = features.mean(axis=axes, dtype=np.float64).astype(np.float32)
        scale = features.std(axis=axes, dtype=np.float64).astype(np.float32)
        # Sabit (varyanssız) boyutlarda sıfıra bölmeyi önle.
        scale = np.where(scale < epsilon, 1.0, scale)
        return cls(mean=mean, scale=scale, feature_axis=feature_axis)

    def transform(self, features: np.ndarray) -> np.ndarray:
        '''(x - ortalama) / std dönüşümünü doğru eksende uygular.'''

        features = np.asarray(features, dtype=np.float32)
        # mean/scale vektörlerini yayın (broadcast) için doğru şekle getir.
        shape = [1] * features.ndim
        shape[self.feature_axis] = self.mean.shape[0]
        mean = self.mean.reshape(shape)
        scale = self.scale.reshape(shape)
        return np.ascontiguousarray((features - mean) / scale, dtype=np.float32)


class ArrayDataset(Dataset):
    '''Hiperparametre denemeleri boyunca yeniden kullanılan bellek-içi tensörler.

    Tüm öznitelikler zaten RAM'e sığdığı için diskten tekrar tekrar okumak
    yerine bir kez tensöre çevirip DataLoader'a veriyoruz — denemeler arası
    en hızlı yol bu.
    '''

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        features = np.ascontiguousarray(features, dtype=np.float32)
        labels = np.ascontiguousarray(labels, dtype=np.int64)
        if labels.ndim != 1 or len(features) != len(labels):
            raise ValueError(
                f'Geçersiz öznitelik/etiket boyutları: {features.shape}, {labels.shape}.'
            )
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]
