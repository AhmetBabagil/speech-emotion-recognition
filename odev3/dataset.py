'''Öznitelik önbelleği, yalnızca eğitim kümesiyle normalizasyon ve PyTorch veri kümeleri.

Bu modül üç sorunu çözer:

1. **Önbellek (cache)**: Mel öznitelik çıkarımı yavaştır (her dosya için ses
   yükleme + STFT). Aynı dosyayı her deneyde yeniden işlemek yerine sonucu
   diske .npy olarak kaydedip sonraki çalıştırmalarda doğrudan okuruz.
2. **Standardizasyon**: Öznitelik boyutlarını z-score ile ölçekleriz; ama
   ortalama/std SADECE eğitim kümesinden hesaplanır. Doğrulama/test
   istatistiklerini karıştırmak "veri sızıntısı" (data leakage) olur ve
   skorları yapay olarak şişirirdi.
3. **Dataset sınıfları**: PyTorch DataLoader'ın beklediği arayüzü sağlayan
   iki varyant — diskten tembel okuyan ``EmotionDataset`` ve tümüyle RAM'de
   duran hızlı ``ArrayDataset``.
'''

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import threading
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from odev3.features_melspec import DEFAULT_CONFIG, MelSpecConfig, extract_melspec


def feature_cache_path(
    audio_path: str | Path,
    cache_dir: str | Path,
    config: MelSpecConfig = DEFAULT_CONFIG,
) -> Path:
    '''Dosyaya VE öznitelik ayarlarına özgü, kararlı bir önbellek yolu döndürür.

    Önbellek anahtarı iki şeyden oluşur:
    - ``config.fingerprint``: öznitelik ayarlarının kısa özeti — ayar değişirse
      alt klasör değişir, eski önbellek yanlışlıkla kullanılamaz.
    - Kaynak dosyanın kimliği: normalize edilmiş tam yol + boyut + değişiklik
      zamanı (nanosaniye). Dosya güncellenirse mtime değişir, dolayısıyla
      özet de değişir ve bayat önbellek otomatik olarak geçersiz kalır.
    '''

    audio_path = Path(audio_path)
    source_stat = audio_path.stat()
    # normcase: Windows'ta büyük/küçük harf farkını yok sayar; aynı dosyaya
    # farklı yazımlarla erişmek aynı önbellek girdisini üretsin diye.
    normalized = os.path.normcase(str(audio_path.resolve()))
    # '\0' ayracı, alanların birbirine karışmasını imkansız kılar (dosya
    # adlarında null karakter olamaz).
    source_identity = (
        f'{normalized}\0{source_stat.st_size}\0{source_stat.st_mtime_ns}'
    )
    digest = hashlib.sha1(source_identity.encode('utf-8')).hexdigest()[:16]
    # Dosya adının başına okunabilir bir parça (stem) koyuyoruz ki önbellek
    # klasörüne bakan biri hangi girdinin hangi kayda ait olduğunu görebilsin;
    # 48 karakter sınırı Windows'un yol uzunluğu limitine takılmamak için.
    safe_stem = audio_path.stem[:48]
    return Path(cache_dir) / config.fingerprint / f'{safe_stem}_{digest}.npy'


def _valid_vector(vector: np.ndarray, config: MelSpecConfig) -> bool:
    # Önbellekten okunan verinin hâlâ kullanılabilir olduğunu doğrular:
    # doğru uzunlukta mı, sayısal mı, NaN/inf içermiyor mu? Bozuk bir .npy
    # dosyası (örn. yarım yazılmış) bu kontrole takılır ve yeniden üretilir.
    return (
        vector.shape == (config.vector_size,)
        and np.issubdtype(vector.dtype, np.number)
        and bool(np.isfinite(vector).all())
    )


def load_or_extract(
    audio_path: str | Path,
    cache_dir: str | Path,
    config: MelSpecConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    '''Geçerli bir önbellek girdisini yükler; yoksa atomik olarak üretip kaydeder.

    "Atomik" yazma şu demek: sonucu önce geçici bir dosyaya yazar, sonra tek
    bir ``os.replace`` ile gerçek adına taşırız. Böylece işlem yarıda kesilse
    bile önbellekte asla yarım dosya kalmaz; paralel çalışan iş parçacıkları
    da birbirinin dosyasını bozamaz (her biri kendi PID+thread-id'li geçici
    dosyasına yazar).
    '''

    cache_path = feature_cache_path(audio_path, cache_dir, config)
    if cache_path.is_file():
        try:
            # allow_pickle=False: .npy içinde rastgele Python nesnesi
            # çalıştırılmasını engeller (güvenlik + sadece dizi bekliyoruz).
            cached = np.asarray(np.load(cache_path, allow_pickle=False), dtype=np.float32)
            if _valid_vector(cached, config):
                return cached
        except (OSError, ValueError):
            # Yarım kalmış/bayat önbellek aşağıda yenisiyle değiştirilir.
            pass

    vector = extract_melspec(audio_path, config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Geçici dosya adına süreç ve iş parçacığı kimliği ekliyoruz; aynı anda
    # birden çok worker aynı kaydı işlese bile çakışma olmaz.
    temp_path = cache_path.with_suffix(
        f'.{os.getpid()}.{threading.get_ident()}.tmp.npy'
    )
    np.save(temp_path, vector, allow_pickle=False)
    # os.replace aynı disk bölümünde atomiktir: ya eski dosya ya yeni dosya
    # görünür, asla karışım görünmez.
    os.replace(temp_path, cache_path)
    return vector


@dataclass(frozen=True)
class FeatureStandardizer:
    '''Yalnızca eğitim katmanında (fold) öğrenilen, boyut başına z-score parametreleri.

    z-score: her öznitelik boyutundan ortalamasını çıkarıp standart sapmasına
    böleriz; sonuçta her boyut ~0 ortalama ve ~1 varyansa gelir. Bu, MLP
    eğitimini hızlandırır ve öğrenme oranı seçimini kolaylaştırır.

    KRİTİK KURAL: ``fit`` yalnızca eğitim verisiyle çağrılır; doğrulama ve
    test verisi yalnızca ``transform``'dan geçer. Aksi veri sızıntısı olur.
    '''

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray, epsilon: float = 1e-6) -> 'FeatureStandardizer':
        features = np.asarray(features)
        if features.ndim != 2 or len(features) == 0:
            raise ValueError(f'Expected a non-empty 2-D matrix, got {features.shape}.')
        if not np.isfinite(features).all():
            raise ValueError('Training features must contain only finite values.')
        # Ara hesapları float64'te yapıp sonucu float32'ye indiriyoruz:
        # binlerce örneğin toplamında float32 hassasiyet kaybı yaşayabilir.
        mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
        # Neredeyse sabit boyutlarda std ~ 0 olur; sıfıra bölmemek için bu
        # boyutların ölçeğini 1.0 yaparız (değer zaten bilgi taşımıyor).
        scale[scale < epsilon] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        # shape[-1] kullanımı sayesinde hem tek vektör (1-D) hem matris (2-D)
        # aynı kodla dönüştürülebilir.
        if features.shape[-1] != self.mean.shape[0]:
            raise ValueError(
                f'Feature size {features.shape[-1]} does not match scaler '
                f'size {self.mean.shape[0]}.'
            )
        if not np.isfinite(features).all():
            raise ValueError('Features to transform must contain only finite values.')
        # ascontiguousarray: PyTorch'a verilecek dizinin bellekte bitişik
        # olmasını garanti eder (torch.from_numpy bunu ister).
        return np.ascontiguousarray((features - self.mean) / self.scale, dtype=np.float32)


class EmotionDataset(Dataset):
    '''Not defterleri ve küçük etkileşimli deneyler için tutulan tembel (lazy) veri kümesi.

    "Tembel" olması şu demek: öznitelikler ancak ``__getitem__`` çağrıldığında
    diskten yüklenir/çıkarılır. Bellek dostu ama tekrar tekrar eğitim yapılan
    hiperparametre aramasında yavaş kalır; orada ``ArrayDataset`` tercih edilir.
    '''

    def __init__(
        self,
        records,
        cache_dir: str | Path,
        config: MelSpecConfig = DEFAULT_CONFIG,
        standardizer: FeatureStandardizer | None = None,
    ) -> None:
        # reset_index + copy: dışarıdaki DataFrame'e bağımlılığı koparır ve
        # konumsal (iloc) erişimin 0..N-1 aralığında güvenli olmasını sağlar.
        self.records = records.reset_index(drop=True).copy()
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.standardizer = standardizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.records.iloc[index]
        vector = load_or_extract(row['path'], self.cache_dir, self.config)
        if self.standardizer is not None:
            vector = self.standardizer.transform(vector)
        # DataLoader'ın beklediği biçim: (öznitelik tensörü, etiket tensörü).
        # Etiket long (int64) olmalı çünkü CrossEntropyLoss bunu bekler.
        return (
            torch.from_numpy(np.asarray(vector, dtype=np.float32)),
            torch.tensor(int(row['label_idx']), dtype=torch.long),
        )


class ArrayDataset(Dataset):
    '''Tekrarlı hiperparametre denemelerinde kullanılan, tamamı bellekte tensörler.

    Tüm öznitelik matrisi bir kez RAM'e alınır; her ``__getitem__`` yalnızca
    hazır tensörden dilim döndürür. Onlarca aday eğitilirken disk I/O'sunu
    tamamen ortadan kaldırdığı için aramayı ciddi biçimde hızlandırır.
    '''

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        features = np.ascontiguousarray(features, dtype=np.float32)
        labels = np.ascontiguousarray(labels, dtype=np.int64)
        if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
            raise ValueError(
                f'Invalid feature/label shapes: {features.shape}, {labels.shape}.'
            )
        # torch.from_numpy kopyasız çalışır: numpy dizisiyle belleği paylaşır.
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def _progress(iterator: Iterable, total: int, description: str, enabled: bool):
    # İsteğe bağlı ilerleme çubuğu: tqdm kuruluysa sarmalar, kurulu değilse
    # ya da kapalıysa iteratörü olduğu gibi döndürür. Böylece tqdm zorunlu
    # bir bağımlılık olmaktan çıkar.
    if not enabled:
        return iterator
    try:
        from tqdm import tqdm

        return tqdm(iterator, total=total, desc=description, unit='ses')
    except ImportError:
        return iterator


def load_feature_matrix(
    records,
    cache_dir: str | Path,
    config: MelSpecConfig = DEFAULT_CONFIG,
    *,
    workers: int = 1,
    show_progress: bool = True,
    description: str = 'Mel özellikleri',
) -> tuple[np.ndarray, np.ndarray]:
    '''Tüm kayıtları BİR KEZ yükleyip önbelleğe alır; denemeler disk çıkarımını tekrarlamaz.

    Hiperparametre aramasında aynı veri onlarca kez kullanılır. Bu fonksiyon
    özellik çıkarımını aramanın en başında bir defa yapar ve (özellik matrisi,
    etiket dizisi) çiftini bellekte döndürür. ``workers > 1`` verilirse
    dosyalar iş parçacığı havuzuyla paralel işlenir — işin darboğazı disk ve
    librosa'nın C tarafı olduğu için thread'ler burada gerçekten hız kazandırır.
    '''

    records = records.reset_index(drop=True)
    paths = records['path'].astype(str).tolist()
    labels = records['label_idx'].to_numpy(dtype=np.int64, copy=True)
    if len(paths) == 0:
        raise ValueError('Cannot build a feature matrix from an empty fold.')

    def load_one(path: str) -> np.ndarray:
        return load_or_extract(path, cache_dir, config)

    if workers > 1:
        # executor.map sırayı korur: vectors[i] her zaman paths[i]'ye karşılık
        # gelir; bu yüzden etiketlerle hizalama bozulmaz.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            vectors = list(
                _progress(
                    executor.map(load_one, paths),
                    len(paths),
                    description,
                    show_progress,
                )
            )
    else:
        vectors = list(
            _progress(map(load_one, paths), len(paths), description, show_progress)
        )

    # Vektör listesini (N, vector_size) tek matrise yığ ve şekli doğrula.
    features = np.stack(vectors).astype(np.float32, copy=False)
    if features.shape != (len(records), config.vector_size):
        raise ValueError(f'Unexpected feature matrix shape: {features.shape}.')
    return features, labels
