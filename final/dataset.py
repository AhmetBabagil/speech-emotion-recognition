# Öznitelik önbelleği, yalnız-eğitimle normalizasyon ve bellek-içi veri kümeleri.
#
# İki yöntem de kayıt başına sabit boyutlu bir float32 dizi ürettiği için tek bir önbellek ve tek bir yükleyici, hem CNN görüntülerini hem RNN serilerini karşılar. Öznitelik çıkarımı pahalı olduğundan her sonuç diske yazılır ve sonraki denemelerde saniyeler içinde geri okunur.

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

from concurrent.futures import ThreadPoolExecutor  # dosyaları paralel (thread) yüklemek için
from dataclasses import dataclass  # Standardizer'ı kolay yazmak için
import hashlib  # dosya kimliğinden özet üretmek için
import os  # dosya işlemleri (atomik replace, pid) için
from pathlib import Path  # dosya yolları için
import threading  # geçici dosya adında thread kimliği için
from typing import Callable, Iterable  # tip ipuçları: fonksiyon ve gezilebilir

import numpy as np  # sayısal diziler
import torch  # tensörler
from torch.utils.data import Dataset  # PyTorch veri kümesi taban sınıfı


def feature_cache_path(
    audio_path: str | Path,  # kaynak ses dosyası
    cache_dir: str | Path,  # önbellek kök klasörü
    fingerprint: str,  # öznitelik ayarlarının kimliği
) -> Path:
    # Kaynak dosyaya ve öznitelik ayarlarına özel, kararlı önbellek yolu üretir.
    #
    # Yol iki şeye bağlıdır:
    # 1. fingerprint: öznitelik ayarlarının kimliği (klasör adı) — farklı ayarların çıktıları karışamaz.
    # 2. Kaynak dosyanın yolu + boyutu + değişiklik zamanı — ses dosyası değişirse eski önbellek otomatik geçersizleşir.

    audio_path = Path(audio_path)  # yolu nesneye çevir
    stat = audio_path.stat()  # dosya bilgisi (boyut, değişiklik zamanı)
    identity = (  # dosyayı benzersiz tanımlayan metin
        f'{os.path.normcase(str(audio_path.resolve()))}\0{stat.st_size}\0{stat.st_mtime_ns}'  # yol + boyut + zaman
    )
    digest = hashlib.sha1(identity.encode('utf-8')).hexdigest()[:16]  # kimliğin 16 haneli özeti
    return Path(cache_dir) / fingerprint / f'{audio_path.stem[:48]}_{digest}.npy'  # önbellek/fingerprint/dosya.npy


def load_or_extract(
    audio_path: str | Path,  # ses dosyası
    cache_dir: str | Path,  # önbellek klasörü
    fingerprint: str,  # ayar kimliği
    extract: Callable[[str | Path], np.ndarray],  # öznitelik çıkaran fonksiyon
    expected_shape: tuple[int, ...],  # beklenen çıktı boyutu
) -> np.ndarray:  # Geçerli bir önbellek kaydı varsa onu döndürür; yoksa hesaplayıp saklar.

    cache_path = feature_cache_path(audio_path, cache_dir, fingerprint)  # bu dosyanın önbellek yolu
    if cache_path.is_file():  # önbellek zaten var mı
        try:
            cached = np.asarray(np.load(cache_path, allow_pickle=False), dtype=np.float32)  # önbelleği oku
            # Boyut ve sonluluk kontrolü: bozuk kayıt varsa aşağıda yeniden üretilir.
            if cached.shape == expected_shape and np.isfinite(cached).all():  # boyut doğru + bozuk değil mi
                return cached  # geçerliyse önbelleği döndür (hızlı yol)
        except (OSError, ValueError):  # okuma hatası olursa
            pass  # yarım kalmış/bozuk önbellek dosyası aşağıda değiştirilir

    array = extract(audio_path)  # önbellek yok/bozuk -> özniteliği taze çıkar
    cache_path.parent.mkdir(parents=True, exist_ok=True)  # önbellek klasörünü oluştur
    # Önce geçici ada yazıp sonra atomik os.replace: iki işlem aynı anda
    # yazsa bile dosya asla yarım hâlde okunmaz.
    temp_path = cache_path.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.tmp.npy')  # benzersiz geçici ad
    np.save(temp_path, array, allow_pickle=False)  # önce geçici dosyaya yaz
    os.replace(temp_path, cache_path)  # atomik olarak asıl ada taşı (yarım okuma olmaz)
    return array  # taze özniteliği döndür


def _progress(iterator: Iterable, total: int, description: str, enabled: bool):  # tqdm kuruluysa ilerleme çubuğu göster; değilse sessizce devam et.

    if not enabled:  # ilerleme istenmiyorsa
        return iterator  # olduğu gibi döndür
    try:
        from tqdm import tqdm  # ilerleme çubuğu kütüphanesi

        return tqdm(iterator, total=total, desc=description, unit='ses')  # çubuklu sarmalayıcı
    except ImportError:  # tqdm kurulu değilse
        return iterator  # sessizce devam


def load_feature_tensor(
    records,  # bir katmanın (fold) kayıtları (yol + etiket)
    cache_dir: str | Path,  # önbellek klasörü
    fingerprint: str,  # ayar kimliği
    extract: Callable[[str | Path], np.ndarray],  # öznitelik çıkaran fonksiyon
    expected_shape: tuple[int, ...],  # beklenen kayıt-başı boyut
    *,
    workers: int = 1,  # paralel yükleme işçisi sayısı
    show_progress: bool = True,  # ilerleme çubuğu göster mi
    description: str = 'Öznitelikler',  # çubuk açıklaması
) -> tuple[np.ndarray, np.ndarray]:
    # Bir katmanın (fold) tüm kayıtlarını tek bir [N, ...] tensöre toplar.
    #
    # workers > 1 verilirse dosyalar iş parçacıklarıyla paralel işlenir (öznitelik çıkarımı G/Ç + librosa ağırlıklı olduğu için thread yeterli).

    records = records.reset_index(drop=True)  # satır indekslerini sıfırla
    paths = records['path'].astype(str).tolist()  # ses dosyası yolları
    labels = records['label_idx'].to_numpy(dtype=np.int64, copy=True)  # etiketler (sayısal)
    if not paths:  # katman boşsa
        raise ValueError('Boş katmandan öznitelik üretilemez.')  # hata

    def load_one(path: str) -> np.ndarray:  # tek bir kaydı yükleyen iç fonksiyon
        return load_or_extract(path, cache_dir, fingerprint, extract, expected_shape)  # önbellekten ya da taze

    if workers > 1:  # paralel isteniyorsa
        with ThreadPoolExecutor(max_workers=workers) as executor:  # iş parçacığı havuzu
            arrays = list(
                _progress(executor.map(load_one, paths), len(paths), description, show_progress)  # paralel yükle
            )
    else:  # tek işçi
        arrays = list(_progress(map(load_one, paths), len(paths), description, show_progress))  # sırayla yükle

    features = np.stack(arrays).astype(np.float32, copy=False)  # tüm kayıtları [N, ...] tensörde dizle
    if features.shape != (len(records), *expected_shape):  # boyut beklenen mi
        raise ValueError(f'Beklenmeyen öznitelik tensörü boyutu: {features.shape}.')  # değilse hata
    return features, labels  # öznitelikler + etiketler


@dataclass(frozen=True)  # kilitli (değiştirilemez) sınıf
class Standardizer:
    # YALNIZCA eğitim katmanından öğrenilen z-skor parametreleri.
    #
    # Neden önemli: ortalama/std'yi tüm veriden öğrenmek, test bilgisinin eğitime sızması demektir (data leakage). Burada fit() sadece eğitim verisiyle çağrılır; geçerleme ve test aynı parametrelerle dönüştürülür.
    #
    # Eksen mantığı: [N, mels, T] mel görüntülerinde istatistik mel bandı başına (feature_axis=1), [N, T, D] serilerde öznitelik boyutu başına (feature_axis=2) tutulur; kalan eksenler (örnek + zaman) üzerinden ortalama alınır.

    mean: np.ndarray  # öznitelik başına ortalama
    scale: np.ndarray  # öznitelik başına standart sapma
    feature_axis: int  # istatistiğin hangi eksende tutulacağı

    @classmethod
    def fit(cls, features: np.ndarray, feature_axis: int, epsilon: float = 1e-6) -> 'Standardizer':  # eğitimden ortalama/std öğrenir
        features = np.asarray(features)  # diziye çevir
        if features.ndim != 3 or len(features) == 0:  # 3 boyutlu ve boş olmayan mı
            raise ValueError(f'Boş olmayan 3 boyutlu tensör bekleniyor: {features.shape}.')  # değilse hata
        if not np.isfinite(features).all():  # bozuk değer var mı
            raise ValueError('Eğitim öznitelikleri sonlu olmalı.')  # varsa hata
        axes = tuple(axis for axis in range(features.ndim) if axis != feature_axis)  # feature_axis dışı eksenler
        mean = features.mean(axis=axes, dtype=np.float64).astype(np.float32)  # öznitelik başına ortalama
        scale = features.std(axis=axes, dtype=np.float64).astype(np.float32)  # öznitelik başına std
        # Sabit (varyanssız) boyutlarda sıfıra bölmeyi önle.
        scale = np.where(scale < epsilon, 1.0, scale)  # çok küçük std -> 1 (sıfıra bölme yok)
        return cls(mean=mean, scale=scale, feature_axis=feature_axis)  # öğrenilmiş standartlaştırıcı

    def transform(self, features: np.ndarray) -> np.ndarray:  # (x - ortalama) / std dönüşümünü doğru eksende uygular.

        features = np.asarray(features, dtype=np.float32)  # diziye çevir
        # mean/scale vektörlerini yayın (broadcast) için doğru şekle getir.
        shape = [1] * features.ndim  # her eksen 1
        shape[self.feature_axis] = self.mean.shape[0]  # yalnız feature_axis öznitelik boyutunda
        mean = self.mean.reshape(shape)  # ortalamayı yayına uygun şekle sok
        scale = self.scale.reshape(shape)  # std'yi yayına uygun şekle sok
        return np.ascontiguousarray((features - mean) / scale, dtype=np.float32)  # z-skor: (x - ort) / std


class ArrayDataset(Dataset):
    # Hiperparametre denemeleri boyunca yeniden kullanılan bellek-içi tensörler.
    #
    # Tüm öznitelikler zaten RAM'e sığdığı için diskten tekrar tekrar okumak yerine bir kez tensöre çevirip DataLoader'a veriyoruz — denemeler arası en hızlı yol bu.

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:  # öznitelik + etiket alır
        features = np.ascontiguousarray(features, dtype=np.float32)  # bellekte bitişik float32'ye çevir
        labels = np.ascontiguousarray(labels, dtype=np.int64)  # etiketleri int64'e çevir
        if labels.ndim != 1 or len(features) != len(labels):  # boyutlar uyumlu mu
            raise ValueError(
                f'Geçersiz öznitelik/etiket boyutları: {features.shape}, {labels.shape}.'  # değilse hata
            )
        self.features = torch.from_numpy(features)  # numpy -> torch tensör (öznitelikler)
        self.labels = torch.from_numpy(labels)  # numpy -> torch tensör (etiketler)

    def __len__(self) -> int:  # veri kümesindeki örnek sayısı
        return self.labels.shape[0]  # etiket sayısı

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:  # index'inci (öznitelik, etiket) çiftini verir
        return self.features[index], self.labels[index]  # DataLoader bunu yığınlar
