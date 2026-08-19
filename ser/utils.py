# Ortak yardımcılar: cihaz seçimi, tohumlama (seed), loglama ve küçük araçlar.
#
# Bu modül bilinçli olarak "hafif" tutulmuştur: torch gibi ağır kütüphaneler modülün en üstünde değil, yalnızca ihtiyaç duyan fonksiyonun İÇİNDE import edilir. Böylece torch kurulu olmayan bir ortamda bile manifest oluşturma gibi torch gerektirmeyen araçlar sorunsuz çalışabilir.

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np

# logging.basicConfig'in yalnızca BİR kez çağrıldığından emin olmak için modül
# düzeyinde bayrak. Birden çok modül get_logger çağırdığında format ayarı
# tekrar tekrar uygulanmaz (ikinci çağrılar zaten sessizce yok sayılırdı ama
# bu bayrak niyeti açıkça belgeliyor).
_LOG_CONFIGURED = False


def get_logger(name: str = "ser") -> logging.Logger:
    # İsimlendirilmiş bir logger döndürür; ilk çağrıda kök formatı kurar.
    #
    # print() yerine logging kullanmanın nedeni: zaman damgası + seviye + kaynak modül bilgisi otomatik eklenir ve seviye filtrelemesi (INFO/DEBUG) tek merkezden yönetilebilir.
    global _LOG_CONFIGURED
    if not _LOG_CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            # Örnek satır: "12:34:56 | INFO    | ser.train | epoch 3 | ..."
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _LOG_CONFIGURED = True
    return logging.getLogger(name)


def set_seed(seed: int = 42) -> None:
    # Python, NumPy ve (varsa) torch'un rastgeleliğini tohumlar.
    #
    # Amaç tekrarlanabilirlik: aynı seed ile aynı veri bölmesi, aynı ağırlık başlangıcı ve aynı augmentasyon sırası elde edilir; iki deney arasındaki fark gerçekten değiştirdiğimiz şeyden kaynaklanır, şanstan değil.
    random.seed(seed)          # Python'un yerleşik random modülü
    np.random.seed(seed)       # NumPy'nin eski (global) RNG'si
    # Hash tohumlaması: set/dict sıralamasına dayanan olası farkları sabitler.
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)           # CPU tarafı RNG
        torch.cuda.manual_seed_all(seed)  # tüm GPU'ların RNG'si
    except Exception:
        # torch kurulu değilse sorun yok: sklearn/numpy tarafı yine tohumlandı.
        pass


def get_device(prefer: str = "auto") -> "object":
    # Bir torch.device döndürür; CUDA'yı otomatik algılar.
    #
    # ``prefer`` şu değerleri alabilir: 'auto' | 'cuda' | 'cpu'. CUDA istenmiş ama mevcut değilse hata fırlatmak yerine uyarı loglayıp CPU'ya düşer — böylece aynı config dosyası GPU'lu ve GPU'suz makinede değişiklik gerekmeden çalışır.
    import torch

    log = get_logger()
    if prefer == "cpu":
        # Kullanıcı açıkça CPU istedi; GPU olsa bile ona saygı duy.
        return torch.device("cpu")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        log.info("Using CUDA device: %s", name)
        return torch.device("cuda")
    if prefer == "cuda":
        log.warning("CUDA requested but not available -- falling back to CPU.")
    else:
        log.info("CUDA not available -- using CPU.")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    # Klasörün var olduğunu garanti eder (yoksa üst klasörleriyle oluşturur).
    #
    # ``exist_ok=True`` sayesinde klasör zaten varsa hata çıkmaz; dönen Path ile ``out = ensure_dir(...) / "dosya.txt"`` gibi zincirleme kullanım rahattır.
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_params(model) -> int:
    # Modelin EĞİTİLEBİLİR parametre sayısını döndürür.
    #
    # ``requires_grad`` filtresi önemli: wav2vec2'de dondurulmuş katmanlar sayılmaz, böylece log'da görülen sayı gerçekten öğrenilen parametredir.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
