# Ortak yardımcılar: cihaz seçimi, tohumlama (seed), loglama ve küçük araçlar.
#
# Bu modül bilinçli olarak "hafif" tutulmuştur: torch gibi ağır kütüphaneler modülün en üstünde değil, yalnızca ihtiyaç duyan fonksiyonun İÇİNDE import edilir. Böylece torch kurulu olmayan bir ortamda bile manifest oluşturma gibi torch gerektirmeyen araçlar sorunsuz çalışabilir.

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

import logging  # günlükleme
import os  # ortam değişkenleri (hash seed)
import random  # Python rastgeleliği
from pathlib import Path  # dosya yolları

import numpy as np  # NumPy rastgeleliği

# logging.basicConfig'in yalnızca BİR kez çağrıldığından emin olmak için modül
# düzeyinde bayrak. Birden çok modül get_logger çağırdığında format ayarı
# tekrar tekrar uygulanmaz (ikinci çağrılar zaten sessizce yok sayılırdı ama
# bu bayrak niyeti açıkça belgeliyor).
_LOG_CONFIGURED = False  # log formatı bir kez kuruldu mu


def get_logger(name: str = "ser") -> logging.Logger:
    # İsimlendirilmiş bir logger döndürür; ilk çağrıda kök formatı kurar.
    #
    # print() yerine logging kullanmanın nedeni: zaman damgası + seviye + kaynak modül bilgisi otomatik eklenir ve seviye filtrelemesi (INFO/DEBUG) tek merkezden yönetilebilir.
    global _LOG_CONFIGURED  # modül-genel bayrak
    if not _LOG_CONFIGURED:  # ilk çağrıysa
        logging.basicConfig(  # kök log formatını kur
            level=logging.INFO,  # INFO ve üstünü göster
            # Örnek satır: "12:34:56 | INFO    | ser.train | epoch 3 | ..."
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",  # satır formatı
            datefmt="%H:%M:%S",  # zaman formatı
        )
        _LOG_CONFIGURED = True  # bir daha kurma
    return logging.getLogger(name)  # isimlendirilmiş logger


def set_seed(seed: int = 42) -> None:
    # Python, NumPy ve (varsa) torch'un rastgeleliğini tohumlar.
    #
    # Amaç tekrarlanabilirlik: aynı seed ile aynı veri bölmesi, aynı ağırlık başlangıcı ve aynı augmentasyon sırası elde edilir; iki deney arasındaki fark gerçekten değiştirdiğimiz şeyden kaynaklanır, şanstan değil.
    random.seed(seed)          # Python'un yerleşik random modülü
    np.random.seed(seed)       # NumPy'nin eski (global) RNG'si
    # Hash tohumlaması: set/dict sıralamasına dayanan olası farkları sabitler.
    os.environ["PYTHONHASHSEED"] = str(seed)  # hash tohumu
    try:
        import torch  # (varsa) torch

        torch.manual_seed(seed)           # CPU tarafı RNG
        torch.cuda.manual_seed_all(seed)  # tüm GPU'ların RNG'si
    except Exception:  # torch yoksa
        # torch kurulu değilse sorun yok: sklearn/numpy tarafı yine tohumlandı.
        pass  # sessizce geç


def get_device(prefer: str = "auto") -> "object":
    # Bir torch.device döndürür; CUDA'yı otomatik algılar.
    #
    # ``prefer`` şu değerleri alabilir: 'auto' | 'cuda' | 'cpu'. CUDA istenmiş ama mevcut değilse hata fırlatmak yerine uyarı loglayıp CPU'ya düşer — böylece aynı config dosyası GPU'lu ve GPU'suz makinede değişiklik gerekmeden çalışır.
    import torch  # (fonksiyon içi) torch

    log = get_logger()  # günlükleyici
    if prefer == "cpu":  # açıkça CPU istendiyse
        # Kullanıcı açıkça CPU istedi; GPU olsa bile ona saygı duy.
        return torch.device("cpu")  # CPU döndür
    if torch.cuda.is_available():  # GPU varsa
        name = torch.cuda.get_device_name(0)  # GPU adı
        log.info("Using CUDA device: %s", name)  # logla
        return torch.device("cuda")  # GPU döndür
    if prefer == "cuda":  # GPU istendi ama yoksa
        log.warning("CUDA requested but not available -- falling back to CPU.")  # uyar
    else:  # otomatik
        log.info("CUDA not available -- using CPU.")  # bilgilendir
    return torch.device("cpu")  # CPU'ya düş


def ensure_dir(path: str | Path) -> Path:
    # Klasörün var olduğunu garanti eder (yoksa üst klasörleriyle oluşturur).
    #
    # ``exist_ok=True`` sayesinde klasör zaten varsa hata çıkmaz; dönen Path ile ``out = ensure_dir(...) / "dosya.txt"`` gibi zincirleme kullanım rahattır.
    p = Path(path)  # yolu nesneye çevir
    p.mkdir(parents=True, exist_ok=True)  # üst klasörlerle birlikte oluştur (varsa hata verme)
    return p  # yolu döndür (zincirleme için)


def count_params(model) -> int:
    # Modelin EĞİTİLEBİLİR parametre sayısını döndürür.
    #
    # ``requires_grad`` filtresi önemli: wav2vec2'de dondurulmuş katmanlar sayılmaz, böylece log'da görülen sayı gerçekten öğrenilen parametredir.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # eğitilebilir ağırlıkları say
