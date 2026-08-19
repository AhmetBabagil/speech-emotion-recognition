# Train/val/test bölme stratejileri.
#
# Config'teki eğitim/değerlendirme korpuslarına bakılarak iki rejimden biri otomatik seçilir:
#
# * KORPUS-İÇİ (within-corpus, train_corpora == eval_corpora): seçilen korpus 3 parçaya bölünür. ``split="speaker"`` ile bölme konuşmacı-bağımsızdır (hiçbir konuşmacı birden fazla foldda yer almaz) — SER için DOĞRU protokol budur, çünkü aynı konuşmacı hem train hem test'te olursa model duyguyu değil o kişinin ses rengini ezberleyerek şişirilmiş skor alır. ``split="meld_official"`` ile MELD'in kendi train/dev/test foldları kullanılır.
#
# * KORPUSLAR-ARASI (cross-corpus, train_corpora != eval_corpora): eğitim korpusu konuşmacı-bağımsız şekilde train/val'e bölünür ve değerlendirme korpusunun TAMAMI test kümesi olur. Bu, farklı kayıt koşulları/konuşmacılar arasında genellemeyi ölçer — test korpusundan tek bir örnek bile eğitime sızmaz.

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

import numpy as np  # rastgele karıştırma + dilimleme
import pandas as pd  # tablo (manifest) işlemleri

from ..constants import NUM_CLASSES  # sınıf sayısı (6)
from ..utils import get_logger  # günlükleyici

log = get_logger(__name__)  # bu modülün günlükleyicisi


def _valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Ortak altı sınıf dışındaki satırları atar ve speaker sütununu dizgeye çevirir.
    #
    # ``label_idx`` 0..5 aralığı dışında olan (eşlenememiş) satırlar elenir. ``speaker`` tip standardizasyonu önemlidir: CREMA-D oyuncu id'leri CSV'den sayı olarak okunabilir; küme karşılaştırmalarında "1001" ile 1001 farklı şeyler olurdu — hepsini string'e çevirmek bu tuzağı kapatır.
    df = df[df["label_idx"].between(0, NUM_CLASSES - 1)].copy()  # yalnız 0..5 etiketli satırlar
    df["speaker"] = df["speaker"].astype(str)  # konuşmacı id'lerini metne çevir (tutarlı karşılaştırma)
    return df  # temizlenmiş tablo


def _check_nonempty(train_df, val_df, test_df):
    # Herhangi bir fold boş kaldıysa net bir mesajla hemen hata fırlatır.
    #
    # "Fail fast" ilkesi: boş fold, eğitimin çok sonrasında anlaşılması zor hatalara (örn. boş dizide metrik hesabı) yol açar; sorunu kaynağında, açıklayıcı bir mesajla yakalamak saatlerce hata ayıklamadan kurtarır.
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):  # her fold için
        if len(part) == 0:  # boşsa
            raise ValueError(  # açık hata ver
                f"Split produced an empty {name} fold "
                f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)}). "
                "Likely too few speakers for the requested fractions, or a "
                "missing corpus/split column."
            )
    return train_df, val_df, test_df  # üç fold sağlamsa döndür


def _speaker_partition(df: pd.DataFrame, fractions: list[float], seed: int) -> list[pd.DataFrame]:
    # ``df``'yi KONUŞMACI bütünlüğünü koruyarak len(fractions) folda böler.
    #
    # fractions toplamı 1'dir; konuşmacılar deterministik olarak karıştırılır ve oranlara göre dilimlenir.
    #
    # Kritik nokta: oranlar kayıt sayısına değil KONUŞMACI sayısına uygulanır ve bir konuşmacının bütün kayıtları aynı folda girer. Böylece train/val/test arasında kimlik (ses) sızıntısı oluşmaz — model test konuşmacısının sesini eğitimde hiç duymamış olur. (Bedeli: konuşmacı başına kayıt sayısı değişken olduğundan fold büyüklükleri hedef oranlardan biraz sapabilir.)
    # Önce sıralamak, aynı seed ile platformdan/pandas sürümünden bağımsız aynı
    # başlangıç sırasını garanti eder; shuffle bu sıralı liste üzerinde yapılır.
    speakers = sorted(df["speaker"].unique())  # benzersiz konuşmacılar (sıralı = deterministik başlangıç)
    rng = np.random.default_rng(seed)  # sabit tohumlu üreteç
    rng.shuffle(speakers)  # konuşmacıları karıştır
    n = len(speakers)  # konuşmacı sayısı
    if n < len(fractions):  # istenen fold sayısı kadar konuşmacı yoksa
        # 3 fold istenip 2 konuşmacı varsa bölme matematiksel olarak imkânsız.
        raise ValueError(  # hata
            f"Need at least {len(fractions)} distinct speakers for a "
            f"speaker-independent split, but only found {n}."
        )
    # Konuşmacı sayısını oranlara göre kümülatif kesme sınırlarına çevir.
    # Örn. 20 konuşmacı, [0.7, 0.15, 0.15] -> sınırlar [14, 17, 20].
    bounds = np.cumsum([int(round(f * n)) for f in fractions])  # kümülatif konuşmacı sınırları
    bounds[-1] = n  # yuvarlama artıklarını son folda yedir (kimse dışarıda kalmasın)
    folds, start = [], 0  # foldlar + başlangıç indeksi
    for end in bounds:  # her sınır için
        fold_speakers = set(speakers[start:end])  # bu foldun konuşmacıları
        # O foldun konuşmacılarına ait TÜM satırları seç.
        folds.append(df[df["speaker"].isin(fold_speakers)].copy())  # o konuşmacıların tüm kayıtları
        start = end  # bir sonraki dilime geç
    return folds  # konuşmacı-bütünlüğü korunmuş foldlar


def _random_partition(df: pd.DataFrame, fractions: list[float], seed: int) -> list[pd.DataFrame]:
    # Satırları tamamen rastgele böler.
    #
    # Aynı konuşmacı farklı foldlara düşebileceği için ana protokol DEĞİLDİR; yalnızca kıyas/ablasyon amaçlıdır ("konuşmacı sızıntısı skoru ne kadar şişiriyor?" sorusuna cevap vermek için random ile speaker karşılaştırılır).
    idx = np.arange(len(df))  # satır indeksleri
    rng = np.random.default_rng(seed)  # sabit tohumlu üreteç
    rng.shuffle(idx)  # satırları karıştır
    n = len(df)  # satır sayısı
    # Bu kez sınırlar SATIR sayısı üzerinden hesaplanır (konuşmacı değil).
    bounds = np.cumsum([int(round(f * n)) for f in fractions])  # kümülatif satır sınırları
    bounds[-1] = n  # artıkları son folda yedir
    folds, start = [], 0  # foldlar + başlangıç
    for end in bounds:  # her sınır için
        folds.append(df.iloc[idx[start:end]].copy())  # o satır dilimini al
        start = end  # ilerle
    return folds  # rastgele foldlar


def prepare_splits(manifest: pd.DataFrame, data_cfg, seed: int = 42):
    # Config'e göre ``(train_df, val_df, test_df)`` üreten ana bölme fonksiyonu.
    #
    # Karar ağacı:
    # 1. train ve eval korpusları FARKLI ise -> cross-corpus rejimi: eğitim korpusu konuşmacı-bağımsız train/val'e bölünür, eval korpusunun tamamı test olur.
    # 2. Aynı ise -> korpus-içi rejim: config'teki ``split`` stratejisine göre (meld_official / random / speaker) üç fold üretilir.
    #
    # Ödev 1 ve Ödev 2, ``split="speaker"`` vererek bu fonksiyonu ortak kullanır; yani tüm deneyler aynı bölme mantığından geçer ve karşılaştırılabilir kalır.
    df = _valid_rows(manifest)  # geçerli satırları temizle
    # set'e çevirme: ("cremad",) ile ["cremad"] gibi farklı gösterimler eşitlenir
    # ve karşılaştırma sıradan bağımsız olur.
    train_corpora = set(data_cfg.train_corpora)  # eğitim korpusları kümesi
    eval_corpora = set(data_cfg.eval_corpora)  # değerlendirme korpusları kümesi

    train_pool = df[df["corpus"].isin(train_corpora)].copy()  # eğitim korpusu satırları
    eval_pool = df[df["corpus"].isin(eval_corpora)].copy()  # değerlendirme korpusu satırları
    if len(train_pool) == 0:  # eğitim verisi yoksa
        raise ValueError(f"No rows for train_corpora={train_corpora}. "  # hata
                         f"Available: {sorted(df['corpus'].unique())}")

    cross = train_corpora != eval_corpora  # korpuslar-arası mı
    vf, tf = data_cfg.val_fraction, data_cfg.test_fraction  # geçerleme + test oranları

    if cross:  # --- KORPUSLAR-ARASI rejim ---
        # --- KORPUSLAR-ARASI rejim -------------------------------------------
        if len(eval_pool) == 0:  # değerlendirme verisi yoksa
            raise ValueError(f"No rows for eval_corpora={eval_corpora}. "  # hata
                             f"Available: {sorted(df['corpus'].unique())}")
        # Eğitim korpusundan val ayrılır (erken durdurma için gerekli);
        # test = diğer korpusun TAMAMI. Test korpusundan eğitime hiçbir şey sızmaz.
        train_df, val_df = _speaker_partition(train_pool, [1 - vf, vf], seed)  # eğitim -> train/val
        test_df = eval_pool  # test = diğer korpusun tamamı
        log.info("CROSS-CORPUS: train=%s eval=%s | train=%d val=%d test=%d",  # logla
                 sorted(train_corpora), sorted(eval_corpora),
                 len(train_df), len(val_df), len(test_df))
        return _check_nonempty(train_df, val_df, test_df)  # foldları döndür

    # --- KORPUS-İÇİ rejim ----------------------------------------------------
    if data_cfg.split == "meld_official" and "split" in train_pool.columns:  # MELD resmî foldu istendiyse
        # NOT: MELD'in resmî foldları DİYALOG bazlıdır, konuşmacı bazlı değil;
        # aynı dizi karakteri train/dev/test'in üçünde de görülebilir. Bu,
        # standart MELD benchmark protokolüdür (literatürle karşılaştırılabilir)
        # ama konuşmacı-bağımsız DEĞİLDİR. Konuşmacı-bağımsız bir MELD
        # değerlendirmesi için split="speaker" kullanın.
        sp = train_pool["split"].astype(str)  # resmî fold etiketi sütunu
        train_df = train_pool[sp == "train"].copy()  # resmî train
        val_df = train_pool[sp == "dev"].copy()  # resmî dev (geçerleme)
        test_df = train_pool[sp == "test"].copy()  # resmî test
        if len(val_df) == 0 or len(test_df) == 0:  # resmî fold eksikse
            # Resmî fold bilgisi eksik (örn. manifest yalnız train içeriyor):
            # çökmek yerine konuşmacı bölmesine geri düş.
            log.warning("meld_official split incomplete; falling back to speaker split")  # uyar, aşağıya düş
        else:  # resmî foldlar tamsa
            log.info("MELD-OFFICIAL (dialogue-based, not speaker-independent) | "  # logla
                     "train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))
            return _check_nonempty(train_df, val_df, test_df)  # resmî foldları döndür

    # Üçlü oran: train = geriye kalan pay (1 - val - test).
    fractions = [1 - vf - tf, vf, tf]  # train/val/test oranları
    if data_cfg.split == "random":  # rastgele bölme istendiyse
        train_df, val_df, test_df = _random_partition(train_pool, fractions, seed)  # satır-rastgele böl
        proto = "RANDOM"  # protokol etiketi
    else:  # varsayılan
        # Varsayılan ve önerilen yol: konuşmacı-bağımsız bölme.
        train_df, val_df, test_df = _speaker_partition(train_pool, fractions, seed)  # konuşmacı-bağımsız böl
        proto = "SPEAKER-INDEPENDENT"  # protokol etiketi
    log.info("%s split | train=%d val=%d test=%d", proto,  # protokolü logla
             len(train_df), len(val_df), len(test_df))
    return _check_nonempty(train_df, val_df, test_df)  # foldları döndür
