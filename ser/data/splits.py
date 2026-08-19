# Train/val/test bölme stratejileri.
#
# Config'teki eğitim/değerlendirme korpuslarına bakılarak iki rejimden biri otomatik seçilir:
#
# * KORPUS-İÇİ (within-corpus, train_corpora == eval_corpora): seçilen korpus
# 3 parçaya bölünür. ``split="speaker"`` ile bölme konuşmacı-bağımsızdır
# (hiçbir konuşmacı birden fazla foldda yer almaz) — SER için DOĞRU protokol
# budur, çünkü aynı konuşmacı hem train hem test'te olursa model duyguyu değil
# o kişinin ses rengini ezberleyerek şişirilmiş skor alır. ``split="meld_official"``
# ile MELD'in kendi train/dev/test foldları kullanılır.
#
# * KORPUSLAR-ARASI (cross-corpus, train_corpora != eval_corpora): eğitim korpusu
# konuşmacı-bağımsız şekilde train/val'e bölünür ve değerlendirme korpusunun
# TAMAMI test kümesi olur. Bu, farklı kayıt koşulları/konuşmacılar arasında
# genellemeyi ölçer — test korpusundan tek bir örnek bile eğitime sızmaz.

from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import NUM_CLASSES
from ..utils import get_logger

log = get_logger(__name__)


def _valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Ortak altı sınıf dışındaki satırları atar ve speaker sütununu dizgeye çevirir.
    #
    # ``label_idx`` 0..5 aralığı dışında olan (eşlenememiş) satırlar elenir. ``speaker`` tip standardizasyonu önemlidir: CREMA-D oyuncu id'leri CSV'den sayı olarak okunabilir; küme karşılaştırmalarında "1001" ile 1001 farklı şeyler olurdu — hepsini string'e çevirmek bu tuzağı kapatır.
    df = df[df["label_idx"].between(0, NUM_CLASSES - 1)].copy()
    df["speaker"] = df["speaker"].astype(str)
    return df


def _check_nonempty(train_df, val_df, test_df):
    # Herhangi bir fold boş kaldıysa net bir mesajla hemen hata fırlatır.
    #
    # "Fail fast" ilkesi: boş fold, eğitimin çok sonrasında anlaşılması zor hatalara (örn. boş dizide metrik hesabı) yol açar; sorunu kaynağında, açıklayıcı bir mesajla yakalamak saatlerce hata ayıklamadan kurtarır.
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        if len(part) == 0:
            raise ValueError(
                f"Split produced an empty {name} fold "
                f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)}). "
                "Likely too few speakers for the requested fractions, or a "
                "missing corpus/split column."
            )
    return train_df, val_df, test_df


def _speaker_partition(df: pd.DataFrame, fractions: list[float], seed: int) -> list[pd.DataFrame]:
    # ``df``'yi KONUŞMACI bütünlüğünü koruyarak len(fractions) folda böler.
    #
    # fractions toplamı 1'dir; konuşmacılar deterministik olarak karıştırılır ve oranlara göre dilimlenir.
    #
    # Kritik nokta: oranlar kayıt sayısına değil KONUŞMACI sayısına uygulanır ve bir konuşmacının bütün kayıtları aynı folda girer. Böylece train/val/test arasında kimlik (ses) sızıntısı oluşmaz — model test konuşmacısının sesini eğitimde hiç duymamış olur. (Bedeli: konuşmacı başına kayıt sayısı değişken olduğundan fold büyüklükleri hedef oranlardan biraz sapabilir.)
    # Önce sıralamak, aynı seed ile platformdan/pandas sürümünden bağımsız aynı
    # başlangıç sırasını garanti eder; shuffle bu sıralı liste üzerinde yapılır.
    speakers = sorted(df["speaker"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(speakers)
    n = len(speakers)
    if n < len(fractions):
        # 3 fold istenip 2 konuşmacı varsa bölme matematiksel olarak imkânsız.
        raise ValueError(
            f"Need at least {len(fractions)} distinct speakers for a "
            f"speaker-independent split, but only found {n}."
        )
    # Konuşmacı sayısını oranlara göre kümülatif kesme sınırlarına çevir.
    # Örn. 20 konuşmacı, [0.7, 0.15, 0.15] -> sınırlar [14, 17, 20].
    bounds = np.cumsum([int(round(f * n)) for f in fractions])
    bounds[-1] = n  # yuvarlama artıklarını son folda yedir (kimse dışarıda kalmasın)
    folds, start = [], 0
    for end in bounds:
        fold_speakers = set(speakers[start:end])
        # O foldun konuşmacılarına ait TÜM satırları seç.
        folds.append(df[df["speaker"].isin(fold_speakers)].copy())
        start = end
    return folds


def _random_partition(df: pd.DataFrame, fractions: list[float], seed: int) -> list[pd.DataFrame]:
    # Satırları tamamen rastgele böler.
    #
    # Aynı konuşmacı farklı foldlara düşebileceği için ana protokol DEĞİLDİR; yalnızca kıyas/ablasyon amaçlıdır ("konuşmacı sızıntısı skoru ne kadar şişiriyor?" sorusuna cevap vermek için random ile speaker karşılaştırılır).
    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n = len(df)
    # Bu kez sınırlar SATIR sayısı üzerinden hesaplanır (konuşmacı değil).
    bounds = np.cumsum([int(round(f * n)) for f in fractions])
    bounds[-1] = n
    folds, start = [], 0
    for end in bounds:
        folds.append(df.iloc[idx[start:end]].copy())
        start = end
    return folds


def prepare_splits(manifest: pd.DataFrame, data_cfg, seed: int = 42):
    # Config'e göre ``(train_df, val_df, test_df)`` üreten ana bölme fonksiyonu.
    #
    # Karar ağacı:
    # 1. train ve eval korpusları FARKLI ise -> cross-corpus rejimi:
    # eğitim korpusu konuşmacı-bağımsız train/val'e bölünür, eval korpusunun
    # tamamı test olur.
    # 2. Aynı ise -> korpus-içi rejim: config'teki ``split`` stratejisine göre
    # (meld_official / random / speaker) üç fold üretilir.
    #
    # Ödev 1 ve Ödev 2, ``split="speaker"`` vererek bu fonksiyonu ortak kullanır; yani tüm deneyler aynı bölme mantığından geçer ve karşılaştırılabilir kalır.
    df = _valid_rows(manifest)
    # set'e çevirme: ("cremad",) ile ["cremad"] gibi farklı gösterimler eşitlenir
    # ve karşılaştırma sıradan bağımsız olur.
    train_corpora = set(data_cfg.train_corpora)
    eval_corpora = set(data_cfg.eval_corpora)

    train_pool = df[df["corpus"].isin(train_corpora)].copy()
    eval_pool = df[df["corpus"].isin(eval_corpora)].copy()
    if len(train_pool) == 0:
        raise ValueError(f"No rows for train_corpora={train_corpora}. "
                         f"Available: {sorted(df['corpus'].unique())}")

    cross = train_corpora != eval_corpora
    vf, tf = data_cfg.val_fraction, data_cfg.test_fraction

    if cross:
        # --- KORPUSLAR-ARASI rejim -------------------------------------------
        if len(eval_pool) == 0:
            raise ValueError(f"No rows for eval_corpora={eval_corpora}. "
                             f"Available: {sorted(df['corpus'].unique())}")
        # Eğitim korpusundan val ayrılır (erken durdurma için gerekli);
        # test = diğer korpusun TAMAMI. Test korpusundan eğitime hiçbir şey sızmaz.
        train_df, val_df = _speaker_partition(train_pool, [1 - vf, vf], seed)
        test_df = eval_pool
        log.info("CROSS-CORPUS: train=%s eval=%s | train=%d val=%d test=%d",
                 sorted(train_corpora), sorted(eval_corpora),
                 len(train_df), len(val_df), len(test_df))
        return _check_nonempty(train_df, val_df, test_df)

    # --- KORPUS-İÇİ rejim ----------------------------------------------------
    if data_cfg.split == "meld_official" and "split" in train_pool.columns:
        # NOT: MELD'in resmî foldları DİYALOG bazlıdır, konuşmacı bazlı değil;
        # aynı dizi karakteri train/dev/test'in üçünde de görülebilir. Bu,
        # standart MELD benchmark protokolüdür (literatürle karşılaştırılabilir)
        # ama konuşmacı-bağımsız DEĞİLDİR. Konuşmacı-bağımsız bir MELD
        # değerlendirmesi için split="speaker" kullanın.
        sp = train_pool["split"].astype(str)
        train_df = train_pool[sp == "train"].copy()
        val_df = train_pool[sp == "dev"].copy()
        test_df = train_pool[sp == "test"].copy()
        if len(val_df) == 0 or len(test_df) == 0:
            # Resmî fold bilgisi eksik (örn. manifest yalnız train içeriyor):
            # çökmek yerine konuşmacı bölmesine geri düş.
            log.warning("meld_official split incomplete; falling back to speaker split")
        else:
            log.info("MELD-OFFICIAL (dialogue-based, not speaker-independent) | "
                     "train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))
            return _check_nonempty(train_df, val_df, test_df)

    # Üçlü oran: train = geriye kalan pay (1 - val - test).
    fractions = [1 - vf - tf, vf, tf]
    if data_cfg.split == "random":
        train_df, val_df, test_df = _random_partition(train_pool, fractions, seed)
        proto = "RANDOM"
    else:
        # Varsayılan ve önerilen yol: konuşmacı-bağımsız bölme.
        train_df, val_df, test_df = _speaker_partition(train_pool, fractions, seed)
        proto = "SPEAKER-INDEPENDENT"
    log.info("%s split | train=%d val=%d test=%d", proto,
             len(train_df), len(val_df), len(test_df))
    return _check_nonempty(train_df, val_df, test_df)
