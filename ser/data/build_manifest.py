# CREMA-D ve MELD'den birleşik manifest CSV'sini üretir.
#
# "Manifest" nedir ve neden var? Projede tüm eğitim/değerlendirme kodu, ham veri klasörlerini doğrudan taramak yerine TEK bir CSV'den beslenir. İki korpusun dosya düzeni ve etiketleme biçimi tamamen farklıdır (CREMA-D etiketi dosya adında taşır, MELD ayrı CSV'lerde tutar); bu farklılık burada, bir kez çözülür ve geri kalan kod tek tip satırlarla çalışır.
#
# Çıktı sütunları (kullanılabilir her kayıt için bir satır):
# path        WAV dosyasının (mutlak/göreli) yolu
# corpus      'cremad' | 'meld'
# speaker     CREMA-D oyuncu id'si / MELD konuşmacı adı
# (konuşmacı-bağımsız bölmeler bu sütuna dayanır)
# split       CREMA-D için '' (resmî fold'u yok), MELD için 'train'/'dev'/'test'
# (resmî fold bilgisi; meld_official bölmesi bunu kullanır)
# orig_label  veri kümesinin kendi etiketi (kod ya da dizge) — izlenebilirlik için
# emotion     kanonik etiket (angry/disgust/fear/happy/neutral/sad)
# label_idx   kanonik sınıf indeksi 0..5
#
# Yalnızca ortak altı duygu tutulur; MELD'in 'surprise' satırları atılır.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..constants import (
    CREMAD_CODE_TO_CANONICAL,
    EMOTION_TO_IDX,
    MELD_LABEL_TO_CANONICAL,
    CORPUS_CREMAD,
    CORPUS_MELD,
)
from ..utils import get_logger, ensure_dir

log = get_logger(__name__)

# MELD'in resmî fold'u -> o foldun etiket CSV'sinin dosya adı.
SPLIT_CSV = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv", "test": "test_sent_emo.csv"}


def cremad_rows(audiowav_dir: str | Path) -> list[dict]:
    # CREMA-D WAV klasörünü tarayıp manifest satırlarını üretir.
    #
    # CREMA-D'de bütün bilgi dosya adındadır: ``<ActorID>_<Sentence>_<Emotion>_<Level>.wav`` (örn. 1001_DFA_ANG_XX.wav). Ayrı bir etiket dosyası yoktur; adı parçalayarak hem konuşmacıyı hem duyguyu çıkarırız. sorted(): dosya sistemi sırasına bağımlı kalmamak için — aynı klasörden her platformda aynı sırayla aynı manifest üretilsin.
    audiowav_dir = Path(audiowav_dir)
    rows = []
    for wav in sorted(audiowav_dir.glob("*.wav")):
        parts = wav.stem.split("_")
        if len(parts) < 3:
            continue  # desene uymayan (bozuk adlandırılmış) dosyayı atla
        # parts[0]=oyuncu id, parts[1]=cümle kodu (kullanılmıyor), parts[2]=duygu kodu
        actor, _sentence, code = parts[0], parts[1], parts[2]
        canon = CREMAD_CODE_TO_CANONICAL.get(code.upper())
        if canon is None:
            continue  # tanınmayan duygu kodu -> manifest'e alma
        rows.append({
            "path": str(wav),
            "corpus": CORPUS_CREMAD,
            "speaker": actor,        # konuşmacı-bağımsız bölme bu alana dayanır
            "split": "",             # CREMA-D'nin resmî fold'u yok
            "orig_label": code.upper(),
            "emotion": canon,
            "label_idx": EMOTION_TO_IDX[canon],
        })
    log.info("CREMA-D: %d usable rows from %s", len(rows), audiowav_dir)
    return rows


def meld_rows(csv_dir: str | Path, audio_root: str | Path) -> list[dict]:
    # MELD'in üç fold CSV'sini okuyup manifest satırlarını üretir.
    #
    # MELD'de etiketler CSV'de, sesler ise (bizim ffmpeg adımımızın ürettiği) ``audio/<split>/diaX_uttY.wav`` dosyalarındadır. CSV satırı ile ses dosyası burada eşleştirilir; sesi diskte OLMAYAN satırlar sessizce atlanır — böylece manifest her zaman gerçekten açılabilir dosyaları listeler.
    csv_dir = Path(csv_dir)
    audio_root = Path(audio_root)
    rows = []
    for split, csv_name in SPLIT_CSV.items():
        csv_path = csv_dir / csv_name
        if not csv_path.exists():
            log.warning("MELD: missing %s, skipping %s", csv_name, split)
            continue
        df = pd.read_csv(csv_path)
        kept = 0
        # itertuples: iterrows'a göre çok daha hızlıdır (satırlar namedtuple gelir);
        # _asdict() ile sütunlara isimle erişilir.
        for r in df.itertuples(index=False):
            d = r._asdict()
            emotion = str(d["Emotion"]).strip().lower()
            canon = MELD_LABEL_TO_CANONICAL.get(emotion)
            if canon is None:
                continue  # 'surprise' veya bilinmeyen etiket -> ortak altıda yok, atla
            # MELD'in dosya adlandırma kuralı: dia<DialogueID>_utt<UtteranceID>.wav
            key = f"dia{int(d['Dialogue_ID'])}_utt{int(d['Utterance_ID'])}"
            wav = audio_root / split / f"{key}.wav"
            if not wav.exists():
                continue  # ses çıkarılmamış (bozuk klip ya da ffmpeg adımı henüz koşmadı)
            rows.append({
                "path": str(wav),
                "corpus": CORPUS_MELD,
                "speaker": str(d["Speaker"]).strip(),  # dizi karakteri adı (örn. "Joey")
                "split": split,                          # MELD'in resmî fold bilgisi
                "orig_label": emotion,
                "emotion": canon,
                "label_idx": EMOTION_TO_IDX[canon],
            })
            kept += 1
        log.info("MELD %s: %d usable rows", split, kept)
    return rows


def build_manifest(
    cremad_dir: str | Path | None = "data/raw/cremad/AudioWAV",
    meld_csv_dir: str | Path | None = None,
    meld_audio_root: str | Path | None = "data/raw/meld/audio",
    out_path: str | Path = "data/processed/manifest.csv",
) -> pd.DataFrame:
    # İki korpusun satırlarını toplayıp tek CSV'ye yazar; DataFrame'i döndürür.
    #
    # Esnek davranır: korpuslardan biri diskte yoksa uyarı verip diğeriyle devam eder (örneğin yalnızca CREMA-D indirilmişse CREMA-D-only deneyler yine çalışabilsin). İkisi de yoksa anlamlı bir hata fırlatılır.
    rows: list[dict] = []

    if cremad_dir and Path(cremad_dir).is_dir():
        rows += cremad_rows(cremad_dir)
    else:
        log.warning("CREMA-D dir not found: %s", cremad_dir)

    # MELD CSV klasörü verilmemişse otomatik bul: arşivin açıldığı derinlik
    # kuruluma göre değişebildiğinden, imza dosyası (train_sent_emo.csv)
    # özyinelemeli aranır ve bulunduğu klasör kullanılır.
    if meld_csv_dir is None:
        guess = Path("data/raw/meld")
        found = list(guess.rglob("train_sent_emo.csv"))
        meld_csv_dir = found[0].parent if found else None
    if meld_csv_dir and meld_audio_root and Path(meld_audio_root).is_dir():
        rows += meld_rows(meld_csv_dir, meld_audio_root)
    else:
        log.warning("MELD audio/CSV not found (csv_dir=%s audio=%s)", meld_csv_dir, meld_audio_root)

    if not rows:
        raise RuntimeError("No data found for either corpus. Run the download scripts first.")

    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)
    log.info("Manifest written: %s (%d rows)", out_path, len(df))
    # Sınıf dağılımını logla: dengesizlik (özellikle MELD'de) daha bu aşamada görülsün.
    log.info("Class distribution:\n%s", df.groupby(["corpus", "emotion"]).size())
    return df
