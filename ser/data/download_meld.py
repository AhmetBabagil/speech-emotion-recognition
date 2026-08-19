# MELD ham verisini indirir ve ffmpeg ile kayıt-başına ses çıkarır.
#
# MELD, "Friends" dizisinden alınmış ÇOK KİPLİ (video+ses+metin) bir veri kümesidir; bize ise yalnızca ses lazım. Bu modülün işi ham .mp4 kliplerden projenin kullandığı 16 kHz mono WAV'ları üretmektir.
#
# İşlem hattı:
# 1. MELD.Raw.tar.gz'yi indir (~10 GB; kaldığı yerden devam edebilir).
# 2. Dış arşivi aç, sonra iç içe train/dev/test tarball'larını
# ``<dest>/video/<split>/`` altına çıkar.
# 3. Duygusu ortak altıda olan HER kayıt için .mp4'ü ffmpeg ile
# ``<dest>/audio/<split>/diaX_uttY.wav`` konumunda 16 kHz mono WAV'a çevir.
#
# Yalnızca ortak-altı kayıtlar dönüştürülür (``surprise`` atlanır); bu hem zaman hem disk tasarrufudur — nasılsa o klipler manifest'e hiç girmeyecek. Bozuk/sessiz klipler loglanıp atlanır, koşuyu düşürmez.

from __future__ import annotations

import subprocess
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

from ..constants import MELD_LABEL_TO_CANONICAL
from ..utils import get_logger, ensure_dir

log = get_logger(__name__)

MELD_URL = "http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz"
# Fold adı -> o foldun etiket CSV'si (arşivin içinden çıkar).
SPLIT_CSV = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv", "test": "test_sent_emo.csv"}


# --------------------------------------------------------------------------- #
# İndirme
# --------------------------------------------------------------------------- #
def _download(url: str, out_path: Path) -> Path:
    # Tek indirme geçişi; dosya varsa HTTP Range ile kaldığı yerden sürdürür.
    #
    # 10 GB'lık bir dosyada bağlantı kopması neredeyse kesindir; Range başlığı sayesinde her deneme sıfırdan değil, eksik kısımdan başlar.
    from tqdm import tqdm

    out_path = Path(out_path)
    # Diskte ne kadarı var? O bayttan devam etmeyi isteyeceğiz.
    resume_from = out_path.stat().st_size if out_path.exists() else 0
    req = urllib.request.Request(url)
    if resume_from:
        req.add_header("Range", f"bytes={resume_from}-")
        log.info("Resuming MELD download from %.2f GB", resume_from / 1e9)

    with urllib.request.urlopen(req, timeout=60) as resp:
        # Sunucu Range'i desteklemiyorsa 206 yerine 200 döner -> dosyanın
        # tamamını baştan gönderiyor demektir; ekleme yerine sıfırdan yaz.
        if resume_from and resp.status == 200:
            log.warning("Server ignored resume; restarting download.")
            resume_from = 0
        # Content-Length yalnız KALAN kısmın boyutudur; toplam için eldekini ekle.
        total = int(resp.headers.get("Content-Length", 0)) + resume_from
        mode = "ab" if resume_from else "wb"  # devamsa dosyanın sonuna ekle
        with open(out_path, mode) as f, tqdm(
            total=total or None, initial=resume_from, unit="B", unit_scale=True,
            desc="MELD.Raw.tar.gz",
        ) as bar:
            # 1 MB'lık parçalar hâlinde akıt: belleği şişirmeden ilerleme göster.
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    return out_path


def _remote_size(url: str) -> int | None:
    # Uzak dosyanın Content-Length'i; öğrenilemezse None.
    #
    # HEAD isteği gövdeyi indirmeden yalnızca başlıkları alır — "indirme bitti mi?" sorusunu cevaplamak için hedef boyutu böyle öğreniriz.
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception as e:
        log.warning("Could not HEAD %s: %s", url, e)
        return None


def _ensure_complete_download(url: str, out_path: Path, max_retries: int = 10) -> None:
    # Dosya uzak boyuta ulaşana dek devam-etmeli (resume) indirme + tekrar dener.
    #
    # Kopan bağlantı diskte KESİK bir dosya bırakır; kesik bir tar'ı açmaya kalkmak kafa karıştıran hatalar üretir. Bu yüzden çıkarmadan ÖNCE, disk boyutu Content-Length'e eşitlenene kadar HTTP Range ile devam ederiz.
    expected = _remote_size(url)
    for attempt in range(1, max_retries + 1):
        have = out_path.stat().st_size if out_path.exists() else 0
        if expected and have >= expected:
            log.info("Archive complete: %.2f GB.", have / 1e9)
            return  # hedefe ulaşıldı: indirme tamam
        if expected:
            log.info("Have %.2f / %.2f GB -- download attempt %d/%d",
                     have / 1e9, expected / 1e9, attempt, max_retries)
        try:
            _download(url, out_path)
        except Exception as e:  # ağ kesintisi -> bir sonraki turda devam et
            log.warning("Download attempt %d failed: %s", attempt, e)
        if expected is None:
            return  # boyut doğrulanamıyor; tek geçişin yettiğini varsay
    # Döngü bitti ama dosya hâlâ eksik: durumu açıkça bildir.
    have = out_path.stat().st_size if out_path.exists() else 0
    if expected and have < expected:
        raise RuntimeError(
            f"MELD download still incomplete after {max_retries} attempts "
            f"({have}/{expected} bytes). Re-run to resume."
        )


def _safe_extract(tar_path: Path, out_dir: Path) -> None:
    # Tar arşivini GÜVENLİ şekilde açar (path traversal saldırısına karşı).
    #
    # Klasik "tar bombası" hilesi: arşive ``../../etc/passwd`` gibi üyeler koyup hedef klasörün DIŞINA yazdırmak. Açmadan önce her üyenin çözülmüş yolunun hedef klasörün altında kaldığı doğrulanır; değilse hata fırlatılır.
    ensure_dir(out_dir)
    with tarfile.open(tar_path, "r:*") as tar:
        members = tar.getmembers()
        base = out_dir.resolve()
        for m in members:
            target = (out_dir / m.name).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError(f"Unsafe path in tar: {m.name}")
        tar.extractall(out_dir)


def _classify_split(name: str) -> str | None:  # Dosya adından hangi folda ait olduğunu kestirir (train/dev/test/None).
    n = name.lower()
    if "train" in n:
        return "train"
    if "dev" in n:
        return "dev"
    if "test" in n:
        return "test"
    return None


# MELD.Raw yerleşimi: dış arşiv, içinde dev_sent_emo.csv, test_sent_emo.csv ve
# üç iç içe tar bulunan bir "MELD.Raw" klasörüne açılır. train.tar.gz ayrıca
# train_sent_emo.csv'yi de içerir; fold başına video klasörleri sırasıyla
# train_splits/, dev_splits_complete/ ve output_repeated_splits_test/ adlarını
# taşır (adlandırma tutarsızlığı MELD'in kendisinden gelir, bu yüzden fold
# başına birden çok "ipucu" adı denenir).
_SPLIT_DIR_HINTS = {
    "train": ("train_splits", "train"),
    "dev": ("dev_splits_complete", "dev_splits", "dev"),
    "test": ("output_repeated_splits_test", "test_splits", "test"),
}


def download_meld(dest: str = "data/raw/meld", keep_archive: bool = True) -> Path:
    # MELD ham verisini indirir + açar. CSV'leri içeren klasörü döndürür.
    #
    # İdempotenttir: veri zaten açılmışsa indirme/açma adımları atlanır; iç tarball'lar açıldıkça silinir ki ~10 GB'lık disk alanı geri kazanılsın.
    dest = ensure_dir(dest)
    archive = dest / "MELD.Raw.tar.gz"

    raw_root = _find_raw_root(dest)
    if raw_root is None:
        # Çıkarmadan ÖNCE indirme TAMAMLANMALI (kesik dosya varsa devam ettirilir);
        # yarım tar'ı açmak anlaşılmaz hatalar üretir.
        _ensure_complete_download(MELD_URL, archive)
        log.info("Extracting outer archive ...")
        _safe_extract(archive, dest)
        raw_root = _find_raw_root(dest)
        if not keep_archive and archive.exists():
            archive.unlink()  # video tarlarını açmadan önce ~11 GB alan boşalt
    if raw_root is None:
        raise RuntimeError(f"Could not find the MELD.Raw layout under {dest} after extraction.")

    # İç içe fold tarball'larını YERİNDE aç. train.tar.gz ayrıca
    # train_sent_emo.csv'yi de çıkarır. Her tar, açıldıktan sonra disk alanı
    # için silinir; daha önce açılmışsa (video klasörü doluysa) yalnızca silinir.
    for nested in sorted(raw_root.glob("*.tar.gz")):
        split = _classify_split(nested.name)
        if split is None:
            continue  # fold'a benzemeyen bir tar: karışma
        if _split_video_dir(raw_root, split) is not None:
            nested.unlink(missing_ok=True)
            continue  # zaten açılmış: tekrar açma, sadece arşivi temizle
        log.info("Extracting %s ...", nested.name)
        _safe_extract(nested, raw_root)
        nested.unlink(missing_ok=True)

    missing = [c for c in SPLIT_CSV.values() if not (raw_root / c).exists()]
    if missing:
        log.warning("MELD CSVs still missing after extraction: %s", missing)
    log.info("MELD raw ready. CSV dir: %s", raw_root)
    return raw_root


def _find_raw_root(root: Path) -> Path | None:
    # İç tar'ları ya da fold CSV'lerini barındıran klasörü bulur.
    #
    # Arşivin hangi derinliğe açıldığı kuruluma göre değişebildiğinden, bilinen "imza" dosyaları özyinelemeli aranır ve ilkinin klasörü döndürülür.
    for marker in ("train.tar.gz", "train_sent_emo.csv", "dev_sent_emo.csv"):
        hits = list(root.rglob(marker))
        if hits:
            return hits[0].parent
    return None


def _split_video_dir(raw_root: Path, split: str) -> Path | None:
    # Bir foldun video klasörünü bulur; içinde gerçekten .mp4 varsa döndürür.
    #
    # Boş klasör "açılmış" sayılmaz — yarım kalmış bir çıkarma sonrası yeniden açma bu sayede tetiklenir.
    for hint in _SPLIT_DIR_HINTS[split]:
        d = raw_root / hint
        if d.is_dir() and any(d.rglob("*.mp4")):
            return d
    return None


def _split_of_mp4(path: Path) -> str | None:
    # Bir .mp4 yolunun hangi folda ait olduğunu klasör adından çıkarır.
    #
    # replace("\\", "/"): Windows ve POSIX yol ayraçlarını eşitleyip alt dizge aramasını platformdan bağımsız hâle getirir.
    s = str(path).replace("\\", "/").lower()
    if "train_splits" in s:
        return "train"
    if "dev_splits" in s:
        return "dev"
    if "output_repeated_splits_test" in s or "test_splits" in s:
        return "test"
    return None


# --------------------------------------------------------------------------- #
# Ses çıkarımı
# --------------------------------------------------------------------------- #
def extract_meld_audio(
    csv_dir: str | Path,
    dest: str = "data/raw/meld",
    ffmpeg_bin: str = "ffmpeg",
    limit: int | None = None,
) -> Path:
    # Ortak-altı MELD kayıtlarını 16 kHz mono WAV'a dönüştürür.
    #
    # ``limit`` parametresi hızlı denemeler içindir (fold başına en çok o kadar dönüşüm yapılır). Fold alt klasörlerini içeren ses kök dizinini (``<dest>/audio``) döndürür.
    from tqdm import tqdm

    csv_dir = Path(csv_dir)
    dest = Path(dest)
    audio_root = ensure_dir(dest / "audio")

    # (split, stem) ikilisiyle anahtarlanmış global mp4 indeksi. dia/utt
    # kimlikleri foldlar arasında TEKRARLANDIĞI için split anahtarın parçası
    # olmak zorundadır; yoksa dia0_utt0 üç ayrı foldda çakışırdı.
    # Önceden indekslemek, CSV'deki her satır için diski yeniden taramaktan
    # (O(N^2) davranış) kurtarır.
    index: dict[tuple[str, str], Path] = {}
    for mp4 in csv_dir.rglob("*.mp4"):
        sp = _split_of_mp4(mp4)
        if sp:
            index[(sp, mp4.stem)] = mp4

    for split, csv_name in SPLIT_CSV.items():
        csv_path = csv_dir / csv_name
        if not csv_path.exists():
            log.warning("Missing %s -- skipping %s split", csv_name, split)
            continue
        df = pd.read_csv(csv_path)
        out_dir = ensure_dir(audio_root / split)

        done = 0
        n_ok = n_skip = n_fail = 0  # fold sonu özeti için sayaçlar
        for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"MELD {split} audio"):
            r = row._asdict()
            emotion = str(r["Emotion"]).strip().lower()
            if MELD_LABEL_TO_CANONICAL.get(emotion) is None:
                n_skip += 1
                continue  # surprise / bilinmeyen etiket: dönüştürmeye değmez
            key = f"dia{int(r['Dialogue_ID'])}_utt{int(r['Utterance_ID'])}"
            out_wav = out_dir / f"{key}.wav"
            if out_wav.exists():
                n_ok += 1
                continue  # idempotentlik: daha önce dönüştürülmüş, tekrar etme
            mp4 = index.get((split, key))
            if mp4 is None:
                n_fail += 1
                continue  # CSV'de var ama videosu diskte yok
            if _ffmpeg_to_wav(ffmpeg_bin, mp4, out_wav):
                n_ok += 1
            else:
                n_fail += 1
            done += 1
            if limit is not None and done >= limit:
                break  # hızlı deneme modu: fold başına limit doldu
        log.info("MELD %s: ok=%d skipped=%d failed/missing=%d", split, n_ok, n_skip, n_fail)
    return audio_root


def _ffmpeg_to_wav(ffmpeg_bin: str, mp4: Path, out_wav: Path) -> bool:
    # Tek bir .mp4'ün ses izini 16 kHz mono WAV'a çevirir; başarıyı döndürür.
    #
    # ffmpeg bayrakları: -y üzerine yaz, -loglevel error yalnız hataları göster, -vn videoyu atla (yalnız ses), -ac 1 monoya indir, -ar 16000 örnekleme frekansını projenin standardı 16 kHz'e ayarla.
    cmd = [ffmpeg_bin, "-y", "-loglevel", "error", "-i", str(mp4),
           "-vn", "-ac", "1", "-ar", "16000", str(out_wav)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        # Çift kontrol: ffmpeg 0 dönse bile çıktı boş olabilir; boyutu doğrula.
        return out_wav.exists() and out_wav.stat().st_size > 0
    except subprocess.CalledProcessError as e:
        # stderr'in ilk 200 karakteri debug loguna: sorun ayıklamaya yeter,
        # logu boğmaz. Yarım kalmış çıktı dosyası silinir ki bir sonraki koşu
        # onu "tamamlanmış" sanmasın (idempotentlik korunur).
        log.debug("ffmpeg failed for %s: %s", mp4.name, e.stderr.decode(errors="ignore")[:200])
        if out_wav.exists():
            out_wav.unlink()
        return False


if __name__ == "__main__":
    # Doğrudan çalıştırma: önce indir/aç, sonra sesleri çıkar.
    csv_dir = download_meld()
    extract_meld_audio(csv_dir)
