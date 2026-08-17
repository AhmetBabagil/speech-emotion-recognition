"""CREMA-D ses verisini indirir (7.442 WAV, orijinal dosya adları korunarak).

İki yöntem vardır:

* ``"hf"`` (varsayılan, güvenilir) — WAV'ları Hugging Face aynası
  ``AbstractTTS/CREMA-D``'den çeker. Her satırda ORİJİNAL dosya adını taşıyan
  bir ``file`` alanı (örn. ``1001_DFA_ANG_XX.wav`` — oyuncu id + cümle + duygu
  kodlanmıştır) ve ham WAV baytları bulunur; bunları ``<dest>/AudioWAV/`` altına
  doğrudan yazarız. Ses ÇÖZME (decode) gerekmez, bu yüzden Windows'ta ek codec
  derdi olmadan temiz çalışır. ``datasets`` paketi gerekir
  (``pip install datasets``).

* ``"lfs"`` (alternatif) — kanonik GitHub deposundan, yalnızca ``AudioWAV``
  klasörünü çeken bir git-lfs "sparse checkout". Doğru bir yöntemdir ama
  GitHub'ın bu depo için LFS bant genişliği sık sık aşırı kısıtlandığından
  varsayılan yapılmamıştır.

Neden dosya adı bu kadar önemli? CREMA-D'nin etiketi VE konuşmacı kimliği
yalnızca dosya adında durur. Orijinal adlar her iki yöntemde de korunduğundan,
oyuncu id'sine dayanan konuşmacı-bağımsız bölme sorunsuz çalışır.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..utils import get_logger, ensure_dir

log = get_logger(__name__)

REPO_URL = "https://github.com/CheyneyComputerScience/CREMA-D.git"
HF_REPO = "AbstractTTS/CREMA-D"
# Bütünlük eşiği: tam set 7.442 dosyadır; birkaç eksiğe tolerans tanımak için
# alt sınır 7.000 seçildi. Bu sayının altı "indirme yarım kalmış" demektir.
EXPECTED_MIN_WAVS = 7000


def _count_wavs(audiowav: Path) -> int:
    """Klasördeki .wav sayısı (klasör yoksa 0) — hem 'zaten indirilmiş mi?'
    kontrolünde hem indirme sonrası doğrulamada kullanılır."""
    return len(list(audiowav.glob("*.wav"))) if audiowav.is_dir() else 0


def download_cremad(dest: str = "data/raw/cremad", method: str = "hf") -> Path:
    """CREMA-D AudioWAV'ın ``dest`` altında var olmasını garanti eder.

    İdempotenttir: yeterli sayıda WAV zaten diskteyse hiçbir şey indirmez —
    betiği tekrar tekrar çalıştırmak güvenlidir. AudioWAV yolunu döndürür.
    """
    dest = ensure_dir(dest)
    audiowav = dest / "AudioWAV"
    if _count_wavs(audiowav) >= EXPECTED_MIN_WAVS:
        log.info("CREMA-D already present (%d wavs) at %s", _count_wavs(audiowav), audiowav)
        return audiowav

    if method == "hf":
        return _download_via_hf(audiowav)
    if method == "lfs":
        return _download_via_gitlfs(dest, audiowav)
    raise ValueError(f"Unknown method={method!r} (expected 'hf' or 'lfs')")


def _download_via_hf(audiowav: Path) -> Path:
    """Hugging Face aynasından WAV baytlarını indirip diske yazar."""
    try:
        from datasets import load_dataset, Audio
    except ImportError as e:  # pragma: no cover
        # Eksik bağımlılığı, çözüm önerisiyle birlikte NET bir mesajla bildir.
        raise ImportError(
            "The 'hf' download method needs the `datasets` package. "
            "Install it (`pip install datasets`) or use method='lfs'."
        ) from e
    from tqdm import tqdm

    ensure_dir(audiowav)
    log.info("Downloading CREMA-D from Hugging Face mirror '%s' ...", HF_REPO)
    # decode=False -> datasets sesi çözmeye çalışmaz, ham WAV baytlarını verir.
    # Böylece torchcodec/soundfile gibi çözücülere ihtiyaç kalmaz; baytlar
    # olduğu gibi diske yazılır (bit düzeyinde birebir kopya).
    ds = load_dataset(HF_REPO, split="train").cast_column("audio", Audio(decode=False))
    n = 0
    for row in tqdm(ds, total=len(ds), desc="CREMA-D wavs"):
        audio = row["audio"]
        data = audio.get("bytes")
        # Orijinal dosya adı: önce 'file' alanı, yoksa audio'nun 'path'i.
        name = row.get("file") or audio.get("path")
        if data is None or not name:
            continue  # eksik kayıt: atla
        if not str(name).lower().endswith(".wav"):
            name = f"{name}.wav"
        # Path(name).name: olası klasör öneklerini temizle, sadece dosya adı kalsın.
        (audiowav / Path(name).name).write_bytes(data)
        n += 1
    if n < EXPECTED_MIN_WAVS:
        raise RuntimeError(f"Only wrote {n} CREMA-D wavs (expected >= {EXPECTED_MIN_WAVS}).")
    log.info("CREMA-D ready: %d wavs at %s", n, audiowav)
    return audiowav


def _run(cmd: list[str], env: dict | None = None) -> None:
    """Bir alt komutu loglayarak çalıştırır; hata kodunda exception fırlatır
    (check=True). Kullanıcı, hangi git komutunun koştuğunu logda görebilir."""
    log.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def _download_via_gitlfs(dest: Path, audiowav: Path) -> Path:
    """GitHub deposundan yalnızca AudioWAV klasörünü indirir (sparse + LFS).

    Adım adım strateji (depo dev boyutta olduğu için hepsi "az indir" amaçlı):
      1. GIT_LFS_SKIP_SMUDGE=1  -> clone sırasında LFS dosyaları İNDİRİLMESİN
         (yalnızca küçük işaretçi dosyaları gelsin).
      2. --no-checkout --filter=blob:none -> geçmişteki dosya içerikleri de gelmesin.
      3. sparse-checkout ile çalışma ağacını yalnızca AudioWAV'a daralt.
      4. En son, `git lfs pull --include AudioWAV/**` ile SADECE ihtiyaç duyulan
         gerçek ses dosyalarını çek.
    """
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    if not (dest / ".git").exists():
        _run(["git", "clone", "--no-checkout", "--filter=blob:none", REPO_URL, str(dest)], env=env)
    _run(["git", "-C", str(dest), "sparse-checkout", "init", "--cone"], env=env)
    _run(["git", "-C", str(dest), "sparse-checkout", "set", "AudioWAV"], env=env)
    # Varsayılan dal adını sormaya çalış (master/main olabilir); öğrenilemezse
    # bu depo için doğru olan "master" varsayımıyla devam et.
    branch = "master"
    try:
        out = subprocess.run(
            ["git", "-C", str(dest), "rev-parse", "--abbrev-ref", "origin/HEAD"],
            capture_output=True, text=True, check=True, env=env,
        ).stdout.strip()
        if "/" in out:
            branch = out.split("/", 1)[1]  # "origin/master" -> "master"
    except subprocess.CalledProcessError:
        pass  # öğrenilemedi: varsayılan "master" ile dene
    _run(["git", "-C", str(dest), "checkout", branch], env=env)
    _run(["git", "-C", str(dest), "lfs", "pull", "--include", "AudioWAV/**"])
    n = _count_wavs(audiowav)
    if n < EXPECTED_MIN_WAVS:
        raise RuntimeError(
            f"Expected >= {EXPECTED_MIN_WAVS} CREMA-D wavs but found {n}. "
            "GitHub LFS may be throttled -- try the default method='hf' instead."
        )
    log.info("CREMA-D ready: %d wavs at %s", n, audiowav)
    return audiowav


if __name__ == "__main__":
    # Modül doğrudan çalıştırılırsa (python -m ser.data.download_cremad)
    # varsayılan ayarlarla indirme yapılır.
    download_cremad()
