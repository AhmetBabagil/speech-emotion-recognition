"""SENTETİK ses üzerinde uçtan uca sağlamlık testi (hiçbir indirme gerektirmez).

"Smoke test" (duman testi) fikri donanım dünyasından gelir: cihazı prize tak,
duman çıkmıyorsa temel devreler sağlamdır. Buradaki karşılığı: gerçek veriyi
indirmeyi beklemeden, boru hattının TAMAMININ — veri -> özellik -> eğitim ->
değerlendirme -> rapor — çalıştığını ve çıktı dosyalarının üretildiğini
birkaç dakikada doğrulamak.

Bunun için her (duygu, konuşmacı) çifti adına birbirinden ayırt edilebilir
birkaç saniyelik yapay ses üretir, gerçek şemayla bir manifest yazar, sonra
klasik baseline'ı ve 2 epoch'luk bir CNN'i GERÇEK kod yollarından geçirir.
Buradaki doğruluk değerleri gerçek başarının ölçüsü DEĞİLDİR; yalnızca kodun
kırık olmadığını gösterir.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Proje kökünü import arama yoluna ekle (ser paketi için).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ser.config import Config  # noqa: E402
from ser.constants import CANONICAL_EMOTIONS, EMOTION_TO_IDX, CORPUS_CREMAD  # noqa: E402
from ser.utils import get_logger, ensure_dir  # noqa: E402

log = get_logger("smoke")

# Sentetik veri üretim ayarları: 16 kHz örnekleme (projenin standart hızı),
# 2 saniyelik klipler, 6 konuşmacı, her (duygu, konuşmacı) çifti için 3 tekrar
# -> 6 duygu x 6 konuşmacı x 3 = 108 klip. Konuşmacı sayısı bilerek 1'den çok:
# konuşmacı-bağımsız bölünme train/val/test'e FARKLI konuşmacılar dağıtabilsin.
# Tüm çıktılar data/smoke altında toplanır (gerçek veri klasörleri kirlenmez).
SR = 16000
DUR = 2.0
N_SPEAKERS = 6
REPS = 3
ROOT = Path("data/smoke")


def _synth_clip(emotion_idx: int, speaker_idx: int, rep: int) -> np.ndarray:
    """Spektral içeriği DUYGUYA bağlı bir klip üretir (model sınıfları gerçekten
    ayırabilsin diye), üzerine konuşmacıya bağlı tını ve hafif gürültü ekler.

    Mantık: her duyguya farklı bir temel frekans (perde), her konuşmacıya
    farklı bir "formant" frekansı atanır. Böylece sınıflar arasında öğrenilebilir
    bir fark, konuşmacılar arasında da gerçekçi bir çeşitlilik oluşur. Rastgelelik
    tohumu üç indeksten türetildiği için aynı klip her çalıştırmada birebir aynı
    üretilir (yeniden üretilebilirlik: test sonuçları koşudan koşuya oynamaz).
    """
    rng = np.random.default_rng(1000 * emotion_idx + 10 * speaker_idx + rep)
    # t: 0'dan DUR'a, örnekleme hızıyla eşit aralıklı zaman ekseni.
    t = np.linspace(0, DUR, int(SR * DUR), endpoint=False)
    base = 180.0 + 90.0 * emotion_idx          # duygu -> perde (temel frekans)
    formant = 500.0 + 120.0 * speaker_idx        # konuşmacı -> tını
    # Üç sinüsün toplamı: temel ton + konuşmacı formantı + bir oktav üstü
    # harmonik (2*base). Ağırlıklar (0.6/0.3/0.1) kabaca doğal sese öykünür.
    sig = (
        0.6 * np.sin(2 * np.pi * base * t)
        + 0.3 * np.sin(2 * np.pi * formant * t)
        + 0.1 * np.sin(2 * np.pi * (2 * base) * t)
    )
    # Duyguya bağlı genlik zarfı (tempo/enerji ipucu): duygu indeksi büyüdükçe
    # ses daha hızlı "dalgalanır". Üstüne %2 genlikte Gauss gürültüsü biner.
    env = 0.5 + 0.5 * np.sin(2 * np.pi * (1 + emotion_idx) * t / DUR)
    sig = sig * env + 0.02 * rng.standard_normal(t.shape)
    # Tepe genliğe bölerek sinyali [-1, 1] aralığına normalle (+1e-8, tamamen
    # sessiz sinyalde sıfıra bölünme koruması). float32 ses/torch standardıdır.
    return (sig / np.max(np.abs(sig) + 1e-8)).astype(np.float32)


def build_synthetic_dataset() -> Path:
    """Sentetik klipleri diske yazar ve gerçek şemada bir manifest CSV üretir."""
    # soundfile'ı fonksiyon içinde import ediyoruz (tembel import): modül
    # yüklenirken değil, yalnızca gerçekten veri üretilirken gereksin.
    import soundfile as sf

    audio_dir = ensure_dir(ROOT / "audio")
    rows = []
    # Üç iç içe döngü = her duygu x her konuşmacı x her tekrar için bir klip.
    for e_idx, emotion in enumerate(CANONICAL_EMOTIONS):
        for spk in range(N_SPEAKERS):
            for rep in range(REPS):
                clip = _synth_clip(e_idx, spk, rep)
                fname = f"spk{spk:02d}_{emotion}_{rep}.wav"
                path = audio_dir / fname
                sf.write(path, clip, SR)
                # Manifest satırı, gerçek manifest'le AYNI sütunları taşır;
                # böylece eğitim kodu sentetik/gerçek veri ayrımını hiç bilmez
                # (testin bütün amacı gerçek kod yolunu sınamaktır).
                rows.append({
                    "path": str(path), "corpus": CORPUS_CREMAD, "speaker": f"spk{spk:02d}",
                    "split": "", "orig_label": emotion, "emotion": emotion,
                    "label_idx": EMOTION_TO_IDX[emotion],
                })
    manifest = ROOT / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    log.info("Synthetic dataset: %d clips -> %s", len(rows), manifest)
    return manifest


def main():
    """Sentetik veri kurar, baseline + kısa CNN eğitir, özet metrikleri basar."""
    manifest = build_synthetic_dataset()

    # Smoke config'ini yükleyip tüm yolları data/smoke altına yönlendir:
    # gerçek deneylerin cache'i ve outputs/ klasörü asla kirlenmesin.
    cfg = Config.from_yaml("configs/smoke.yaml")
    cfg.data.manifest = str(manifest)
    cfg.data.cache_dir = str(ROOT / "cache")
    cfg.output_dir = str(ROOT / "outputs")

    # 1) Klasik baseline (lojistik regresyon): saniyeler içinde biter ve
    #    scikit-learn tarafındaki özellik çıkarma + eğitim + değerlendirme
    #    yolunun tamamını sınar.
    from ser.train import train_baseline, train_torch
    log.info("--- baseline (logreg) ---")
    cfg.experiment = "smoke_baseline"
    bm = train_baseline(cfg, kind="logreg")

    # 2) CNN (yalnızca 2 epoch): amaç iyi öğrenmek değil; PyTorch veri
    #    yükleyicisinin, eğitim döngüsünün ve checkpoint kaydının çalıştığını
    #    görmek. 2 epoch bunun için yeterli, fazlası zaman kaybı olur.
    log.info("--- cnn (2 epochs) ---")
    cfg.experiment = "smoke_cnn"
    cm = train_torch(cfg)

    # İki eğitim fonksiyonunun döndürdüğü metrik sözlüklerinden kısa özet:
    # sayılar "yüksek" olmak zorunda değil, sadece üretilebilmiş olmalı.
    print("\n=== SMOKE TEST OK ===")
    print(f"baseline: acc={bm['accuracy']:.3f}  macroF1={bm['macro_f1']:.3f}")
    print(f"cnn     : acc={cm['accuracy']:.3f}  macroF1={cm['macro_f1']:.3f}")
    print("Artifacts under", ROOT / "outputs")


if __name__ == "__main__":
    main()
