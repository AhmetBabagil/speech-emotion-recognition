"""Eğitilmiş bir modelle TEK bir ses dosyasının duygusunu tahmin eder.

Bu betik, proje önerisindeki girdi->çıktı akışının canlı halidir: içeri bir
WAV dosyası girer, dışarı tahmin edilen duygu + tüm sınıflar üzerindeki
olasılık dağılımı çıkar (yalnızca akustikten; metin/transkript kullanılmaz).

En kritik ilke: buradaki ön işleme, eğitimde kullanılanla BİREBİR aynı olmak
zorundadır. Model hangi örnekleme hızı, hangi özellik parametreleriyle
eğitildiyse tahminde de aynıları uygulanır; bu yüzden her iki yol da ayarları
modelin yanına kaydedilmiş config'ten okur.

Örnekler
--------
    # Eğitilmiş bir CNN / wav2vec2 checkpoint'i ile:
    python scripts/predict.py --checkpoint outputs/cnn_cremad/best.pt --audio some.wav

    # Klasik (MFCC) baseline ile:
    python scripts/predict.py --baseline outputs/baseline_cremad/baseline.joblib --audio some.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Proje kökünü import arama yoluna ekle (ser paketi için).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ser.config import Config  # noqa: E402
from ser.constants import CANONICAL_EMOTIONS, NUM_CLASSES  # noqa: E402
from ser.features.io import load_audio, fix_length  # noqa: E402
from ser.features.melspec import (  # noqa: E402
    log_mel_spectrogram, fixed_num_frames, fix_frames, standardize,
)
from ser.features.mfcc import mfcc_statistics  # noqa: E402


def _print_distribution(probs: np.ndarray) -> None:
    """Olasılık dağılımını büyükten küçüğe, ASCII çubuk grafikle yazdırır.

    `np.argsort` indeksleri küçükten büyüğe sıralar; `[::-1]` ile ters
    çevirince en olası duygu başa gelir. Çubuk uzunluğu olasılıkla orantılıdır
    (30 karakter = olasılık 1.0); böylece dağılım terminalde tek bakışta okunur.
    """
    order = np.argsort(probs)[::-1]
    print(f"\nPredicted emotion: {CANONICAL_EMOTIONS[order[0]]}  "
          f"(p={probs[order[0]]:.3f})\n")
    print("Full distribution:")
    for i in order:
        bar = "#" * int(round(probs[i] * 30))
        print(f"  {CANONICAL_EMOTIONS[i]:<8} {probs[i]:6.3f} {bar}")


def predict_torch(checkpoint: str, audio: str) -> np.ndarray:
    """PyTorch checkpoint'inden (CNN veya wav2vec2) olasılık vektörü üretir."""
    # torch'u fonksiyonun içinde import ediyoruz (tembel import): yalnızca
    # baseline kullanan biri, PyTorch kurulu olmasa da bu betiği çalıştırabilsin.
    import torch
    from ser.models import build_model

    # Checkpoint yalnızca ağırlıkları değil, eğitimdeki config'i de taşır.
    # Modeli aynı mimari ve özellik ayarlarıyla yeniden kurmak için önce onu
    # okuyoruz. map_location="cpu": GPU'da eğitilmiş model CPU'da da açılsın.
    # weights_only=False: dosyada tensörlerin yanında config sözlüğü de var.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    model = build_model(cfg, NUM_CLASSES)
    model.load_state_dict(ckpt["model"])
    # eval(): dropout gibi yalnızca eğitimde aktif katmanları kapatır —
    # çıkarım (inference) modunda bunlar açık kalırsa tahmin bozulur.
    model.eval()

    # Sesi, modelin eğitildiği örnekleme hızında yükle (gerekirse yeniden örnekler).
    wav = load_audio(audio, cfg.audio.sample_rate)
    if cfg.model.name.lower() == "wav2vec2":
        # wav2vec2 ham dalga formuyla çalışır: önce sabit uzunluğa getir
        # (kırp/doldur), sonra ortalaması 0 - std'si 1 olacak şekilde normalle
        # (+1e-5, sessiz kayıtta sıfıra bölünmeyi önleyen küçük sabit).
        # [None, :] tek örneğe batch boyutu ekler -> şekil (1, örnek_sayısı).
        wav = fix_length(wav, cfg.audio.num_samples)
        wav = (wav - wav.mean()) / (wav.std() + 1e-5)
        x = torch.from_numpy(wav.astype(np.float32))[None, :]
    else:
        # CNN ise log-mel spektrogram bekler. Eğitimdekiyle aynı zincir:
        # spektrogram çıkar -> kare sayısını eğitimdeki sabit değere getir ->
        # standardize et. [None, None, :, :] -> (batch=1, kanal=1, mel, zaman);
        # 2B CNN'ler girdiyi tek kanallı "görüntü" gibi bu şekilde alır.
        spec = log_mel_spectrogram(wav, cfg.feature, cfg.audio.sample_rate)
        nf = fixed_num_frames(cfg.audio.num_samples, cfg.feature.hop_length)
        spec = standardize(fix_frames(spec, nf))
        x = torch.from_numpy(np.ascontiguousarray(spec))[None, None, :, :]

    # no_grad: sadece tahmin yapıyoruz; gradyan takibi kapatılınca hem bellek
    # hem zaman kazanılır. softmax, modelin ham skorlarını (logit) toplamı 1
    # olan olasılıklara dönüştürür; [0] batch'teki tek örneği seçer.
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].numpy()
    return probs


def predict_baseline(model_path: str, audio: str) -> np.ndarray:
    """Klasik (scikit-learn) baseline modeliyle olasılık vektörü üretir."""
    # joblib da tembel import: torch tarafını kullananlara yük olmasın.
    import joblib

    pipe = joblib.load(model_path)
    # Modelin eğitildiği özellik parametrelerini birebir kullan (modelin
    # yanına kaydedilen config.yaml'dan) — predict_torch ile aynı ilke.
    # Config dosyası yoksa yalnızca o zaman varsayılanlara geri düş.
    cfg_path = Path(model_path).parent / "config.yaml"
    cfg = Config.from_yaml(cfg_path) if cfg_path.exists() else Config()
    wav = load_audio(audio, cfg.audio.sample_rate)
    # MFCC istatistik vektörünü çıkar; [None, :] tek örneği (1, özellik) şeklinde
    # 2B matrise çevirir çünkü scikit-learn modelleri her zaman 2B girdi bekler.
    feat = mfcc_statistics(wav, cfg.feature, cfg.audio.sample_rate)[None, :]
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(feat)[0]
    # Olasılık desteği olmayan modellerde (ör. probability=False eğitilmiş SVC)
    # gerçek bir dağılım yoktur: tahmin edilen sınıfa 1.0, diğerlerine 0.0
    # vererek "one-hot" bir sahte dağılım üretiyoruz ki çıktı biçimi hep aynı kalsın.
    pred = int(pipe.predict(feat)[0])
    probs = np.zeros(NUM_CLASSES, dtype=np.float32)
    probs[pred] = 1.0
    return probs


def main():
    """Argümanlara göre doğru tahmin yolunu (torch / baseline) seçip çalıştırır."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True, help="Path to a WAV file.")
    ap.add_argument("--checkpoint", default=None, help="Trained torch checkpoint (best.pt).")
    ap.add_argument("--baseline", default=None, help="Trained baseline (baseline.joblib).")
    args = ap.parse_args()

    # Kullanıcı hatalarını erken ve okunur bir mesajla yakala: SystemExit,
    # uzun bir Python traceback'i yerine tek satırlık açıklama gösterir.
    if not Path(args.audio).exists():
        raise SystemExit(f"Audio file not found: {args.audio}")
    if args.checkpoint:
        probs = predict_torch(args.checkpoint, args.audio)
    elif args.baseline:
        probs = predict_baseline(args.baseline, args.audio)
    else:
        raise SystemExit("Provide --checkpoint <best.pt> or --baseline <baseline.joblib>.")

    _print_distribution(probs)


if __name__ == "__main__":
    main()
