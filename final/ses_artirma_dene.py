'''Ses-uzayı (dalga formu) veri artırma denemesi — Yöntem 2 (BiGRU) için.

Şimdiye kadarki tüm artırmalar spektrogram/öznitelik uzayındaydı (yalnızca
gizleme). Bu deney dalga formunun KENDİSİNİ çeşitlendirir: her eğitim kaydının
pitch-shift edilmiş ve hafif gürültü eklenmiş kopyalarını üreterek eğitim
setine gerçekten yeni varyasyon ekler.

Dürüst protokol: artırma YALNIZCA eğitim kayıtlarına uygulanır; geçerleme ve
test orijinal hâlleriyle kalır. Hazır/önceden eğitilmiş model yok — sadece
librosa ile sinyal işleme (yönerge izinli).

ÖNEMLİ (Windows): torch import'ları main() İÇİNE ertelenmiştir. Böylece
ProcessPoolExecutor işçileri (spawn ile bu modülü yeniden import eder) torch/CUDA
yüklemez; yoksa işçiler CUDA çakışmasından takılıyor. Modül-üstü importlar sadece
numpy + librosa + features (hafif) olmalı.

Örnek:
    python final/ses_artirma_dene.py --kosu 5 --turler pitch noise --islemler 14
'''

from __future__ import annotations

# BLAS/numba iş parçacıklarını numpy/librosa import'undan ÖNCE 1'e sabitle.
import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# HAFİF importlar (işçiler bunları yükler; torch YOK):
import numpy as np  # noqa: E402
from final.features import IntervalConfig, extract_interval_series, _load_audio  # noqa: E402


def _isci_kur() -> None:
    '''Her işçide BLAS iş parçacıklarını 1'e sabitle.'''
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[var] = '1'


def _dalga_artir(audio: np.ndarray, sr: int, tur: str, seed: int) -> np.ndarray:
    '''Dalga formuna tek bir artırma uygular (deterministik, seed'e bağlı).'''
    import librosa
    rng = np.random.default_rng(seed)
    if tur == 'pitch':
        adim = float(rng.choice([-3, -2, -1, 1, 2, 3]))   # ±1..3 yarım ton
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=adim)
    if tur == 'noise':
        rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-8
        snr_db = float(rng.uniform(15, 30))               # 15-30 dB SNR
        gurultu_rms = rms / (10 ** (snr_db / 20))
        return audio + rng.normal(0, gurultu_rms, size=audio.shape).astype(audio.dtype)
    return audio


def _isci(arg):
    '''(yol, etiket, cfg_dict, tur, seed) -> (seri, etiket). torch kullanmaz.'''
    yol, etiket, cfg_dict, tur, seed = arg
    cfg = IntervalConfig(**cfg_dict)
    audio = _load_audio(yol, cfg.sample_rate)
    if tur != 'orig':
        audio = _dalga_artir(audio, cfg.sample_rate, tur, seed).astype(np.float32)
    seri = extract_interval_series(yol, cfg, audio=audio)
    return seri, etiket


def artirilmis_egitim(train_df, cfg, turler, islemler):
    '''Orijinal + artırılmış eğitim serilerini paralel üretir -> (X, y).'''
    isler = []
    for i, row in enumerate(train_df.itertuples()):
        yol = str(row.path); etiket = int(row.label_idx); cd = dict(cfg.__dict__)
        isler.append((yol, etiket, cd, 'orig', 0))
        for j, tur in enumerate(turler):
            isler.append((yol, etiket, cd, tur, i * 31 + j * 7 + 1))
    X, y = [], []
    if islemler <= 1:
        # Tek-süreç: numba+multiprocessing deadlock'unu tamamen atlar (yavaş ama sağlam).
        for k, is_ in enumerate(isler):
            seri, etiket = _isci(is_)
            X.append(seri); y.append(etiket)
            if k % 1000 == 0:
                print(f'    ...{k}/{len(isler)} çıkarıldı', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=islemler, initializer=_isci_kur) as ex:
            for seri, etiket in ex.map(_isci, isler, chunksize=16):
                X.append(seri); y.append(etiket)
    return np.stack(X), np.array(y, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kosu', type=int, default=5)
    parser.add_argument('--turler', nargs='+', default=['pitch', 'noise'],
                        choices=['pitch', 'noise'])
    parser.add_argument('--islemler', type=int, default=14)
    args = parser.parse_args()

    # AĞIR importlar SADECE burada (işçiler bu koda hiç girmez -> torch yüklemez).
    import pandas as pd
    import torch
    from final.ablasyon import MODEL
    from final.dataset import Standardizer
    from final.models import SeqRNN
    from final.pipeline import SplitSettings, _feature_folds
    from final.training import (
        evaluate_arrays, inverse_frequency_weights, train_with_early_stopping)
    from ser.constants import NUM_CLASSES
    from ser.data.splits import prepare_splits
    from ser.evaluate import compute_metrics

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = IntervalConfig(n_intervals=32, interval_ms=200)   # jitter+kontrast (53)
    manifest = pd.read_csv('data/processed/manifest.csv')
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',))
    tr, va, te = prepare_splits(manifest, ayar, seed=42)
    cw = inverse_frequency_weights(tr['label_idx'].to_numpy(), NUM_CLASSES)

    # Val/test: ORİJİNAL öznitelikler (önbellekten). Asla artırılmaz.
    T = _feature_folds({'val': va, 'test': te}, cfg, extract_interval_series,
                       'data/cache/final', workers=8, cache={})
    vx, vy = T['val']; ex, ey = T['test']

    print(f'Eğitim artırılıyor: {len(tr)} × (1 + {len(args.turler)}) = '
          f'{len(tr) * (1 + len(args.turler))} seri (paralel, {args.islemler} işçi)...',
          flush=True)
    tx, ty = artirilmis_egitim(tr, cfg, args.turler, args.islemler)
    print(f'Genişletilmiş eğitim: {tx.shape}', flush=True)

    olcek = Standardizer.fit(tx, feature_axis=2)
    tx_s, vx_s, ex_s = olcek.transform(tx), olcek.transform(vx), olcek.transform(ex)

    accs, f1s = [], []
    for k in range(args.kosu):
        torch.manual_seed(k)
        torch.cuda.manual_seed_all(k)
        outcome = train_with_early_stopping(
            SeqRNN(cfg.feature_dim, NUM_CLASSES, MODEL), tx_s, ty, vx_s, vy,
            MODEL.optim, num_classes=NUM_CLASSES, device=device, max_epochs=60, seed=k)
        _, _, prob = evaluate_arrays(outcome.model, ex_s, ey, class_weights=cw, device=device)
        m = compute_metrics(ey, prob.argmax(axis=1))
        accs.append(m['accuracy']); f1s.append(m['macro_f1'])
        print(f'  koşu {k}: test_acc={m["accuracy"]:.4f}', flush=True)

    import json
    ort = {'turler': args.turler, 'kosu': args.kosu,
           'test_acc_ort': float(np.mean(accs)), 'test_acc_std': float(np.std(accs)),
           'macro_f1_ort': float(np.mean(f1s)), 'artirmasiz_taban': 0.6694}
    Path('final/outputs/cremad/rnn').mkdir(parents=True, exist_ok=True)
    Path('final/outputs/cremad/rnn/ses_artirma_deney.json').write_text(
        json.dumps(ort, indent=2), encoding='utf-8')

    print(f'\n===== SES ARTIRMA ({"+".join(args.turler)}) — {args.kosu} koşu =====', flush=True)
    print(f'  test acc = {np.mean(accs):.4f} ± {np.std(accs):.4f}')
    print(f'  macro-F1 = {np.mean(f1s):.4f}')
    print(f'  KIYAS: artırmasız tek model 5-koşu ort. = 0.6694')
    fark = float(np.mean(accs)) - 0.6694
    sonuc = 'İŞE YARADI' if fark > 0.003 else ('nötr/gürültü' if abs(fark) <= 0.003 else 'ZARAR VERDİ')
    print(f'  fark: {fark:+.4f}  ->  {sonuc}')


if __name__ == '__main__':
    main()
