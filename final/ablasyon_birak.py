# Bırak-birini (leave-one-out) ablasyonu: her öznitelik grubunun GERÇEKTEN gerekli olup olmadığını ölçer.
#
# Kümülatif ablasyon "sırayla eklersem ne olur" der; bu ise tam setten her grubu TEK TEK çıkarıp "onsuz ne kadar kötüleşiyorum" diye bakar. Bir grubu çıkarınca doğruluk düşüyorsa o grup gerekli; değişmiyor/artıyorsa gereksiz.
#
# Tam set: MFCC(39) + skaler(5) + pitch(3) + jitter(2) + kontrast(7) = 56. (delta2 ve bant kümülatif ablasyonda zaten düştüğü için dahil değil.)
#
# Örnek:
# python final/ablasyon_birak.py --kosu 5 --islemler 14

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from final.ablasyon import MODEL, on_cikar_paralel  # noqa: E402
from final.dataset import Standardizer  # noqa: E402
from final.features import IntervalConfig, extract_interval_series  # noqa: E402
from final.models import SeqRNN  # noqa: E402
from final.pipeline import SplitSettings, _feature_folds  # noqa: E402
from final.training import (  # noqa: E402
    evaluate_arrays,
    inverse_frequency_weights,
    train_with_early_stopping,
)
from ser.constants import NUM_CLASSES  # noqa: E402
from ser.data.splits import prepare_splits  # noqa: E402
from ser.evaluate import compute_metrics  # noqa: E402
from ser.utils import get_logger  # noqa: E402

log = get_logger(__name__)

# Tam set = pitch + jitter + kontrast açık (delta2/bant kapalı — zaten kötüydü).
# Her deneme, tam setten YALNIZ birini kapatır.
TAM = dict(use_pitch=True, use_jitter=True, use_contrast=True,
           use_delta2=False, use_bandwidth=False)
GRUPLAR = [
    ('TAM SET (56)', TAM),
    ('- pitch',    {**TAM, 'use_pitch': False}),
    ('- jitter',   {**TAM, 'use_jitter': False}),
    ('- kontrast', {**TAM, 'use_contrast': False}),
]


def egit_test(fcfg, folds, cw, device, kosu, workers, islemler):
    on_cikar_paralel(pd.concat(folds.values())['path'].astype(str).tolist(),
                     dict(fcfg.__dict__), 'data/cache/final', islemler)
    T = _feature_folds(folds, fcfg, extract_interval_series, 'data/cache/final',
                       workers=workers, cache={})
    tx, ty = T['train']; vx, vy = T['val']; ex, ey = T['test']
    olcek = Standardizer.fit(tx, feature_axis=2)
    tx_s, vx_s, ex_s = olcek.transform(tx), olcek.transform(vx), olcek.transform(ex)
    accs = []
    for k in range(kosu):
        torch.manual_seed(k)
        torch.cuda.manual_seed_all(k)
        model = SeqRNN(fcfg.feature_dim, NUM_CLASSES, MODEL)
        res = train_with_early_stopping(model, tx_s, ty, vx_s, vy, MODEL.optim,
                                        num_classes=NUM_CLASSES, device=device,
                                        max_epochs=60, seed=k)
        _, _, prob = evaluate_arrays(res.model, ex_s, ey, class_weights=cw, device=device)
        accs.append(compute_metrics(ey, prob.argmax(axis=1))['accuracy'])
    return float(np.mean(accs)), float(np.std(accs))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kosu', type=int, default=5)
    parser.add_argument('--feature-workers', type=int, default=8)
    parser.add_argument('--islemler', type=int, default=14)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest = pd.read_csv('data/processed/manifest.csv')
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',))
    tr, va, te = prepare_splits(manifest, ayar, seed=42)
    folds = {'train': tr, 'val': va, 'test': te}
    cw = inverse_frequency_weights(tr['label_idx'].to_numpy(), NUM_CLASSES)

    tam_acc = None
    satirlar = []
    for ad, bayrak in GRUPLAR:
        fcfg = IntervalConfig(n_intervals=32, interval_ms=200, **bayrak)
        acc, std = egit_test(fcfg, folds, cw, device, args.kosu,
                             args.feature_workers, args.islemler)
        if tam_acc is None:
            tam_acc = acc
        etki = acc - tam_acc   # tam sete göre: negatifse "o grup gerekliydi"
        satirlar.append((ad, fcfg.feature_dim, acc, std, etki))
        log.info('[%s] boyut=%d | acc=%.4f±%.4f | tam sete göre %+.4f',
                 ad, fcfg.feature_dim, acc, std, etki)

    print('\n===== BIRAK-BİRİNİ ABLASYONU (5 koşu ort.) =====')
    print(f'{"grup":14s} {"boyut":>5s} {"test acc":>14s} {"etki":>9s}')
    for ad, boyut, acc, std, etki in satirlar:
        yorum = ''
        if ad != 'TAM SET (56)':
            yorum = '  <- gerekli' if etki < -0.003 else ('  <- gereksiz' if etki > -0.003 else '')
        print(f'{ad:14s} {boyut:5d} {acc:.4f}±{std:.4f} {etki:+8.4f}{yorum}')
    print('\nYorum: bir grubu ÇIKARINCA doğruluk çok düşüyorsa (negatif etki) o grup')
    print('gereklidir; değişmiyor/artıyorsa çıkarılabilir (gereksiz).')


if __name__ == '__main__':
    main()
