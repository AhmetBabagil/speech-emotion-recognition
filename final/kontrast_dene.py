'''En yalın aday: SADECE spektral kontrast (pitch+jitter olmadan) = 51 boyut.

Bırak-birini ablasyonu pitch ve jitter'ın kontrastla birlikteyken fazlalık
olduğunu gösterdi. Bu betik "taban + yalnız kontrast" setini 5 koşu eğitip,
bu en yalın kombinasyonun en iyisi olup olmadığını test eder.
'''

from __future__ import annotations

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


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest = pd.read_csv('data/processed/manifest.csv')
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',))
    tr, va, te = prepare_splits(manifest, ayar, seed=42)
    folds = {'train': tr, 'val': va, 'test': te}
    cw = inverse_frequency_weights(tr['label_idx'].to_numpy(), NUM_CLASSES)

    # SADECE kontrast: pitch=False, jitter=False, contrast=True.
    fcfg = IntervalConfig(n_intervals=32, interval_ms=200,
                          use_pitch=False, use_jitter=False, use_contrast=True,
                          use_delta2=False, use_bandwidth=False)
    print(f'Sadece-kontrast set: boyut={fcfg.feature_dim}')

    on_cikar_paralel(pd.concat(folds.values())['path'].astype(str).tolist(),
                     dict(fcfg.__dict__), 'data/cache/final', 14)
    T = _feature_folds(folds, fcfg, extract_interval_series, 'data/cache/final',
                       workers=8, cache={})
    tx, ty = T['train']; vx, vy = T['val']; ex, ey = T['test']
    olcek = Standardizer.fit(tx, feature_axis=2)
    tx_s, vx_s, ex_s = olcek.transform(tx), olcek.transform(vx), olcek.transform(ex)

    accs, f1s = [], []
    for k in range(5):
        torch.manual_seed(k)
        torch.cuda.manual_seed_all(k)
        model = SeqRNN(fcfg.feature_dim, NUM_CLASSES, MODEL)
        res = train_with_early_stopping(model, tx_s, ty, vx_s, vy, MODEL.optim,
                                        num_classes=NUM_CLASSES, device=device,
                                        max_epochs=60, seed=k)
        _, _, prob = evaluate_arrays(res.model, ex_s, ey, class_weights=cw, device=device)
        m = compute_metrics(ey, prob.argmax(axis=1))
        accs.append(m['accuracy'])
        f1s.append(m['macro_f1'])

    print(f'\n===== SADECE KONTRAST ({fcfg.feature_dim} boyut) =====')
    print(f'  test acc = {np.mean(accs):.4f} ± {np.std(accs):.4f}')
    print(f'  macro-F1 = {np.mean(f1s):.4f}')
    print('\nKıyas (5 koşu ort.):')
    print('  taban (44)            = 0.6455')
    print('  pitch+jitter+kontrast (56) = 0.6636')
    print('  - pitch (53)          = 0.6694')
    print(f'  SADECE kontrast ({fcfg.feature_dim})    = {np.mean(accs):.4f}  <-- bu koşu')


if __name__ == '__main__':
    main()
