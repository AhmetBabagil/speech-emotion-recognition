'''Parametre deneme kutusu — hoca "şunu değiştir çalıştır" derse.

Aşağıdaki DEĞİŞTİR kutusundaki sayıları değiştir, betiği çalıştır: model tek
sefer eğitilir ve test doğruluğu ekrana basılır. HİÇBİR resmi dosya (kayıtlı
model, matris, JSON) bozulmaz — güvenle deneyebilirsin.

Çalıştırma:  python final/dene_hizli.py
'''

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from final.dataset import Standardizer  # noqa: E402
from final.features import IntervalConfig, extract_interval_series  # noqa: E402
from final.models import OptimSettings, RNNConfig, SeqRNN  # noqa: E402
from final.pipeline import SplitSettings, _feature_folds  # noqa: E402
from final.training import (  # noqa: E402
    evaluate_arrays, inverse_frequency_weights, train_with_early_stopping)
from ser.constants import NUM_CLASSES  # noqa: E402
from ser.data.splits import prepare_splits  # noqa: E402
from ser.evaluate import compute_metrics  # noqa: E402

# ===================== DEĞİŞTİR (sadece bu sayılar) =====================
N_INTERVALS = 32      # kayıt kaç aralığa bölünsün (ödev hiperparametresi)   [resmi: 32]
INTERVAL_MS = 200     # her aralık kaç ms (ödev hiperparametresi)            [resmi: 200]
HIDDEN      = 192     # RNN gizli birim sayısı (model boyutu)                [resmi: 192]
NUM_LAYERS  = 2       # üst üste RNN katmanı                                 [resmi: 2]
DROPOUT     = 0.3     # düzenlileştirme (0-1, büyükse daha çok)              [resmi: 0.3]
LR          = 1e-3    # öğrenme oranı                                        [resmi: 0.001]
# =======================================================================


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fcfg = IntervalConfig(n_intervals=N_INTERVALS, interval_ms=INTERVAL_MS)
    mcfg = RNNConfig(rnn_type='gru', hidden_size=HIDDEN, num_layers=NUM_LAYERS,
                     bidirectional=True, pooling='mean', dropout=DROPOUT,
                     optim=OptimSettings(batch_size=64, learning_rate=LR))
    print(f'Deneme: {N_INTERVALS} aralık × {INTERVAL_MS} ms | GRU {HIDDEN}×{NUM_LAYERS} '
          f'| dropout {DROPOUT} | lr {LR}')

    manifest = pd.read_csv('data/processed/manifest.csv')
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',))
    tr, va, te = prepare_splits(manifest, ayar, seed=42)
    folds = {'train': tr, 'val': va, 'test': te}
    cw = inverse_frequency_weights(tr['label_idx'].to_numpy(), NUM_CLASSES)

    T = _feature_folds(folds, fcfg, extract_interval_series, 'data/cache/final',
                       workers=8, cache={})
    tx, ty = T['train']; vx, vy = T['val']; ex, ey = T['test']
    olcek = Standardizer.fit(tx, feature_axis=2)
    tx_s, vx_s, ex_s = olcek.transform(tx), olcek.transform(vx), olcek.transform(ex)

    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    res = train_with_early_stopping(SeqRNN(fcfg.feature_dim, NUM_CLASSES, mcfg),
                                    tx_s, ty, vx_s, vy, mcfg.optim,
                                    num_classes=NUM_CLASSES, device=device,
                                    max_epochs=60, seed=0)
    _, _, prob = evaluate_arrays(res.model, ex_s, ey, class_weights=cw, device=device)
    m = compute_metrics(ey, prob.argmax(axis=1))
    print(f'\n>>> TEST DOĞRULUĞU: {m["accuracy"]:.4f}  (macro-F1 {m["macro_f1"]:.4f})')
    print('    Kıyas: resmi tek model ~0.669, topluluk 0.686')


if __name__ == '__main__':
    main()
