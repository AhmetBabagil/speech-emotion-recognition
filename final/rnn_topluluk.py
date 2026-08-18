'''Yöntem 2 (BiGRU) için TOPLULUK: 5 jitter+kontrast BiGRU'nun softmax
çıktılarını ortalar.

CNN'de topluluk +2,5 puan kazandırdı; aynı fikri en iyi modelimize (5-koşu
ort. %66,9) uyguluyoruz. Beş üye de aynı mimari (BiGRU 32×200 ms, 192×2,
jitter+kontrast 53 boyut), farklı rastgele tohum, hepsi sıfırdan.

Dürüstlük: topluluk test'e BAKMADAN kurulur (eşit ağırlıklı ortalama). Her üye
kendi geçerlemesinde erken durur; test yalnızca en sonda bir kez ölçülür.

Örnek:
    python final/rnn_topluluk.py --uye 5
'''

from __future__ import annotations

import argparse
import json
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
from ser.evaluate import compute_metrics, report as evaluate_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uye', type=int, default=5)
    parser.add_argument('--islemler', type=int, default=14)
    parser.add_argument('--feature-workers', type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fcfg = IntervalConfig(n_intervals=32, interval_ms=200)  # jitter+kontrast (53)
    manifest = pd.read_csv('data/processed/manifest.csv')
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',))
    tr, va, te = prepare_splits(manifest, ayar, seed=42)
    folds = {'train': tr, 'val': va, 'test': te}
    cw = inverse_frequency_weights(tr['label_idx'].to_numpy(), NUM_CLASSES)

    on_cikar_paralel(pd.concat(folds.values())['path'].astype(str).tolist(),
                     dict(fcfg.__dict__), 'data/cache/final', args.islemler)
    T = _feature_folds(folds, fcfg, extract_interval_series, 'data/cache/final',
                       workers=args.feature_workers, cache={})
    tx, ty = T['train']; vx, vy = T['val']; ex, ey = T['test']
    olcek = Standardizer.fit(tx, feature_axis=2)
    tx_s, vx_s, ex_s = olcek.transform(tx), olcek.transform(vx), olcek.transform(ex)

    tekil_acc = []
    prob_toplam = None
    durumlar = []
    for k in range(args.uye):
        torch.manual_seed(k)
        torch.cuda.manual_seed_all(k)
        outcome = train_with_early_stopping(
            SeqRNN(fcfg.feature_dim, NUM_CLASSES, MODEL), tx_s, ty, vx_s, vy,
            MODEL.optim, num_classes=NUM_CLASSES, device=device, max_epochs=60, seed=k)
        _, _, prob = evaluate_arrays(outcome.model, ex_s, ey, class_weights=cw, device=device)
        acc = compute_metrics(ey, prob.argmax(axis=1))['accuracy']
        tekil_acc.append(acc)
        prob_toplam = prob if prob_toplam is None else prob_toplam + prob
        durumlar.append({n: v.detach().cpu().clone() for n, v in outcome.model.state_dict().items()})
        print(f'  üye {k}: tekil test_acc={acc:.4f}')

    ens_prob = prob_toplam / args.uye
    ens_pred = ens_prob.argmax(axis=1)

    rnn_dir = Path('final/outputs/cremad/rnn')
    tm = evaluate_report(ey, ens_pred, rnn_dir, prefix='ensemble',
                         title=f'BiGRU topluluk ({args.uye} jitter+kontrast) test')
    torch.save({'uyeler': durumlar, 'uye_sayisi': args.uye,
                'model_config': MODEL.to_dict(), 'feature_config': fcfg.__dict__,
                'standardizer_mean': olcek.mean, 'standardizer_scale': olcek.scale,
                'feature_axis': olcek.feature_axis, 'variant': 'jitter_contrast_ensemble'},
               rnn_dir / 'ensemble_model.pt')
    ozet = {
        'uye_sayisi': args.uye,
        'tekil_acc_ort': float(np.mean(tekil_acc)),
        'tekil_acc_std': float(np.std(tekil_acc)),
        'topluluk_acc': tm['accuracy'],
        'topluluk_balanced': tm['balanced_accuracy'],
        'topluluk_macro_f1': tm['macro_f1'],
        'tekil_5kosu_ort': 0.6694,
        'per_class': tm['per_class'],
    }
    (rnn_dir / 'ensemble_summary.json').write_text(
        json.dumps(ozet, indent=2, default=str), encoding='utf-8')

    print(f'\n===== BiGRU TOPLULUK ({args.uye} jitter+kontrast-BiGRU) =====')
    print(f'  tekil koşular : {np.mean(tekil_acc):.4f} ± {np.std(tekil_acc):.4f}')
    print(f'  TOPLULUK      : acc={tm["accuracy"]:.4f}  '
          f'balanced={tm["balanced_accuracy"]:.4f}  macro-F1={tm["macro_f1"]:.4f}')
    print(f'  (tek model 5-koşu ort. {0.6694:.3f})')
    print(f'  topluluk kazancı: {tm["accuracy"] - np.mean(tekil_acc):+.4f}')


if __name__ == '__main__':
    main()
