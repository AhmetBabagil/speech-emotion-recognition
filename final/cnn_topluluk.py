'''Yöntem 1 (CNN) için TOPLULUK (ensemble): 5 SpecAugment-CNN'in softmax
çıktılarını ortalar.

Neden: tek CNN koşusunun test varyansı ±3,9 puan (çok kararsız). Topluluk,
bağımsız koşuların hatalarını birbirine sönümleterek hem varyansı düşürür hem
de tipik olarak tek koşuları geçer. Hazır/önceden eğitilmiş model YOK — beş
model de sıfırdan, aynı mimari, farklı rastgele tohum.

Dürüstlük: topluluk, test'e BAKMADAN kurulur (eşit ağırlıklı ortalama). Her
model kendi geçerlemesinde erken durur; test yalnızca en sonda, bir kez ölçülür.

Örnek:
    python final/cnn_topluluk.py --uye 5
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

from final.augment import SpecMask  # noqa: E402
from final.dataset import Standardizer  # noqa: E402
from final.features import MelImageConfig, extract_mel_image  # noqa: E402
from final.models import CNNConfig, MelCNN, OptimSettings  # noqa: E402
from final.pipeline import SplitSettings, _feature_folds  # noqa: E402
from final.training import (  # noqa: E402
    evaluate_arrays,
    inverse_frequency_weights,
    train_with_early_stopping,
)
from ser.constants import NUM_CLASSES  # noqa: E402
from ser.data.splits import prepare_splits  # noqa: E402
from ser.evaluate import compute_metrics, report as evaluate_report  # noqa: E402

OPT = OptimSettings(batch_size=16, learning_rate=1e-3, weight_decay=1e-4, patience=8)
CFG = CNNConfig(channels=(32, 64, 128), dropout=0.3, optim=OPT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uye', type=int, default=5, help='topluluk üye sayısı')
    parser.add_argument('--feature-workers', type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest = pd.read_csv('data/processed/manifest.csv')
    ayar = SplitSettings(train_corpora=('cremad',), eval_corpora=('cremad',))
    tr, va, te = prepare_splits(manifest, ayar, seed=42)
    folds = {'train': tr, 'val': va, 'test': te}
    cw = inverse_frequency_weights(tr['label_idx'].to_numpy(), NUM_CLASSES)

    fcfg = MelImageConfig()
    T = _feature_folds(folds, fcfg, extract_mel_image, 'data/cache/final',
                       workers=args.feature_workers, cache={})
    tx, ty = T['train']; vx, vy = T['val']; ex, ey = T['test']
    olcek = Standardizer.fit(tx, feature_axis=1)
    tx_s, vx_s, ex_s = olcek.transform(tx), olcek.transform(vx), olcek.transform(ex)

    spec = SpecMask()
    tekil_acc = []
    prob_toplam = None
    durumlar = []
    for k in range(args.uye):
        outcome = train_with_early_stopping(
            MelCNN(NUM_CLASSES, CFG), tx_s, ty, vx_s, vy, CFG.optim,
            num_classes=NUM_CLASSES, device=device, max_epochs=60, seed=k,
            amp=True, train_transform=spec)
        _, _, prob = evaluate_arrays(outcome.model, ex_s, ey, class_weights=cw, device=device)
        acc = compute_metrics(ey, prob.argmax(axis=1))['accuracy']
        tekil_acc.append(acc)
        prob_toplam = prob if prob_toplam is None else prob_toplam + prob
        durumlar.append({n: v.detach().cpu().clone() for n, v in outcome.model.state_dict().items()})
        print(f'  üye {k}: tekil test_acc={acc:.4f}')

    # TOPLULUK: 5 softmax'ın ortalaması -> tek tahmin.
    ens_prob = prob_toplam / args.uye
    ens_pred = ens_prob.argmax(axis=1)

    cnn_dir = Path('final/outputs/cremad/cnn')
    tm = evaluate_report(ey, ens_pred, cnn_dir, prefix='ensemble',
                         title=f'CNN topluluk ({args.uye} SpecAugment) test')
    torch.save({'uyeler': durumlar, 'uye_sayisi': args.uye,
                'model_config': CFG.to_dict(), 'feature_config': fcfg.__dict__,
                'standardizer_mean': olcek.mean, 'standardizer_scale': olcek.scale,
                'feature_axis': olcek.feature_axis, 'variant': 'specaugment_ensemble'},
               cnn_dir / 'ensemble_model.pt')
    ozet = {
        'uye_sayisi': args.uye,
        'tekil_acc_ort': float(np.mean(tekil_acc)),
        'tekil_acc_std': float(np.std(tekil_acc)),
        'topluluk_acc': tm['accuracy'],
        'topluluk_balanced': tm['balanced_accuracy'],
        'topluluk_macro_f1': tm['macro_f1'],
        'taban_5kosu_ort': 0.5585,
        'specaug_5kosu_ort': 0.5989,
        'eski_kazanan_tek_kosu': 0.6264,
        'per_class': tm['per_class'],
    }
    (cnn_dir / 'ensemble_summary.json').write_text(
        json.dumps(ozet, indent=2, default=str), encoding='utf-8')

    print(f'\n===== CNN TOPLULUK ({args.uye} SpecAugment-CNN) =====')
    print(f'  tekil koşular   : {np.mean(tekil_acc):.4f} ± {np.std(tekil_acc):.4f}')
    print(f'  TOPLULUK        : acc={tm["accuracy"]:.4f}  '
          f'balanced={tm["balanced_accuracy"]:.4f}  macro-F1={tm["macro_f1"]:.4f}')
    print(f'  (taban 5-koşu {0.5585:.3f} | specaug 5-koşu {0.5989:.3f} | '
          f'eski kazanan tek koşu {0.6264:.3f})')
    kazanc = tm['accuracy'] - np.mean(tekil_acc)
    print(f'  topluluk kazancı (tekil ort. üzerine): {kazanc:+.4f}')


if __name__ == '__main__':
    main()
